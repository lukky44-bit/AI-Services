import os
import re
import math
import logging
from dataclasses import dataclass
from typing import Optional
import ssl

# Bypass SSL context verification to resolve environment SSL issues when downloading HuggingFace models
ssl._create_default_https_context = ssl._create_unverified_context

import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder

from .config import Config
from .utils import RetrievedChunk, tokenize, format_sources, rrf_fuse, build_context

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.WARNING, format="%(levelname)s  %(message)s")

# Config

# Must match the embedding model used during ingestion in ingestion_code.py
EMBED_MODEL        = "all-MiniLM-L6-v2"
PROSE_COLLECTION   = "k6_prose"
CODE_COLLECTION    = "k6_code"
RERANKER_MODEL     = "cross-encoder/ms-marco-MiniLM-L-6-v2"
VECTOR_TOP_K       = 15
BM25_TOP_K         = 10 
RERANK_TOP_K       = 8
MAX_CONTEXT_TOKENS = 3000

# Shared model caches at module level to guarantee they are loaded exactly ONCE per process
_EMBED_FN = None
_RERANKER = None
_CHROMA_CLIENT = None
_BM25_CORPUS = None
_BM25_INDEX = None


class K6Retriever:

    def __init__(
        self,
        chroma_path: str = None,
        groq_api_key: str = "",
        reranker_model: str = RERANKER_MODEL,
    ):
        global _EMBED_FN, _RERANKER, _CHROMA_CLIENT, _BM25_CORPUS, _BM25_INDEX
        
        chroma_path = chroma_path or Config.CHROMA_DB_PATH

        # Load SentenceTransformer embedding model exactly ONCE per process
        if _EMBED_FN is None:
            log.info(f"Loading SentenceTransformer embedding model: {EMBED_MODEL}")
            _EMBED_FN = SentenceTransformerEmbeddingFunction(model_name=EMBED_MODEL)

        # Initialize Chroma PersistentClient exactly ONCE per process
        if _CHROMA_CLIENT is None:
            log.info(f"Connecting to Chroma persistent client at: {chroma_path}")
            _CHROMA_CLIENT = chromadb.PersistentClient(path=chroma_path)

        self.prose_col = _CHROMA_CLIENT.get_or_create_collection(PROSE_COLLECTION, embedding_function=_EMBED_FN)
        self.code_col  = _CHROMA_CLIENT.get_or_create_collection(CODE_COLLECTION,  embedding_function=_EMBED_FN)

        # Load CrossEncoder reranker model exactly ONCE per process
        if _RERANKER is None:
            log.info(f"Loading CrossEncoder reranker model: {reranker_model}")
            _RERANKER = CrossEncoder(reranker_model)
            
        self.reranker = _RERANKER

        # Build/load BM25 index exactly ONCE per process
        if _BM25_INDEX is None:
            log.info(f"Loaded collections: {self.prose_col.count()} prose, {self.code_col.count()} code chunks")
            self._bm25_corpus = []  # type: list[RetrievedChunk]
            self._bm25_index = None  # type: Optional[BM25Okapi]
            self._build_bm25_index()
            _BM25_CORPUS = self._bm25_corpus
            _BM25_INDEX = self._bm25_index
        else:
            self._bm25_corpus = _BM25_CORPUS
            self._bm25_index = _BM25_INDEX

    # BM25 index

    def _build_bm25_index(self):
        """Load all chunks from Chroma and build a BM25 index in memory."""
        log.info("Building BM25 index...")
        all_chunks = []

        for col in (self.prose_col, self.code_col):
            results = col.get(include=["documents", "metadatas"])
            for doc_id, doc, meta in zip(results["ids"], results["documents"], results["metadatas"]):
                all_chunks.append(RetrievedChunk(
                    chunk_id=doc_id,
                    text=doc,
                    url=meta.get("url", ""),
                    heading_path=meta.get("heading_path", ""),
                    content_type=meta.get("content_type", "prose"),
                    language=meta.get("language", ""),
                ))

        self._bm25_corpus = all_chunks
        if all_chunks:
            tokenized = [tokenize(c.text) for c in all_chunks]
            self._bm25_index = BM25Okapi(tokenized)
            log.info(f"BM25 index built: {len(all_chunks)} total chunks")
        else:
            self._bm25_index = None
            log.info("BM25 index skipped: no chunks found in Chroma database yet")

    # Vector search

    def _vector_search(self, query_text: str, n_results: int) -> list[RetrievedChunk]:
        """Search both collections equally to ensure robust candidates for both types."""
        results: list[RetrievedChunk] = []

        # Prose + table collection
        if self.prose_col.count() > 0:
            pr = self.prose_col.query(
                query_texts=[query_text],
                n_results=min(n_results, self.prose_col.count()),
                include=["documents", "metadatas", "distances"],
            )
            for doc_id, doc, meta, dist in zip(
                pr["ids"][0], pr["documents"][0], pr["metadatas"][0], pr["distances"][0]
            ):
                results.append(RetrievedChunk(
                    chunk_id=doc_id, text=doc,
                    url=meta.get("url",""), heading_path=meta.get("heading_path",""),
                    content_type=meta.get("content_type","prose"), language=meta.get("language",""),
                    score=1.0 - dist,  # cosine distance → similarity
                ))

        # Code collection
        if self.code_col.count() > 0:
            cr = self.code_col.query(
                query_texts=[query_text],
                n_results=min(n_results, self.code_col.count()),
                include=["documents", "metadatas", "distances"],
            )
            for doc_id, doc, meta, dist in zip(
                cr["ids"][0], cr["documents"][0], cr["metadatas"][0], cr["distances"][0]
            ):
                results.append(RetrievedChunk(
                    chunk_id=doc_id, text=doc,
                    url=meta.get("url",""), heading_path=meta.get("heading_path",""),
                    content_type="code", language=meta.get("language",""),
                    score=1.0 - dist,
                ))

        return results

    # BM25 search

    def _bm25_search(self, query: str, n_results: int) -> list[RetrievedChunk]:
        if not self._bm25_index:
            return []
        tokens = tokenize(query)
        scores = self._bm25_index.get_scores(tokens)
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:n_results]
        results = []
        for idx in top_indices:
            chunk = self._bm25_corpus[idx]
            chunk.score = float(scores[idx])
            results.append(chunk)
        return results

    # Cross-encoder reranking

    def _rerank(self, query: str, candidates: list[RetrievedChunk], top_k: int) -> list[RetrievedChunk]:
        """Re-score candidates with a cross-encoder; return top_k."""
        if not candidates:
            return []
        pairs = [[query, c.text] for c in candidates]
        scores = self.reranker.predict(pairs)
        for chunk, score in zip(candidates, scores):
            chunk.score = float(score)
        ranked = sorted(candidates, key=lambda c: c.score, reverse=True)
        return ranked[:top_k]

    # Main retrieve 

    def retrieve(
        self,
        query: str,
        top_k: int = RERANK_TOP_K,
        hyde_doc: str = "",
    ) -> list[RetrievedChunk]:
        """
        Full retrieval pipeline:
          query → vector search (with optional pre-computed HyDE) + BM25 → RRF fusion → cross-encoder rerank
        
        Args:
            query: The user's search query
            top_k: Number of final chunks to return
            hyde_doc: Pre-computed HyDE hypothetical document (from merged rephrase+hyde call).
                      If provided, used for vector search; if empty, uses raw query only.
        """
        # 1. Build search text using pre-computed HyDE doc if available
        if hyde_doc:
            search_text = f"{query}\n\n{hyde_doc}"
            log.info(f"Using pre-computed HyDE doc ({len(hyde_doc)} chars)")
        else:
            search_text = query
            log.info("No HyDE doc provided, using raw query for vector search")

        # 2. Vector search (query both collections equally for VECTOR_TOP_K)
        vector_results = self._vector_search(search_text, VECTOR_TOP_K)
        log.info(f"Vector search returned {len(vector_results)} candidates")

        # 3. BM25 search (on original query for exact-term precision)
        bm25_results = self._bm25_search(query, BM25_TOP_K)
        log.info(f"BM25 search returned {len(bm25_results)} candidates")

        # 4. RRF fusion (contains all candidates sorted by hybrid RRF score)
        fused = rrf_fuse(vector_results, bm25_results)
        log.info(f"After RRF fusion: {len(fused)} unique candidates")

        # Separate candidates into Prose, Table, and Code
        fused_prose = [c for c in fused if c.content_type == "prose"]
        fused_table = [c for c in fused if c.content_type == "table"]
        fused_code  = [c for c in fused if c.content_type == "code"]

        # Rerank ONLY prose using the Cross-Encoder (where deep sentence semantics are essential)
        log.info(f"Reranking {len(fused_prose[:15])} prose candidates with cross-encoder...")
        reranked_prose = self._rerank(query, fused_prose[:15], top_k=15)

        final_selection = []
        final_selection.extend(reranked_prose[:3])
        final_selection.extend(fused_table[:1])
        final_selection.extend(fused_code[:2])

        # If we need more chunks to hit top_k, fill with remaining candidates from the main fused pool in RRF order
        if len(final_selection) < top_k:
            remaining = [c for c in fused if c not in final_selection]
            final_selection.extend(remaining[:(top_k - len(final_selection))])

        # Maintain original fused RRF rank order for final context assembly
        final_selection.sort(key=lambda c: fused.index(c))
        reranked = final_selection[:top_k]

        log.info(f"After reranking (balanced hybrid with RRF code bypass): {len(reranked)} final chunks")
        return reranked




