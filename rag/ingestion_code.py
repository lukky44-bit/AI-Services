"""
k6 Documentation Ingestion Pipeline → ChromaDB (Production-Grade)
================================================================
Strategy:
  - Header-aware chunking  : never splits mid-section (H1→H2→H3 breadcrumb preserved)
  - Code-block awareness   : fenced code blocks are kept atomic, never split
  - Table awareness        : markdown tables kept as single chunks
  - Dual collection store  : prose/tables → "k6_prose", code blocks → "k6_code"
  - Rich metadata          : url, heading_path, content_type, token_count, h1/h2/h3
  - Embedding              : local SentenceTransformer embedding function matching retriever
"""

import re
import sys
import os
import time
import hashlib
import logging
from dataclasses import dataclass
from typing import Optional
import requests
import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from tqdm import tqdm

# Add parent directory of rag_agent to PYTHONPATH to support running directly or as a module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rag_agent.config import Config

# ─── Config ──────────────────────────────────────────────────────────────────

PROSE_COLLECTION   = "k6_prose"   # prose + table chunks
CODE_COLLECTION    = "k6_code"    # fenced code blocks
TARGET_CHUNK_TOKENS = 512          # soft target; code blocks/tables may exceed
CHUNK_OVERLAP_CHARS = 200          # character overlap between prose chunks
MAX_RETRIES         = 3
RETRY_DELAY         = 2.0          # seconds between retries
USER_AGENT          = "k6-rag-ingester/1.0"

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)

# ─── Data structures ─────────────────────────────────────────────────────────

@dataclass
class Chunk:
    text: str
    url: str
    content_type: str 
    h1: str = ""
    h2: str = ""
    h3: str = ""
    language: str = ""
    doc_title: str = ""
    doc_description: str = ""
    token_count: int = 0

    @property
    def heading_path(self) -> str:
        parts = [p for p in [self.h1, self.h2, self.h3] if p]
        return " > ".join(parts)

    @property
    def chunk_id(self) -> str:
        """Stable deterministic ID so re-ingestion is idempotent."""
        digest = hashlib.sha256(f"{self.url}::{self.text}".encode()).hexdigest()[:16]
        return digest

    def get_searchable_text(self) -> str:
        """Format the chunk text with metadata context for dense/sparse retrieval."""
        title = self.doc_title or self.h1 or ""
        path = self.heading_path
        
        context_parts = []
        if title:
            context_parts.append(f"Document: {title}")
        if path:
            context_parts.append(f"Section: {path}")
            
        context_header = " | ".join(context_parts)
        
        if self.content_type == "code":
            # For code, wrap the context in a comment block
            return f"// {context_header}\n{self.text}"
        else:
            # For prose and tables, prepend standard text headers
            return f"[{context_header}]\n{self.text}"

    def to_metadata(self) -> dict:
        return {
            "url":          self.url,
            "content_type": self.content_type,
            "h1":           self.h1,
            "h2":           self.h2,
            "h3":           self.h3,
            "heading_path": self.heading_path,
            "language":     self.language,
            "doc_title":    self.doc_title,
            "doc_description": self.doc_description,
            "token_count":  self.token_count,
        }

# ─── Tokenizer ───────────────────────────────────────────────────────────────

def token_count(text: str) -> int:
    return len(text.split())

# ─── Markdown fetcher ────────────────────────────────────────────────────────

def fetch_markdown(url: str) -> Optional[str]:
    for attempt in range(MAX_RETRIES):
        try:
            r = requests.get(url, timeout=15, headers={"User-Agent": USER_AGENT})
            r.raise_for_status()
            return r.text
        except Exception as e:
            if attempt < MAX_RETRIES - 1:
                log.warning(f"Retry {attempt+1} for {url}: {e}")
                time.sleep(RETRY_DELAY)
            else:
                log.error(f"Failed to fetch {url}: {e}")
                return None

# ─── Markdown pre-cleaner ────────────────────────────────────────────────────

_FRONT_MATTER_RE = re.compile(r"^---\s*\n.*?\n---\s*\n", re.DOTALL)
_IMPORT_RE       = re.compile(r"^import\s+.*?from\s+['\"].*?['\"].*$", re.MULTILINE)
_EDIT_LINK_RE    = re.compile(r"\[.*?edit.*?\]\(.*?\)", re.IGNORECASE)
_HTML_COMMENT_RE = re.compile(r"<!--.*?-->", re.DOTALL)

def parse_frontmatter(raw: str) -> tuple[str, str]:
    """Extract title and description from frontmatter if present."""
    title, desc = "", ""
    m = _FRONT_MATTER_RE.match(raw)
    if m:
        fm_text = m.group(0).replace("---", "").strip()
        for line in fm_text.split("\n"):
            if ":" in line:
                key, val = line.split(":", 1)
                key = key.strip().lower()
                val = val.strip().strip('"').strip("'")
                if key == "title":
                    title = val
                elif key == "description":
                    desc = val
    return title, desc

def clean_markdown(raw: str) -> str:
    """Remove front matter, import lines, HTML comments."""
    text = _FRONT_MATTER_RE.sub("", raw, count=1)
    text = _HTML_COMMENT_RE.sub("", text)
    text = _IMPORT_RE.sub("", text)
    text = _EDIT_LINK_RE.sub("", text)
    return text.strip()

# ─── Header-aware + code-aware chunker ───────────────────────────────────────

_CODE_FENCE_RE  = re.compile(r"```(\w*)\n(.*?)```", re.DOTALL)
_TABLE_ROW_RE   = re.compile(r"^\|.+\|$", re.MULTILINE)
_HEADER_RE      = re.compile(r"^(#{1,3})\s+(.+)$", re.MULTILINE)

class MarkdownChunker:
    """
    Splits markdown into typed chunks while preserving code fences, tables, and sections.
    """
    def __init__(self, target_tokens: int = TARGET_CHUNK_TOKENS, overlap_chars: int = CHUNK_OVERLAP_CHARS):
        self.target_tokens = target_tokens
        self.overlap_chars = overlap_chars

    def chunk(self, text: str, url: str, doc_title: str = "", doc_description: str = "") -> list[Chunk]:
        chunks: list[Chunk] = []
        current_h = {"h1": "", "h2": "", "h3": ""}

        segments = self._segment(text)

        for seg_type, seg_text, headings in segments:
            current_h.update(headings)

            if seg_type == "code":
                lang = seg_text.split("\n")[0].replace("```", "").strip()
                c = Chunk(
                    text=seg_text,
                    url=url,
                    content_type="code",
                    language=lang,
                    doc_title=doc_title,
                    doc_description=doc_description,
                    **current_h,
                )
                c.token_count = token_count(c.text)
                chunks.append(c)

            elif seg_type == "table":
                c = Chunk(
                    text=seg_text,
                    url=url,
                    content_type="table",
                    doc_title=doc_title,
                    doc_description=doc_description,
                    **current_h,
                )
                c.token_count = token_count(c.text)
                chunks.append(c)

            else:  # prose
                prose_chunks = self._split_prose(seg_text, url, current_h.copy(), doc_title=doc_title, doc_description=doc_description)
                chunks.extend(prose_chunks)

        return [c for c in chunks if c.text.strip()]

    def _segment(self, text: str) -> list[tuple[str, str, dict]]:
        """Walk the document, yielding (type, content, heading_delta) segments."""
        segments = []
        current_h = {"h1": "", "h2": "", "h3": ""}

        # Collect code fence positions
        code_spans = []
        for m in _CODE_FENCE_RE.finditer(text):
            code_spans.append((m.start(), m.end(), m.group(1), m.group(2)))

        # Collect table positions
        table_spans = []
        lines = text.split("\n")
        line_offsets = []
        off = 0
        for line in lines:
            line_offsets.append(off)
            off += len(line) + 1

        i = 0
        while i < len(lines):
            if _TABLE_ROW_RE.match(lines[i]):
                start_line = i
                while i < len(lines) and (lines[i].startswith("|") or lines[i].strip() == ""):
                    i += 1
                end_line = i
                ts = line_offsets[start_line]
                te = line_offsets[end_line - 1] + len(lines[end_line - 1])
                table_spans.append((ts, te, "\n".join(lines[start_line:end_line]).strip()))
            else:
                i += 1

        # Merge special spans and sort by position
        special = []
        for (s, e, lang, body) in code_spans:
            full = f"```{lang}\n{body}```" if lang else f"```\n{body}```"
            special.append((s, e, "code", full))
        for (s, e, content) in table_spans:
            special.append((s, e, "table", content))

        # Remove overlapping spans (code wins over table)
        special.sort(key=lambda x: x[0])
        filtered = []
        last_end = 0
        for item in special:
            if item[0] >= last_end:
                filtered.append(item)
                last_end = item[1]
        special = filtered

        # Walk through, extracting prose between special spans
        pos = 0
        for (s, e, stype, content) in special:
            if pos < s:
                prose = text[pos:s]
                h_delta, prose_no_headers = self._extract_headings(prose, current_h)
                current_h.update(h_delta)
                if prose_no_headers.strip():
                    segments.append(("prose", prose_no_headers.strip(), current_h.copy()))
            segments.append((stype, content, current_h.copy()))
            pos = e

        if pos < len(text):
            prose = text[pos:]
            h_delta, prose_no_headers = self._extract_headings(prose, current_h)
            current_h.update(h_delta)
            if prose_no_headers.strip():
                segments.append(("prose", prose_no_headers.strip(), current_h.copy()))

        return segments

    def _extract_headings(self, text: str, current_h: dict) -> tuple[dict, str]:
        """Extract heading lines, update heading state, return (delta, text_without_headings)."""
        delta = {}
        lines_out = []
        for line in text.split("\n"):
            m = _HEADER_RE.match(line)
            if m:
                level = len(m.group(1))
                title = m.group(2).strip()
                if level == 1:
                    delta["h1"] = title
                    delta["h2"] = ""
                    delta["h3"] = ""
                elif level == 2:
                    delta["h2"] = title
                    delta["h3"] = ""
                elif level == 3:
                    delta["h3"] = title
            else:
                lines_out.append(line)
        return delta, "\n".join(lines_out)

    def _split_prose(self, text: str, url: str, headings: dict, doc_title: str = "", doc_description: str = "") -> list[Chunk]:
        """Split long prose text into overlapping token-bounded chunks."""
        if token_count(text) <= self.target_tokens:
            c = Chunk(text=text.strip(), url=url, content_type="prose", doc_title=doc_title, doc_description=doc_description, **headings)
            c.token_count = token_count(c.text)
            return [c]

        sentences = re.split(r"(?<=\.)\s+|\n\n+", text)
        chunks = []
        buffer = []

        for sent in sentences:
            sent = sent.strip()
            if not sent:
                continue
            buffer.append(sent)
            if token_count(" ".join(buffer)) >= self.target_tokens:
                chunk_text = " ".join(buffer)
                c = Chunk(text=chunk_text.strip(), url=url, content_type="prose", doc_title=doc_title, doc_description=doc_description, **headings)
                c.token_count = token_count(c.text)
                chunks.append(c)
                # Keep last ~overlap_chars worth of sentences
                overlap_buf = []
                overlap_len = 0
                for s in reversed(buffer):
                    if overlap_len + len(s) > self.overlap_chars:
                        break
                    overlap_buf.insert(0, s)
                    overlap_len += len(s)
                buffer = overlap_buf

        if buffer:
            chunk_text = " ".join(buffer)
            if chunk_text.strip():
                c = Chunk(text=chunk_text.strip(), url=url, content_type="prose", doc_title=doc_title, doc_description=doc_description, **headings)
                c.token_count = token_count(c.text)
                chunks.append(c)

        return chunks

# ─── ChromaDB setup ──────────────────────────────────────────────────────────

def get_collections(chroma_path: str):
    """Create or retrieve the two ChromaDB collections using Config embedding model."""
    client = chromadb.PersistentClient(path=chroma_path)
    embed_fn = SentenceTransformerEmbeddingFunction(model_name=Config.EMBEDDING_MODEL)

    prose_col = client.get_or_create_collection(
        name=PROSE_COLLECTION,
        embedding_function=embed_fn,
        metadata={"hnsw:space": "cosine", "description": "k6 prose and table chunks"},
    )

    code_col = client.get_or_create_collection(
        name=CODE_COLLECTION,
        embedding_function=embed_fn,
        metadata={"hnsw:space": "cosine", "description": "k6 code block chunks"},
    )

    return prose_col, code_col

# ─── Upsert into ChromaDB ────────────────────────────────────────────────────

def upsert_chunks(collection, chunks: list[Chunk], batch_size: int = 50):
    if not chunks:
        return

    # De-duplicate chunks by chunk_id to prevent DuplicateIDError
    unique_chunks = []
    seen_ids = set()
    for c in chunks:
        if c.chunk_id not in seen_ids:
            seen_ids.add(c.chunk_id)
            unique_chunks.append(c)

    ids       = [c.chunk_id for c in unique_chunks]
    texts     = [c.get_searchable_text() for c in unique_chunks]
    metadatas = [c.to_metadata() for c in unique_chunks]

    for i in range(0, len(unique_chunks), batch_size):
        collection.upsert(
            ids=ids[i:i+batch_size],
            documents=texts[i:i+batch_size],
            metadatas=metadatas[i:i+batch_size],
        )

# ─── Main Ingestion ──────────────────────────────────────────────────────────

def get_all_k6_markdown_urls() -> list[str]:
    """Fetches the official Grafana LLM index and finds all k6 documentation pages."""
    index_url = "https://grafana.com/llms-full.txt"
    log.info("Fetching master documentation index...")
    
    response = requests.get(index_url, headers={"User-Agent": USER_AGENT})
    response.raise_for_status()
    
    pattern = r'https://grafana\.com(/docs/(?:k6/latest|grafana-cloud/testing/k6)[^\s\)\>]*\.md)'
    matches = re.findall(pattern, response.text)
    
    full_urls = [f"https://grafana.com{path}" for path in set(matches)]
    log.info(f"Found {len(full_urls)} documentation pages for k6.")
    return sorted(full_urls)

def run_ingestion():
    chroma_path = Config.CHROMA_DB_PATH
    log.info(f"Starting production-grade ingestion to database: {chroma_path}")
    
    # Fetch URLs dynamically
    urls = get_all_k6_markdown_urls()
    # urls = ["https://grafana.com/docs/k6/latest/using-k6/test-lifecycle.md"]
    if not urls:
        log.warning("No k6 documentation URLs found.")
        return
        
    prose_col, code_col = get_collections(chroma_path)
    chunker = MarkdownChunker()

    total_prose, total_code, total_table = 0, 0, 0

    for url in tqdm(urls, desc="Ingesting"):
        raw = fetch_markdown(url)
        if not raw:
            continue

        title, desc = parse_frontmatter(raw)
        cleaned = clean_markdown(raw)
        chunks  = chunker.chunk(cleaned, url, doc_title=title, doc_description=desc)

        prose_chunks = [c for c in chunks if c.content_type in ("prose", "table")]
        code_chunks  = [c for c in chunks if c.content_type == "code"]

        if prose_chunks:
            upsert_chunks(prose_col, prose_chunks)
            total_prose += sum(1 for c in prose_chunks if c.content_type == "prose")
            total_table += sum(1 for c in prose_chunks if c.content_type == "table")

        if code_chunks:
            upsert_chunks(code_col, code_chunks)
            total_code += len(code_chunks)

    log.info(f"\n✓ Ingestion complete successfully!")
    log.info(f"  Prose chunks : {total_prose}")
    log.info(f"  Table chunks : {total_table}")
    log.info(f"  Code chunks  : {total_code}")
    log.info(f"  ChromaDB path: {chroma_path}")

if __name__ == "__main__":
    run_ingestion()
