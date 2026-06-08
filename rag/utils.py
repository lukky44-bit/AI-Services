import re
from dataclasses import dataclass

@dataclass
class RetrievedChunk:
    chunk_id:     str
    text:         str
    url:          str
    heading_path: str
    content_type: str
    language:     str
    score:        float = 0.0

def tokenize(text: str) -> list[str]:
    """Simple whitespace + punctuation tokenizer for BM25."""
    return re.findall(r"[a-zA-Z0-9_\.]+", text.lower())

def format_sources(chunks: list[RetrievedChunk]) -> list[dict]:
    """Helper to deduplicate and format retrieved sources."""
    seen = set()
    sources = []
    for c in chunks:
        key = (c.url, c.heading_path)
        if key not in seen:
            seen.add(key)
            sources.append({
                "url":          c.url,
                "heading_path": c.heading_path,
                "content_type": c.content_type,
            })
    return sources

def rrf_fuse(
    vector_results: list[RetrievedChunk],
    bm25_results: list[RetrievedChunk],
    k: int = 60,
) -> list[RetrievedChunk]:
    """
    Reciprocal Rank Fusion: RRF(d) = Σ 1/(k + rank_i(d))
    Combines ranked lists from semantic and lexical search.
    """
    rrf_scores: dict[str, float] = {}
    chunk_map:  dict[str, RetrievedChunk] = {}

    for rank, chunk in enumerate(vector_results):
        rrf_scores[chunk.chunk_id] = rrf_scores.get(chunk.chunk_id, 0) + 1.0 / (k + rank + 1)
        chunk_map[chunk.chunk_id] = chunk

    for rank, chunk in enumerate(bm25_results):
        rrf_scores[chunk.chunk_id] = rrf_scores.get(chunk.chunk_id, 0) + 1.0 / (k + rank + 1)
        chunk_map[chunk.chunk_id] = chunk

    sorted_ids = sorted(rrf_scores.keys(), key=lambda cid: rrf_scores[cid], reverse=True)
    fused = []
    for cid in sorted_ids:
        c = chunk_map[cid]
        c.score = rrf_scores[cid]
        fused.append(c)
    return fused

def build_context(chunks: list[RetrievedChunk], max_context_tokens: int = 3000) -> str:
    """
    Separate code blocks from prose/tables in the context string.
    This helps the LLM distinguish between explanations and runnable examples.
    """
    prose_parts = []
    code_parts  = []
    seen_ids    = set()
    total_chars = 0
    char_budget = max_context_tokens * 4  # rough chars → tokens ratio

    for chunk in chunks:
        if chunk.chunk_id in seen_ids:
            continue
        seen_ids.add(chunk.chunk_id)

        header = f"[Source: {chunk.url}]"
        if chunk.heading_path:
            header += f" ({chunk.heading_path})"

        if chunk.content_type == "code":
            lang = chunk.language or ""
            entry = f"{header}\n```{lang}\n{chunk.text}\n```"
            code_parts.append(entry)
        else:
            entry = f"{header}\n{chunk.text}"
            prose_parts.append(entry)

        total_chars += len(entry)
        if total_chars > char_budget:
            break

    sections = []
    if prose_parts:
        sections.append("### Documentation\n\n" + "\n\n---\n\n".join(prose_parts))
    if code_parts:
        sections.append("### Code Examples\n\n" + "\n\n---\n\n".join(code_parts))

    return "\n\n".join(sections)
