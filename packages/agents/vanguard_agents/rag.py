"""Retrieval helpers used by the Support Agent.

Responsibilities:
- Run similarity search against the configured vector store.
- Convert raw results into our ``RetrievedChunk`` typed dict.
- Format chunks for prompt injection and produce the ``Sources:`` footer.

We use ``similarity_search_with_relevance_scores`` so we can apply the
fallback threshold from settings.
"""

from __future__ import annotations

import warnings

from langchain_core.documents import Document

from .factories.vectorstore import build_vectorstore
from .settings import get_settings
from .state import RetrievedChunk

# LangChain's Chroma wrapper emits this warning whenever the relevance score
# is outside [0, 1]. With HuggingFace embeddings + cosine-distance backend,
# scores can dip below 0 for unrelated content. The relative ordering is
# still correct, which is all our threshold logic depends on.
warnings.filterwarnings(
    "ignore",
    message=".*Relevance scores must be between 0 and 1.*",
    category=UserWarning,
)


def retrieve(query: str, *, k: int | None = None) -> list[RetrievedChunk]:
    """Top-k similarity search against the vector store.

    Returns chunks sorted by relevance, descending. Each chunk's score is on
    the relevance scale used by LangChain's
    ``similarity_search_with_relevance_scores`` (higher = more relevant).
    """
    s = get_settings()
    vs = build_vectorstore()
    top_k = k or s.rag_top_k

    results: list[tuple[Document, float]] = vs.similarity_search_with_relevance_scores(
        query, k=top_k
    )

    chunks: list[RetrievedChunk] = []
    for doc, score in results:
        meta = doc.metadata or {}
        chunks.append(
            {
                "text": doc.page_content,
                "source": meta.get("title") or meta.get("doc_id") or "unknown",
                "chunk_id": meta.get("chunk_id", "?"),
                "score": float(score),
                "page": meta.get("page"),
            }
        )
    return chunks


def best_score(chunks: list[RetrievedChunk]) -> float:
    """Highest relevance score among the chunks (or 0.0 if empty)."""
    return max((c["score"] for c in chunks), default=0.0)


def render_chunks_for_prompt(chunks: list[RetrievedChunk]) -> str:
    """Render retrieved chunks for inclusion in the Support Agent prompt."""
    if not chunks:
        return "(no relevant context retrieved)"

    blocks: list[str] = []
    for ch in chunks:
        page = f", page {ch['page']}" if ch.get("page") is not None else ""
        header = f"[{ch['source']} · {ch['chunk_id']}{page}]"
        blocks.append(f"{header}\n{ch['text'].strip()}")
    return "\n\n---\n\n".join(blocks)


def format_citation_line(chunks: list[RetrievedChunk]) -> str:
    """Produce the ``Sources:`` line per the spec.

    Example output:
        Sources: [500 Index Fund Prospectus, chunk_id: VG-PR-002#003]
                 [Expense Ratio Explainer, chunk_id: VG-ED-002#001]
    """
    if not chunks:
        return ""
    parts = [f"[{c['source']}, chunk_id: {c['chunk_id']}]" for c in chunks]
    return "Sources: " + " ".join(parts)
