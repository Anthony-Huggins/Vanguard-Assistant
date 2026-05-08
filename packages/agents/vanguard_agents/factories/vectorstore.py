"""Vector store factory. Returns a LangChain ``VectorStore`` / ``BaseRetriever``."""

from __future__ import annotations

from langchain_core.retrievers import BaseRetriever
from langchain_core.vectorstores import VectorStore

from ..settings import get_settings
from .embedder import build_embedder


def build_vectorstore() -> VectorStore:
    s = get_settings()

    # Validate config BEFORE we do expensive embedder construction —
    # fail fast on misconfig.
    if s.vectorstore_backend == "pinecone":
        if not s.pinecone_api_key or not s.pinecone_index:
            raise RuntimeError(
                "VECTORSTORE_BACKEND=pinecone requires PINECONE_API_KEY and PINECONE_INDEX."
            )
    elif s.vectorstore_backend != "chroma":
        raise ValueError(f"Unsupported vectorstore_backend: {s.vectorstore_backend!r}")

    embedder = build_embedder()

    if s.vectorstore_backend == "chroma":
        from langchain_chroma import Chroma

        # Distance → relevance conversion.  Our embedder
        # (sentence-transformers/all-MiniLM-L6-v2) produces normalized
        # vectors, so cosine distance is in [0, 2] and ``1 - distance``
        # gives a relevance in [-1, 1] — clamp to [0, 1] to match the
        # threshold (0.25) calibrated for embedded mode in M3.
        def _cosine_relevance(distance: float) -> float:
            return max(0.0, 1.0 - distance)

        if s.chroma_mode == "http":
            # Talk to the Chroma service container (M10).  Parse the URL
            # rather than splitting on ":" so trailing slashes and IPv6
            # don't break us.
            from urllib.parse import urlparse

            import chromadb

            parsed = urlparse(s.chroma_url)
            client = chromadb.HttpClient(
                host=parsed.hostname or "localhost",
                port=parsed.port or 8000,
                ssl=parsed.scheme == "https",
            )
            chroma = Chroma(
                collection_name=s.chroma_collection,
                embedding_function=embedder,
                client=client,
            )
        else:
            chroma = Chroma(
                collection_name=s.chroma_collection,
                embedding_function=embedder,
                persist_directory=s.chroma_persist_dir,
            )

        # langchain-chroma 1.1.0 silently drops the ``relevance_score_fn``
        # constructor arg, so its ``_select_relevance_score_fn`` falls
        # back to metric auto-detection from collection metadata.  In
        # chromadb 1.x that metadata no longer carries ``hnsw:space``,
        # so detection raises ``ValueError("...metric of type: None")``.
        # Monkey-patch the instance method to always return our function;
        # bulletproof against either library's churn.
        chroma._select_relevance_score_fn = lambda: _cosine_relevance
        return chroma

    # pinecone (already validated above)
    try:
        from langchain_pinecone import PineconeVectorStore  # type: ignore[import-not-found]
    except ImportError as exc:
        raise RuntimeError(
            "Install the 'pinecone' extra: `pip install -e .[pinecone]`"
        ) from exc

    return PineconeVectorStore(
        index_name=s.pinecone_index,
        embedding=embedder,
        pinecone_api_key=s.pinecone_api_key,
    )


def build_retriever(*, k: int | None = None) -> BaseRetriever:
    s = get_settings()
    vs = build_vectorstore()
    return vs.as_retriever(search_kwargs={"k": k or s.rag_top_k})
