"""Document ingestion pipeline: load → chunk → embed → upsert into Chroma.

Supports PDF, Markdown, and HTML. The synthetic Vanguard knowledge base in
``ingest/data/`` follows the naming convention ``VG-{ED|OP|PR}-NNN_*.{pdf,md,html}``
which we use to derive a stable ``doc_id`` for citations.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from .factories.vectorstore import build_vectorstore
from .settings import get_settings

log = logging.getLogger(__name__)

# Match the corpus naming convention: VG-PR-001_Whatever_Name.ext
_DOC_ID_RE = re.compile(r"^(VG-(?:ED|OP|PR)-\d{3})", re.IGNORECASE)


@dataclass(frozen=True)
class IngestStats:
    """Summary of an ingest run."""

    files_loaded: int
    documents_loaded: int
    chunks_created: int
    chunks_upserted: int


def _doc_id_from_filename(path: Path) -> str:
    """Extract a stable doc id from the filename, falling back to the stem."""
    match = _DOC_ID_RE.match(path.name)
    return match.group(1).upper() if match else path.stem


def _human_title_from_filename(path: Path) -> str:
    """Turn ``VG-PR-002_500_Index_Fund_Prospectus.pdf`` into a title."""
    stem = path.stem
    match = _DOC_ID_RE.match(stem)
    after = stem[match.end() :].lstrip("_- ") if match else stem
    return after.replace("_", " ").strip()


def _load_pdf(path: Path) -> list[Document]:
    from langchain_community.document_loaders import PyPDFLoader

    return PyPDFLoader(str(path)).load()


def _load_markdown(path: Path) -> list[Document]:
    text = path.read_text(encoding="utf-8")
    return [Document(page_content=text, metadata={})]


def _load_html(path: Path) -> list[Document]:
    from langchain_community.document_loaders import BSHTMLLoader

    return BSHTMLLoader(str(path), open_encoding="utf-8").load()


_LOADERS = {
    ".pdf": _load_pdf,
    ".md": _load_markdown,
    ".html": _load_html,
    ".htm": _load_html,
}


def discover_files(data_dir: Path) -> list[Path]:
    """List loadable files under ``data_dir``, excluding the corpus README."""
    files: list[Path] = []
    for ext in _LOADERS:
        files.extend(data_dir.glob(f"*{ext}"))
    # Skip the corpus's own README (not real content).
    files = [f for f in files if f.name.lower() != "readme.md"]
    return sorted(files)


def load_documents(data_dir: Path) -> list[Document]:
    """Load every supported file in ``data_dir`` into LangChain ``Document``s.

    Each Document's metadata gets ``doc_id``, ``title``, and ``source_path``
    stamped on it; PDFs additionally carry ``page`` from the loader.
    """
    docs: list[Document] = []
    for path in discover_files(data_dir):
        loader = _LOADERS[path.suffix.lower()]
        doc_id = _doc_id_from_filename(path)
        title = _human_title_from_filename(path)
        try:
            loaded = loader(path)
        except Exception:
            log.exception("failed to load %s — skipping", path)
            continue
        for d in loaded:
            d.metadata = {
                **d.metadata,
                "doc_id": doc_id,
                "title": title,
                "source_path": str(path),
                "format": path.suffix.lstrip(".").lower(),
            }
        docs.extend(loaded)
        log.info("loaded %d page(s) from %s", len(loaded), path.name)
    return docs


def chunk_documents(
    docs: Iterable[Document],
    *,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> list[Document]:
    """Split documents into overlapping chunks. Each chunk gets a ``chunk_id``."""
    s = get_settings()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size or s.rag_chunk_size,
        chunk_overlap=chunk_overlap or s.rag_chunk_overlap,
        # Token-ish defaults — recursive splitter falls back through these.
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(list(docs))

    # Stamp a chunk_id per (doc_id, position) so citations are stable.
    counters: dict[str, int] = {}
    for ch in chunks:
        doc_id = ch.metadata.get("doc_id", "UNKNOWN")
        n = counters.get(doc_id, 0)
        counters[doc_id] = n + 1
        ch.metadata["chunk_id"] = f"{doc_id}#{n:03d}"
    return chunks


def ingest(data_dir: Path | str, *, reset: bool = False) -> IngestStats:
    """Full pipeline: load → chunk → upsert into the configured vector store.

    ``reset=True`` deletes the existing Chroma collection contents first.
    Useful when re-running after edits to the corpus.
    """
    data_dir = Path(data_dir)
    if not data_dir.is_dir():
        raise FileNotFoundError(f"data_dir not found: {data_dir}")

    files = discover_files(data_dir)
    log.info("ingesting %d files from %s", len(files), data_dir)

    docs = load_documents(data_dir)
    chunks = chunk_documents(docs)

    vs = build_vectorstore()

    if reset:
        # Best-effort wipe — only Chroma exposes ``reset_collection`` cleanly.
        try:
            vs.reset_collection()  # type: ignore[attr-defined]
            log.info("vector store collection reset")
        except AttributeError:
            log.warning(
                "vector store backend does not support reset; "
                "old vectors will remain alongside new ones"
            )

    # Use chunk_id as the deterministic upsert id so re-ingest is idempotent.
    ids = [c.metadata["chunk_id"] for c in chunks]
    vs.add_documents(chunks, ids=ids)

    stats = IngestStats(
        files_loaded=len(files),
        documents_loaded=len(docs),
        chunks_created=len(chunks),
        chunks_upserted=len(ids),
    )
    log.info(
        "ingest complete: files=%d docs=%d chunks=%d",
        stats.files_loaded,
        stats.documents_loaded,
        stats.chunks_created,
    )
    return stats
