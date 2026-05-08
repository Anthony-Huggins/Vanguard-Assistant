"""Unit tests for the ingestion pipeline (no embedding model needed)."""

from __future__ import annotations

from pathlib import Path

import pytest

from vanguard_agents.ingest import (
    _doc_id_from_filename,
    _human_title_from_filename,
    chunk_documents,
    discover_files,
    load_documents,
)


@pytest.fixture
def fake_kb(tmp_path: Path) -> Path:
    """Make a tiny corpus on disk: one md, one html, one named like ours."""
    (tmp_path / "VG-OP-099_Fake_Policy.md").write_text(
        "# Fake Policy\n\nThis is a synthetic test policy used for unit tests.\n",
        encoding="utf-8",
    )
    (tmp_path / "VG-OP-100_Fake_HTML.html").write_text(
        "<html><body><h1>Hello</h1><p>some content</p></body></html>",
        encoding="utf-8",
    )
    # Should be ignored:
    (tmp_path / "README.md").write_text("# corpus readme", encoding="utf-8")
    (tmp_path / "ignore.txt").write_text("not loadable", encoding="utf-8")
    return tmp_path


def test_doc_id_extraction():
    p = Path("VG-PR-002_500_Index_Fund_Prospectus.pdf")
    assert _doc_id_from_filename(p) == "VG-PR-002"
    assert _human_title_from_filename(p) == "500 Index Fund Prospectus"


def test_doc_id_falls_back_to_stem():
    p = Path("uncategorized_doc.md")
    assert _doc_id_from_filename(p) == "uncategorized_doc"


def test_discover_files_skips_readme_and_unknown_ext(fake_kb):
    files = discover_files(fake_kb)
    names = {f.name for f in files}
    assert "VG-OP-099_Fake_Policy.md" in names
    assert "VG-OP-100_Fake_HTML.html" in names
    assert "README.md" not in names
    assert "ignore.txt" not in names


def test_load_stamps_metadata(fake_kb):
    docs = load_documents(fake_kb)
    by_doc_id = {d.metadata["doc_id"]: d for d in docs}
    assert "VG-OP-099" in by_doc_id
    assert "VG-OP-100" in by_doc_id
    md_doc = by_doc_id["VG-OP-099"]
    assert md_doc.metadata["title"] == "Fake Policy"
    assert md_doc.metadata["format"] == "md"
    assert "synthetic test policy" in md_doc.page_content


def test_chunking_assigns_stable_ids(fake_kb):
    docs = load_documents(fake_kb)
    # Use a tiny chunk size to force multiple chunks per doc.
    chunks = chunk_documents(docs, chunk_size=20, chunk_overlap=5)
    ids = [c.metadata["chunk_id"] for c in chunks]
    assert all("#" in cid for cid in ids)
    # No duplicate chunk ids overall.
    assert len(ids) == len(set(ids))
    # Each chunk_id starts with its doc_id.
    for c in chunks:
        assert c.metadata["chunk_id"].startswith(c.metadata["doc_id"])
