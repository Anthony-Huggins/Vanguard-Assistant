"""Settings should load from environment with sensible defaults."""

import os

from vanguard_agents.settings import Settings, reset_settings_for_tests


def test_defaults(monkeypatch):
    # Wipe potentially-set vars so we're testing defaults.
    for var in [
        "LLM_PROVIDER",
        "LLM_MODEL",
        "EMBEDDER_PROVIDER",
        "VECTORSTORE_BACKEND",
        "RAG_TOP_K",
    ]:
        monkeypatch.delenv(var, raising=False)
    reset_settings_for_tests()

    s = Settings(_env_file=None)  # type: ignore[arg-type]
    assert s.llm_provider == "azure_openai"
    assert s.embedder_provider == "huggingface"
    assert s.vectorstore_backend == "chroma"
    assert s.rag_top_k == 5
    assert s.rag_chunk_size == 512


def test_env_overrides(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    monkeypatch.setenv("RAG_TOP_K", "8")
    monkeypatch.setenv("VECTORSTORE_BACKEND", "pinecone")
    reset_settings_for_tests()

    s = Settings(_env_file=None)  # type: ignore[arg-type]
    assert s.llm_provider == "openai"
    assert s.rag_top_k == 8
    assert s.vectorstore_backend == "pinecone"


def test_unknown_extras_ignored(monkeypatch):
    monkeypatch.setenv("SOMETHING_UNRELATED", "yep")
    reset_settings_for_tests()

    s = Settings(_env_file=None)  # type: ignore[arg-type]
    # Just constructing without error is the assertion.
    assert s is not None
