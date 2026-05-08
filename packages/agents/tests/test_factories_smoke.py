"""Smoke tests for factories — verify they return the right interface type.

These tests do NOT make any network calls. They construct the objects and
assert they conform to the LangChain abstract base classes. That's what
proves the pluggability claim: agent code can rely on the interface.
"""

import pytest
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel
from langchain_core.retrievers import BaseRetriever
from langchain_core.vectorstores import VectorStore


@pytest.fixture(autouse=True)
def _isolated_settings(monkeypatch, tmp_path):
    """Each test gets a clean settings env, persistent dir in tmp."""
    from vanguard_agents.settings import reset_settings_for_tests

    monkeypatch.setenv("CHROMA_PERSIST_DIR", str(tmp_path / "chroma"))
    reset_settings_for_tests()
    yield
    reset_settings_for_tests()


def test_chat_model_returns_basechatmodel(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "azure_openai")
    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://example.openai.azure.com")
    monkeypatch.setenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4.1-nano")
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "sk-test-not-real")
    from vanguard_agents.factories.llm import build_chat_model
    from vanguard_agents.settings import reset_settings_for_tests

    reset_settings_for_tests()
    model = build_chat_model()
    assert isinstance(model, BaseChatModel)


def test_embedder_returns_embeddings(monkeypatch):
    # HuggingFace embeddings download a small model the first time. To keep
    # CI fast we skip if the package isn't importable in this env.
    pytest.importorskip("langchain_huggingface")

    from vanguard_agents.factories.embedder import build_embedder

    embedder = build_embedder()
    assert isinstance(embedder, Embeddings)


def test_vectorstore_returns_vectorstore(monkeypatch):
    pytest.importorskip("langchain_chroma")
    pytest.importorskip("langchain_huggingface")

    from vanguard_agents.factories.vectorstore import build_retriever, build_vectorstore

    vs = build_vectorstore()
    assert isinstance(vs, VectorStore)
    retriever = build_retriever()
    assert isinstance(retriever, BaseRetriever)


def test_pinecone_without_keys_raises(monkeypatch):
    monkeypatch.setenv("VECTORSTORE_BACKEND", "pinecone")
    monkeypatch.delenv("PINECONE_API_KEY", raising=False)
    monkeypatch.delenv("PINECONE_INDEX", raising=False)
    from vanguard_agents.factories.vectorstore import build_vectorstore
    from vanguard_agents.settings import reset_settings_for_tests

    reset_settings_for_tests()
    with pytest.raises(RuntimeError, match="PINECONE"):
        build_vectorstore()


def test_chroma_http_mode_constructs_http_client(monkeypatch):
    """M10 — http mode parses the URL and hands an HttpClient to Chroma."""
    pytest.importorskip("langchain_chroma")
    pytest.importorskip("langchain_huggingface")
    pytest.importorskip("chromadb")

    monkeypatch.setenv("CHROMA_MODE", "http")
    monkeypatch.setenv("CHROMA_URL", "http://chroma-host:8123")

    from unittest.mock import MagicMock, patch

    from vanguard_agents.settings import reset_settings_for_tests

    reset_settings_for_tests()

    fake_client = MagicMock()
    with patch("chromadb.HttpClient", return_value=fake_client) as http_client:
        from vanguard_agents.factories.vectorstore import build_vectorstore

        build_vectorstore()
        http_client.assert_called_once()
        kwargs = http_client.call_args.kwargs
        assert kwargs["host"] == "chroma-host"
        assert kwargs["port"] == 8123
        assert kwargs["ssl"] is False
