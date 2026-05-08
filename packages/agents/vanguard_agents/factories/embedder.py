"""Embedder factory. Returns a LangChain ``Embeddings``."""

from __future__ import annotations

from langchain_core.embeddings import Embeddings

from ..settings import get_settings


def build_embedder() -> Embeddings:
    s = get_settings()

    if s.embedder_provider == "huggingface":
        from langchain_huggingface import HuggingFaceEmbeddings

        return HuggingFaceEmbeddings(model_name=s.embedder_model)

    if s.embedder_provider == "openai":
        from langchain_openai import OpenAIEmbeddings

        return OpenAIEmbeddings(model=s.embedder_model)

    if s.embedder_provider == "azure_openai":
        from langchain_openai import AzureOpenAIEmbeddings

        if not s.azure_openai_endpoint:
            raise RuntimeError("AZURE_OPENAI_ENDPOINT must be set for azure_openai embedder.")
        return AzureOpenAIEmbeddings(
            azure_endpoint=s.azure_openai_endpoint,
            api_version=s.azure_openai_api_version,
            azure_deployment=s.embedder_model,
        )

    raise ValueError(f"Unsupported embedder_provider: {s.embedder_provider!r}")
