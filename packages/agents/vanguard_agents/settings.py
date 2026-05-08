"""Typed configuration loaded from environment variables.

Every provider choice (LLM, embedder, vector store) is selected here.
Agent code reads these via ``get_settings()`` and never imports a provider SDK
directly — see ``factories/`` for the construction layer.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


def _find_env_file() -> Path | None:
    """Walk up from this file until we find a .env (or stop at filesystem root)."""
    here = Path(__file__).resolve()
    for parent in [here, *here.parents]:
        candidate = parent / ".env"
        if candidate.is_file():
            return candidate
    return None


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=_find_env_file(),
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # ----- LLM -----
    llm_provider: Literal["azure_openai", "openai", "anthropic"] = "azure_openai"
    # Azure OpenAI: only these two deployments are available to the class.
    # The value should match the AZURE_OPENAI_DEPLOYMENT name in your resource.
    llm_model: Literal["gpt-4.1-nano", "gpt-4.1"] = "gpt-4.1"
    llm_temperature: float = 0.0
    llm_max_tokens_coordinator: int = 1024
    llm_max_tokens_subagent: int = 2048

    # ----- Azure OpenAI -----
    azure_openai_endpoint: str | None = None
    azure_openai_api_version: str = "2024-08-01-preview"
    # Should be one of: "gpt-4.1-nano" | "gpt-4.1"
    azure_openai_deployment: Literal["gpt-4.1-nano", "gpt-4.1"] | None = None
    azure_openai_api_key: str | None = None  # blank → DefaultAzureCredential

    # ----- Embeddings -----
    embedder_provider: Literal["huggingface", "openai", "azure_openai"] = "huggingface"
    embedder_model: str = "sentence-transformers/all-MiniLM-L6-v2"

    # ----- Vector store -----
    vectorstore_backend: Literal["chroma", "pinecone"] = "chroma"
    # ``embedded`` (default) keeps Chroma in-process via ``persist_directory``.
    # ``http`` talks to a Chroma service container (M10) at ``chroma_url``.
    chroma_mode: Literal["embedded", "http"] = "embedded"
    chroma_persist_dir: str = "./.chroma"
    chroma_collection: str = "vanguard_kb"
    # Used only when ``chroma_mode=http`` — the docker-compose stack sets this
    # to ``http://chroma:8000``.
    chroma_url: str = "http://localhost:8000"

    # Pinecone (only when vectorstore_backend == "pinecone")
    pinecone_api_key: str | None = None
    pinecone_index: str | None = None

    # ----- RAG -----
    rag_top_k: int = 5
    rag_chunk_size: int = 512
    rag_chunk_overlap: int = 50
    # Empirically calibrated against the synthetic Vanguard corpus with
    # `sentence-transformers/all-MiniLM-L6-v2`:
    #   real-question best-score range: 0.35 – 0.61
    #   off-topic best-score range:    -0.12 – -0.04
    # 0.25 cleanly separates them. Re-tune if you swap the embedder.
    rag_similarity_threshold: float = 0.25

    # ----- Conversation -----
    history_max_turns: int = 20

    # ----- Local research-tool data -----
    # Path to the SQLite warehouse used by the local ``db_query`` tool.
    # Resolved relative to CWD (run commands from the project root).
    warehouse_db_path: str = "./data/fund_warehouse.sqlite"

    # ----- MCP (action tools only — research tools live in-process) -----
    mcp_url: str = "http://localhost:8765/mcp/"
    mcp_bearer_token: str | None = None

    # ----- Checkpointer (M9) -----
    # Selects the LangGraph checkpoint backend. ``sqlite`` keeps everything
    # local (CLI-friendly default); ``postgres`` uses AsyncPostgresSaver and
    # requires ``postgres_dsn`` to be set.  See
    # ``vanguard_agents.factories.checkpointer.build_checkpointer``.
    checkpointer_backend: Literal["sqlite", "postgres"] = "sqlite"
    checkpoint_sqlite_path: str = ".checkpoints.sqlite"
    # Plain ``postgresql://...`` DSN — psycopg-style, no async dialect prefix.
    postgres_dsn: str | None = None

    # ----- App database (M9) -----
    # SQLAlchemy AsyncEngine DSN for the API's own metadata table(s).
    # Defaults to a local SQLite file so dev works without Postgres; flip to
    # ``postgresql+asyncpg://...`` when ``checkpointer_backend=postgres``.
    app_db_dsn: str = "sqlite+aiosqlite:///.app_db.sqlite"

    # ----- Eval (M12) -----
    eval_judge_model: Literal["gpt-4.1-nano", "gpt-4.1"] = "gpt-4.1"
    eval_data_path: str = "packages/eval/datasets/qa_pairs.jsonl"
    eval_results_dir: str = "packages/eval/results"

    # ----- Misc -----
    log_level: str = "INFO"


_settings: Settings | None = None


def get_settings() -> Settings:
    """Return the process-wide Settings singleton."""
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings


def reset_settings_for_tests() -> None:
    """Force the next ``get_settings()`` to re-read the environment."""
    global _settings
    _settings = None
