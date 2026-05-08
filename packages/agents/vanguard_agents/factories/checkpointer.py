"""Pluggable LangGraph checkpoint-saver factory (M9).

Both the CLI and the FastAPI app build their checkpointer through this
module so the choice of backend (SQLite for local dev, Postgres for
M9+ deployments) is a one-line ``.env`` change.  Mirrors the same
factory pattern used for the LLM, embedder, and vector store.

Usage::

    async with build_checkpointer(get_settings()) as checkpointer:
        graph = build_graph(checkpointer=checkpointer)
        ...

The async context manager owns the underlying connection's lifetime —
yielding the saver and tearing the connection down on exit.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import TYPE_CHECKING, AsyncIterator

if TYPE_CHECKING:
    from langgraph.checkpoint.base import BaseCheckpointSaver

    from ..settings import Settings

log = logging.getLogger(__name__)


@asynccontextmanager
async def build_checkpointer(
    settings: "Settings",
) -> AsyncIterator["BaseCheckpointSaver"]:
    """Yield a checkpoint saver of the configured backend.

    - ``checkpointer_backend="sqlite"`` → ``AsyncSqliteSaver`` over
      ``settings.checkpoint_sqlite_path``.  The directory is created
      lazily so a fresh checkout works without setup.
    - ``checkpointer_backend="postgres"`` → ``AsyncPostgresSaver`` over
      ``settings.postgres_dsn``.  Calls ``setup()`` on first connect to
      create the checkpoint tables if they don't exist.
    """
    backend = settings.checkpointer_backend

    if backend == "sqlite":
        path = settings.checkpoint_sqlite_path
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

        async with AsyncSqliteSaver.from_conn_string(path) as cp:
            log.info("checkpointer: sqlite (%s)", path)
            yield cp
        return

    if backend == "postgres":
        dsn = settings.postgres_dsn
        if not dsn:
            raise RuntimeError(
                "checkpointer_backend=postgres requires POSTGRES_DSN "
                "(plain postgresql://... DSN, no async dialect prefix)."
            )
        from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

        async with AsyncPostgresSaver.from_conn_string(dsn) as cp:
            # Idempotent — creates the checkpoint tables if absent.  Must run
            # before the first invoke or aget_state, so we do it here.
            await cp.setup()
            log.info("checkpointer: postgres (host=%s)", _redact_host(dsn))
            yield cp
        return

    raise ValueError(f"unknown checkpointer_backend: {backend!r}")


def _redact_host(dsn: str) -> str:
    """Best-effort log-safe summary — strip credentials, keep host."""
    try:
        from urllib.parse import urlparse

        parsed = urlparse(dsn)
        return parsed.hostname or "?"
    except Exception:  # noqa: BLE001
        return "?"
