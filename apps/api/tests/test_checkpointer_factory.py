"""Tests for the pluggable checkpointer factory (M9)."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from vanguard_agents.factories import build_checkpointer
from vanguard_agents.settings import Settings


@pytest.mark.asyncio
async def test_sqlite_backend_yields_async_sqlite_saver():
    from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

    with tempfile.TemporaryDirectory() as td:
        cp_path = Path(td) / "cp.sqlite"
        # Build a fresh Settings — model_validate skips env loading so the
        # test isn't affected by the user's .env.
        settings = Settings.model_validate(
            {
                "checkpointer_backend": "sqlite",
                "checkpoint_sqlite_path": str(cp_path),
            }
        )
        async with build_checkpointer(settings) as cp:
            assert isinstance(cp, AsyncSqliteSaver)
        assert cp_path.exists()


@pytest.mark.asyncio
async def test_postgres_backend_without_dsn_raises():
    settings = Settings.model_validate(
        {"checkpointer_backend": "postgres", "postgres_dsn": None}
    )
    with pytest.raises(RuntimeError, match="POSTGRES_DSN"):
        async with build_checkpointer(settings):
            pass


@pytest.mark.asyncio
async def test_unknown_backend_raises():
    settings = Settings.model_construct(checkpointer_backend="badger")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="badger"):
        async with build_checkpointer(settings):
            pass
