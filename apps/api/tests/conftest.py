"""Shared pytest fixtures for the FastAPI app tests.

Mocks the graph + checkpointer so tests don't touch Azure / Chroma / MCP.
``httpx.AsyncClient`` with ``ASGITransport`` calls the app in-process and
does *not* trigger the lifespan handler — which means the real graph is
never built.  We attach the mocks to ``app.state`` directly instead.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import AsyncIterator
from unittest.mock import AsyncMock, MagicMock

# Make ``apps.api.*`` and ``vanguard_agents.*`` importable when tests are
# invoked from anywhere (``pytest apps/api/tests`` from the repo root,
# ``pytest`` from inside the dir, etc.).  Same trick the agents test
# conftest uses — see ``packages/agents/tests/conftest.py:12``.
_REPO_ROOT = Path(__file__).resolve().parents[3]  # vanguard-assistant/
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import pytest  # noqa: E402
import pytest_asyncio  # noqa: E402
from httpx import ASGITransport, AsyncClient  # noqa: E402
from sqlalchemy.ext.asyncio import (  # noqa: E402
    AsyncEngine,
    async_sessionmaker,
    create_async_engine,
)

from apps.api.db import Base  # noqa: E402
from vanguard_agents.state import RoutingDecision  # noqa: E402


# ---------------------------------------------------------------------------
# Canned data — used by both test_chat and test_threads.
# ---------------------------------------------------------------------------

DEFAULT_CHAT_RESULT: dict = {
    "final_response": (
        "[Support Agent] Vanguard's flagship index funds carry an expense "
        "ratio around 0.04% on average.\n\n"
        "Sources: [vanguard-fees.pdf, chunk_id: 12] [vanguard-fees.pdf, chunk_id: 47]"
    ),
    "routing": RoutingDecision(
        agent="support",
        reason="user asked about fee structure — RAG topic",
    ),
    "retrieved_chunks": [
        {
            "text": "Vanguard's expense ratios average...",
            "source": "vanguard-fees.pdf",
            "chunk_id": "12",
            "score": 0.61,
            "page": 3,
        },
        {
            "text": "Index fund expense ratios as low as 0.03%...",
            "source": "vanguard-fees.pdf",
            "chunk_id": "47",
            "score": 0.55,
            "page": 8,
        },
    ],
    "tool_results": [],
}


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_graph() -> AsyncMock:
    """An AsyncMock graph whose ``ainvoke`` returns the canned support reply.

    Tests that need a different result should overwrite ``return_value``.
    """
    graph = AsyncMock()
    graph.ainvoke.return_value = DEFAULT_CHAT_RESULT

    # No interrupts pending — the auto-confirm loop checks aget_state.
    snapshot = MagicMock()
    snapshot.tasks = []
    graph.aget_state.return_value = snapshot
    return graph


@pytest.fixture
def mock_checkpointer() -> MagicMock:
    """A MagicMock checkpointer with no persisted threads by default.

    Threads-test cases override ``alist`` / ``aget_tuple`` to seed data.
    """
    cp = MagicMock()

    async def _empty_alist(config=None, **_):  # noqa: ANN001 — duck-typing
        if False:  # pragma: no cover — make this an async generator with no yields
            yield

    cp.alist = _empty_alist
    cp.aget_tuple = AsyncMock(return_value=None)
    return cp


@pytest_asyncio.fixture
async def app_db_engine() -> AsyncIterator[AsyncEngine]:
    """In-memory SQLite engine + schema, freshly created per test.

    Sufficient for everything the routes do (a single ``threads`` table).
    Avoids touching the on-disk ``.app_db.sqlite`` Alembic created.
    """
    engine = create_async_engine("sqlite+aiosqlite:///:memory:", future=True)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    try:
        yield engine
    finally:
        await engine.dispose()


@pytest_asyncio.fixture
async def client(
    mock_graph, mock_checkpointer, app_db_engine
) -> AsyncIterator[AsyncClient]:
    """Async HTTP client wired to the FastAPI app with mocks injected.

    ``ASGITransport`` does not run FastAPI's lifespan, so the real graph
    + app DB are never constructed — we attach the mocks (and an in-memory
    SQLite session factory) to ``app.state`` ourselves.
    """
    from apps.api.main import app

    app.state.graph = mock_graph
    app.state.checkpointer = mock_checkpointer
    app.state.app_db_engine = app_db_engine
    app.state.app_db_sessionmaker = async_sessionmaker(
        app_db_engine, expire_on_commit=False
    )

    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as ac:
        yield ac
