"""FastAPI dependencies that read graph + checkpointer + db off ``app.state``.

Built once in the lifespan, then injected into route handlers via ``Depends``.
"""

from __future__ import annotations

from typing import AsyncIterator

from fastapi import Request
from sqlalchemy.ext.asyncio import AsyncSession


def get_graph(request: Request):
    """Return the compiled LangGraph stored on ``app.state.graph``."""
    return request.app.state.graph


def get_checkpointer(request: Request):
    """Return the checkpoint saver stored on ``app.state.checkpointer``."""
    return request.app.state.checkpointer


async def get_db(request: Request) -> AsyncIterator[AsyncSession]:
    """Yield an ``AsyncSession`` bound to the API's app DB.

    Sessions are scoped per-request — the dependency closes them on exit.
    Routes call ``await session.commit()`` themselves; on exception the
    session is rolled back automatically.
    """
    factory = request.app.state.app_db_sessionmaker
    async with factory() as session:
        try:
            yield session
        except Exception:
            await session.rollback()
            raise
