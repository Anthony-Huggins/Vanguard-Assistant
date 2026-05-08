"""Threads CRUD — keep it boring, no clever query DSL.

The router calls :func:`upsert_thread_metadata` after every successful
turn (via either ``/api/chat`` or ``/api/chat/stream``) so the sidebar
sort order tracks the most recent activity.  ``display_name`` is set
once on create from the user's first message; subsequent turns leave
it alone.
"""

from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from .models import Thread


_DISPLAY_NAME_LIMIT = 80


def _truncate(text: str, limit: int = _DISPLAY_NAME_LIMIT) -> str:
    cleaned = text.strip().splitlines()[0] if text else ""
    if len(cleaned) <= limit:
        return cleaned
    return cleaned[: limit - 1].rstrip() + "…"


async def upsert_thread_metadata(
    session: AsyncSession,
    *,
    thread_id: str,
    user_message: str,
) -> None:
    """Create the thread row on first turn; bump ``last_message_at`` after."""
    now = datetime.now(timezone.utc)

    existing = await session.get(Thread, thread_id)
    if existing is None:
        session.add(
            Thread(
                thread_id=thread_id,
                display_name=_truncate(user_message),
                last_message_at=now,
            )
        )
    else:
        existing.last_message_at = now

    await session.commit()


async def list_threads(session: AsyncSession) -> list[Thread]:
    """All threads, most recently active first."""
    result = await session.execute(
        select(Thread).order_by(
            Thread.last_message_at.desc().nullslast(),
            Thread.created_at.desc(),
        )
    )
    return list(result.scalars())


async def get_thread(session: AsyncSession, thread_id: str) -> Thread | None:
    return await session.get(Thread, thread_id)
