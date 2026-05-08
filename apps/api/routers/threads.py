"""GET /api/threads — list threads (joined with the threads metadata table),
GET /api/threads/{id} — load history from the LangGraph checkpointer.

M6 listed threads by iterating every checkpoint row.  M9 introduces the
``threads`` SQLAlchemy table, so the list pulls human-readable display
names from there and only queries the checkpointer for ``message_count``
on a per-thread basis.  When the threads table is empty (fresh DB,
no chats yet) we fall back to the M6 iteration so the route still works.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from langchain_core.messages import BaseMessage
from sqlalchemy.ext.asyncio import AsyncSession

from apps.api.db.crud import list_threads as list_threads_db
from apps.api.deps import get_checkpointer, get_db
from apps.api.schemas import ThreadDetail, ThreadMessage, ThreadSummary

log = logging.getLogger(__name__)

router = APIRouter()


def _parse_ts(raw: Any) -> datetime:
    if isinstance(raw, datetime):
        return raw
    if isinstance(raw, str):
        try:
            return datetime.fromisoformat(raw)
        except ValueError:
            pass
    return datetime.fromtimestamp(0, tz=timezone.utc)


def _thread_id_from(config: dict) -> str | None:
    return (config or {}).get("configurable", {}).get("thread_id")


def _message_to_schema(msg: Any) -> ThreadMessage | None:
    if isinstance(msg, BaseMessage):
        content = msg.content if isinstance(msg.content, str) else str(msg.content)
        return ThreadMessage(role=msg.type, content=content)
    if isinstance(msg, dict):
        role = msg.get("type") or msg.get("role")
        content = msg.get("content")
        if role and isinstance(content, str):
            return ThreadMessage(role=role, content=content)
    return None


async def _message_count(checkpointer, thread_id: str) -> int:
    """Pull the latest checkpoint for a thread and count its messages."""
    config = {"configurable": {"thread_id": thread_id}}
    tup = await checkpointer.aget_tuple(config)
    if tup is None:
        return 0
    messages = (tup.checkpoint.get("channel_values") or {}).get("messages") or []
    return len(messages)


@router.get(
    "/threads",
    response_model=list[ThreadSummary],
    summary="List known threads (most recent first).",
)
async def list_threads(
    checkpointer=Depends(get_checkpointer),
    session: AsyncSession = Depends(get_db),
) -> list[ThreadSummary]:
    rows = await list_threads_db(session)
    if rows:
        out: list[ThreadSummary] = []
        for row in rows:
            count = await _message_count(checkpointer, row.thread_id)
            out.append(
                ThreadSummary(
                    thread_id=row.thread_id,
                    display_name=row.display_name,
                    last_updated=row.last_message_at or row.created_at,
                    message_count=count,
                )
            )
        return out

    # Fallback: threads table is empty (e.g. checkpoint state pre-dates M9).
    # Iterate the checkpointer like M6 did.  Slow on large checkpoint stores,
    # but only fires once before the first /api/chat call.
    latest: dict[str, ThreadSummary] = {}
    async for tup in checkpointer.alist(config=None):
        tid = _thread_id_from(tup.config)
        if not tid:
            continue
        ts = _parse_ts(tup.checkpoint.get("ts"))
        messages = (tup.checkpoint.get("channel_values") or {}).get("messages") or []
        existing = latest.get(tid)
        if existing is None or ts > existing.last_updated:
            latest[tid] = ThreadSummary(
                thread_id=tid,
                display_name=None,
                last_updated=ts,
                message_count=len(messages),
            )
    return sorted(latest.values(), key=lambda s: s.last_updated, reverse=True)


@router.get(
    "/threads/{thread_id}",
    response_model=ThreadDetail,
    summary="Return the message history for one thread.",
    responses={404: {"description": "Thread not found"}},
)
async def get_thread(thread_id: str, checkpointer=Depends(get_checkpointer)) -> ThreadDetail:
    config = {"configurable": {"thread_id": thread_id}}
    tup = await checkpointer.aget_tuple(config)
    if tup is None:
        raise HTTPException(status_code=404, detail="thread not found")

    raw_messages = (tup.checkpoint.get("channel_values") or {}).get("messages") or []
    messages: list[ThreadMessage] = []
    for raw in raw_messages:
        coerced = _message_to_schema(raw)
        if coerced is not None:
            messages.append(coerced)

    return ThreadDetail(thread_id=thread_id, messages=messages)
