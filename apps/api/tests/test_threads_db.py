"""Threads-table CRUD tests (M9).

Uses the in-memory SQLite from the shared ``app_db_engine`` fixture so
no Postgres or on-disk file is required.
"""

from __future__ import annotations

import pytest
from sqlalchemy.ext.asyncio import async_sessionmaker

from apps.api.db.crud import (
    get_thread,
    list_threads,
    upsert_thread_metadata,
)


@pytest.mark.asyncio
async def test_upsert_creates_and_then_updates_a_thread(app_db_engine):
    Session = async_sessionmaker(app_db_engine, expire_on_commit=False)

    async with Session() as s:
        await upsert_thread_metadata(
            s, thread_id="t1", user_message="What does Vanguard charge?"
        )
        first = await get_thread(s, "t1")
        assert first is not None
        assert first.display_name == "What does Vanguard charge?"
        first_seen = first.last_message_at
        assert first_seen is not None

    async with Session() as s:
        # Subsequent turn updates last_message_at but leaves display_name alone.
        await upsert_thread_metadata(s, thread_id="t1", user_message="follow-up")
        second = await get_thread(s, "t1")
        assert second is not None
        assert second.display_name == "What does Vanguard charge?"
        assert second.last_message_at >= first_seen


@pytest.mark.asyncio
async def test_display_name_is_truncated(app_db_engine):
    Session = async_sessionmaker(app_db_engine, expire_on_commit=False)
    long_msg = "x" * 500
    async with Session() as s:
        await upsert_thread_metadata(s, thread_id="long", user_message=long_msg)
        row = await get_thread(s, "long")
    assert row is not None
    assert row.display_name is not None
    assert len(row.display_name) <= 80
    assert row.display_name.endswith("…")


@pytest.mark.asyncio
async def test_list_threads_orders_by_recency(app_db_engine):
    import asyncio

    Session = async_sessionmaker(app_db_engine, expire_on_commit=False)
    async with Session() as s:
        await upsert_thread_metadata(s, thread_id="A", user_message="first")
        # Sleep so the timestamps differ; SQLite stores microseconds.
        await asyncio.sleep(0.01)
        await upsert_thread_metadata(s, thread_id="B", user_message="second")
        await asyncio.sleep(0.01)
        await upsert_thread_metadata(s, thread_id="A", user_message="third turn on A")

    async with Session() as s:
        rows = await list_threads(s)

    # A was bumped most recently → it should be first.
    assert [r.thread_id for r in rows] == ["A", "B"]
