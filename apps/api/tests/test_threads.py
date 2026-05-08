"""GET /api/threads and GET /api/threads/{id} — list + detail tests.

The checkpointer is mocked: ``alist`` yields fake CheckpointTuples, and
``aget_tuple`` returns a single tuple by thread_id.  No real SQLite file is
opened.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from langchain_core.messages import AIMessage, HumanMessage


def _tup(thread_id: str, ts: str, messages: list) -> SimpleNamespace:
    """Stand-in for ``langgraph.checkpoint.base.CheckpointTuple`` — only the
    fields the router reads need to be present."""
    return SimpleNamespace(
        config={"configurable": {"thread_id": thread_id}},
        checkpoint={"ts": ts, "channel_values": {"messages": messages}},
        metadata={},
        parent_config=None,
        pending_writes=[],
    )


@pytest.mark.asyncio
async def test_list_threads_groups_by_id_and_sorts_desc(client, mock_checkpointer):
    """Two checkpoints for thread A, one for B → newest-first list."""
    rows = [
        _tup("A", "2026-05-01T10:00:00+00:00", [HumanMessage(content="hi")]),
        _tup(
            "A",
            "2026-05-02T12:00:00+00:00",
            [HumanMessage(content="hi"), AIMessage(content="hello")],
        ),
        _tup("B", "2026-05-03T08:00:00+00:00", [HumanMessage(content="hey")]),
    ]

    async def _alist(config=None, **_):
        for r in rows:
            yield r

    mock_checkpointer.alist = _alist

    response = await client.get("/api/threads")
    assert response.status_code == 200
    body = response.json()

    assert [t["thread_id"] for t in body] == ["B", "A"]
    a = next(t for t in body if t["thread_id"] == "A")
    assert a["message_count"] == 2  # latest checkpoint for A had 2 messages


@pytest.mark.asyncio
async def test_get_thread_returns_404_for_unknown(client):
    response = await client.get("/api/threads/does-not-exist")
    assert response.status_code == 404


@pytest.mark.asyncio
async def test_get_thread_returns_messages(client, mock_checkpointer):
    """Known thread returns its message list with role/content."""
    tup = _tup(
        "demo",
        "2026-05-06T09:00:00+00:00",
        [
            HumanMessage(content="What does Vanguard charge?"),
            AIMessage(content="Around 0.04% on average."),
        ],
    )
    mock_checkpointer.aget_tuple = AsyncMock(return_value=tup)

    response = await client.get("/api/threads/demo")
    assert response.status_code == 200
    body = response.json()

    assert body["thread_id"] == "demo"
    assert len(body["messages"]) == 2
    assert body["messages"][0]["role"] == "human"
    assert body["messages"][0]["content"] == "What does Vanguard charge?"
    assert body["messages"][1]["role"] == "ai"
