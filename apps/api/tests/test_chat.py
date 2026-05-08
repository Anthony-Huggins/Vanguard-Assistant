"""POST /api/chat — happy-path test against a mocked graph."""

from __future__ import annotations

import pytest


@pytest.mark.asyncio
async def test_chat_returns_structured_response(client, mock_graph):
    """Posting a message yields the agent's reply with sources + routing.

    The mocked graph returns the canned Support Agent reply from conftest.
    Verifies that ``[Agent Name]`` and ``Sources: [...]`` decorations are
    stripped from ``reply`` and that ``sources`` mirrors
    ``state['retrieved_chunks']``.
    """
    response = await client.post(
        "/api/chat",
        json={"thread_id": "thread-abc", "message": "What does Vanguard charge?"},
    )

    assert response.status_code == 200
    body = response.json()

    assert body["agent_used"] == "support"
    assert "fee structure" in body["routing_reason"]

    # Reply is decoration-free.
    assert not body["reply"].startswith("[Support Agent]")
    assert "Sources:" not in body["reply"]
    assert "expense ratio" in body["reply"]

    # Sources reflect retrieved_chunks.
    assert len(body["sources"]) == 2
    first = body["sources"][0]
    assert first["source"] == "vanguard-fees.pdf"
    assert first["chunk_id"] == "12"
    assert first["page"] == 3
    assert first["score"] == pytest.approx(0.61)

    # The graph was called exactly once with the right config.
    mock_graph.ainvoke.assert_awaited_once()
    _state, kwargs = mock_graph.ainvoke.await_args
    assert kwargs["config"]["configurable"]["thread_id"] == "thread-abc"


@pytest.mark.asyncio
async def test_chat_validates_request_body(client):
    """Empty thread_id and empty message both fail Pydantic validation."""
    bad = await client.post("/api/chat", json={"thread_id": "", "message": "hi"})
    assert bad.status_code == 422

    bad2 = await client.post("/api/chat", json={"thread_id": "ok", "message": ""})
    assert bad2.status_code == 422
