"""Unit tests for the LangGraph → SSE event translator.

The translator is the brittle piece — it depends on the exact shape of
``astream_events(version="v2")`` output.  Pin the contract here so future
LangChain bumps that change the event shape fail loudly.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from apps.api.streaming import (
    EventTranslator,
    extract_pending_interrupt,
    translate_stream,
)


def _stream_event(name: str) -> object:
    """Stand-in for an LLM ``BaseMessageChunk`` with ``.content``."""
    return SimpleNamespace(content=name)


@pytest.mark.asyncio
async def test_translator_emits_agent_start_with_routing_reason():
    """Coordinator pulses on every turn (M11). Routing reason captured at
    its ``on_chain_end`` and forwarded into the next subagent's
    ``agent_start``."""
    raw = [
        {
            "event": "on_chain_start",
            "name": "coordinator",
            "data": {},
            "metadata": {"langgraph_node": "coordinator"},
        },
        {
            "event": "on_chain_end",
            "name": "coordinator",
            "data": {
                "output": {
                    "routing": SimpleNamespace(
                        agent="support", reason="user asked about fees"
                    )
                }
            },
            "metadata": {},
        },
        {
            "event": "on_chain_start",
            "name": "support",
            "data": {},
            "metadata": {"langgraph_node": "support"},
        },
    ]

    t = EventTranslator()
    out: list[dict] = []
    for r in raw:
        out.extend(t.feed(r))

    assert out == [
        {
            "type": "agent_start",
            "data": {"agent": "coordinator", "routing_reason": ""},
        },
        {
            "type": "agent_end",
            "data": {"agent": "coordinator"},
        },
        {
            "type": "agent_start",
            "data": {"agent": "support", "routing_reason": "user asked about fees"},
        },
    ]


@pytest.mark.asyncio
async def test_translator_pulses_coordinator_on_every_turn():
    """The coordinator emits ``agent_start``/``agent_end`` even when no
    subagent runs (e.g. malformed routing). M11 graph viz needs this."""
    raw = [
        {
            "event": "on_chain_start",
            "name": "coordinator",
            "data": {},
            "metadata": {"langgraph_node": "coordinator"},
        },
        {
            "event": "on_chain_end",
            "name": "coordinator",
            "data": {"output": {}},
            "metadata": {},
        },
    ]

    t = EventTranslator()
    out = [e for r in raw for e in t.feed(r)]

    assert [e["type"] for e in out] == ["agent_start", "agent_end"]
    assert out[0]["data"]["agent"] == "coordinator"
    assert out[1]["data"]["agent"] == "coordinator"


@pytest.mark.asyncio
async def test_translator_drops_coordinator_tokens_keeps_subagent_tokens():
    """Coordinator's structured-output tokens must NOT reach the client."""
    raw = [
        # Coordinator's LLM stream — should be dropped (it's classification).
        {
            "event": "on_chat_model_stream",
            "name": "ChatOpenAI",
            "data": {"chunk": _stream_event("{")},
            "metadata": {"langgraph_node": "coordinator"},
        },
        # Subagent starts.
        {
            "event": "on_chain_start",
            "name": "support",
            "data": {},
            "metadata": {"langgraph_node": "support"},
        },
        # Subagent's LLM stream — should pass through as token events.
        {
            "event": "on_chat_model_stream",
            "name": "ChatOpenAI",
            "data": {"chunk": _stream_event("Hello")},
            "metadata": {"langgraph_node": "support"},
        },
        {
            "event": "on_chat_model_stream",
            "name": "ChatOpenAI",
            "data": {"chunk": _stream_event(" world")},
            "metadata": {"langgraph_node": "support"},
        },
    ]

    t = EventTranslator()
    out: list[dict] = []
    for r in raw:
        out.extend(t.feed(r))

    types = [e["type"] for e in out]
    assert types == ["agent_start", "token", "token"]
    assert out[1]["data"] == {"text": "Hello"}
    assert out[2]["data"] == {"text": " world"}


@pytest.mark.asyncio
async def test_translator_emits_citation_then_agent_end():
    raw = [
        {
            "event": "on_chain_start",
            "name": "support",
            "data": {},
            "metadata": {"langgraph_node": "support"},
        },
        {
            "event": "on_chain_end",
            "name": "support",
            "data": {
                "output": {
                    "retrieved_chunks": [
                        {
                            "text": "...",
                            "source": "fees.pdf",
                            "chunk_id": "12",
                            "score": 0.55,
                            "page": 3,
                        }
                    ],
                    "final_response": "ignored",
                }
            },
            "metadata": {},
        },
    ]

    t = EventTranslator()
    out: list[dict] = []
    for r in raw:
        out.extend(t.feed(r))

    types = [e["type"] for e in out]
    assert types == ["agent_start", "citation", "agent_end"]
    sources = out[1]["data"]["sources"]
    assert sources == [
        {"source": "fees.pdf", "chunk_id": "12", "page": 3, "score": 0.55}
    ]


@pytest.mark.asyncio
async def test_translator_emits_token_for_clarify_no_llm():
    """clarify_node never calls an LLM, so there are no on_chat_model_stream
    events.  The translator must synthesise a token from the node's
    final_response so the frontend bubble isn't empty."""
    clarify_text = "I need more info — could you rephrase your question?"
    raw = [
        {
            "event": "on_chain_start",
            "name": "clarify",
            "data": {},
            "metadata": {"langgraph_node": "clarify"},
        },
        {
            "event": "on_chain_end",
            "name": "clarify",
            "data": {"output": {"final_response": clarify_text}},
            "metadata": {},
        },
    ]

    t = EventTranslator()
    out: list[dict] = []
    for r in raw:
        out.extend(t.feed(r))

    types = [e["type"] for e in out]
    assert types == ["agent_start", "token", "agent_end"]
    assert out[1]["data"]["text"] == clarify_text


@pytest.mark.asyncio
async def test_translator_handles_tool_start_and_tool_end():
    raw = [
        {
            "event": "on_tool_start",
            "name": "send_email",
            "data": {"input": {"recipient": "x@y.com", "subject": "hi"}},
            "metadata": {},
        },
        {
            "event": "on_tool_end",
            "name": "send_email",
            "data": {"output": "OK queued"},
            "metadata": {},
        },
    ]

    t = EventTranslator()
    out = [e for r in raw for e in t.feed(r)]

    assert [e["type"] for e in out] == ["tool_start", "tool_end"]
    assert out[0]["data"]["tool"] == "send_email"
    assert out[0]["data"]["args"]["recipient"] == "x@y.com"
    assert out[1]["data"]["output_preview"] == "OK queued"


@pytest.mark.asyncio
async def test_translate_stream_passes_events_in_order():
    """Smoke test of the async generator."""

    async def fake_raw():
        yield {
            "event": "on_chain_start",
            "name": "support",
            "data": {},
            "metadata": {"langgraph_node": "support"},
        }
        yield {
            "event": "on_chat_model_stream",
            "name": "x",
            "data": {"chunk": _stream_event("hi")},
            "metadata": {"langgraph_node": "support"},
        }
        yield {
            "event": "on_chain_end",
            "name": "support",
            "data": {"output": {}},
            "metadata": {},
        }

    out: list[dict] = []
    async for ev in translate_stream(fake_raw()):
        out.append(ev)

    assert [e["type"] for e in out] == ["agent_start", "token", "agent_end"]


def test_extract_pending_interrupt_returns_none_when_clean():
    snapshot = SimpleNamespace(tasks=[])
    assert extract_pending_interrupt(snapshot) is None


def test_extract_pending_interrupt_returns_first_dict_payload():
    interrupt = SimpleNamespace(
        value={
            "question": "Send email to x@y.com?",
            "tool": "send_email",
            "args": {"recipient": "x@y.com"},
        }
    )
    snapshot = SimpleNamespace(
        tasks=[SimpleNamespace(interrupts=[interrupt])]
    )
    pending = extract_pending_interrupt(snapshot)
    assert pending is not None
    assert pending["question"] == "Send email to x@y.com?"
    assert pending["tool"] == "send_email"
    assert pending["args"] == {"recipient": "x@y.com"}


@pytest.mark.asyncio
async def test_chat_stream_route_sends_done_when_no_interrupts(
    client, mock_graph, mock_checkpointer
):
    """End-to-end-ish: post to /api/chat/stream with a mocked graph that
    yields a small event sequence, expect the SSE body to contain
    ``agent_start`` ... ``done``."""

    async def fake_astream(payload, config, version):
        yield {
            "event": "on_chain_start",
            "name": "support",
            "data": {},
            "metadata": {"langgraph_node": "support"},
        }
        yield {
            "event": "on_chat_model_stream",
            "name": "ChatOpenAI",
            "data": {"chunk": _stream_event("done.")},
            "metadata": {"langgraph_node": "support"},
        }
        yield {
            "event": "on_chain_end",
            "name": "support",
            "data": {"output": {}},
            "metadata": {},
        }

    mock_graph.astream_events = fake_astream
    mock_graph.aget_state = AsyncMock(return_value=SimpleNamespace(tasks=[]))

    async with client.stream(
        "POST",
        "/api/chat/stream",
        json={"thread_id": "t1", "message": "hi"},
    ) as response:
        assert response.status_code == 200
        body = await response.aread()

    text = body.decode()
    assert "event: agent_start" in text
    assert "event: token" in text
    assert "event: agent_end" in text
    assert "event: done" in text
    # And no spurious confirmation_required.
    assert "event: confirmation_required" not in text
