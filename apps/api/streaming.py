"""Translate raw LangGraph ``astream_events(version="v2")`` output to a
clean SSE event taxonomy for the frontend.

The raw event stream is verbose and tightly coupled to LangChain internals
(``on_chat_model_stream``, ``on_chain_start`` per node, ``on_tool_start``
per tool, …).  The frontend needs a small, stable taxonomy that won't
churn every time we tweak a node:

  - ``agent_start``         {agent, routing_reason}
  - ``agent_end``           {agent}
  - ``token``               {text}                          (assistant reply text only)
  - ``tool_start``          {tool, args}
  - ``tool_end``            {tool, output_preview}
  - ``citation``            {sources: Source[]}             (Support agent only)
  - ``confirmation_required`` {question, tool, args}        (after stream pauses on interrupt)
  - ``done``                {}
  - ``error``               {message}

This module owns *only* the translator.  The router decides when to emit
``confirmation_required`` (after the iterator drains, by querying
``graph.aget_state``) and ``done`` (when there are no pending interrupts).
"""

from __future__ import annotations

import logging
from typing import Any, AsyncIterator, Iterable

log = logging.getLogger(__name__)

# Names of the four subagent nodes whose token streams we forward as
# user-facing assistant text.  The coordinator runs an LLM too (for routing)
# but its tokens are structured-output classifications — never user-facing
# reply text.
SUBAGENT_NODES: frozenset[str] = frozenset({"support", "research", "action", "clarify"})

# Subagents that intentionally skip the LLM and return final_response as a
# plain Python string.  For these nodes the EventTranslator synthesises a
# single ``token`` event from the node's output dict so the frontend bubble
# isn't empty (there are no ``on_chat_model_stream`` events for them).
NO_LLM_SUBAGENTS: frozenset[str] = frozenset({"clarify"})

# Nodes whose ``on_chain_*`` events drive ``agent_start`` / ``agent_end``.
# Wider than ``SUBAGENT_NODES`` because the M11 graph viz wants the
# coordinator to pulse on every turn, even though its tokens stay filtered.
TRACED_NODES: frozenset[str] = SUBAGENT_NODES | frozenset({"coordinator"})

_PREVIEW_CHARS = 240


def _preview(value: Any) -> str:
    text = str(value)
    if len(text) <= _PREVIEW_CHARS:
        return text
    return text[: _PREVIEW_CHARS - 1] + "…"


def _serialize_chunk(chunk: dict | Any) -> dict:
    """Map a ``RetrievedChunk`` (TypedDict) into the API ``Source`` shape."""
    if not isinstance(chunk, dict):
        return {}
    return {
        "source": chunk.get("source", "?"),
        "chunk_id": str(chunk.get("chunk_id", "?")),
        "page": chunk.get("page"),
        "score": float(chunk.get("score", 0.0)),
    }


class EventTranslator:
    """Stateful translator: feed it raw events, get back zero or more
    translated SSE events per call.

    Tracks the active subagent so we can attribute streamed tokens
    (the coordinator's LLM tokens are filtered out by checking
    ``langgraph_node`` metadata) and so ``agent_start`` carries the
    routing reason captured from the coordinator's output.
    """

    def __init__(self) -> None:
        self.active_agent: str | None = None
        self.routing_reason: str = ""

    def feed(self, raw: dict) -> Iterable[dict]:
        ev = raw.get("event")
        name = raw.get("name")
        data = raw.get("data") or {}
        meta = raw.get("metadata") or {}
        node = meta.get("langgraph_node")

        # Traced node started (coordinator or subagent) — emit agent_start.
        if ev == "on_chain_start" and name in TRACED_NODES:
            if name in SUBAGENT_NODES:
                self.active_agent = name
            yield {
                "type": "agent_start",
                "data": {"agent": name, "routing_reason": self.routing_reason},
            }

        # Coordinator finished — stash the routing reason so the next
        # subagent's agent_start can carry it, then emit agent_end for the
        # coordinator itself.
        if ev == "on_chain_end" and name == "coordinator":
            output = data.get("output") or {}
            routing = output.get("routing") if isinstance(output, dict) else None
            reason = getattr(routing, "reason", None)
            if reason is None and isinstance(routing, dict):
                reason = routing.get("reason")
            if reason:
                self.routing_reason = str(reason)
            yield {"type": "agent_end", "data": {"agent": name}}

        # Subagent finished — emit any retrieved chunks then agent_end.
        # For agents that intentionally skip the LLM (NO_LLM_SUBAGENTS, i.e.
        # clarify), synthesise a single token from final_response so the
        # frontend bubble isn't empty (there are no on_chat_model_stream
        # events for them).
        if ev == "on_chain_end" and name in SUBAGENT_NODES:
            output = data.get("output") if isinstance(data.get("output"), dict) else {}
            if name in NO_LLM_SUBAGENTS:
                final_text = (output or {}).get("final_response")
                if isinstance(final_text, str) and final_text.strip():
                    yield {"type": "token", "data": {"text": final_text.strip()}}
            chunks = (output or {}).get("retrieved_chunks") or []
            if chunks:
                yield {
                    "type": "citation",
                    "data": {"sources": [_serialize_chunk(c) for c in chunks]},
                }
            yield {"type": "agent_end", "data": {"agent": name}}
            self.active_agent = None

        # LLM token — only forward if a subagent is active. Coordinator
        # tokens (structured-output) have node="coordinator" and are dropped.
        if ev == "on_chat_model_stream" and node in SUBAGENT_NODES:
            chunk = data.get("chunk")
            text = getattr(chunk, "content", None)
            if isinstance(text, str) and text:
                yield {"type": "token", "data": {"text": text}}

        # Tool calls — name is the tool name in v2 events.
        if ev == "on_tool_start":
            yield {
                "type": "tool_start",
                "data": {
                    "tool": name or "?",
                    "args": data.get("input") if isinstance(data.get("input"), dict) else {},
                },
            }
        if ev == "on_tool_end":
            yield {
                "type": "tool_end",
                "data": {
                    "tool": name or "?",
                    "output_preview": _preview(data.get("output", "")),
                },
            }


async def translate_stream(
    raw_events: AsyncIterator[dict],
) -> AsyncIterator[dict]:
    """Async generator: feed raw events through an EventTranslator."""
    translator = EventTranslator()
    async for raw in raw_events:
        for ev in translator.feed(raw):
            yield ev


def extract_pending_interrupt(snapshot: Any) -> dict | None:
    """Pull the first pending interrupt off a graph state snapshot.

    Returns ``None`` if the graph has no pending interrupts.  Used by the
    streaming router to decide between emitting ``confirmation_required``
    or ``done`` after the event iterator drains.
    """
    pending: list = []
    for task in getattr(snapshot, "tasks", []) or []:
        pending.extend(getattr(task, "interrupts", []) or [])
    if not pending:
        return None

    first = pending[0]
    details = getattr(first, "value", first)
    if isinstance(details, dict):
        return {
            "question": details.get("question", "Confirm action?"),
            "tool": details.get("tool", "?"),
            "args": details.get("args") if isinstance(details.get("args"), dict) else {},
        }
    return {"question": str(details), "tool": "?", "args": {}}
