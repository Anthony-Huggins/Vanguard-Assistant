"""SSE streaming routes (M8) — incremental tokens + tool/agent traces.

``POST /api/chat/stream``
    Start a new turn.  Streams translated events (see ``apps.api.streaming``).
    If the graph pauses on an action confirmation, emits
    ``confirmation_required`` and closes.  The graph state stays durable in
    the checkpointer so a later POST to ``/api/chat/resume`` can finish it.

``POST /api/chat/resume``
    Resume a paused turn with the user's answer ("yes" / "no").  Same SSE
    taxonomy.  Emits ``done`` once the graph finishes.

The SSE protocol is one-way (server → client), so we model the
human-in-the-loop confirmation as two separate stream calls rather than
trying to round-trip an answer over the same connection.
"""

from __future__ import annotations

import json
import logging
from typing import Any, AsyncIterator

from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from sse_starlette.sse import EventSourceResponse

from vanguard_agents.state import fresh_turn_state

from apps.api.db.crud import upsert_thread_metadata
from apps.api.deps import get_db, get_graph
from apps.api.schemas import ChatRequest, ResumeRequest
from apps.api.streaming import extract_pending_interrupt, translate_stream

log = logging.getLogger(__name__)

router = APIRouter()


def _sse(event: str, payload: Any) -> dict:
    """Render an SSE record in the shape ``EventSourceResponse`` expects."""
    return {"event": event, "data": json.dumps(payload, default=str)}


async def _stream_turn(
    graph,
    payload: Any,
    config: dict,
) -> AsyncIterator[dict]:
    """Drive one graph invocation through the translator + handle the tail.

    Yields SSE-shaped dicts.  Catches exceptions to emit a single ``error``
    event so the client never sees a truncated stream silently.

    When LangGraph hits an ``interrupt()`` call it raises ``GraphInterrupt``
    through ``astream_events`` — that is NOT a real error.  We catch it
    separately so we still fall through to the ``aget_state`` probe that
    emits the ``confirmation_required`` event.
    """
    # Import once so the isinstance check below stays cheap.
    try:
        from langgraph.errors import GraphInterrupt  # LangGraph >= 0.2
    except ImportError:
        GraphInterrupt = None  # type: ignore[assignment,misc]

    try:
        raw_events = graph.astream_events(payload, config=config, version="v2")
        async for ev in translate_stream(raw_events):
            yield _sse(ev["type"], ev["data"])
    except Exception as exc:  # noqa: BLE001
        if GraphInterrupt is not None and isinstance(exc, GraphInterrupt):
            # Graph paused at an interrupt — fall through to aget_state.
            log.debug("astream_events raised GraphInterrupt; probing for pending interrupt")
        else:
            log.exception("astream_events failed")
            yield _sse("error", {"message": f"{type(exc).__name__}: {exc}"})
            return

    # The iterator drained (or raised GraphInterrupt) — check for pending interrupt.
    try:
        snapshot = await graph.aget_state(config)
    except Exception as exc:  # noqa: BLE001
        log.exception("aget_state failed after stream drain")
        yield _sse("error", {"message": f"{type(exc).__name__}: {exc}"})
        return

    pending = extract_pending_interrupt(snapshot)
    if pending is not None:
        yield _sse("confirmation_required", pending)
    else:
        yield _sse("done", {})


@router.post(
    "/chat/stream",
    summary="Start a turn and stream translated agent/token events.",
    description=(
        "Server-Sent Events.  Event taxonomy: ``agent_start``, ``agent_end``, "
        "``token``, ``tool_start``, ``tool_end``, ``citation``, "
        "``confirmation_required``, ``done``, ``error``.  If the Action "
        "agent needs confirmation, the stream emits ``confirmation_required`` "
        "and closes — POST to ``/api/chat/resume`` to finish the turn."
    ),
)
async def chat_stream(
    req: ChatRequest,
    graph=Depends(get_graph),
    session: AsyncSession = Depends(get_db),
):
    # Upsert thread metadata up-front so the sidebar reflects activity even
    # if the stream errors mid-flight.  ``display_name`` is set on first turn
    # and left alone on subsequent ones.
    await upsert_thread_metadata(
        session, thread_id=req.thread_id, user_message=req.message
    )

    state = fresh_turn_state(req.message, thread_id=req.thread_id)
    config = {
        "configurable": {"thread_id": req.thread_id},
        "run_name": f"chat/{req.thread_id[:8]}",
        "tags": [f"thread:{req.thread_id}"],
        "metadata": {"thread_id": req.thread_id, "turn": "initial"},
    }
    return EventSourceResponse(_stream_turn(graph, state, config))


@router.post(
    "/chat/resume",
    summary="Answer a pending confirmation and stream the rest of the turn.",
)
async def chat_resume(req: ResumeRequest, graph=Depends(get_graph)):
    from langgraph.types import Command

    config = {
        "configurable": {"thread_id": req.thread_id},
        "run_name": f"chat/{req.thread_id[:8]} (resume)",
        "tags": [f"thread:{req.thread_id}"],
        "metadata": {"thread_id": req.thread_id, "turn": "resume"},
    }
    payload = Command(resume=req.answer)
    return EventSourceResponse(_stream_turn(graph, payload, config))
