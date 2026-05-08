"""POST /api/chat — non-streaming turn through the multi-agent graph.

Action-tool confirmation interrupts (``send_email`` / ``create_ticket``) are
auto-confirmed in M6 so a single HTTP request always returns a complete
response.  M8 will add SSE-driven HITL on a separate ``/api/chat/stream``
route — see the milestone plan.  The audit trail (``log_action`` JSONL) is
preserved either way, so "what did the assistant just do" is always
recoverable from disk.
"""

from __future__ import annotations

import logging
import re

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from vanguard_agents.state import fresh_turn_state

from apps.api.db.crud import upsert_thread_metadata
from apps.api.deps import get_db, get_graph
from apps.api.schemas import ChatRequest, ChatResponse, Source

log = logging.getLogger(__name__)

router = APIRouter()


# Same regexes the CLI uses to clean the response body for terminal display.
# Duplicated here (rather than imported) because ``vanguard_agents.cli`` is a
# script entry point, not a library module — coupling the API to it would be
# the wrong direction.  See ``cli.py:137``.
_BADGE_RE = re.compile(r"^\[([^\]]+)\](?:\s*\(\d+\s+tool\s+calls?\))?\s*")
_SOURCES_RE = re.compile(r"\n\nSources:\s*((?:\[[^\]]+\]\s*)+)\s*$")


def _strip_decorations(response: str) -> str:
    """Remove the ``[Agent Name] (N tool calls)`` prefix and the trailing
    ``Sources: [...]`` block from the agent's ``final_response``.  Both are
    presentation details — the API exposes structured ``agent_used`` and
    ``sources`` fields instead."""
    response = _BADGE_RE.sub("", response, count=1)
    response = _SOURCES_RE.sub("", response).rstrip()
    return response


def _build_sources(state: dict) -> list[Source]:
    """Map ``state['retrieved_chunks']`` to API ``Source`` models.

    Empty for non-Support routes — the coordinator clears the list each turn.
    """
    chunks = state.get("retrieved_chunks") or []
    out: list[Source] = []
    for chunk in chunks:
        out.append(
            Source(
                source=chunk.get("source", "?"),
                chunk_id=str(chunk.get("chunk_id", "?")),
                page=chunk.get("page"),
                score=float(chunk.get("score", 0.0)),
            )
        )
    return out


async def _invoke_with_auto_confirm(graph, initial_state: dict, config: dict) -> dict:
    """Drive the graph, auto-confirming any action-tool interrupts with "yes".

    Mirrors the CLI's :func:`vanguard_agents.cli._invoke_with_interrupt_loop`
    but without the ``input()`` prompt — there is no interactive terminal in
    a single HTTP request.  Real HITL lands in M8 over SSE.
    """
    try:
        from langgraph.types import Command
    except ImportError:
        return await graph.ainvoke(initial_state, config=config)

    try:
        from langgraph.errors import GraphInterrupt
    except ImportError:
        GraphInterrupt = None  # type: ignore[assignment, misc]

    payload: dict | "Command" = initial_state

    while True:
        try:
            result = await graph.ainvoke(payload, config=config)
            raised: list = []
        except Exception as exc:
            if GraphInterrupt is not None and isinstance(exc, GraphInterrupt):
                result = None
                raised = list(exc.interrupts)
            else:
                raise

        pending = list(raised)
        if not pending:
            snapshot = await graph.aget_state(config)
            for task in getattr(snapshot, "tasks", []) or []:
                pending.extend(getattr(task, "interrupts", []) or [])

        if not pending:
            return result  # graph finished

        for interrupt_info in pending:
            details = getattr(interrupt_info, "value", interrupt_info)
            log.info("auto-confirming action interrupt: %s", details)
        payload = Command(resume="yes")


@router.post(
    "/chat",
    response_model=ChatResponse,
    summary="Send a chat message; receive the assistant's full reply.",
    description=(
        "Runs one turn of the multi-agent graph.  Conversation history is "
        "keyed on ``thread_id`` and persisted via the LangGraph checkpointer "
        "— supplying the same id on a follow-up request continues the same "
        "thread.\n\n"
        "Action-tool confirmation interrupts (``send_email`` / "
        "``create_ticket``) are auto-confirmed in this endpoint; M8 adds "
        "real human-in-the-loop over SSE."
    ),
)
async def chat(
    req: ChatRequest,
    graph=Depends(get_graph),
    session: AsyncSession = Depends(get_db),
) -> ChatResponse:
    state = fresh_turn_state(req.message, thread_id=req.thread_id)
    config = {"configurable": {"thread_id": req.thread_id}}

    try:
        result = await _invoke_with_auto_confirm(graph, state, config)
    except Exception as exc:  # noqa: BLE001
        log.exception("graph.ainvoke failed")
        raise HTTPException(status_code=500, detail=f"{type(exc).__name__}: {exc}") from exc

    if not isinstance(result, dict):
        # Defensive: the auto-confirm loop should always return a dict on success.
        raise HTTPException(status_code=500, detail="graph returned no result")

    routing = result.get("routing")
    if routing is None:
        raise HTTPException(status_code=500, detail="graph produced no routing decision")

    # Record thread metadata so the sidebar can display human-readable names.
    # Done after the turn succeeds — if the graph blew up, we don't want to
    # leak a half-baked thread row.
    await upsert_thread_metadata(
        session, thread_id=req.thread_id, user_message=req.message
    )

    final_response = result.get("final_response") or "(no response)"
    return ChatResponse(
        reply=_strip_decorations(final_response),
        sources=_build_sources(result),
        agent_used=routing.agent,
        routing_reason=routing.reason or "",
    )
