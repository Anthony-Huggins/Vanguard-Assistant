"""Shared ReAct-style tool-calling loop for Research and Action subagents.

Both subagents have the same shape:

    system prompt + user message
        ↓
    LLM with tools bound
        ↓ (if tool_calls present)
    execute tool(s) → append ToolMessage(s) → call LLM again
        ↓
    final assistant message with no tool_calls

Factored out so the per-subagent module only owns its prompt and tool list.

Confirmation interrupts (M5)
-----------------------------
Pass ``confirm_tools`` with a set of tool names that require the user to
confirm before execution (e.g. ``send_email``, ``create_ticket``).  When any
of those tools is about to fire, ``langgraph.types.interrupt()`` is called.

LangGraph catches the resulting ``NodeInterrupt``, persists graph state, and
returns from ``ainvoke`` / raises ``GraphInterrupt``.  The CLI (or API) then
prompts the user, then resumes via ``graph.ainvoke(Command(resume=answer), …)``.
The ``interrupt()`` call returns the user's answer string; the loop proceeds
only if it starts with "y".

A checkpointer is required for interrupts to be resumable.  If the graph was
compiled without one (``--no-checkpoint`` flag), the interrupt still fires and
the user is prompted, but the state cannot be persisted between the pause and
the resume.  In practice this means ``--no-checkpoint`` disables confirmation
for action tools — the tool runs without asking.  This is intentional: dev
sessions shouldn't be interrupted by prompts during automated smoke tests.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.tools import BaseTool

from ..factories.llm import build_chat_model
from ..retry import wrap_with_retry
from ..settings import get_settings
from ..state import ToolResult

log = logging.getLogger(__name__)

# Hard cap on tool-call rounds to prevent runaway loops.
MAX_TOOL_ROUNDS = 6


def _normalize_tool_output(output: Any) -> str:
    """Coerce various tool-output shapes into a clean string for the LLM.

    MCP tools return a list of content blocks like
    ``[{"type": "text", "text": "...", "id": "..."}]``. We surface just
    the text payload(s) so the LLM doesn't have to peel the envelope.
    """
    if isinstance(output, str):
        return output
    if (
        isinstance(output, list)
        and output
        and all(isinstance(b, dict) for b in output)
    ):
        texts = [b.get("text", "") for b in output if b.get("type") == "text"]
        if texts:
            return "\n".join(texts)
    return json.dumps(output, default=str)


async def run_tool_loop(
    *,
    system_prompt: str,
    tools: list[BaseTool],
    user_message: str,
    history: list[BaseMessage] | None = None,
    confirm_tools: frozenset[str] = frozenset(),
) -> tuple[str, list[BaseMessage], list[ToolResult]]:
    """Run the agent → tools → agent loop until the LLM stops calling tools.

    Args:
        system_prompt:  System message text for the LLM.
        tools:          LangChain tools the LLM can call.
        user_message:   The user's request for this turn (used as a fallback
                        when ``history`` is empty / not supplied).
        history:        Full conversation history from ``state["messages"]``.
                        When provided, the LLM sees prior turns so multi-turn
                        flows work — e.g. the user supplying an email body
                        in a follow-up message after the agent asked for it.
        confirm_tools:  Tool names that require user confirmation before
                        executing.  Triggers a LangGraph ``interrupt()`` so
                        the CLI / API can prompt the user.

    Returns:
        final_text:    The assistant's final answer text.
        new_messages:  Every message produced this turn (AIMessages and
                       ToolMessages), suitable for appending to graph state.
        tool_results:  Structured record of each tool call for tracing / UI.
    """
    s = get_settings()
    model = build_chat_model(max_tokens=s.llm_max_tokens_subagent)
    llm_with_tools = model.bind_tools(tools) if tools else model
    llm_with_retry = wrap_with_retry(llm_with_tools)

    by_name: dict[str, BaseTool] = {t.name: t for t in tools}
    convo: list[BaseMessage] = [SystemMessage(content=system_prompt)]
    if history:
        convo.extend(history)
    else:
        convo.append(HumanMessage(content=user_message))
    new_messages: list[BaseMessage] = []
    tool_results: list[ToolResult] = []

    for round_idx in range(MAX_TOOL_ROUNDS):
        response = await llm_with_retry.ainvoke(convo)
        convo.append(response)
        new_messages.append(response)

        tool_calls = getattr(response, "tool_calls", None) or []
        if not tool_calls:
            break

        for call in tool_calls:
            name = call["name"]
            args = call.get("args") or {}

            # ── Confirmation interrupt ────────────────────────────────────
            if name in confirm_tools:
                answer = _maybe_interrupt(name, args)
                if answer is not None and not answer.lower().startswith("y"):
                    output = f"[cancelled] The user declined to run {name!r}."
                    error: str | None = "user_cancelled"
                    tool_msg = ToolMessage(content=output, tool_call_id=call["id"])
                    convo.append(tool_msg)
                    new_messages.append(tool_msg)
                    tool_results.append(
                        {"name": name, "args": args, "output": output, "error": error}
                    )
                    continue  # skip to next tool call

            # ── Execute ───────────────────────────────────────────────────
            tool = by_name.get(name)
            if tool is None:
                output = f"[error] unknown tool: {name}"
                error = output
            else:
                try:
                    raw = await tool.ainvoke(args)
                    output = _normalize_tool_output(raw)
                    error = None
                except Exception as exc:  # noqa: BLE001
                    log.exception("tool %s failed", name)
                    output = f"[error] {type(exc).__name__}: {exc}"
                    error = str(exc)

            tool_msg = ToolMessage(content=output, tool_call_id=call["id"])
            convo.append(tool_msg)
            new_messages.append(tool_msg)
            tool_results.append(
                {"name": name, "args": args, "output": output, "error": error}
            )
    else:
        # Loop exhausted without a tool-free response — synthesize a stop.
        log.warning("tool loop hit MAX_TOOL_ROUNDS=%d", MAX_TOOL_ROUNDS)
        stop = AIMessage(content="(stopped: too many tool-call rounds)")
        convo.append(stop)
        new_messages.append(stop)
        return stop.content, new_messages, tool_results

    final = convo[-1]
    final_text = final.content if isinstance(final.content, str) else str(final.content)
    return final_text, new_messages, tool_results


def _maybe_interrupt(tool_name: str, args: dict) -> str | None:
    """Call LangGraph's ``interrupt()`` to pause for user confirmation.

    Returns the user's answer string (lowercased), or ``None`` if interrupts
    are not available (e.g. very old LangGraph install).

    The caller should treat ``None`` as "proceed" so that the tool executes
    without asking in environments where interrupts cannot be wired up.
    """
    try:
        from langgraph.types import interrupt  # LangGraph >= 0.2.50
    except ImportError:
        log.warning("langgraph.types.interrupt not available — skipping confirmation for %s", tool_name)
        return None

    # Build a human-readable summary of the pending action.
    arg_summary = ", ".join(f"{k}={v!r}" for k, v in list(args.items())[:3])
    question = f"About to call {tool_name}({arg_summary}). Proceed?"

    answer: str = interrupt({
        "tool": tool_name,
        "args": args,
        "question": question,
    })
    return str(answer).lower()


def _format_tool_call_summary(results: list[ToolResult]) -> str:
    """Compact debug-friendly trace of which tools fired with what args."""
    if not results:
        return ""
    lines = []
    for r in results:
        status = "ok" if r["error"] is None else f"ERR: {r['error']}"
        lines.append(f"  - {r['name']}({r['args']}) → {status}")
    return "Tool trace:\n" + "\n".join(lines)


# Re-exported helper so subagent modules can decorate their final response
# with a brief "[Research Agent]" badge — useful before we have a real UI.
def with_agent_badge(label: str, body: str, results: list[ToolResult]) -> str:
    badge = f"[{label}]" + (f" ({len(results)} tool call{'s' if len(results) != 1 else ''})" if results else "")
    return f"{badge} {body}"
