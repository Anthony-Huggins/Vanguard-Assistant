"""Action Agent — performs side-effects (email, tickets, audit logs) via MCP.

Before executing ``send_email`` or ``create_ticket`` the agent pauses and asks
the user to confirm via a LangGraph interrupt.  The CLI and future streaming
API catch the interrupt, prompt the user, and resume the graph with the user's
answer.  ``log_action`` never requires confirmation because it is read-only
from the user's perspective (audit-trail writes are always safe to proceed).
"""

from __future__ import annotations

import logging

from langchain_core.messages import AIMessage

from ..mcp_client import get_action_tools
from ..state import ConversationState
from ._tool_loop import run_tool_loop, with_agent_badge

log = logging.getLogger(__name__)

# Tools that mutate external systems and therefore require explicit user
# confirmation before execution.
_CONFIRM_TOOLS: frozenset[str] = frozenset({"send_email", "create_ticket"})

ACTION_SYSTEM_PROMPT = """\
You are the Vanguard Action Agent. You execute side-effecting tasks on
behalf of the user.

You have three tools:

  - send_email(recipient, subject, body): send an email.
  - create_ticket(summary, description):  open a Jira-style ticket.
  - log_action(action, details):          append a structured audit entry
                                          (``details`` must be a JSON string).

Before calling any tool, confirm you have the required arguments. If the
user's request is missing a required field (e.g. an email recipient), ask
them for it instead of guessing. Always call ``log_action`` after a
mutating call so the audit trail is complete.

After executing the tool(s), summarize what was done in plain English and
include any returned ids (ticket id, send_at timestamp).
"""


async def action_node(state: ConversationState) -> dict:
    user_message = state["user_message"]

    try:
        tools = await get_action_tools()
    except Exception as exc:  # noqa: BLE001
        log.exception("could not connect to MCP server for action tools")
        msg = (
            "I can't reach the action tools right now "
            f"({type(exc).__name__}). The MCP server may be offline."
        )
        return {
            "final_response": msg,
            "messages": [AIMessage(content=msg)],
        }

    final_text, new_messages, tool_results = await run_tool_loop(
        system_prompt=ACTION_SYSTEM_PROMPT,
        tools=tools,
        user_message=user_message,
        history=state.get("messages"),
        confirm_tools=_CONFIRM_TOOLS,
    )
    decorated = with_agent_badge("Action Agent", final_text, tool_results)

    return {
        "final_response": decorated,
        "messages": new_messages,
        "tool_results": tool_results,
    }
