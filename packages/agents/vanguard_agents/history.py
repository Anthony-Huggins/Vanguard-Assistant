"""History management for the conversation graph.

Provides a ``trim_history_node`` that keeps the ``messages`` list in the graph
state from growing without bound.  It runs as the first node every turn:

    START → trim_history → coordinator → subagents → END

How trimming works
------------------
``history_max_turns`` (default 20, set in .env) controls how many turns to
retain.  One "turn" = one HumanMessage + one AIMessage = 2 messages.

When the list exceeds that window the oldest messages are dropped.  LangGraph's
``add_messages`` reducer recognises ``RemoveMessage`` objects — returning them
from a node deletes the matching messages from the checkpoint without
touching the rest of the list.

Summarisation note
------------------
A common enhancement is to summarise the dropped messages with an LLM call and
inject the summary as a SystemMessage so long-range context is preserved.  That
is deferred to Phase C (M15 polish) because:

  • It requires an extra LLM call every time the window is exceeded.
  • The subagents in Phase A only use ``user_message``, not ``state["messages"]``,
    so there is nothing to summarise for them yet.
  • Phase B's streaming API, once built, will be the right place to wire this in
    because the summarisation call can overlap with the streamed response.

For now the node simply drops the overflow; a TODO comment marks the hook.
"""

from __future__ import annotations

import logging

from langchain_core.messages import BaseMessage

from .settings import get_settings
from .state import ConversationState

log = logging.getLogger(__name__)

try:
    from langgraph.graph.message import RemoveMessage  # LangGraph >= 0.2
except ImportError:  # pragma: no cover — only hit on very old installs
    RemoveMessage = None  # type: ignore[assignment, misc]


def trim_history_node(state: ConversationState) -> dict:
    """Drop messages beyond the configured turn window.

    Returns an empty dict (no-op) when the history is within the limit.
    Returns a ``messages`` list of ``RemoveMessage`` objects otherwise, which
    causes the ``add_messages`` reducer to delete the old messages from the
    LangGraph checkpoint.
    """
    if RemoveMessage is None:
        # Very old LangGraph install — skip silently rather than crashing.
        return {}

    messages: list[BaseMessage] = state.get("messages", [])  # type: ignore[assignment]
    s = get_settings()
    # Each turn produces 2 messages (HumanMessage + AIMessage).
    max_messages = s.history_max_turns * 2

    if len(messages) <= max_messages:
        return {}

    overflow = len(messages) - max_messages
    to_remove = messages[:overflow]

    log.debug(
        "trim_history: dropping %d old messages (limit=%d)",
        overflow,
        max_messages,
    )

    # TODO (Phase C / M15): summarise to_remove with a brief LLM call and
    # inject the summary as a SystemMessage before the kept window so that
    # long-range context is not completely lost.

    return {"messages": [RemoveMessage(id=m.id) for m in to_remove]}
