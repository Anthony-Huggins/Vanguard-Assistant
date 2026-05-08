"""Wire the multi-agent StateGraph and expose a ``build_graph`` factory.

Edges:
    START → trim_history → coordinator → (support | research | action | clarify) → END

``trim_history`` runs first every turn so the ``messages`` list in the
checkpoint never exceeds ``settings.history_max_turns * 2`` entries.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from langgraph.graph import END, START, StateGraph

from .coordinator import coordinator_node, route_after_coordinator
from .history import trim_history_node
from .state import ConversationState
from .subagents import action_node, clarify_node, research_node, support_node

if TYPE_CHECKING:
    from langgraph.checkpoint.base import BaseCheckpointSaver


def build_graph(*, checkpointer: "BaseCheckpointSaver | None" = None):
    """Compile the multi-agent graph.

    Pass a ``checkpointer`` (e.g. ``SqliteSaver``) to enable durable
    multi-turn memory keyed on ``thread_id``.  Pass ``None`` for one-shot
    invocations and unit tests.

    The confirmation interrupt in the Action Agent requires a checkpointer to
    be resumable — if ``checkpointer=None`` the interrupt still fires but the
    graph state cannot be saved/restored between the pause and the resume.
    """
    builder = StateGraph(ConversationState)

    builder.add_node("trim_history", trim_history_node)
    builder.add_node("coordinator", coordinator_node)
    builder.add_node("support", support_node)
    builder.add_node("research", research_node)
    builder.add_node("action", action_node)
    builder.add_node("clarify", clarify_node)

    builder.add_edge(START, "trim_history")
    builder.add_edge("trim_history", "coordinator")
    builder.add_conditional_edges(
        "coordinator",
        route_after_coordinator,
        {
            "support": "support",
            "research": "research",
            "action": "action",
            "clarify": "clarify",
        },
    )
    for terminal in ("support", "research", "action", "clarify"):
        builder.add_edge(terminal, END)

    return builder.compile(checkpointer=checkpointer)
