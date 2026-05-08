"""Clarify node — handles ambiguous, harmful, or off-topic messages.

This is an addition to the original spec. Without it, the coordinator is
forced to pick one of {support, research, action} for any input, which
encourages hallucinated routing on garbage prompts.
"""

from __future__ import annotations

import logging

from langchain_core.messages import AIMessage

from ..state import ConversationState

log = logging.getLogger(__name__)


def clarify_node(state: ConversationState) -> dict:
    routing = state.get("routing")

    # Defensively extract reason — after a checkpointer round-trip the
    # RoutingDecision may come back as a plain dict instead of the Pydantic
    # model, so we try both attribute and key access.
    reason: str = "I'm not sure how to help with that."
    if routing is not None:
        if hasattr(routing, "reason"):
            reason = routing.reason or reason
        elif isinstance(routing, dict):
            reason = routing.get("reason") or reason

    reply = (
        f"I'd like to help, but I need a bit more to go on — {reason}\n\n"
        "Could you rephrase, or tell me whether you're after policy info, "
        "market data, or an action like sending an email?"
    )
    log.info("clarify_node: routing reason=%r", reason[:80])
    return {
        "final_response": reply,
        "messages": [AIMessage(content=reply)],
    }
