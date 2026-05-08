"""Coordinator node — emits a structured RoutingDecision and nothing else."""

from __future__ import annotations

import logging

from langchain_core.messages import HumanMessage, SystemMessage

from .factories.llm import build_chat_model
from .retry import wrap_with_retry
from .settings import get_settings
from .state import ConversationState, RoutingDecision

log = logging.getLogger(__name__)

COORDINATOR_SYSTEM_PROMPT = """\
You are a routing coordinator. Your ONLY job is to decide which agent should
handle the user message. You do NOT answer the user. You output a
RoutingDecision and nothing else.

Agents available:
  "support"  — policy, FAQ, account, document questions; anything that
               should be answered from Vanguard's internal knowledge base.
  "research" — market data, fund performance, live lookups; anything that
               needs an external search or a database query.
  "action"   — send email, create ticket, log something; the user is asking
               for an action to be performed.
  "clarify"  — the message is ambiguous, harmful, off-topic, or adversarial;
               we should ask the user a follow-up instead of acting.

Pick exactly ONE agent. Default to "clarify" when in genuine doubt — never
to "support" — so the user is invited to refine their question rather than
receiving a guess.

Provide a short ``reason`` (one sentence, ≤ 240 chars).
"""


def coordinator_node(state: ConversationState) -> dict:
    """Classify the user message and emit a RoutingDecision.

    Uses the FULL conversation history (``state["messages"]``) — not just the
    current turn — so follow-up replies are routed in context.  Without this,
    a follow-up like "the body should say X" looks ambiguous and gets sent to
    ``clarify`` because the coordinator can't see the previous "what should
    the body say?" question.

    Also clears per-turn fields so stale data from a prior turn (under the
    same thread_id checkpoint) doesn't leak into this turn's subagent.
    """
    s = get_settings()
    model = build_chat_model(max_tokens=s.llm_max_tokens_coordinator)
    classifier = wrap_with_retry(model.with_structured_output(RoutingDecision))

    history = state.get("messages") or []
    if not history:
        # First-turn safety: fall back to the raw user_message field.
        history = [HumanMessage(content=state["user_message"])]

    messages = [SystemMessage(content=COORDINATOR_SYSTEM_PROMPT), *history]
    decision: RoutingDecision = classifier.invoke(messages)  # type: ignore[assignment]
    log.info("coordinator → %s  reason=%r", decision.agent, decision.reason[:60])

    return {
        "routing": decision,
        "retrieved_chunks": [],
        "tool_results": [],
        "final_response": None,
    }


def route_after_coordinator(state: ConversationState) -> str:
    """Conditional-edge selector. Returns the name of the next node."""
    routing = state.get("routing")
    if routing is None:
        return "clarify"
    return routing.agent
