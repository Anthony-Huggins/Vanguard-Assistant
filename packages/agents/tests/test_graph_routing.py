"""Graph-level tests with a mocked LLM.

These verify that the conditional edge correctly routes on the agent value
returned by the coordinator, without making any real LLM call.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from vanguard_agents.graph import build_graph
from vanguard_agents.state import RoutingDecision, fresh_turn_state


def _stub_classifier(decision: RoutingDecision) -> MagicMock:
    """Return a chain whose .invoke() yields the given decision.

    ``.with_retry()`` returns ``self`` so the retry wrapper added in M5 is
    transparent in tests — the same mock handles the final ``.invoke()`` call.
    """
    chain = MagicMock()
    chain.invoke.return_value = decision
    chain.with_retry.return_value = chain
    return chain


@pytest.fixture
def patched_coordinator():
    """Patch the coordinator's LLM construction; the test sets the decision."""
    with patch("vanguard_agents.coordinator.build_chat_model") as build:
        model = MagicMock()
        build.return_value = model
        yield model


@pytest.fixture(autouse=True)
def _stub_subagent_externals():
    """Mock out everything external the subagents touch so routing tests
    are pure routing tests:
    - Support: force the fallback path (no Chroma).
    - Research: short-circuit the tool loop (research tools are local but
                we don't want to actually run them in routing tests).
    - Action: short-circuit the tool loop AND the MCP discovery.
    """
    from langchain_core.messages import AIMessage

    async def fake_loop(*, system_prompt, tools, user_message, **kwargs):
        # **kwargs absorbs confirm_tools and any future additions.
        msg = AIMessage(content="(stub) ok")
        return msg.content, [msg], []

    async def fake_get_action_tools():
        return []

    with (
        patch("vanguard_agents.subagents.support.retrieve", return_value=[]),
        patch("vanguard_agents.subagents.research.run_tool_loop", new=fake_loop),
        patch("vanguard_agents.subagents.action.run_tool_loop", new=fake_loop),
        patch("vanguard_agents.subagents.action.get_action_tools", new=fake_get_action_tools),
    ):
        yield


@pytest.mark.parametrize(
    "agent_choice,expected_substring",
    [
        ("support", "knowledge base"),  # support fallback message
        ("research", "Research Agent"),  # badge prefix from with_agent_badge
        ("action", "Action Agent"),
        ("clarify", "rephrase"),
    ],
)
async def test_routes_to_correct_node(patched_coordinator, agent_choice, expected_substring):
    decision = RoutingDecision(agent=agent_choice, reason="test")
    patched_coordinator.with_structured_output.return_value = _stub_classifier(decision)

    graph = build_graph()  # no checkpointer for tests
    state = fresh_turn_state("does not matter")
    result = await graph.ainvoke(state)

    assert result["routing"].agent == agent_choice
    assert expected_substring in result["final_response"]


async def test_routing_clears_per_turn_fields(patched_coordinator):
    """The coordinator must reset retrieved_chunks/tool_results each turn."""
    decision = RoutingDecision(agent="support", reason="policy")
    patched_coordinator.with_structured_output.return_value = _stub_classifier(decision)

    graph = build_graph()
    initial = fresh_turn_state("hi")
    initial["retrieved_chunks"] = [
        {"text": "stale", "source": "old", "chunk_id": "1", "score": 0.9, "page": None}
    ]
    initial["tool_results"] = [{"name": "old", "args": {}, "output": "stale", "error": None}]

    result = await graph.ainvoke(initial)

    assert result["retrieved_chunks"] == []
    assert result["tool_results"] == []
