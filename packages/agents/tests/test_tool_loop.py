"""Tests for the shared ReAct tool-calling loop.

We mock the LLM and tools so these tests are deterministic, fast, and free.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.messages import AIMessage, ToolMessage

from vanguard_agents.subagents._tool_loop import (
    MAX_TOOL_ROUNDS,
    run_tool_loop,
    with_agent_badge,
)


def _ai(content: str, tool_calls: list[dict] | None = None) -> AIMessage:
    msg = AIMessage(content=content, tool_calls=tool_calls or [])
    return msg


def _make_tool(name: str, output: str = "ok"):
    """Build a fake LangChain BaseTool that returns a fixed string."""
    tool = MagicMock()
    tool.name = name
    tool.ainvoke = AsyncMock(return_value=output)
    return tool


@pytest.fixture
def patched_llm():
    """Patch ``build_chat_model`` and let each test set the LLM behaviour.

    ``.bind_tools()`` and ``.with_retry()`` both return ``self`` so that
    ``wrap_with_retry(model.bind_tools(tools))`` is transparent — the same
    mock handles all ``ainvoke`` calls that the tests set up.
    """
    fake = MagicMock()
    fake.bind_tools.return_value = fake
    fake.with_retry.return_value = fake   # makes wrap_with_retry a no-op in tests
    fake.ainvoke = AsyncMock()
    with patch(
        "vanguard_agents.subagents._tool_loop.build_chat_model", return_value=fake
    ):
        yield fake


async def test_loop_returns_immediately_when_no_tool_calls(patched_llm):
    patched_llm.ainvoke.return_value = _ai("here is your answer")
    text, msgs, results = await run_tool_loop(
        system_prompt="sys", tools=[], user_message="hello"
    )
    assert text == "here is your answer"
    assert results == []
    assert len(msgs) == 1


async def test_loop_executes_tools_then_finishes(patched_llm):
    """LLM emits a tool_call once, gets the result, then returns final text."""
    weather = _make_tool("weather", output='{"temp": 72}')

    patched_llm.ainvoke.side_effect = [
        _ai(
            "calling tool…",
            tool_calls=[
                {"id": "c1", "name": "weather", "args": {"city": "Boston"}}
            ],
        ),
        _ai("It is 72 degrees in Boston."),
    ]

    text, msgs, results = await run_tool_loop(
        system_prompt="sys", tools=[weather], user_message="weather in Boston?"
    )

    assert text == "It is 72 degrees in Boston."
    assert len(results) == 1
    assert results[0]["name"] == "weather"
    assert results[0]["args"] == {"city": "Boston"}
    assert results[0]["error"] is None
    # Sequence of new messages: AI(tool_call), Tool(result), AI(final).
    assert isinstance(msgs[0], AIMessage)
    assert isinstance(msgs[1], ToolMessage)
    assert isinstance(msgs[-1], AIMessage)
    weather.ainvoke.assert_awaited_once_with({"city": "Boston"})


async def test_loop_records_tool_error(patched_llm):
    """When a tool raises, the loop records the error and continues."""
    bad = _make_tool("bad")
    bad.ainvoke = AsyncMock(side_effect=RuntimeError("boom"))

    patched_llm.ainvoke.side_effect = [
        _ai(
            "trying…",
            tool_calls=[{"id": "c1", "name": "bad", "args": {}}],
        ),
        _ai("Sorry, that tool failed."),
    ]

    _, _, results = await run_tool_loop(
        system_prompt="sys", tools=[bad], user_message="go"
    )
    assert results[0]["error"] == "boom"
    assert "boom" in results[0]["output"]


async def test_loop_records_unknown_tool(patched_llm):
    """If the LLM hallucinates a tool name, we record it without crashing."""
    real = _make_tool("real")
    patched_llm.ainvoke.side_effect = [
        _ai(
            "trying…",
            tool_calls=[{"id": "c1", "name": "ghost", "args": {}}],
        ),
        _ai("That tool isn't available."),
    ]

    _, _, results = await run_tool_loop(
        system_prompt="sys", tools=[real], user_message="go"
    )
    assert results[0]["name"] == "ghost"
    assert "unknown tool" in results[0]["error"]


async def test_loop_caps_at_max_rounds(patched_llm):
    """Defensive: never loop forever even if the LLM keeps calling tools."""
    forever = _make_tool("forever")
    patched_llm.ainvoke.return_value = _ai(
        "and again",
        tool_calls=[{"id": "c", "name": "forever", "args": {}}],
    )

    text, _, results = await run_tool_loop(
        system_prompt="sys", tools=[forever], user_message="loop"
    )
    assert "stopped" in text.lower()
    assert len(results) == MAX_TOOL_ROUNDS  # one tool call per round, all rounds used


def test_with_agent_badge_singular_and_plural():
    assert with_agent_badge("X", "hi", []) == "[X] hi"
    assert with_agent_badge("X", "hi", [{"name": "a", "args": {}, "output": "", "error": None}]) == "[X] (1 tool call) hi"
    two = [{"name": "a", "args": {}, "output": "", "error": None}] * 2
    assert with_agent_badge("X", "hi", two) == "[X] (2 tool calls) hi"
