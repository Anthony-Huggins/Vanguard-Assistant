"""Tests for the history-trim node in history.py."""

from __future__ import annotations

from unittest.mock import patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from vanguard_agents.history import trim_history_node


def _make_messages(n_turns: int) -> list:
    """Build n_turns * 2 messages (HumanMessage + AIMessage alternating)."""
    msgs = []
    for i in range(n_turns):
        msgs.append(HumanMessage(content=f"user turn {i}", id=f"h{i}"))
        msgs.append(AIMessage(content=f"asst turn {i}", id=f"a{i}"))
    return msgs


class TestTrimHistoryNode:
    def test_no_trim_when_within_limit(self):
        """Under the limit → empty dict (no-op)."""
        msgs = _make_messages(5)  # 10 messages, default limit is 20 turns = 40 msgs
        result = trim_history_node({"messages": msgs})
        assert result == {}

    def test_no_trim_at_exact_limit(self):
        """Exactly at the limit → no-op."""
        msgs = _make_messages(20)  # 40 messages = 20 turns (the default)
        result = trim_history_node({"messages": msgs})
        assert result == {}

    def test_trim_when_over_limit(self):
        """One turn over the limit → removes the two oldest messages."""
        msgs = _make_messages(21)  # 42 messages > 40 limit
        result = trim_history_node({"messages": msgs})
        assert "messages" in result
        # Should remove exactly 2 messages (42 - 40 = 2 overflow)
        remove_msgs = result["messages"]
        assert len(remove_msgs) == 2
        # They should be RemoveMessage objects targeting the oldest messages
        removed_ids = {m.id for m in remove_msgs}
        assert "h0" in removed_ids
        assert "a0" in removed_ids

    def test_trim_removes_oldest_first(self):
        """The oldest messages (index 0, 1) should be the ones scheduled for removal."""
        msgs = _make_messages(25)  # 50 messages; 10 messages over the 40-message limit
        result = trim_history_node({"messages": msgs})
        assert "messages" in result
        remove_msgs = result["messages"]
        assert len(remove_msgs) == 10
        removed_ids = {m.id for m in remove_msgs}
        # First 5 turns (h0..h4, a0..a4) should be removed
        for i in range(5):
            assert f"h{i}" in removed_ids
            assert f"a{i}" in removed_ids

    def test_trim_respects_custom_limit(self):
        """Custom ``history_max_turns`` setting is respected."""
        msgs = _make_messages(6)  # 12 messages
        with patch("vanguard_agents.history.get_settings") as mock_settings:
            mock_settings.return_value.history_max_turns = 5  # 10-message window
            result = trim_history_node({"messages": msgs})
        assert "messages" in result
        remove_msgs = result["messages"]
        assert len(remove_msgs) == 2  # 12 - 10 = 2

    def test_empty_messages_is_noop(self):
        result = trim_history_node({"messages": []})
        assert result == {}

    def test_missing_messages_key_is_noop(self):
        result = trim_history_node({})
        assert result == {}
