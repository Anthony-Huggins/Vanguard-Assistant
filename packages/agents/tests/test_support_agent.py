"""Tests for the Support Agent's RAG path with retrieve + LLM both mocked."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from langchain_core.messages import AIMessage

from vanguard_agents.state import RetrievedChunk, fresh_turn_state
from vanguard_agents.subagents.support import support_node


def _chunk(score: float, doc_id: str = "VG-PR-002", text: str = "VTI ER is 0.03%") -> RetrievedChunk:
    return {
        "text": text,
        "source": "500 Index Fund Prospectus",
        "chunk_id": f"{doc_id}#001",
        "score": score,
        "page": 1,
    }


def test_fallback_when_no_chunks():
    state = fresh_turn_state("anything")
    with patch("vanguard_agents.subagents.support.retrieve", return_value=[]):
        result = support_node(state)
    assert "knowledge base" in result["final_response"]
    assert isinstance(result["messages"][0], AIMessage)


def test_fallback_when_below_threshold(monkeypatch):
    """Even with chunks, if best score < threshold, take the fallback path."""
    monkeypatch.setenv("RAG_SIMILARITY_THRESHOLD", "0.95")
    from vanguard_agents.settings import reset_settings_for_tests

    reset_settings_for_tests()

    state = fresh_turn_state("anything")
    chunks = [_chunk(0.5), _chunk(0.6)]
    with patch("vanguard_agents.subagents.support.retrieve", return_value=chunks):
        result = support_node(state)

    assert "knowledge base" in result["final_response"]
    assert result["retrieved_chunks"] == chunks  # still recorded for tracing


def test_grounded_answer_with_citations(monkeypatch):
    """When score passes threshold, LLM is called and citation line appended."""
    monkeypatch.setenv("RAG_SIMILARITY_THRESHOLD", "0.5")
    from vanguard_agents.settings import reset_settings_for_tests

    reset_settings_for_tests()

    chunks = [_chunk(0.9), _chunk(0.8, doc_id="VG-ED-002")]

    fake_model = MagicMock()
    fake_model.invoke.return_value = AIMessage(content="The expense ratio is 0.03%.")
    # wrap_with_retry calls .with_retry() on the model; make it a no-op so the
    # same mock's .invoke() is what actually gets called.
    fake_model.with_retry.return_value = fake_model

    state = fresh_turn_state("What is the V500 expense ratio?")
    with (
        patch("vanguard_agents.subagents.support.retrieve", return_value=chunks),
        patch("vanguard_agents.subagents.support.build_chat_model", return_value=fake_model),
    ):
        result = support_node(state)

    body = result["final_response"]
    assert "expense ratio is 0.03%" in body
    assert "Sources:" in body
    assert "VG-PR-002#001" in body
    assert "VG-ED-002#001" in body
