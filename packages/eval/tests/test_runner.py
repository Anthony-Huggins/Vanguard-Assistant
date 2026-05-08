"""Unit tests for the M12 eval harness.

All external calls (support_node, Ragas evaluate, judge LLM) are mocked so
the suite runs offline without Azure or Chroma.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# replay_support
# ---------------------------------------------------------------------------


def test_replay_support_extracts_answer_and_contexts():
    """replay_support maps support_node output to answer/contexts/sources."""
    from run_eval import replay_support

    fake_result = {
        "final_response": "VFIAX expense ratio is 0.04%.",
        "retrieved_chunks": [
            {"text": "The expense ratio is 0.04%.", "source": "VG-PR-002", "chunk_id": "1", "score": 0.7, "page": 1},
        ],
    }

    # support_node is imported inside replay_support — patch at source module.
    with patch("vanguard_agents.subagents.support.support_node", return_value=fake_result):
        out = replay_support("What is the VFIAX expense ratio?")

    assert out["answer"] == "VFIAX expense ratio is 0.04%."
    assert out["contexts"] == ["The expense ratio is 0.04%."]
    assert out["sources"] == ["VG-PR-002"]


def test_replay_support_handles_empty_chunks():
    """replay_support works when support_node returns no retrieved chunks."""
    from run_eval import replay_support

    fake_result = {
        "final_response": "I don't have that information.",
        "retrieved_chunks": [],
    }

    with patch("vanguard_agents.subagents.support.support_node", return_value=fake_result):
        out = replay_support("What is the meaning of life?")

    assert out["contexts"] == []
    assert out["sources"] == []


# ---------------------------------------------------------------------------
# ragas_runner.score_rows
# ---------------------------------------------------------------------------


def _make_rows(n: int = 2) -> list[dict]:
    return [
        {
            "question": f"Q{i}",
            "answer": f"A{i}",
            "contexts": [f"ctx{i}"],
            "ground_truth": f"GT{i}",
            "sources": ["VG-PR-002"],
            "expected_sources": ["VG-PR-002"],
        }
        for i in range(n)
    ]


def test_score_rows_adds_ragas_fields(monkeypatch):
    """score_rows returns rows with faithfulness/answer_relevancy/context_precision.

    ragas and datasets are not installed in the test venv, so we inject them
    into sys.modules before importing score_rows.
    """
    import sys

    score_data = [
        {"faithfulness": 0.9, "answer_relevancy": 0.85, "context_precision": 0.8},
        {"faithfulness": 0.7, "answer_relevancy": 0.75, "context_precision": 0.7},
    ]
    row_mocks = []
    for d in score_data:
        m = MagicMock()
        m.get.side_effect = lambda k, default=None, _d=d: _d.get(k, default)
        row_mocks.append(m)

    fake_df = MagicMock()
    fake_df.iloc.__getitem__.side_effect = lambda i: row_mocks[i]
    fake_result = MagicMock()
    fake_result.to_pandas.return_value = fake_df

    mock_datasets = MagicMock()
    mock_ragas = MagicMock()
    mock_ragas_metrics = MagicMock()
    mock_ragas_llms = MagicMock()
    mock_ragas.evaluate.return_value = fake_result
    mock_datasets.Dataset.from_list.return_value = MagicMock()

    for mod, obj in [
        ("datasets", mock_datasets),
        ("ragas", mock_ragas),
        ("ragas.metrics", mock_ragas_metrics),
        ("ragas.llms", mock_ragas_llms),
    ]:
        monkeypatch.setitem(sys.modules, mod, obj)

    # Force re-import so the patched sys.modules take effect.
    if "ragas_runner" in sys.modules:
        monkeypatch.delitem(sys.modules, "ragas_runner")

    from ragas_runner import score_rows

    rows = _make_rows(2)
    scored = score_rows(rows, judge_model=MagicMock())

    assert len(scored) == 2
    assert scored[0]["faithfulness"] == pytest.approx(0.9)
    assert scored[0]["answer_relevancy"] == pytest.approx(0.85)
    assert scored[0]["context_precision"] == pytest.approx(0.8)
    assert scored[0]["citation_hit"] is True


def test_safe_float_handles_nan():
    """_safe_float converts float('nan') to 0.0 instead of propagating NaN."""
    import math
    from ragas_runner import _safe_float

    assert _safe_float(float("nan")) == 0.0
    assert _safe_float(None) == 0.0
    assert _safe_float(0.9) == pytest.approx(0.9)


def test_score_rows_skips_empty_context_rows(monkeypatch):
    """Rows with empty contexts get faithfulness=0.0; Ragas only runs on scorable rows."""
    import sys

    score_data = [{"faithfulness": 0.8, "answer_relevancy": 0.9, "context_precision": 0.7}]
    m = MagicMock()
    m.get.side_effect = lambda k, default=None, _d=score_data[0]: _d.get(k, default)

    fake_df = MagicMock()
    fake_df.iloc.__getitem__.side_effect = lambda i: m
    fake_result = MagicMock()
    fake_result.to_pandas.return_value = fake_df

    mock_datasets = MagicMock()
    mock_ragas = MagicMock()
    mock_ragas_metrics = MagicMock()
    mock_ragas_llms = MagicMock()
    mock_ragas.evaluate.return_value = fake_result
    mock_datasets.Dataset.from_list.return_value = MagicMock()

    for mod, obj in [
        ("datasets", mock_datasets),
        ("ragas", mock_ragas),
        ("ragas.metrics", mock_ragas_metrics),
        ("ragas.llms", mock_ragas_llms),
    ]:
        monkeypatch.setitem(sys.modules, mod, obj)

    if "ragas_runner" in sys.modules:
        monkeypatch.delitem(sys.modules, "ragas_runner")

    from ragas_runner import score_rows

    rows = [
        {"question": "Q1", "answer": "A1", "contexts": ["ctx1"], "ground_truth": "GT1",
         "sources": ["VG-PR-002"], "source_ids": ["VG-PR-002"], "expected_sources": ["VG-PR-002"]},
        {"question": "Q2", "answer": "I don't have that information.", "contexts": [],
         "ground_truth": "GT2", "sources": [], "source_ids": [], "expected_sources": ["VG-PR-002"]},
    ]
    scored = score_rows(rows, judge_model=MagicMock())

    # Row 0 has context — should get Ragas scores
    assert scored[0]["faithfulness"] == pytest.approx(0.8)
    # Row 1 has no context — faithfulness must be 0.0, not NaN
    assert scored[1]["faithfulness"] == 0.0
    assert scored[1]["answer_relevancy"] == 0.0

    # Ragas evaluate() called once, only with the 1 scorable row
    call_args = mock_datasets.Dataset.from_list.call_args[0][0]
    assert len(call_args) == 1


def test_citation_hit_true_when_expected_source_in_actual():
    from ragas_runner import _citation_hit

    assert _citation_hit(["VG-PR-002_500_Index"], ["VG-PR-002"]) is True


def test_citation_hit_false_when_no_match():
    from ragas_runner import _citation_hit

    assert _citation_hit(["VG-OP-001"], ["VG-PR-002"]) is False


def test_citation_hit_true_when_no_expected():
    from ragas_runner import _citation_hit

    assert _citation_hit(["anything"], []) is True


# ---------------------------------------------------------------------------
# run_eval.py — qa_pairs loading
# ---------------------------------------------------------------------------


def test_load_qa_pairs(tmp_path):
    """_load_qa_pairs reads a valid JSONL file correctly."""
    import json

    from run_eval import _load_qa_pairs

    jl = tmp_path / "qa.jsonl"
    rows = [{"question": "Q1", "ground_truth": "A1", "expected_sources": ["VG-PR-001"]}]
    jl.write_text("\n".join(json.dumps(r) for r in rows))

    loaded = _load_qa_pairs(jl)
    assert len(loaded) == 1
    assert loaded[0]["question"] == "Q1"


def test_qa_pairs_jsonl_has_twenty_rows():
    """The shipped dataset has exactly 20 rows."""
    import json

    data_path = Path(__file__).resolve().parents[1] / "datasets" / "qa_pairs.jsonl"
    rows = [json.loads(l) for l in data_path.read_text().splitlines() if l.strip()]
    assert len(rows) == 20
    for row in rows:
        assert "question" in row
        assert "ground_truth" in row
        assert "expected_sources" in row
