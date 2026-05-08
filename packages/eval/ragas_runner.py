"""Ragas scoring wrapper for the M12 eval harness.

Takes replayed eval rows and scores them with three Ragas metrics:
  - faithfulness:       is the answer grounded in the retrieved context?
  - answer_relevancy:   does the answer address the question?
  - context_precision:  does the retrieved context contain the ground truth?

Plus three retrieval metrics computed directly against expected_sources
(no LLM calls):
  - retrieval_precision:  fraction of retrieved docs that were expected
  - retrieval_recall:     fraction of expected docs that were retrieved
  - citation_hit:         binary — at least one expected doc was retrieved
"""

from __future__ import annotations

import logging
import math
from typing import Any

log = logging.getLogger(__name__)


def score_rows(
    rows: list[dict[str, Any]],
    judge_model: Any,
) -> list[dict[str, Any]]:
    """Score a list of replayed eval rows with Ragas.

    Each row must contain: question, answer, contexts (list[str]), ground_truth.
    Returns the same rows extended with faithfulness, answer_relevancy,
    context_precision, and citation_hit fields.
    """
    from datasets import Dataset  # type: ignore[import-untyped]
    from ragas import evaluate  # type: ignore[import-untyped]
    from ragas.metrics import (  # type: ignore[import-untyped]
        answer_relevancy,
        context_precision,
        faithfulness,
    )

    # Faithfulness requires non-empty contexts — rows with no retrieved chunks
    # (fallback "I don't have that information" answers) are skipped for that
    # metric and scored 0.0.  All other metrics still run on the full set.
    scorable_indices = [i for i, r in enumerate(rows) if r.get("contexts")]
    scorable_rows = [rows[i] for i in scorable_indices]

    ragas_scores: dict[int, dict[str, float]] = {}
    if scorable_rows:
        dataset = Dataset.from_list(
            [
                {
                    "question": r["question"],
                    "answer": r["answer"],
                    "contexts": r["contexts"],
                    "ground_truth": r["ground_truth"],
                }
                for r in scorable_rows
            ]
        )

        eval_kwargs: dict[str, Any] = {
            "metrics": [faithfulness, answer_relevancy, context_precision],
        }
        try:
            from ragas.llms import LangchainLLMWrapper  # type: ignore[import-untyped]
            from ragas.embeddings import LangchainEmbeddingsWrapper  # type: ignore[import-untyped]
            from vanguard_agents.factories.embedder import build_embedder

            eval_kwargs["llm"] = LangchainLLMWrapper(judge_model)
            eval_kwargs["embeddings"] = LangchainEmbeddingsWrapper(build_embedder())
        except ImportError:
            log.warning("ragas.llms/embeddings unavailable — judge LLM not set; scores may degrade")

        result = evaluate(dataset, **eval_kwargs)
        scored_df = result.to_pandas()

        for df_i, orig_i in enumerate(scorable_indices):
            s = scored_df.iloc[df_i]
            ragas_scores[orig_i] = {
                "faithfulness": _safe_float(s.get("faithfulness")),
                "answer_relevancy": _safe_float(s.get("answer_relevancy")),
                "context_precision": _safe_float(s.get("context_precision")),
            }

    out: list[dict[str, Any]] = []
    for i, row in enumerate(rows):
        # Use doc IDs (extracted from chunk_ids) for retrieval metrics —
        # the human-readable source titles don't contain the VG-XX-NNN IDs.
        source_ids = row.get("source_ids", row.get("sources", []))
        expected = row.get("expected_sources", [])
        scores = ragas_scores.get(i, {"faithfulness": 0.0, "answer_relevancy": 0.0, "context_precision": 0.0})
        out.append(
            {
                **row,
                **scores,
                "retrieval_precision": retrieval_precision(source_ids, expected),
                "retrieval_recall": retrieval_recall(source_ids, expected),
                "citation_hit": _citation_hit(source_ids, expected),
            }
        )
    return out


def _safe_float(val: Any) -> float:
    try:
        f = float(val)
        return 0.0 if math.isnan(f) else f
    except (TypeError, ValueError):
        return 0.0


def _matches(actual_source: str, expected_id: str) -> bool:
    """True if expected_id appears as a substring of actual_source (case-insensitive)."""
    return expected_id.lower() in actual_source.lower()


def retrieval_precision(actual: list[str], expected: list[str]) -> float:
    """Fraction of retrieved sources that match at least one expected doc ID.

    E.g. if 3 of 5 retrieved chunks came from expected docs → 0.6.
    Returns 1.0 when expected is empty (nothing to miss).
    """
    if not actual:
        return 0.0
    if not expected:
        return 1.0
    hits = sum(1 for src in actual if any(_matches(src, exp) for exp in expected))
    return hits / len(actual)


def retrieval_recall(actual: list[str], expected: list[str]) -> float:
    """Fraction of expected doc IDs that appear in at least one retrieved source.

    E.g. if 2 of 3 expected docs were retrieved → 0.667.
    Returns 1.0 when expected is empty.
    """
    if not expected:
        return 1.0
    found = sum(1 for exp in expected if any(_matches(src, exp) for src in actual))
    return found / len(expected)


def _citation_hit(actual: list[str], expected: list[str]) -> bool:
    """True if at least one expected source doc ID was retrieved."""
    if not expected:
        return True
    return any(_matches(src, exp) for exp in expected for src in actual)
