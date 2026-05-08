"""CLI eval runner for the M12 Ragas harness.

Usage (from vanguard-assistant/ root):
    ./.venv/Scripts/python.exe packages/eval/run_eval.py

Replays each question in datasets/qa_pairs.jsonl through the Support agent
directly (bypassing coordinator routing), scores the results with Ragas, and
writes a timestamped JSON file to packages/eval/results/.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

# Put vanguard-assistant/ and packages/eval/ on sys.path so both
# vanguard_agents and the local ragas_runner module resolve.
_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parents[1]
for _p in (_REPO, _HERE):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

log = logging.getLogger(__name__)

# Ragas asks Azure for n=3 completions but Azure returns 1 — harmless fallback.
logging.getLogger("ragas").setLevel(logging.ERROR)

DEFAULT_DATA_PATH = _HERE / "datasets" / "qa_pairs.jsonl"
DEFAULT_RESULTS_DIR = _HERE / "results"


def _load_qa_pairs(path: Path) -> list[dict]:
    rows = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def replay_support(question: str) -> dict:
    """Call the Support agent node directly with a synthetic state.

    Returns dict with keys: answer, contexts (list[str]), sources (list[str]).
    Bypasses coordinator routing so the eval stays deterministic.
    """
    from langchain_core.messages import HumanMessage
    from vanguard_agents.subagents.support import support_node

    state = {
        "user_message": question,
        "messages": [HumanMessage(content=question)],
        "routing": None,
        "retrieved_chunks": [],
        "tool_results": [],
        "final_response": None,
        "thread_id": "eval",
    }
    result = support_node(state)
    chunks = result.get("retrieved_chunks", [])
    # chunk_id format is "VG-PR-002#003" — the prefix before # is the doc ID.
    doc_ids = [c["chunk_id"].split("#")[0] for c in chunks]
    return {
        "answer": result.get("final_response", ""),
        "contexts": [c["text"] for c in chunks],
        "sources": [c["source"] for c in chunks],
        "source_ids": doc_ids,
    }


def run_eval(data_path: Path, results_dir: Path) -> Path:
    import ragas_runner  # local module in packages/eval/

    from vanguard_agents.factories.judge import build_judge_model

    qa_pairs = _load_qa_pairs(data_path)
    log.info("Loaded %d Q/A pairs from %s", len(qa_pairs), data_path)

    # Cache replayed rows to disk so a scoring failure doesn't waste the
    # replay API calls on the next run.
    replay_cache = results_dir / "_replay_cache.json"
    results_dir.mkdir(parents=True, exist_ok=True)

    if replay_cache.exists():
        log.info("Loading cached replay results from %s (delete to re-replay)", replay_cache)
        replayed = json.loads(replay_cache.read_text())
    else:
        replayed = []
        for i, row in enumerate(qa_pairs, 1):
            log.info("[%d/%d] %s", i, len(qa_pairs), row["question"][:70])
            replay = replay_support(row["question"])
            replayed.append(
                {
                    "question": row["question"],
                    "ground_truth": row["ground_truth"],
                    "expected_sources": row.get("expected_sources", []),
                    **replay,
                }
            )
        replay_cache.write_text(json.dumps(replayed, indent=2))
        log.info("Replay cached to %s", replay_cache)

    log.info("Scoring %d rows with Ragas…", len(replayed))
    judge = build_judge_model()
    scored = ragas_runner.score_rows(replayed, judge)

    n = len(scored)
    summary = {
        "faithfulness": sum(r["faithfulness"] for r in scored) / n,
        "answer_relevancy": sum(r["answer_relevancy"] for r in scored) / n,
        "context_precision": sum(r["context_precision"] for r in scored) / n,
        "retrieval_precision": sum(r["retrieval_precision"] for r in scored) / n,
        "retrieval_recall": sum(r["retrieval_recall"] for r in scored) / n,
        "citation_hit_rate": sum(r["citation_hit"] for r in scored) / n,
    }

    run_ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    payload = {"timestamp": run_ts, "summary": summary, "rows": scored}

    results_dir.mkdir(parents=True, exist_ok=True)
    out_path = results_dir / f"{run_ts}.json"
    out_path.write_text(json.dumps(payload, indent=2))

    log.info("Results → %s", out_path)
    log.info("Summary: %s", {k: f"{v:.3f}" for k, v in summary.items()})
    return out_path


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    p = argparse.ArgumentParser(description="Run Ragas eval against the Support agent.")
    p.add_argument("--data", type=Path, default=DEFAULT_DATA_PATH)
    p.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    args = p.parse_args()
    run_eval(args.data, args.results_dir)


if __name__ == "__main__":
    main()
