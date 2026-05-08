"""GET /api/evals/latest — returns the most recent Ragas eval run.

Reads the newest timestamped JSON file from the eval results directory
configured via EVAL_RESULTS_DIR (default: packages/eval/results/).
Returns 404 if no runs exist yet.
"""

from __future__ import annotations

import json
from pathlib import Path

from fastapi import APIRouter, HTTPException

from vanguard_agents.settings import get_settings

router = APIRouter()

# Repo root is two levels above this file (apps/api/routers/evals.py → vanguard-assistant/)
_REPO_ROOT = Path(__file__).resolve().parents[3]


def _resolve_results_dir(s) -> Path:
    p = Path(s.eval_results_dir)
    return p if p.is_absolute() else _REPO_ROOT / p


@router.get("/evals/latest", tags=["evals"])
async def get_latest_eval() -> dict:
    s = get_settings()
    results_dir = _resolve_results_dir(s)

    if not results_dir.exists():
        raise HTTPException(status_code=404, detail="No eval results found. Run packages/eval/run_eval.py first.")

    runs = sorted(results_dir.glob("[0-9]*.json"), reverse=True)
    if not runs:
        raise HTTPException(status_code=404, detail="No eval results found. Run packages/eval/run_eval.py first.")

    return json.loads(runs[0].read_text())


@router.get("/evals/runs", tags=["evals"])
async def list_eval_runs() -> list[str]:
    """List all eval run timestamps, newest first."""
    s = get_settings()
    results_dir = _resolve_results_dir(s)

    if not results_dir.exists():
        return []

    return [f.stem for f in sorted(results_dir.glob("[0-9]*.json"), reverse=True)]
