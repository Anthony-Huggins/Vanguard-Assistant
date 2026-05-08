"""Tests for GET /api/evals/latest and GET /api/evals/runs."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest


@pytest.mark.asyncio
async def test_get_latest_eval_returns_404_when_no_results(client):
    with patch("apps.api.routers.evals.Path") as mock_path_cls:
        mock_dir = mock_path_cls.return_value
        mock_dir.exists.return_value = False

        resp = await client.get("/api/evals/latest")

    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_get_latest_eval_returns_newest_file(client, tmp_path):
    run = {
        "timestamp": "20240101T120000Z",
        "summary": {
            "faithfulness": 0.85,
            "answer_relevancy": 0.80,
            "context_precision": 0.75,
            "citation_hit_rate": 0.90,
        },
        "rows": [],
    }
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    (results_dir / "20240101T120000Z.json").write_text(json.dumps(run))

    from vanguard_agents.settings import get_settings

    original = get_settings().eval_results_dir
    get_settings().eval_results_dir = str(results_dir)
    try:
        resp = await client.get("/api/evals/latest")
    finally:
        get_settings().eval_results_dir = original

    assert resp.status_code == 200
    body = resp.json()
    assert body["timestamp"] == "20240101T120000Z"
    assert body["summary"]["faithfulness"] == pytest.approx(0.85)


@pytest.mark.asyncio
async def test_list_eval_runs_returns_empty_when_no_dir(client):
    with patch("apps.api.routers.evals.Path") as mock_path_cls:
        mock_dir = mock_path_cls.return_value
        mock_dir.exists.return_value = False

        resp = await client.get("/api/evals/runs")

    assert resp.status_code == 200
    assert resp.json() == []
