"""Tests for the local research tools (web_search, db_query).

We test ``db_query`` against a real on-disk SQLite fixture (it's small and
deterministic). ``web_search`` is exercised via its stub fallback so the
tests don't hit the network.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest


@pytest.fixture
def tmp_warehouse(tmp_path: Path, monkeypatch) -> Path:
    """Create a tiny SQLite warehouse and point settings at it."""
    db = tmp_path / "fund_warehouse.sqlite"
    conn = sqlite3.connect(db)
    try:
        conn.executescript(
            """
            CREATE TABLE funds (
                ticker TEXT PRIMARY KEY,
                expense_ratio REAL NOT NULL
            );
            CREATE TABLE fund_returns (
                ticker TEXT NOT NULL,
                year INTEGER NOT NULL,
                total_return REAL NOT NULL
            );
            INSERT INTO funds VALUES ('V500', 0.03), ('VTBM', 0.04);
            INSERT INTO fund_returns VALUES
                ('V500', 2024, 26.10),
                ('V500', 2025, 12.04),
                ('VTBM', 2024,  1.30);
            """
        )
        conn.commit()
    finally:
        conn.close()

    monkeypatch.setenv("WAREHOUSE_DB_PATH", str(db))
    from vanguard_agents.settings import reset_settings_for_tests

    reset_settings_for_tests()
    return db


async def test_db_query_select_returns_rows(tmp_warehouse):
    from vanguard_agents.tools.research import db_query

    raw = await db_query.ainvoke({"sql": "SELECT ticker, expense_ratio FROM funds ORDER BY ticker"})
    payload = json.loads(raw)
    assert payload["row_count"] == 2
    assert payload["columns"] == ["ticker", "expense_ratio"]
    assert payload["rows"][0] == {"ticker": "V500", "expense_ratio": 0.03}


async def test_db_query_rejects_non_select(tmp_warehouse):
    from vanguard_agents.tools.research import db_query

    raw = await db_query.ainvoke({"sql": "DELETE FROM funds"})
    payload = json.loads(raw)
    assert "error" in payload
    assert "SELECT" in payload["error"]


async def test_db_query_missing_warehouse(tmp_path, monkeypatch):
    """Helpful error when the SQLite file isn't seeded yet."""
    monkeypatch.setenv("WAREHOUSE_DB_PATH", str(tmp_path / "nonexistent.sqlite"))
    from vanguard_agents.settings import reset_settings_for_tests
    from vanguard_agents.tools.research import db_query

    reset_settings_for_tests()
    raw = await db_query.ainvoke({"sql": "SELECT 1"})
    payload = json.loads(raw)
    assert "not seeded" in payload["error"]


async def test_db_query_sql_error(tmp_warehouse):
    """Bad SQL surfaces as an error field, not an exception."""
    from vanguard_agents.tools.research import db_query

    raw = await db_query.ainvoke({"sql": "SELECT * FROM no_such_table"})
    payload = json.loads(raw)
    assert "error" in payload
    assert "SQL error" in payload["error"]


async def test_web_search_stub_when_ddgs_missing(monkeypatch):
    """If ``ddgs`` import fails, we get a clearly-labelled stub result."""
    import builtins

    real_import = builtins.__import__

    def blocked_import(name, *args, **kwargs):
        if name == "ddgs":
            raise ImportError("simulated")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", blocked_import)

    from vanguard_agents.tools.research import web_search

    raw = await web_search.ainvoke({"query": "anything"})
    payload = json.loads(raw)
    assert payload["stub"] is True
    assert "anything" in payload["results"][0]["snippet"]
