"""Research tools — web_search and db_query.

These are read-only and have no side effects, so they live in-process and
get bound to the Research Agent's LLM directly. (Action tools, which DO
have side effects, live in the separate MCP server — see ``apps/mcp_server``.)
"""

from __future__ import annotations

import json
import logging
import sqlite3
from pathlib import Path

from langchain_core.tools import tool

from ..settings import get_settings

log = logging.getLogger(__name__)


def get_warehouse_db_path() -> Path:
    """Resolve the SQLite warehouse path from settings.

    Defaults to ``data/fund_warehouse.sqlite`` relative to the current working
    directory, which is the convention when running from the project root.
    """
    s = get_settings()
    return Path(s.warehouse_db_path).resolve()


@tool
async def web_search(query: str, max_results: int = 5) -> str:
    """Run a public web search and return the top results as JSON.

    Falls back to a clearly-labelled stub when the optional ``ddgs`` package
    isn't installed, so the agent loop is still usable offline.

    Args:
        query: The search query.
        max_results: Maximum number of results (default 5).

    Returns:
        A JSON string ``{"results": [{title, url, snippet}, ...], "stub": bool}``.
    """
    try:
        from ddgs import DDGS  # type: ignore[import-not-found]
    except ImportError:
        log.warning("ddgs not installed — returning stub web_search results")
        return json.dumps(
            {
                "results": [
                    {
                        "title": "[stub] web_search not configured",
                        "url": "",
                        "snippet": (
                            f"Real web search unavailable. Query was: {query!r}. "
                            "Install `ddgs` to enable live search."
                        ),
                    }
                ],
                "stub": True,
            }
        )

    try:
        with DDGS() as ddgs:
            raw = list(ddgs.text(query, max_results=max_results))
    except Exception as exc:  # noqa: BLE001
        log.exception("web_search failed")
        return json.dumps({"error": f"{type(exc).__name__}: {exc}"})

    results = [
        {
            "title": r.get("title", ""),
            "url": r.get("href", ""),
            "snippet": (r.get("body", "") or "")[:300],
        }
        for r in raw
    ]
    return json.dumps({"results": results, "stub": False})


@tool
async def db_query(sql: str) -> str:
    """Execute a read-only SELECT against the Vanguard fund warehouse.

    The warehouse is a small SQLite file with two tables:

      funds(ticker, share_class, name, expense_ratio, asset_class,
            inception_year, net_assets_billions)
      fund_returns(ticker, year, total_return)

    Any statement that doesn't start with SELECT is rejected.

    Args:
        sql: A SELECT statement.

    Returns:
        A JSON string ``{"columns": [...], "rows": [...], "row_count": N}``
        or ``{"error": "..."}`` on failure.
    """
    if not sql.strip().upper().startswith("SELECT"):
        return json.dumps({"error": "Only SELECT statements are allowed."})

    db_path = get_warehouse_db_path()
    if not db_path.is_file():
        return json.dumps(
            {
                "error": (
                    f"Fund warehouse not seeded at {db_path}. Run "
                    f"`python data/seed_warehouse.py` first."
                )
            }
        )

    try:
        # `mode=ro` enforces read-only at the SQLite layer too — defence
        # in depth alongside the SELECT-prefix check.
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        conn.row_factory = sqlite3.Row
        try:
            cursor = conn.execute(sql)
            rows = [dict(r) for r in cursor.fetchall()]
            cols = [d[0] for d in cursor.description] if cursor.description else []
        finally:
            conn.close()
    except sqlite3.Error as exc:
        return json.dumps({"error": f"SQL error: {exc}"})

    return json.dumps({"columns": cols, "rows": rows, "row_count": len(rows)})


# Convenience: the canonical tool list for the Research Agent.
RESEARCH_TOOLS = [web_search, db_query]
