"""Tool registration for the Vanguard MCP server.

Only **side-effecting (action) tools** live here — emails, tickets, audit
logs. Read-only research tools (web_search, db_query) are bound directly
to the Research Agent's LLM in-process; see
``packages/agents/vanguard_agents/tools/research.py``.
"""

from . import action

__all__ = ["action"]


def register_all(mcp) -> None:
    """Attach every action tool to the FastMCP instance."""
    action.register(mcp)
