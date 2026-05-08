"""Local LangChain tools owned by the agents package.

Only side-effecting tools (send_email / create_ticket / log_action) live in
the separate MCP server. Read-only tools — like the research ones here —
ride directly on the LLM via ``llm.bind_tools(...)`` and don't need cross-
process isolation.
"""

from .research import RESEARCH_TOOLS, db_query, get_warehouse_db_path, web_search

__all__ = [
    "RESEARCH_TOOLS",
    "db_query",
    "get_warehouse_db_path",
    "web_search",
]
