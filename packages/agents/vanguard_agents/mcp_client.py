"""MCP client wrapper.

The MCP server hosts the **action tools only** (send_email, create_ticket,
log_action). Research tools are local — see ``vanguard_agents.tools``.

Why a wrapper around ``MultiServerMCPClient``?

1. One place that knows the server URL and the bearer token.
2. Cache the discovered tool list across turns so we're not re-handshaking
   the MCP server on every user message.
"""

from __future__ import annotations

import logging
from typing import Any

from langchain_core.tools import BaseTool

from .settings import get_settings

log = logging.getLogger(__name__)

# Names of the tools exposed by the MCP server. Used to assert we received
# what we expected after discovery.
EXPECTED_ACTION_TOOLS: frozenset[str] = frozenset(
    {"send_email", "create_ticket", "log_action"}
)

# Cache the discovered tools across turns. ``None`` means "not yet loaded".
_cached_tools: list[BaseTool] | None = None


def _client_config() -> dict[str, Any]:
    s = get_settings()
    config: dict[str, Any] = {
        "url": s.mcp_url,
        "transport": "streamable_http",
    }
    if s.mcp_bearer_token:
        config["headers"] = {"Authorization": f"Bearer {s.mcp_bearer_token}"}
    return config


async def get_action_tools(*, refresh: bool = False) -> list[BaseTool]:
    """Discover and return the action tools exposed by the MCP server.

    Set ``refresh=True`` to force a fresh handshake (e.g. after deploying
    new tools to the server).
    """
    global _cached_tools
    if _cached_tools is not None and not refresh:
        return _cached_tools

    from langchain_mcp_adapters.client import MultiServerMCPClient

    client = MultiServerMCPClient({"vanguard-tools": _client_config()})
    try:
        tools = await client.get_tools()
    except Exception:  # noqa: BLE001
        log.exception("failed to discover tools from MCP server")
        raise

    names = {t.name for t in tools}
    missing = EXPECTED_ACTION_TOOLS - names
    if missing:
        log.warning(
            "MCP server is missing expected action tools: %s "
            "(discovered: %s)",
            sorted(missing),
            sorted(names),
        )

    log.info("discovered %d MCP action tools: %s", len(tools), sorted(names))
    _cached_tools = tools
    return tools


def reset_cache_for_tests() -> None:
    """Force the next ``get_action_tools()`` to re-discover."""
    global _cached_tools
    _cached_tools = None
