"""Vanguard tools MCP server.

Runs as a separate Python process. Exposes the research and action tools
over **streamable HTTP** transport (chosen so it works across Docker
containers later — stdio MCP doesn't cross container boundaries cleanly).

Auth: every inbound request must carry ``Authorization: Bearer <token>``.
The expected token comes from the ``MCP_BEARER_TOKEN`` environment variable
(or ``vanguard-assistant/.env``).

Run:
    ./.venv/Scripts/python.exe apps/mcp_server/server.py

or after ``pip install -e packages/agents``:
    ./.venv/Scripts/vanguard-mcp.exe
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

# Ensure ``vanguard-assistant/`` is importable so ``apps.*`` resolves
# regardless of how the file is invoked (script vs module vs console-script).
_THIS = Path(__file__).resolve()
_REPO = _THIS.parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from mcp.server.fastmcp import FastMCP  # noqa: E402
from starlette.middleware.base import BaseHTTPMiddleware  # noqa: E402
from starlette.responses import JSONResponse  # noqa: E402

from apps.mcp_server.tools import register_all  # noqa: E402

log = logging.getLogger(__name__)

DEFAULT_HOST = "0.0.0.0"
DEFAULT_PORT = 8765
DEFAULT_TOKEN = "dev-token-change-me"


def _load_env() -> None:
    """Load ``vanguard-assistant/.env`` so env vars set there are visible."""
    try:
        from dotenv import load_dotenv

        env_path = _REPO / ".env"
        if env_path.is_file():
            load_dotenv(env_path)
    except ImportError:
        pass  # python-dotenv is optional here


class BearerAuthMiddleware(BaseHTTPMiddleware):
    """Reject any request that doesn't present the expected bearer token."""

    def __init__(self, app, token: str):
        super().__init__(app)
        self._token = token

    async def dispatch(self, request, call_next):
        # Permit health probes without auth so docker-compose healthchecks work.
        if request.url.path == "/healthz":
            return JSONResponse({"ok": True})

        auth = request.headers.get("authorization", "")
        if auth != f"Bearer {self._token}":
            log.warning("rejected unauthenticated request to %s", request.url.path)
            return JSONResponse({"error": "unauthorized"}, status_code=401)
        return await call_next(request)


def build_mcp() -> FastMCP:
    # Pass host="0.0.0.0" so FastMCP does NOT auto-enable localhost-only
    # DNS-rebinding protection.  The server listens on all interfaces (Docker
    # needs this) so "127.0.0.1" is wrong.  Bearer-token auth in the
    # BearerAuthMiddleware already protects the endpoint.
    mcp = FastMCP(
        name="vanguard-tools",
        host="0.0.0.0",
        instructions=(
            "Vanguard agentic-AI training server. Exposes research tools "
            "(web_search, db_query) and action tools (send_email, "
            "create_ticket, log_action). Bearer-token auth required."
        ),
    )
    register_all(mcp)
    return mcp


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    _load_env()

    token = os.environ.get("MCP_BEARER_TOKEN") or DEFAULT_TOKEN
    if token == DEFAULT_TOKEN:
        log.warning(
            "MCP_BEARER_TOKEN not set in env — using insecure default. "
            "Set MCP_BEARER_TOKEN in .env for any non-trivial use."
        )

    host = os.environ.get("MCP_HOST", DEFAULT_HOST)
    port = int(os.environ.get("MCP_PORT", DEFAULT_PORT))

    mcp = build_mcp()

    # ``streamable_http_app`` returns the underlying Starlette ASGI app so we
    # can wrap it in our auth middleware before serving.
    app = mcp.streamable_http_app()
    app.add_middleware(BearerAuthMiddleware, token=token)

    log.info("starting MCP server on http://%s:%d/mcp/", host, port)

    import uvicorn

    uvicorn.run(app, host=host, port=port, log_level="info")
    return 0


if __name__ == "__main__":
    sys.exit(main())
