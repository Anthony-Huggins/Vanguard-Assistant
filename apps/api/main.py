"""FastAPI app factory for the Vanguard Assistant.

Wraps the multi-agent LangGraph from ``vanguard_agents`` behind an HTTP +
SSE surface.  As of M9 the checkpointer is pluggable (sqlite | postgres)
via ``vanguard_agents.factories.build_checkpointer``, and the API also
owns a small SQLAlchemy database for thread metadata.

Run:
    ./.venv/Scripts/python.exe -m uvicorn apps.api.main:app --reload --port 8000

The graph, checkpointer, and app-DB engine are expensive to construct, so
they're built once in ``lifespan`` and stashed on ``app.state``.  Route
handlers grab them via ``Depends`` — see ``apps/api/deps.py``.
"""

from __future__ import annotations

import logging
import os
import sys
import warnings
from contextlib import asynccontextmanager
from pathlib import Path

# Ensure ``vanguard-assistant/`` is on ``sys.path`` so ``apps.*`` and
# ``vanguard_agents.*`` both resolve regardless of how the file is invoked
# (``uvicorn apps.api.main:app`` vs ``python apps/api/main.py``).  Same trick
# the MCP server uses — see ``apps/mcp_server/server.py:27``.
_THIS = Path(__file__).resolve()
_REPO = _THIS.parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

# Silence the same harmless serialization warnings the CLI silences before
# importing anything LangGraph-adjacent — see ``cli.py:38``.
warnings.filterwarnings("ignore", message=".*PydanticSerializationUnexpectedValue.*")
warnings.filterwarnings("ignore", category=UserWarning, module="pydantic.*")

from fastapi import FastAPI  # noqa: E402
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402

from vanguard_agents.factories import build_checkpointer  # noqa: E402
from vanguard_agents.graph import build_graph  # noqa: E402
from vanguard_agents.guardrails import PIIRedactingFilter  # noqa: E402
from vanguard_agents.settings import get_settings  # noqa: E402

from apps.api.db import build_engine, build_sessionmaker  # noqa: E402
from apps.api.routers import chat, evals, stream, threads  # noqa: E402

log = logging.getLogger(__name__)

DEFAULT_CORS_ORIGIN = "http://localhost:5173"

API_DESCRIPTION = """\
HTTP surface for the Vanguard multi-agent assistant.

**Action confirmation** — the Action agent normally asks the user to confirm
before sending email or creating a Jira ticket.  In the non-streaming
``/api/chat`` endpoint M6 auto-confirms with "yes" so a single HTTP call
returns a complete response.  Real human-in-the-loop arrives in M8 over SSE.
The audit trail is preserved via the ``log_action`` JSONL.
"""


def _configure_logging() -> None:
    """Match the CLI's logging setup so the same PII filter applies."""
    s = get_settings()
    level = getattr(logging, s.log_level.upper(), logging.INFO)
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)-7s %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    )
    logging.getLogger().addFilter(PIIRedactingFilter())

    for _noisy in (
        "azure",
        "httpx",
        "httpcore",
        "sentence_transformers",
        "huggingface_hub",
        "chromadb",
        "urllib3",
    ):
        logging.getLogger(_noisy).setLevel(logging.WARNING)
    logging.getLogger("langgraph.checkpoint.serde.jsonplus").setLevel(logging.ERROR)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Build the graph + checkpointer + app DB engine once at startup."""
    settings = get_settings()

    # Honor the legacy API_CHECKPOINT_PATH env var (M6 default) by writing it
    # back to the settings before building the checkpointer.  M9+ users
    # should prefer CHECKPOINT_SQLITE_PATH, but we don't break old configs.
    legacy_path = os.environ.get("API_CHECKPOINT_PATH")
    if legacy_path and settings.checkpointer_backend == "sqlite":
        settings.checkpoint_sqlite_path = legacy_path

    engine = build_engine(settings.app_db_dsn)
    session_factory = build_sessionmaker(engine)

    async with build_checkpointer(settings) as checkpointer:
        app.state.graph = build_graph(checkpointer=checkpointer)
        app.state.checkpointer = checkpointer
        app.state.app_db_engine = engine
        app.state.app_db_sessionmaker = session_factory
        if os.environ.get("LANGSMITH_TRACING") == "true":
            log.info("LangSmith tracing enabled  ·  project=%s", os.environ.get("LANGSMITH_PROJECT"))
        log.info(
            "API ready  ·  checkpointer=%s  ·  app_db=%s",
            settings.checkpointer_backend,
            settings.app_db_dsn.split("://", 1)[0],
        )
        try:
            yield
        finally:
            await engine.dispose()


def _cors_origins() -> list[str]:
    raw = os.environ.get("API_CORS_ORIGINS", DEFAULT_CORS_ORIGIN)
    return [o.strip() for o in raw.split(",") if o.strip()]


def create_app() -> FastAPI:
    _configure_logging()
    app = FastAPI(
        title="Vanguard Assistant API",
        description=API_DESCRIPTION,
        version="0.1.0",
        lifespan=lifespan,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=_cors_origins(),
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.include_router(chat.router, prefix="/api", tags=["chat"])
    app.include_router(stream.router, prefix="/api", tags=["stream"])
    app.include_router(threads.router, prefix="/api", tags=["threads"])
    app.include_router(evals.router, prefix="/api", tags=["evals"])

    @app.get("/api/healthz", tags=["meta"])
    async def healthz() -> dict[str, bool]:
        return {"ok": True}

    return app


app = create_app()
