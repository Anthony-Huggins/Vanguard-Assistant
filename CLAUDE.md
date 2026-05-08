# Vanguard Assistant — Claude operating context

This file is loaded automatically by Claude Code when working in this project.
It captures conventions, the plan, and the current state so a fresh session
can pick up exactly where the last one stopped.

## What this project is

Final project for the EY/Skillstorm Agentic AI class (course root:
`C:\Users\langn\Skillstorm\Training\ai\20260330-EY-Agentic-AI\`). A multi-agent
chatbot combining (a) LangGraph orchestrator-subagent routing,
(b) RAG over a Vanguard document corpus, and (c) tool use exposed through an
MCP server. Designed as a Python CLI first, then layered into a full-stack web
app with React + Vite frontend, FastAPI backend, SSE streaming, agent graph
visualization, Ragas evals, voice mode, and PDF citation highlighting.

The student is **deliberately learning new things** — full-stack, streaming,
observability, deployment — beyond their current Python/LangGraph comfort
zone. Plan accordingly when suggesting approaches.

## Where to find things

- **The plan**: `C:\Users\langn\.claude\plans\i-need-your-help-immutable-panda.md`
  Contains the milestone-by-milestone build plan (M1 through M15) plus the
  spec critique (what we changed from the original outline and why). Read this
  first.
- **Original course spec** (the assignment outline) lives in the user's first
  message of the session that produced this project — not on disk. The plan
  file's "Spec critique" section captures the deviations.
- **Class repo root** (one level up): contains `00. Labs/`, `01. Lecture/`,
  `Presentations/`. Lab 05 (Tools/MCP) and Lab 07 (Chunking) are the most
  reusable patterns.

## Repo layout

```
vanguard-assistant/
  packages/
    agents/                       # CORE - reusable Python package
      vanguard_agents/
        settings.py               # Pydantic BaseSettings — env-driven config
        factories/                # LLM/embedder/vectorstore — pluggability layer
        state.py                  # ConversationState TypedDict + RoutingDecision
        coordinator.py            # routing node
        subagents/                # support, research, action, clarify
        graph.py                  # StateGraph wiring
        cli.py                    # interactive driver (entry: vanguard-cli)
      tests/
      pyproject.toml
    eval/                         # Ragas harness (Phase C)
  apps/
    cli/main.py                   # thin shim → vanguard_agents.cli:main
    api/                          # FastAPI service (Phase B, not yet built)
    mcp_server/                   # FastMCP tools server (M4)
    web/                          # React + Vite SPA (Phase B)
  ingest/                         # document ingestion scripts (M3)
  scripts/                        # one-off developer scripts (seed DB, smoke tests)
  infra/                          # Docker Compose (Phase B finale)
  .env / .env.example             # config
  .venv/                          # Python virtualenv (gitignored)
```

## Hard rules

1. **The pluggability contract**: agent code MUST go through
   `vanguard_agents.factories.{build_chat_model, build_embedder, build_vectorstore, build_retriever}`.
   It MUST NOT import `openai`, `anthropic`, `chromadb`, `pinecone`, or
   `azure.*` directly. Swapping providers must be a `.env` change.
1a. **Tool location split**: side-effecting (action) tools live in the
   separate MCP server (`apps/mcp_server/`). Read-only research tools live
   in-process at `vanguard_agents/tools/`. Don't move tools across this
   boundary without a deliberate reason.
2. **State is a TypedDict, not a Pydantic model**. `ConversationState` in
   `state.py`. The `messages` field uses `Annotated[list[BaseMessage], add_messages]`
   so LangGraph appends instead of overwriting.
3. **Pydantic is for boundaries**: tool inputs, structured LLM outputs
   (`RoutingDecision`), config (`Settings`). Not for state.
4. **Routing has 4 routes, not 3**: `support | research | action | clarify`.
   The `clarify` route is our addition to the spec — it lets the coordinator
   bail out on ambiguous input instead of guessing.
5. **No server-side conversation memory in the application layer**. Use
   LangGraph's checkpointer (`SqliteSaver` in dev, `PostgresSaver` later)
   keyed on `thread_id`. The original spec says "fresh state per request" —
   we explicitly diverge.
6. **Models**: only `gpt-4.1-nano` and `gpt-4.1` are available on the Azure
   deployment. `Literal["gpt-4.1-nano", "gpt-4.1"]` is enforced in settings.
7. **Don't write *.md docs unless asked.** This file and the plan file are
   the only docs. Keep code self-documenting.

## How to run things

Working venv lives at `vanguard-assistant/.venv/`. From the project root:

```bash
# 1. Start the MCP tools server in one terminal (only Action tools need it)
./.venv/Scripts/python.exe apps/mcp_server/server.py

# 2. In another terminal, run the CLI
./.venv/Scripts/vanguard-cli.exe

# Run the test suite (no servers needed; everything is mocked)
./.venv/Scripts/python.exe -m pytest packages/agents -q

# Run the API test suite (mocked graph; no Azure / MCP needed)
./.venv/Scripts/python.exe -m pytest apps/api/tests -q

# Boot the FastAPI app (M6) — http://localhost:8000/docs for OpenAPI
./.venv/Scripts/python.exe -m uvicorn apps.api.main:app --reload --port 8000

# Boot the React + Vite SPA (M7) — http://localhost:5173/
# Run from apps/web/. Dev proxies /api/* to localhost:8000 so the API
# must already be running for chat to work.
cd apps/web && npm run dev

# Run the web test suite (Vitest + Testing Library)
cd apps/web && npm test

# Type-check + production build
cd apps/web && npm run build

# --- M9: Postgres + Alembic ---
# Bring up Postgres for dev (M10 extends this with the full stack):
docker compose -f infra/docker-compose.dev.yml up -d
# Run the app DB migrations (works on sqlite-default and Postgres alike):
./.venv/Scripts/python.exe -m alembic upgrade head
# Switch the whole stack to Postgres:
#   CHECKPOINTER_BACKEND=postgres
#   POSTGRES_DSN=postgresql://vg:vg@localhost:5432/vanguard
#   APP_DB_DSN=postgresql+asyncpg://vg:vg@localhost:5432/vanguard
# (set in .env, then re-run uvicorn / vanguard-cli — same factory drives both)

# --- M10: full Docker Compose stack ---
# Boot api + web + mcp + chroma + postgres in one shot.  Reads creds (Azure,
# SendGrid, Jira) from the host .env so secrets stay out of the images.
docker compose -f infra/docker-compose.yml up --build

# One-time KB ingest (populates the chroma container).  Safe to re-run.
docker compose -f infra/docker-compose.yml run --rm ingest

# SPA at http://localhost:5173, OpenAPI at http://localhost:8000/docs.
# Tear down + drop volumes:
docker compose -f infra/docker-compose.yml down -v

# Re-seed the fund warehouse SQLite (only needed once / after schema changes)
./.venv/Scripts/python.exe scripts/seed_warehouse.py

# Re-ingest the knowledge base into Chroma (idempotent)
./.venv/Scripts/python.exe ingest/scripts/ingest_kb.py --reset

# Test real SendGrid / Jira credentials (edit SMOKE_RECIPIENT first)
./.venv/Scripts/python.exe scripts/smoke_test_integrations.py

# Phase A spec smoke test (all 10 checklist rows; needs Azure + Chroma live)
./.venv/Scripts/python.exe scripts/smoke_test_spec.py
# Skip the live API tests (unit checks only):
# SKIP_LIVE_TESTS=1 ./.venv/Scripts/python.exe scripts/smoke_test_spec.py

# Quick sanity check that settings loaded the .env correctly
./.venv/Scripts/python.exe -c "from vanguard_agents.settings import get_settings; s=get_settings(); print(s.azure_openai_endpoint, s.azure_openai_deployment, s.llm_model)"
```

The CLI silently degrades if the MCP server is down: Support and Research
keep working (their tools are local / in-process), but Action will reply
with "I can't reach the tools right now."

The `.env` file is at `vanguard-assistant/.env`. The settings module walks up
from `settings.py`'s install path to find it, so it works regardless of CWD.

## Auth

`.env` either sets `AZURE_OPENAI_API_KEY` (simple) or leaves it blank, in which
case the LLM factory falls back to `DefaultAzureCredential` from
`azure-identity` — meaning `az login` must have run in the user's shell. This
is the same pattern used in the class labs.

## Current build state

Tracked in the conversation's TodoWrite list. As of the last update:

- **Phase A (M1–M5: core spec — Python CLI multi-agent assistant)**
  - [x] M1: Foundation — settings, factories, pyproject, tests
  - [x] M2: LangGraph skeleton — coordinator, 4 stub subagents, graph wiring,
        SqliteSaver checkpointer, CLI; verified against real Azure
  - [x] M3: RAG pipeline — ingest 27 Vanguard docs (PDF/MD/HTML) → 544 chunks
        in Chroma; Support Agent retrieves + cites; threshold calibrated to
        0.25 for `all-MiniLM-L6-v2` (was 0.75 in the original spec — see
        Spec critique #2)
  - [x] M4: Tools split by side-effect profile.
        - **MCP server** (FastMCP, streamable HTTP, bearer-token auth) hosts
          the 3 action tools:
          - `send_email` → SendGrid API (`SENDGRID_API_KEY` +
            `SENDGRID_FROM_EMAIL`); falls back to a JSONL+console stub when
            creds are absent.
          - `create_ticket` → Jira REST v3 (`JIRA_URL` + `JIRA_USER_EMAIL` +
            `JIRA_API_TOKEN` + optional `JIRA_PROJECT_KEY`); JSONL stub fallback.
          - `log_action` → always local JSONL audit log.
          API failures (e.g. invalid SendGrid key) are RAISED, not silently
          stubbed — masking a misconfigured production integration would be
          worse than failing visibly.
        - **Local in-process** (LangChain `@tool` bound directly to the LLM):
          web_search, db_query.
        Both paths use the shared ReAct tool-calling loop capped at 6 rounds.
        Graph is async (CLI uses `ainvoke`).
  - [x] M5: Hardening — history trim node (trim_history_node wired before
        coordinator), tenacity retries via wrap_with_retry() on all LLM call
        sites (coordinator, support, tool loop), PII masking (PIIRedactingFilter
        installed on root logger in cli.py), confirmation interrupt for
        send_email/create_ticket (LangGraph interrupt() + CLI Command(resume=)
        loop), pluggability proof (Pinecone factory + clear missing-keys error)
- **Phase B (M6–M10: full-stack web app)**
  - [x] M6: FastAPI backend at `apps/api/`. `POST /api/chat`,
        `GET /api/threads`, `GET /api/threads/{id}`, `GET /api/healthz`.
        OpenAPI docs at `/docs`. CORS allows `http://localhost:5173` by
        default (override via `API_CORS_ORIGINS`). Graph + AsyncSqliteSaver
        built once in lifespan, injected via `app.state` + FastAPI
        dependencies. Action-tool confirmation interrupts are auto-confirmed
        in this endpoint — see "Known quirks" below. Tests in
        `apps/api/tests/` use `httpx.AsyncClient` + `ASGITransport` with the
        graph mocked.
  - [x] M7: React + Vite + TypeScript SPA at `apps/web/`. TailwindCSS v4 +
        shadcn/ui (button, card, input, scroll-area, dialog, badge). Zustand
        store for chat state. Single chat page; thread id stored in
        `?thread=` URL param (UUID generated on first visit). Vite dev
        proxies `/api/*` to `http://localhost:8000` so the SPA can use
        relative URLs. Vitest + Testing Library smoke-test
        `MessageBubble`. SSE streaming arrives in M8.
  - [x] M8: SSE streaming. New `POST /api/chat/stream` and
        `POST /api/chat/resume` routes emit a translated event taxonomy
        (`agent_start` / `agent_end` / `token` / `tool_start` / `tool_end` /
        `citation` / `confirmation_required` / `done` / `error`) over
        `sse-starlette`. The translator (`apps/api/streaming.py`) folds
        `astream_events(version="v2")` and filters out coordinator tokens
        via `metadata.langgraph_node`. Frontend uses
        `@microsoft/fetch-event-source` (POST + headers; native EventSource
        is GET-only); `lib/eventReducer.ts` folds events into the Zustand
        store. Action confirmations now show a real modal — clicking
        Confirm/Deny POSTs to `/api/chat/resume` which `Command(resume=...)`s
        the same checkpoint. The non-streaming `/api/chat` endpoint stays
        as-is (handy for `curl` and integration tests).
  - [x] M9: Postgres persistence + thread metadata table.
        - **Pluggable checkpointer**:
          `vanguard_agents.factories.build_checkpointer(settings)` returns
          either `AsyncSqliteSaver` or `AsyncPostgresSaver` keyed on
          `CHECKPOINTER_BACKEND` (default `sqlite`). Both the API lifespan
          and the CLI's `_build_checkpointer` go through this factory —
          one env var swap moves the whole stack to Postgres.
        - **App DB**: SQLAlchemy 2 (async) + Alembic for the API's own
          metadata. The `threads` table holds `display_name`, `created_at`,
          `last_message_at` per `thread_id`. Migrations live in `migrations/`
          with `alembic.ini` at repo root. Run
          `./.venv/Scripts/python.exe -m alembic upgrade head` after
          changing `APP_DB_DSN`.
        - **Routers**: `/api/chat` and `/api/chat/stream` both upsert the
          thread row after a successful turn (display name = first user
          message, truncated to 80 chars). `/api/threads` joins the metadata
          table with the checkpointer for `message_count`; the M6 fallback
          (iterate the saver) still kicks in when the table is empty.
        - **Frontend**: `ThreadSidebar` lists past conversations with
          relative timestamps and message counts. Clicking a thread updates
          `?thread=...`; `Chat` rehydrates from `GET /api/threads/{id}` on
          thread switch. Historical assistant messages show as plain cards
          (no badge / no sources — those aren't recoverable from the
          checkpointer's BaseMessages).
        - **Dev infra**: `infra/docker-compose.dev.yml` brings up Postgres
          alone for M9 dev; M10 will extend it with the rest of the stack.
  - [x] M10: Docker Compose ties the whole stack together. One
        `docker compose -f infra/docker-compose.yml up --build` boots
        postgres + chroma + mcp + api + web with healthcheck-gated
        `depends_on`. Highlights:
        - **Per-service Dockerfiles** in `infra/`: `Dockerfile.api` and
          `Dockerfile.mcp` share `python:3.11-slim` + `pip install -e
          packages/agents`; `Dockerfile.web` is multi-stage
          (`node:20-alpine` builder → `nginx:alpine` runtime).
        - **API entrypoint** (`infra/api-entrypoint.sh`) runs
          `alembic upgrade head` before launching uvicorn, so a fresh boot
          provisions the threads schema automatically. Idempotent on
          subsequent boots.
        - **Chroma over HTTP**: new `chroma_mode=http` branch in the
          vectorstore factory + `CHROMA_URL` setting let the api container
          talk to a stand-alone Chroma service. Existing embedded mode is
          unchanged for local dev.
        - **Nginx** (`infra/nginx.conf`) serves the SPA build and proxies
          `/api/*` → `http://api:8000` with SSE-friendly settings
          (`proxy_buffering off`, 1-hour read timeout) so streaming chat
          works through the prod proxy.
        - **One-shot ingest service** (compose profile `ingest`) runs the
          KB ingestion against the chroma container — exercised once per
          fresh stack with `docker compose run --rm ingest`.
        - **Sentence-transformers** model is pre-downloaded into the API
          image at build time so the first chat doesn't pay the model-load
          tax.
        - `.dockerignore` keeps `.venv`, `node_modules`, local sqlite/chroma
          data, and `.env` out of the build context.
- **Phase C (M11–M15: stretch goals — graph viz, Ragas, voice, PDF viewer)**
  - [x] M11: Live LangGraph agent visualization. Static topology
        (coordinator → 4 subagents → 5 tools) defined in
        `apps/web/src/graph/topology.ts`; runtime state overlay driven by
        the existing SSE event reducer.
        - **Backend**: `apps/api/streaming.py` widened with a `TRACED_NODES`
          set so the coordinator emits `agent_start`/`agent_end` on every
          turn. Token forwarding still gated to `SUBAGENT_NODES` only —
          coordinator's structured-output classification tokens stay
          filtered.
        - **Frontend**: `<AgentTrace />` panel slotted between MessageList
          and the input form in `apps/web/src/components/Chat.tsx`. Uses
          `@xyflow/react` (React Flow) with pan/zoom disabled — read-only
          canvas. Node styling keyed on `NodeState`
          (`idle`/`active`/`done`/`error`) from
          `apps/web/src/types/trace.ts`. Edges between active+done nodes
          turn green; edge into the active downstream node animates.
        - **Reducer**: `lib/eventReducer.ts` updates `nodeStates` from
          `agent_start`/`agent_end`/`tool_start`/`tool_end`/`error` events.
          Coordinator's `agent_start` updates the trace but NOT a streaming
          bubble (it has no user-facing tokens). New turns call
          `resetNodeStates()` from `Chat.tsx` to clear the canvas.
        - **Side panel**: top-right of the trace shows the routing reason
          captured from `RoutingDecision` (already on `agent_start.routing_reason`).
        - **Toggle**: a switch in the AgentTrace header collapses the canvas
          to just the header bar (local React state, not persisted).
  - [x] M12: Ragas eval harness. Support-only replay, gpt-4.1 judge, 20 Q/A
        pairs, timestamped JSON results.
        - **Eval package**: `packages/eval/` — `run_eval.py` replays each
          question through `support_node()` directly (bypasses coordinator),
          scores via Ragas (faithfulness, answer_relevancy, context_precision)
          + custom citation_hit metric. Results in
          `packages/eval/results/<timestamp>.json`.
        - **Judge factory**: `packages/agents/vanguard_agents/factories/judge.py`
          wraps `build_chat_model(model_override=eval_judge_model)`. Adds
          `model_override` kwarg to `build_chat_model` so M12 can target
          gpt-4.1 regardless of the app's default deployment.
        - **API**: `GET /api/evals/latest` and `GET /api/evals/runs` in
          `apps/api/routers/evals.py`; reads from `EVAL_RESULTS_DIR`.
        - **SPA**: "Evals" tab in the header nav switches to `EvalDashboard`
          (same-page toggle via `?view=evals`; no router library). Shows
          summary score cards + per-question table.
        - **Run command**:
          `./.venv/Scripts/python.exe packages/eval/run_eval.py`
          (needs `pip install ragas datasets` first).

## Known quirks

- The non-streaming `POST /api/chat` endpoint auto-confirms Action-tool
  interrupts (`send_email` / `create_ticket`) with `Command(resume="yes")`
  so a single HTTP request always returns a complete response. The
  streaming `POST /api/chat/stream` (M8) keeps the human-in-the-loop:
  emits `confirmation_required`, closes the stream, and waits for the
  client to POST `/api/chat/resume` with `{thread_id, answer}`. Audit
  trail is preserved via the `log_action` JSONL either way.
- A harmless `PydanticSerializationUnexpectedValue` warning fires from the
  LangGraph checkpointer when serializing `RoutingDecision`. Suppressed in
  `cli.py` via `warnings.filterwarnings`. The fix lives upstream in
  `langchain-openai`; revisit when bumping that pin.
- Tests use `pytest.importorskip` for `langchain_huggingface` and
  `langchain_chroma` so they pass even before M3's heavy deps are installed.
- The minimal venv install used `--no-deps` for the agents package; `pip`
  reports "missing" warnings for `sentence-transformers`, `structlog`,
  `unstructured`. Those install during M3.

## When in doubt

- Read the plan file (`~/.claude/plans/i-need-your-help-immutable-panda.md`)
  before suggesting any architectural change.
- Don't break the pluggability contract.
- Don't introduce new top-level docs.
- If something feels redundant with what's in the plan, link the plan instead
  of restating it.
