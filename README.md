# Vanguard Assistant

Multi-agent AI assistant combining LangGraph orchestration, RAG over a Vanguard document corpus, and MCP-exposed tools. Final project for the EY/Skillstorm Agentic AI class.
Core Agent code in packages/agents


**Agents:** coordinator → support (RAG) · research (web search + fund DB) · action (email / ticket) · clarify  
**Stack:** Python 3.11 · LangGraph · FastAPI · React + Vite · ChromaDB · Postgres · Docker Compose

---

## Running with Docker Compose (recommended)

### Prerequisites

- [Docker Desktop](https://www.docker.com/products/docker-desktop/) running
- Azure OpenAI resource with a `gpt-4.1` or `gpt-4.1-nano` deployment
- Source PDFs placed in `ingest/data/` (see *Ingesting documents* below)

### 1. Configure environment

```bash
cp .env.example .env
```

Open `.env` and fill in your Azure OpenAI values:

```
AZURE_OPENAI_ENDPOINT=https://<your-resource>.openai.azure.com/
AZURE_OPENAI_DEPLOYMENT=gpt-4.1
# Leave AZURE_OPENAI_API_KEY blank to use az login (device-code flow)
# OR set it to your key to skip the az login step entirely
AZURE_OPENAI_API_KEY=
```

### 2. Build and start the stack

```bash
docker compose -f infra/docker-compose.yml up --build
```

This starts: **postgres · chroma · mcp · api · web**  
The API container automatically runs database migrations on boot.

> **First boot takes a while** — the API image installs all Python deps and pre-downloads the sentence-transformers embedder model (~90 MB).  Subsequent builds are much faster.

### 3. Authenticate with Azure (if not using an API key)

If `AZURE_OPENAI_API_KEY` is blank in your `.env`, the API container uses `DefaultAzureCredential` → `AzureCliCredential`. Run this once after the stack is up and follow the device-code link in your browser:

```bash
docker exec -it vanguard-api az login --use-device-code
```

The token is stored in a named Docker volume (`vanguard_az_cache`) and persists across container restarts.

### 4. Ingest the knowledge base (first time only)

Drop Vanguard PDF/HTML/Markdown documents into `ingest/data/`, then run:

```bash
docker compose -f infra/docker-compose.yml run --rm ingest
```

This is idempotent — safe to re-run if you add more documents.

### 5. Open the app

| Service | URL |
|---|---|
| Web UI (chat) | http://localhost:5173 |
| API (OpenAPI docs) | http://localhost:8000/docs |

### Tear down

```bash
# Stop containers (preserves volumes / data)
docker compose -f infra/docker-compose.yml down

# Stop and delete all data (volumes)
docker compose -f infra/docker-compose.yml down -v
```

---

## Repo layout

```
vanguard-assistant/
  packages/
    agents/         # Core Python package — LangGraph graph, factories, subagents
    eval/           # Ragas evaluation harness (M12)
  apps/
    api/            # FastAPI service — chat, streaming SSE, thread history
    mcp_server/     # FastMCP tools server — send_email, create_ticket, log_action
    web/            # React + Vite SPA — streaming chat, agent graph viz, eval dashboard
  ingest/           # Document ingestion scripts (PDF → Chroma)
  scripts/          # Dev utilities — seed DB, smoke tests
  infra/            # Dockerfiles, Docker Compose, nginx config
  data/             # fund_warehouse.sqlite — seed data for the research agent's db_query tool
```

---

## Running locally without Docker (CLI only)

```bash
# 1. Create virtualenv and install deps
cd vanguard-assistant
python -m venv .venv
.venv/Scripts/activate          # Windows
pip install -e "packages/agents[websearch]"

# 2. Configure
cp .env.example .env
# Edit .env — set AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_DEPLOYMENT

# 3. Ingest documents
python ingest/scripts/ingest_kb.py

# 4. Start the MCP tools server (separate terminal — needed for action tools)
python apps/mcp_server/server.py

# 5. Run the CLI
vanguard-cli
```

---

## Build phases

- **Phase A (M1–M5)** — Python CLI: LangGraph multi-agent graph, RAG pipeline, MCP tools, history management, retries, PII guardrails
- **Phase B (M6–M10)** — Full-stack: FastAPI + SSE streaming + React SPA + Postgres + Docker Compose
- **Phase C (M11–M12)** — Stretch goals: live agent graph visualization (React Flow), Ragas eval dashboard
