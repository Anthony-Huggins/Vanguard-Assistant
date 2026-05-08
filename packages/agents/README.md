# vanguard-agents

The reusable LangGraph runtime for the Vanguard Assistant. Imported by the CLI, the FastAPI service, and the Ragas eval harness.

## Public API (so far)

```python
from vanguard_agents.settings import get_settings
from vanguard_agents.factories.llm import build_chat_model
from vanguard_agents.factories.embedder import build_embedder
from vanguard_agents.factories.vectorstore import build_vectorstore, build_retriever
```

## Pluggability contract

Agent code MUST go through these factories. It MUST NOT import provider SDKs (`openai`, `anthropic`, `chromadb`, `pinecone`) directly. Swapping providers must be done through settings.
