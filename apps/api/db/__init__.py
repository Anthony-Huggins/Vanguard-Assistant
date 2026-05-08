"""SQLAlchemy 2 (async) layer for the API's own database (M9).

Distinct from the LangGraph checkpointer — that owns its own tables,
managed by ``langgraph-checkpoint-postgres``/``-sqlite``.  These tables
hold *application* metadata (currently just a ``threads`` row per
conversation for human-readable display names).  Migrations are managed
by Alembic at the repo root.
"""

from .base import Base, build_engine, build_sessionmaker
from .models import Thread

__all__ = ["Base", "Thread", "build_engine", "build_sessionmaker"]
