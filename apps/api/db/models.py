"""ORM models for the API's own metadata tables.

Currently just ``threads`` — one row per conversation, joined with the
LangGraph checkpointer to produce ``GET /api/threads`` summaries with
human-readable display names.
"""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import DateTime, String
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy.sql import func

from .base import Base


class Thread(Base):
    """A user-visible chat thread.

    Created lazily on the first chat turn for a thread_id; updated each
    subsequent turn so ``last_message_at`` drives the sidebar's sort.
    The actual conversation history lives in the LangGraph checkpointer
    keyed on the same ``thread_id``.
    """

    __tablename__ = "threads"

    thread_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    display_name: Mapped[str | None] = mapped_column(String(200), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )
    last_message_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )
