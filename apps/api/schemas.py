"""Pydantic request/response models for the FastAPI app.

These mirror (a subset of) ``vanguard_agents.state`` types but are explicitly
HTTP-boundary models — the agents package uses TypedDicts for state, Pydantic
only at module boundaries.  Hand-typed mirror in ``apps/web/src/types/api.ts``.
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


AgentName = Literal["support", "research", "action", "clarify"]


class ChatRequest(BaseModel):
    thread_id: str = Field(..., min_length=1, max_length=128)
    message: str = Field(..., min_length=1, max_length=4000)


class Source(BaseModel):
    """One retrieved chunk surfaced to the client as a citation."""

    source: str
    chunk_id: str
    page: int | None = None
    score: float


class ChatResponse(BaseModel):
    reply: str
    sources: list[Source]
    agent_used: AgentName
    routing_reason: str


class ThreadSummary(BaseModel):
    thread_id: str
    display_name: str | None = None
    last_updated: datetime
    message_count: int


class ThreadMessage(BaseModel):
    role: str  # "human" | "ai" | "system" | "tool"
    content: str


class ThreadDetail(BaseModel):
    thread_id: str
    messages: list[ThreadMessage]


class ResumeRequest(BaseModel):
    """Body for POST /api/chat/resume — answer a pending action confirmation."""

    thread_id: str = Field(..., min_length=1, max_length=128)
    answer: str = Field(..., min_length=1, max_length=32)  # "yes" | "no"
