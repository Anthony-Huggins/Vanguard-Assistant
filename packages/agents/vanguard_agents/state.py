"""LangGraph state schema and the structured types it carries.

The graph state is a plain ``TypedDict`` (per the spec). Pydantic models are
used for boundary types — the coordinator's routing decision and the chunk /
tool-result records — but state itself stays untyped at the runtime level so
LangGraph's reducers work cleanly.
"""

from __future__ import annotations

from typing import Annotated, Any, Literal, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field


class RoutingDecision(BaseModel):
    """Coordinator output. Bound to the LLM via ``with_structured_output``.

    The ``clarify`` route is an addition to the spec — it lets the coordinator
    bail out instead of inventing a route on ambiguous input.
    """

    agent: Literal["support", "research", "action", "clarify"] = Field(
        description="Which subagent should handle the user message."
    )
    reason: str = Field(
        description="One short sentence explaining the choice.",
        max_length=240,
    )


class RetrievedChunk(TypedDict):
    """A chunk returned from the vector store, with citation metadata."""

    text: str
    source: str
    chunk_id: str
    score: float
    page: int | None


class ToolResult(TypedDict):
    """Recorded result of a tool invocation (for tracing / UI badges)."""

    name: str
    args: dict[str, Any]
    output: str
    error: str | None


class ConversationState(TypedDict):
    """The state object that flows through every node in the graph."""

    # Conversation history. The reducer appends instead of overwriting.
    messages: Annotated[list[BaseMessage], add_messages]

    # Set fresh each turn.
    user_message: str
    routing: RoutingDecision | None
    retrieved_chunks: list[RetrievedChunk]
    tool_results: list[ToolResult]
    final_response: str | None

    # Thread identifier — set by the application layer when invoking.
    thread_id: str | None


def fresh_turn_state(user_message: str, thread_id: str | None = None) -> ConversationState:
    """Build the per-turn input dict to feed ``graph.invoke``.

    The conversation history (``messages``) is loaded from the checkpointer
    when a ``thread_id`` is reused; the new HumanMessage is appended via the
    ``add_messages`` reducer.
    """
    from langchain_core.messages import HumanMessage

    return {  # type: ignore[typeddict-item]
        "user_message": user_message,
        "messages": [HumanMessage(content=user_message)],
        "routing": None,
        "retrieved_chunks": [],
        "tool_results": [],
        "final_response": None,
        "thread_id": thread_id,
    }
