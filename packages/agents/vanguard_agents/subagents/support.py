"""Support Agent — RAG-grounded answers from the Vanguard knowledge base.

Pipeline:
1. Retrieve top-k chunks for the user's question.
2. If the best chunk's score is below ``rag_similarity_threshold``,
   short-circuit with the "I don't know" fallback (and log the query).
3. Otherwise, call the LLM with the retrieved context. The system prompt
   constrains it to answer ONLY from the provided chunks.
4. Append the ``Sources:`` line to the response.
"""

from __future__ import annotations

import logging

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from ..factories.llm import build_chat_model
from ..retry import wrap_with_retry
from ..rag import (
    best_score,
    format_citation_line,
    render_chunks_for_prompt,
    retrieve,
)
from ..settings import get_settings
from ..state import ConversationState

log = logging.getLogger(__name__)

SUPPORT_SYSTEM_PROMPT = """\
You are the Vanguard Support Agent. Answer the user's question DIRECTLY using
ONLY the information in the context below.

Rules:
1. Give the actual answer.  State the facts, requirements, steps, numbers, and
   policies that are in the context.  Quote them when helpful.
2. NEVER tell the user to "refer to the document," "see the document for
   details," or "the document describes…".  The user is asking YOU because
   they don't want to read the document — extract and present the content.
3. If the context contains a list (account types, required documents,
   procedures), reproduce that list in your answer.
4. If the context truly does not answer the question, say so explicitly with
   "I don't have that information in my knowledge base" — never invent
   details.
5. Do NOT include inline citations like ``[Title · chunk_id]`` in your answer.
   The application appends a formal ``Sources:`` block after your reply
   automatically.

Length: as long as needed to fully answer the question.  Brevity is good but
completeness is more important — it's better to give the user the actual
content than to be terse and unhelpful.

Context:
{retrieved_chunks}
"""

NO_CONTEXT_FALLBACK = (
    "I don't have that information in my knowledge base. "
    "Would you like me to escalate this to a human advisor?"
)


def support_node(state: ConversationState) -> dict:
    s = get_settings()
    user_message = state["user_message"]

    chunks = retrieve(user_message)
    top_score = best_score(chunks)

    if not chunks or top_score < s.rag_similarity_threshold:
        log.info(
            "support fallback: query=%r best_score=%.3f threshold=%.3f",
            user_message,
            top_score,
            s.rag_similarity_threshold,
        )
        reply = NO_CONTEXT_FALLBACK
        return {
            "retrieved_chunks": chunks,
            "final_response": reply,
            "messages": [AIMessage(content=reply)],
        }

    rendered = render_chunks_for_prompt(chunks)
    model = wrap_with_retry(build_chat_model(max_tokens=s.llm_max_tokens_subagent))

    # Pass the full conversation history so the LLM can answer follow-ups
    # like "tell me more about that" or "compare it to the previous one".
    # The current user message is already the last entry in state["messages"]
    # (added by fresh_turn_state and persisted by add_messages).
    history = state.get("messages") or [HumanMessage(content=user_message)]
    messages = [
        SystemMessage(content=SUPPORT_SYSTEM_PROMPT.format(retrieved_chunks=rendered)),
        *history,
    ]
    response = model.invoke(messages)
    answer_body = response.content if isinstance(response.content, str) else str(response.content)

    citation = format_citation_line(chunks)
    final = f"{answer_body}\n\n{citation}" if citation else answer_body

    return {
        "retrieved_chunks": chunks,
        "final_response": final,
        "messages": [AIMessage(content=final)],
    }
