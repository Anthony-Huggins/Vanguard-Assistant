"""Judge LLM factory for the M12 Ragas eval harness.

Uses gpt-4.1 (full) regardless of the app's default deployment because
gpt-4.1-nano is too weak for Ragas judge prompts (hallucination on scores).
The deployment is overridable via EVAL_JUDGE_MODEL.
"""

from __future__ import annotations

from langchain_core.language_models import BaseChatModel

from ..settings import get_settings
from .llm import build_chat_model


def build_judge_model() -> BaseChatModel:
    """Construct the judge LLM for Ragas scoring.

    Defaults to gpt-4.1; override with EVAL_JUDGE_MODEL in .env.
    """
    s = get_settings()
    return build_chat_model(model_override=s.eval_judge_model)
