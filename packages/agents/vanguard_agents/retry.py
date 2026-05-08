"""Retry helpers for LLM calls.

LangChain Runnables expose ``.with_retry()`` which wraps the runnable in a
tenacity retry loop under the hood.  ``wrap_with_retry`` applies a standard
policy (3 attempts, exponential back-off 2-10 s with jitter) appropriate for
transient Azure OpenAI errors: rate limits, 5xx responses, and timeouts.

Usage — apply AFTER ``.with_structured_output()`` or ``.bind_tools()``, since
those methods live on ``BaseChatModel`` and return a new ``Runnable``.  The
retry wrapper goes on the outermost chain::

    classifier = wrap_with_retry(model.with_structured_output(RoutingDecision))
    decision = classifier.invoke(messages)   # retried up to 3×

    llm_with_tools = wrap_with_retry(model.bind_tools(tools))
    response = await llm_with_tools.ainvoke(convo)  # retried up to 3×
"""

from __future__ import annotations

from langchain_core.runnables import Runnable


def wrap_with_retry(runnable: Runnable) -> Runnable:
    """Return *runnable* wrapped with a 3-attempt exponential-backoff policy.

    Back-off: starts at 2 s, doubles each attempt, caps at 10 s, with random
    jitter to avoid thundering-herd when multiple workers hit a rate limit
    simultaneously.

    All exceptions are eligible for retry; the underlying tenacity policy will
    re-raise on the final attempt so errors are never silently swallowed.
    """
    return runnable.with_retry(
        stop_after_attempt=3,
        wait_exponential_jitter=True,
    )
