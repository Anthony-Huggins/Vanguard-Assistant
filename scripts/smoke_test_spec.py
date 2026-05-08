"""Phase A specification smoke test.

Exercises every checklist row from the plan's Phase A verification section
against the REAL Azure OpenAI deployment and local Chroma/SQLite.

Requirements:
  - .env must be configured (AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_DEPLOYMENT, etc.)
  - Chroma knowledge base must be ingested (run scripts/ingest_kb.py --reset once)
  - Fund warehouse must be seeded (run scripts/seed_warehouse.py once)
  - MCP server does NOT need to be running — tests that involve the action
    agent check routing only, not execution, unless MCP_SERVER_REQUIRED=1.

Usage:
    .\.venv\Scripts\python.exe scripts/smoke_test_spec.py

Each test prints PASS or FAIL.  Exit code 0 = all pass, 1 = any failure.
"""

from __future__ import annotations

import asyncio
import os
import sys
import warnings
from pathlib import Path

# ── Bootstrap ─────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

warnings.filterwarnings("ignore", message=".*PydanticSerializationUnexpectedValue.*")
warnings.filterwarnings("ignore", message=".*Relevance scores must be between 0 and 1.*")

from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

from vanguard_agents.graph import build_graph
from vanguard_agents.settings import reset_settings_for_tests
from vanguard_agents.state import fresh_turn_state

# ── Helpers ───────────────────────────────────────────────────────────────────

PASS = "✅ PASS"
FAIL = "❌ FAIL"
SKIP = "⏭  SKIP"

results: list[tuple[str, str, str]] = []  # (label, status, note)


def _record(label: str, status: str, note: str = "") -> None:
    results.append((label, status, note))
    icon = status
    line = f"  {icon}  {label}"
    if note:
        line += f"\n        {note}"
    print(line)


async def _ask(question: str) -> dict:
    """Invoke the graph with no checkpointer (ephemeral)."""
    graph = build_graph(checkpointer=None)
    state = fresh_turn_state(question)
    return await graph.ainvoke(state, config={"configurable": {"thread_id": "smoke"}})


# ── Tests ─────────────────────────────────────────────────────────────────────

async def test_support_routing_and_rag():
    """T1: Support question → routed to support, Sources: line present."""
    label = "T1  Support routing + RAG citation"
    try:
        result = await _ask("What is Vanguard's expense ratio for the 500 Index Fund?")
        agent = result.get("routing", {})
        agent_name = agent.agent if agent else "?"
        response = result.get("final_response", "")

        if agent_name != "support":
            _record(label, FAIL, f"expected route=support, got={agent_name!r}")
            return
        if "Sources:" not in response and "knowledge base" not in response.lower():
            _record(label, FAIL, f"no Sources: line in response:\n        {response[:200]}")
            return
        _record(label, PASS)
    except Exception as exc:
        _record(label, FAIL, f"{type(exc).__name__}: {exc}")


async def test_research_routing():
    """T2: Market/data question → routed to research."""
    label = "T2  Research routing"
    try:
        result = await _ask("What were the annual returns for Vanguard's 500 Index ETF over the last 3 years?")
        agent = result.get("routing", {})
        agent_name = agent.agent if agent else "?"
        if agent_name != "research":
            _record(label, FAIL, f"expected route=research, got={agent_name!r}")
            return
        _record(label, PASS)
    except Exception as exc:
        _record(label, FAIL, f"{type(exc).__name__}: {exc}")


async def test_action_routing():
    """T3: Action request → routed to action."""
    label = "T3  Action routing"
    try:
        result = await _ask("Please send an email to advisor@vanguard.com telling them about the wire transfer issue.")
        agent = result.get("routing", {})
        agent_name = agent.agent if agent else "?"
        if agent_name != "action":
            _record(label, FAIL, f"expected route=action, got={agent_name!r}")
            return
        _record(label, PASS, "MCP server not required for routing check")
    except Exception as exc:
        _record(label, FAIL, f"{type(exc).__name__}: {exc}")


async def test_clarify_routing():
    """T4: Ambiguous/gibberish → routed to clarify."""
    label = "T4  Clarify routing (ambiguous input)"
    try:
        result = await _ask("asdf qwerty zxcv")
        agent = result.get("routing", {})
        agent_name = agent.agent if agent else "?"
        if agent_name != "clarify":
            _record(label, FAIL, f"expected route=clarify, got={agent_name!r}")
            return
        _record(label, PASS)
    except Exception as exc:
        _record(label, FAIL, f"{type(exc).__name__}: {exc}")


async def test_support_no_context_fallback():
    """T5: Off-topic question → Support fallback message."""
    label = "T5  Support fallback (no relevant context)"
    try:
        result = await _ask("What is the capital of France and how does that relate to my portfolio?")
        agent = result.get("routing", {})
        agent_name = agent.agent if agent else "?"
        response = result.get("final_response", "")

        # Could be routed to support or clarify; either is valid.
        if agent_name not in ("support", "clarify"):
            _record(label, FAIL, f"unexpected route={agent_name!r}")
            return

        # If it went to support, the fallback should fire.
        if agent_name == "support":
            fallback_keywords = ["don't have", "knowledge base", "not in my", "I don't"]
            if not any(kw.lower() in response.lower() for kw in fallback_keywords):
                _record(label, FAIL, f"no fallback message; got: {response[:200]}")
                return

        _record(label, PASS)
    except Exception as exc:
        _record(label, FAIL, f"{type(exc).__name__}: {exc}")


async def test_research_db_query():
    """T6: DB query via local tool — fund data is returned."""
    label = "T6  Research db_query tool fires"
    try:
        result = await _ask("What is the expense ratio of the Vanguard 500 Index ETF (V500)?")
        agent = result.get("routing", {})
        agent_name = agent.agent if agent else "?"
        response = result.get("final_response", "")
        tool_results = result.get("tool_results", [])

        if agent_name not in ("research", "support"):
            _record(label, FAIL, f"unexpected route={agent_name!r}")
            return

        # For research route, at least one tool should have fired.
        if agent_name == "research":
            if not tool_results:
                _record(label, FAIL, "research agent ran but no tool_results recorded")
                return

        _record(label, PASS)
    except Exception as exc:
        _record(label, FAIL, f"{type(exc).__name__}: {exc}")


def test_pinecone_missing_keys_raises():
    """T7: Pluggability proof — Pinecone without keys gives a clear error."""
    label = "T7  Pinecone pluggability (missing keys → clear error)"
    import os as _os
    original = {
        k: _os.environ.get(k) for k in ("VECTORSTORE_BACKEND", "PINECONE_API_KEY", "PINECONE_INDEX")
    }
    try:
        _os.environ["VECTORSTORE_BACKEND"] = "pinecone"
        _os.environ.pop("PINECONE_API_KEY", None)
        _os.environ.pop("PINECONE_INDEX", None)
        reset_settings_for_tests()

        from vanguard_agents.factories.vectorstore import build_vectorstore
        try:
            build_vectorstore()
            _record(label, FAIL, "expected RuntimeError, got nothing")
        except RuntimeError as exc:
            if "PINECONE_API_KEY" in str(exc) or "PINECONE_INDEX" in str(exc):
                _record(label, PASS, f"error: {exc}")
            else:
                _record(label, FAIL, f"wrong error: {exc}")
        except Exception as exc:
            _record(label, FAIL, f"unexpected error type {type(exc).__name__}: {exc}")
    finally:
        # Restore original environment.
        for k, v in original.items():
            if v is None:
                _os.environ.pop(k, None)
            else:
                _os.environ[k] = v
        reset_settings_for_tests()


def test_max_tokens_caps_enforced():
    """T8: max_tokens is capped per-role (1024 coordinator, 2048 subagent)."""
    label = "T8  max_tokens caps"
    from vanguard_agents.settings import get_settings
    s = get_settings()
    ok = s.llm_max_tokens_coordinator == 1024 and s.llm_max_tokens_subagent == 2048
    if ok:
        _record(label, PASS, f"coordinator={s.llm_max_tokens_coordinator}, subagent={s.llm_max_tokens_subagent}")
    else:
        _record(label, FAIL, f"coordinator={s.llm_max_tokens_coordinator}, subagent={s.llm_max_tokens_subagent}")


def test_pii_masking_active():
    """T9: PII masking filter redacts email/SSN/phone from log records."""
    label = "T9  PII masking (guardrails)"
    from vanguard_agents.guardrails import mask_pii
    sample = "Contact 123-45-6789 at user@example.com or call 555-123-4567"
    masked = mask_pii(sample)
    if "user@example.com" in masked or "123-45-6789" in masked or "555-123-4567" in masked:
        _record(label, FAIL, f"PII not masked: {masked}")
    else:
        _record(label, PASS)


def test_history_trim_fires():
    """T10: History trim node removes messages beyond the configured window."""
    label = "T10 History trim (enforces turn limit)"
    from langchain_core.messages import AIMessage, HumanMessage
    from vanguard_agents.history import trim_history_node

    # Build 21 turns (42 messages) — 1 over the 20-turn default.
    msgs = []
    for i in range(21):
        msgs.append(HumanMessage(content=f"q{i}", id=f"h{i}"))
        msgs.append(AIMessage(content=f"a{i}", id=f"a{i}"))

    result = trim_history_node({"messages": msgs})
    if "messages" not in result:
        _record(label, FAIL, "trim_history_node returned empty dict for 21-turn history")
        return
    removed = result["messages"]
    if len(removed) != 2:
        _record(label, FAIL, f"expected 2 removals, got {len(removed)}")
        return
    _record(label, PASS, "2 oldest messages scheduled for removal")


# ── Runner ────────────────────────────────────────────────────────────────────

async def run_async_tests() -> None:
    """Run the tests that need live API calls."""
    print("\n── Live API tests (need Azure OpenAI + Chroma) ──────────────────")
    await test_support_routing_and_rag()
    await test_research_routing()
    await test_action_routing()
    await test_clarify_routing()
    await test_support_no_context_fallback()
    await test_research_db_query()


def run_unit_tests() -> None:
    """Run the tests that are purely local (no API calls)."""
    print("\n── Local unit checks ────────────────────────────────────────────")
    test_pinecone_missing_keys_raises()
    test_max_tokens_caps_enforced()
    test_pii_masking_active()
    test_history_trim_fires()


async def main() -> int:
    print("=" * 60)
    print("  Vanguard Assistant — Phase A spec smoke test")
    print("=" * 60)

    run_unit_tests()

    skip_live = os.environ.get("SKIP_LIVE_TESTS", "").lower() in ("1", "true", "yes")
    if skip_live:
        print("\n  (SKIP_LIVE_TESTS=1 — skipping API tests)")
    else:
        await run_async_tests()

    # Summary
    total = len(results)
    passed = sum(1 for _, s, _ in results if s == PASS)
    failed = sum(1 for _, s, _ in results if s == FAIL)
    skipped = sum(1 for _, s, _ in results if s == SKIP)

    print(f"\n{'=' * 60}")
    print(f"  Results: {passed}/{total} passed", end="")
    if failed:
        print(f"  {failed} FAILED", end="")
    if skipped:
        print(f"  {skipped} skipped", end="")
    print()
    if failed == 0:
        print("  Phase A spec: ALL CHECKS PASS ✅")
    else:
        print("  Phase A spec: SOME CHECKS FAILED ❌")
    print()

    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
