"""Interactive CLI driver for the multi-agent assistant.

Usage:
    python -m vanguard_agents.cli                  # ephemeral in-memory thread
    python -m vanguard_agents.cli --thread MY_ID   # resume a named thread
    python -m vanguard_agents.cli --no-checkpoint  # no persistence at all

Interrupt / confirmation flow
------------------------------
When the Action Agent is about to run ``send_email`` or ``create_ticket`` it
calls LangGraph's ``interrupt()``, which pauses the graph and surfaces the
pending action for the user to confirm.  The CLI catches the ``GraphInterrupt``
exception, prints the confirmation prompt, reads the user's answer, and resumes
the graph via ``Command(resume=answer)``.

Note: ``--no-checkpoint`` disables resumable interrupts.  The action tools will
still execute, but the confirmation loop is skipped.  This is intentional for
automated tests and dev smoke-runs.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import re
import sys
import uuid
import warnings
from contextlib import asynccontextmanager
from pathlib import Path

# Silence a known harmless serialization warning from langchain-openai's
# ``with_structured_output`` interaction with Pydantic when the parsed model
# is round-tripped through the LangGraph checkpointer.
# We filter on both the message pattern AND by module so the filter fires
# regardless of how the warning text is split across lines.
warnings.filterwarnings(
    "ignore",
    message=".*PydanticSerializationUnexpectedValue.*",
)
warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    module="pydantic.*",
)

from .factories import build_checkpointer  # noqa: E402
from .graph import build_graph  # noqa: E402  (after warnings filter)
from .guardrails import PIIRedactingFilter  # noqa: E402
from .settings import get_settings  # noqa: E402
from .state import fresh_turn_state  # noqa: E402

DEFAULT_CHECKPOINT_PATH = ".checkpoints.sqlite"


def _configure_logging() -> None:
    """Set up root logging with PII redaction.

    The PII filter is attached to the root logger so every module's
    ``log = logging.getLogger(__name__)`` inherits it automatically.

    Third-party libraries (azure-identity, httpx, sentence-transformers,
    huggingface_hub, chromadb) are noisy at INFO level.  We push them up to
    WARNING so the CLI output stays readable.  Our own ``vanguard_agents.*``
    loggers still respect the configured LOG_LEVEL.
    """
    s = get_settings()
    level = getattr(logging, s.log_level.upper(), logging.INFO)
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)-7s %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    )
    logging.getLogger().addFilter(PIIRedactingFilter())

    # Silence chatty third-party loggers (push above their default emit level).
    for _noisy in (
        "azure",
        "httpx",
        "httpcore",
        "sentence_transformers",
        "huggingface_hub",
        "chromadb",
        "urllib3",
        "asyncio",
    ):
        logging.getLogger(_noisy).setLevel(logging.WARNING)

    # This one logs at WARNING (a cosmetic deprecation notice, not an error),
    # so silencing it requires going up to ERROR.
    logging.getLogger("langgraph.checkpoint.serde.jsonplus").setLevel(logging.ERROR)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Interactive Vanguard Assistant CLI.",
    )
    parser.add_argument(
        "--thread",
        default=None,
        help="Thread id to resume (default: random uuid).",
    )
    parser.add_argument(
        "--checkpoint-path",
        default=DEFAULT_CHECKPOINT_PATH,
        help="SQLite file for the LangGraph checkpointer.",
    )
    parser.add_argument(
        "--no-checkpoint",
        action="store_true",
        help="Run with no checkpointer (state lost between turns; "
             "action-tool confirmation prompts are also disabled).",
    )
    return parser.parse_args(argv)


@asynccontextmanager
async def _build_checkpointer(path: str | None):
    """Yield a checkpointer respecting CLI flags + Settings.

    ``--no-checkpoint`` (``path is None``) → yield ``None`` (no persistence).
    Otherwise delegate to :func:`vanguard_agents.factories.build_checkpointer`
    so the CLI honors the same pluggability contract as the FastAPI app.
    The CLI's ``--checkpoint-path`` flag overrides
    ``settings.checkpoint_sqlite_path`` for backward compatibility but only
    applies when the configured backend is sqlite.
    """
    if path is None:
        yield None
        return

    settings = get_settings()
    if settings.checkpointer_backend == "sqlite":
        # Honor the CLI flag's explicit path.
        settings.checkpoint_sqlite_path = path

    async with build_checkpointer(settings) as cp:
        yield cp


_RULE = "─" * 68

# Matches the "[Research Agent] (2 tool calls) " prefix added by with_agent_badge.
_BADGE_RE = re.compile(r"^\[([^\]]+)\](?:\s*\(\d+\s+tool\s+calls?\))?\s*")

# Matches the "\n\nSources: [doc, chunk_id: X] [...]" block at the end of RAG answers.
_SOURCES_RE = re.compile(r"\n\nSources:\s*((?:\[[^\]]+\]\s*)+)\s*$")
_SOURCE_ITEM_RE = re.compile(r"\[([^,\]]+),\s*chunk_id:\s*([^\]]+)\]")


def _format_response(result: dict | None) -> str:
    if result is None:
        return f"\n{_RULE}\n(no result returned from graph)\n{_RULE}"

    routing = result.get("routing")
    # After a checkpointer round-trip the RoutingDecision may come back as a
    # plain dict instead of the Pydantic model — handle both forms.
    if routing is None:
        agent, reason = "?", ""
    elif isinstance(routing, dict):
        agent = routing.get("agent", "?")
        reason = routing.get("reason", "") or ""
    else:
        agent = routing.agent
        reason = (routing.reason or "") if hasattr(routing, "reason") else ""

    response = result.get("final_response") or "(no response)"

    # Strip [Agent Name] (N tool calls) badge and note the count for the footer.
    tool_calls_label = ""
    badge_match = _BADGE_RE.match(response)
    if badge_match:
        tc_match = re.search(r"\((\d+)\s+tool\s+calls?\)", response[: badge_match.end()])
        if tc_match:
            n = int(tc_match.group(1))
            tool_calls_label = f"  ·  {n} tool call{'s' if n != 1 else ''}"
        response = response[badge_match.end():]

    # Separate answer body from the Sources block.
    source_match = _SOURCES_RE.search(response)
    if source_match:
        body = response[: source_match.start()].strip()
        sources = _SOURCE_ITEM_RE.findall(source_match.group(1))
    else:
        body = response.strip()
        sources = []

    lines = [f"\n{_RULE}", body]

    if sources:
        lines.append("\n  Sources:")
        for doc_name, chunk_id in sources:
            lines.append(f"    ▸ {doc_name.strip():<46} {chunk_id.strip()}")

    # Compact routing footer — truncate long reasons to keep it one line.
    reason_display = reason[:70] + "…" if len(reason) > 70 else reason
    lines.append(f'\n  ↳ {agent}{tool_calls_label}  ·  "{reason_display}"')
    lines.append(_RULE)

    return "\n".join(lines)


async def _invoke_with_interrupt_loop(
    graph,
    initial_state: dict,
    config: dict,
    has_checkpointer: bool,
) -> dict:
    """Invoke the graph, handling any confirmation interrupts in a loop.

    Different LangGraph versions surface interrupts differently:

      1. Older versions raise ``GraphInterrupt`` from ``ainvoke``.
      2. Newer versions (0.2.50+) return a partial result silently and store
         the pending interrupt in the graph's checkpoint state.

    We handle both: catch the exception OR query ``graph.aget_state(config)``
    after each call.  If pending interrupts are present, prompt the user and
    resume with ``Command(resume=answer)`` until the graph runs to completion.

    Without a checkpointer, interrupts cannot be resumed reliably — we
    auto-confirm in that mode.
    """
    try:
        from langgraph.types import Command
    except ImportError:
        # Very old LangGraph; just invoke directly.
        return await graph.ainvoke(initial_state, config=config)

    try:
        from langgraph.errors import GraphInterrupt
    except ImportError:
        GraphInterrupt = None  # type: ignore[assignment, misc]

    input_payload: dict | Command = initial_state

    while True:
        # 1. Drive the graph forward.
        try:
            result = await graph.ainvoke(input_payload, config=config)
            raised_interrupts: list = []
        except Exception as exc:
            if GraphInterrupt is not None and isinstance(exc, GraphInterrupt):
                result = None
                raised_interrupts = list(exc.interrupts)
            else:
                raise

        # 2. Look for any pending interrupts the graph is paused on.
        pending = list(raised_interrupts)
        if has_checkpointer and not pending:
            state_snapshot = await graph.aget_state(config)
            for task in getattr(state_snapshot, "tasks", []) or []:
                pending.extend(getattr(task, "interrupts", []) or [])

        if not pending:
            return result  # Graph finished normally.

        # 3. Prompt the user for each pending confirmation.
        if not has_checkpointer:
            print(
                "\n[warning] Action tools require a checkpointer for "
                "confirmation prompts. Auto-confirming.",
                file=sys.stderr,
            )
            input_payload = Command(resume="yes")
            continue

        answer = "yes"
        for interrupt_info in pending:
            payload = getattr(interrupt_info, "value", interrupt_info)
            if isinstance(payload, dict):
                question = payload.get(
                    "question",
                    f"Proceed with {payload.get('tool', 'action')}?",
                )
            else:
                question = str(payload)
            print("\n⚠  Action requires confirmation:")
            print(f"   {question}")
            answer = (await asyncio.to_thread(input, "   Confirm? (yes/no): ")).strip() or "no"

        input_payload = Command(resume=answer)


async def _run_loop(
    *, thread_id: str, cp_path: str | None
) -> int:
    """The interactive REPL.  Async because some subagent nodes are async."""
    print(f"Vanguard Assistant — thread {thread_id[:8]}. Ctrl-D / Ctrl-C to exit.")
    if cp_path:
        print(f"  checkpoint: {cp_path}")
        print("  confirmation prompts: enabled (action tools will ask before sending)")
    else:
        print("  checkpoint: disabled  (use --checkpoint-path to enable multi-turn memory)")

    has_checkpointer = cp_path is not None

    async with _build_checkpointer(cp_path) as checkpointer:
        graph = build_graph(checkpointer=checkpointer)
        config = {"configurable": {"thread_id": thread_id}}

        while True:
            try:
                user = await asyncio.to_thread(input, "\nyou> ")
            except (EOFError, KeyboardInterrupt):
                print("\nbye.")
                return 0
            user = user.strip()
            if not user:
                continue

            state = fresh_turn_state(user, thread_id=thread_id)
            try:
                result = await _invoke_with_interrupt_loop(
                    graph, state, config, has_checkpointer=has_checkpointer
                )
            except Exception as exc:  # noqa: BLE001 — surface any error in the CLI
                print(f"\n[error] {type(exc).__name__}: {exc}", file=sys.stderr)
                continue

            print(_format_response(result))


def main(argv: list[str] | None = None) -> int:
    # Suppress tqdm progress bars (sentence-transformers model loading, etc.)
    # Must be set before sentence_transformers is imported (it's lazy here).
    import os
    os.environ.setdefault("TQDM_DISABLE", "1")

    _configure_logging()
    args = _parse_args(argv)
    thread_id = args.thread or str(uuid.uuid4())
    cp_path = None if args.no_checkpoint else args.checkpoint_path
    try:
        return asyncio.run(_run_loop(thread_id=thread_id, cp_path=cp_path))
    except KeyboardInterrupt:
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
