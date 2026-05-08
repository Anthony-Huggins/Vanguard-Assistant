"""Smoke test for SendGrid and Jira real-API integrations.

Run this script BEFORE testing through the full agent stack to confirm your
credentials are valid.  It calls _send_via_sendgrid and _create_via_jira
directly — no MCP server, no LangGraph, no OpenAI.

Usage:
    .\.venv\Scripts\python.exe scripts/smoke_test_integrations.py

The script reads credentials from .env (same as the rest of the project).
Edit SMOKE_RECIPIENT below to choose where the test email lands.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

# ── Resolve project root so we can load .env and import the tool helpers ──────
ROOT = Path(__file__).resolve().parent.parent   # scripts/ -> project root
sys.path.insert(0, str(ROOT))

# Load .env before importing the action module (it reads from os.environ)
from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

from apps.mcp_server.tools.action import _create_via_jira, _send_via_sendgrid

# ── Change this to an email address you can check ─────────────────────────────
SMOKE_RECIPIENT = "test@example.com"   # <── UPDATE THIS


def _ok(label: str) -> None:
    print(f"  ✅  {label}")


def _fail(label: str, err: str) -> None:
    print(f"  ❌  {label}")
    print(f"      {err}")


async def test_sendgrid() -> bool:
    import os
    key = os.environ.get("SENDGRID_API_KEY", "")
    from_email = os.environ.get("SENDGRID_FROM_EMAIL", "")

    print("\n── SendGrid ─────────────────────────────────────────")
    if not key or not from_email:
        print("  ⚠️   SENDGRID_API_KEY or SENDGRID_FROM_EMAIL not set — skipping")
        return False

    print(f"  Key   : {key[:8]}...{key[-4:]}")
    print(f"  From  : {from_email}")
    print(f"  To    : {SMOKE_RECIPIENT}")

    try:
        result = await _send_via_sendgrid(
            recipient=SMOKE_RECIPIENT,
            subject="[Vanguard Assistant smoke test]",
            body="If you see this, SendGrid is wired up correctly. 🎉",
        )
        _ok(f"Email sent  |  message_id={result.get('message_id')}")
        return True
    except RuntimeError as exc:
        _fail("SendGrid call failed", str(exc))
        _diagnose_sendgrid(str(exc))
        return False
    except Exception as exc:
        _fail("Unexpected error", f"{type(exc).__name__}: {exc}")
        return False


def _diagnose_sendgrid(msg: str) -> None:
    if "401" in msg:
        print("      → The API key looks invalid. Check SENDGRID_API_KEY in .env")
        print("        (it should start with 'SG.' and be ~70 chars long)")
    elif "403" in msg:
        print("      → 403 Forbidden — the 'from' address may not be a verified sender.")
        print("        Verify it at https://app.sendgrid.com/settings/sender_auth")
    elif "400" in msg:
        print("      → Bad request. Check that SMOKE_RECIPIENT is a real email address")
        print("        and that SENDGRID_FROM_EMAIL is a verified sender.")


async def test_jira() -> bool:
    import os
    base_url = os.environ.get("JIRA_URL", "")
    email = os.environ.get("JIRA_USER_EMAIL", "")
    token = os.environ.get("JIRA_API_TOKEN", "")
    project = os.environ.get("JIRA_PROJECT_KEY", "VG")

    print("\n── Jira ─────────────────────────────────────────────")
    if not base_url or not email or not token:
        print("  ⚠️   JIRA_URL, JIRA_USER_EMAIL, or JIRA_API_TOKEN not set — skipping")
        return False

    print(f"  URL     : {base_url}")
    print(f"  Email   : {email}")
    print(f"  Token   : {token[:8]}...{token[-4:]}")
    print(f"  Project : {project}")

    try:
        result = await _create_via_jira(
            summary="[Smoke test] Vanguard Assistant integration check",
            description=(
                "This ticket was created automatically by the smoke test script. "
                "It confirms that the Jira REST API v3 integration is working. "
                "Feel free to delete it."
            ),
        )
        _ok(f"Ticket created  |  id={result.get('id')}  url={result.get('url')}")
        return True
    except RuntimeError as exc:
        _fail("Jira call failed", str(exc))
        _diagnose_jira(str(exc))
        return False
    except Exception as exc:
        _fail("Unexpected error", f"{type(exc).__name__}: {exc}")
        return False


def _diagnose_jira(msg: str) -> None:
    if "401" in msg:
        print("      → Authentication failed. Double-check JIRA_USER_EMAIL and JIRA_API_TOKEN.")
        print("        The token should be an Atlassian API token, not your password.")
        print("        Get one at: https://id.atlassian.com/manage-profile/security/api-tokens")
    elif "400" in msg and ("project" in msg.lower() or "does not exist" in msg.lower()):
        print(f"      → Project key issue. Check JIRA_PROJECT_KEY in .env")
        print("        Make sure the project exists and your account has permission to create issues.")
    elif "400" in msg:
        print("      → Bad request. The full error above has details.")
    elif "404" in msg:
        print("      → Wrong URL. Check JIRA_URL in .env — it should be")
        print("        https://YOUR-DOMAIN.atlassian.net (no trailing slash needed)")


async def main() -> None:
    print("=" * 55)
    print("  Vanguard Assistant — integration smoke test")
    print("=" * 55)

    sg_ok = await test_sendgrid()
    jira_ok = await test_jira()

    print("\n── Summary ──────────────────────────────────────────")
    if sg_ok:
        print("  SendGrid : ✅ working")
    else:
        print("  SendGrid : ❌ skipped or failed (see above)")

    if jira_ok:
        print("  Jira     : ✅ working")
    else:
        print("  Jira     : ❌ skipped or failed (see above)")

    if sg_ok and jira_ok:
        print("\n  All integrations OK. You can test through the CLI now.")
    else:
        print("\n  Fix the issues above before testing through the CLI.")

    print()


if __name__ == "__main__":
    asyncio.run(main())
