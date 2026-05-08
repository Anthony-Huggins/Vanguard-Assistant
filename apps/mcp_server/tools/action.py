"""Action tools — send_email (SendGrid), create_ticket (Jira), log_action.

Each side-effecting tool tries the real provider first. If the required
credentials aren't set in the environment, the tool falls back to a local
stub (JSONL append + console print) so dev and tests keep working.

Required env vars (per provider):

  SendGrid:  SENDGRID_API_KEY, SENDGRID_FROM_EMAIL
  Jira:      JIRA_URL, JIRA_USER_EMAIL, JIRA_API_TOKEN, JIRA_PROJECT_KEY (default "VG")

A failed real-API call is surfaced as a tool error so the agent loop can
react — we deliberately do NOT silently fall back to the stub on failure,
because that would mask a misconfigured production integration.
"""

from __future__ import annotations

import json
import logging
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import httpx

log = logging.getLogger(__name__)

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
EMAILS_FILE = DATA_DIR / "sent_emails.jsonl"
TICKETS_FILE = DATA_DIR / "tickets.jsonl"
AUDIT_FILE = DATA_DIR / "audit.jsonl"

SENDGRID_URL = "https://api.sendgrid.com/v3/mail/send"
HTTP_TIMEOUT = 15.0


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _append_jsonl(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


# ---------------------------------------------------------------------------
# SendGrid integration
# ---------------------------------------------------------------------------

async def _send_via_sendgrid(
    recipient: str, subject: str, body: str
) -> dict | None:
    """Send an email via SendGrid.

    Returns:
        Provider response dict on success, ``None`` if credentials are not
        set (caller should fall back to the stub).

    Raises:
        RuntimeError: when credentials are set but SendGrid rejects the
            request (the agent loop surfaces this to the LLM).
    """
    api_key = os.environ.get("SENDGRID_API_KEY")
    from_email = os.environ.get("SENDGRID_FROM_EMAIL")
    if not api_key or not from_email:
        return None

    payload = {
        "personalizations": [{"to": [{"email": recipient}]}],
        "from": {"email": from_email},
        "subject": subject,
        "content": [{"type": "text/plain", "value": body}],
    }

    async with httpx.AsyncClient(timeout=HTTP_TIMEOUT) as client:
        resp = await client.post(
            SENDGRID_URL,
            json=payload,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
        )

    if resp.status_code >= 300:
        # SendGrid returns useful JSON error payloads — include them.
        raise RuntimeError(
            f"SendGrid {resp.status_code}: {resp.text[:500]}"
        )

    return {
        "status": "sent",
        "provider": "sendgrid",
        "message_id": resp.headers.get("X-Message-Id"),
        "from": from_email,
    }


# ---------------------------------------------------------------------------
# Jira integration
# ---------------------------------------------------------------------------

def _adf_paragraph(text: str) -> dict:
    """Render plain text as Atlassian Document Format (required by Jira REST v3)."""
    return {
        "type": "doc",
        "version": 1,
        "content": [
            {
                "type": "paragraph",
                "content": [{"type": "text", "text": text}],
            }
        ],
    }


async def _create_via_jira(summary: str, description: str) -> dict | None:
    """Create a Jira issue via the REST API v3.

    Returns:
        Provider response dict on success, ``None`` if credentials are not set.

    Raises:
        RuntimeError: when credentials are set but Jira rejects the request.
    """
    base_url = os.environ.get("JIRA_URL")
    email = os.environ.get("JIRA_USER_EMAIL")
    token = os.environ.get("JIRA_API_TOKEN")
    project_key = os.environ.get("JIRA_PROJECT_KEY", "VG")

    if not base_url or not email or not token:
        return None

    payload = {
        "fields": {
            "project": {"key": project_key},
            "summary": summary,
            "description": _adf_paragraph(description),
            "issuetype": {"name": "Task"},
        }
    }

    url = f"{base_url.rstrip('/')}/rest/api/3/issue"

    async with httpx.AsyncClient(timeout=HTTP_TIMEOUT) as client:
        resp = await client.post(
            url,
            json=payload,
            auth=(email, token),
            headers={"Content-Type": "application/json"},
        )

    if resp.status_code >= 300:
        raise RuntimeError(
            f"Jira {resp.status_code}: {resp.text[:500]}"
        )

    data = resp.json()
    issue_key = data.get("key", "<unknown>")
    return {
        "status": "created",
        "provider": "jira",
        "id": issue_key,
        "url": f"{base_url.rstrip('/')}/browse/{issue_key}",
    }


# ---------------------------------------------------------------------------
# Tool registration
# ---------------------------------------------------------------------------

def register(mcp) -> None:
    @mcp.tool()
    async def send_email(recipient: str, subject: str, body: str) -> str:
        """Send an email. Uses SendGrid when credentials are configured;
        otherwise writes to a local JSONL stub.

        Args:
            recipient: Email address of the recipient.
            subject:   Email subject line.
            body:      Plain-text email body.

        Returns:
            JSON string ``{status, provider, recipient, subject, sent_at, ...}``.
        """
        real = await _send_via_sendgrid(recipient, subject, body)

        if real is not None:
            record = {
                **real,
                "recipient": recipient,
                "subject": subject,
                "body_preview": body[:200],
                "sent_at": _utc_now(),
            }
            _append_jsonl(EMAILS_FILE, record)
            log.info("send_email via sendgrid -> %s: %r", recipient, subject)
            return json.dumps(record)

        # Stub fallback
        record = {
            "status": "sent (stub)",
            "provider": "stub",
            "recipient": recipient,
            "subject": subject,
            "body_preview": body[:200],
            "sent_at": _utc_now(),
        }
        _append_jsonl(EMAILS_FILE, record)
        log.warning(
            "send_email used stub (SendGrid creds not set: "
            "SENDGRID_API_KEY / SENDGRID_FROM_EMAIL)"
        )
        print(
            f"\n[STUB EMAIL]\n  to: {recipient}\n  subject: {subject}\n  body:\n{body}\n"
        )
        return json.dumps(record)

    @mcp.tool()
    async def create_ticket(summary: str, description: str) -> str:
        """Create a Jira ticket. Uses the Jira REST API v3 when credentials
        are configured; otherwise writes to a local JSONL stub.

        Args:
            summary:     One-line ticket summary.
            description: Longer ticket description.

        Returns:
            JSON string with the created ticket including its id and url.
        """
        real = await _create_via_jira(summary, description)

        if real is not None:
            record = {
                **real,
                "summary": summary,
                "description": description,
                "created_at": _utc_now(),
            }
            _append_jsonl(TICKETS_FILE, record)
            log.info("create_ticket via jira -> %s: %r", record["id"], summary)
            return json.dumps(record)

        # Stub fallback
        ticket = {
            "id": f"VG-{int(time.time())}",
            "provider": "stub",
            "summary": summary,
            "description": description,
            "status": "open",
            "created_at": _utc_now(),
        }
        _append_jsonl(TICKETS_FILE, ticket)
        log.warning(
            "create_ticket used stub (Jira creds not set: "
            "JIRA_URL / JIRA_USER_EMAIL / JIRA_API_TOKEN)"
        )
        return json.dumps(ticket)

    @mcp.tool()
    async def log_action(action: str, details: str = "{}") -> str:
        """Append an audit log entry.

        Args:
            action:  Short label for the action being logged.
            details: JSON string with structured details (default ``{}``).

        Returns:
            JSON string with the recorded audit entry.
        """
        try:
            parsed_details = json.loads(details) if details else {}
        except json.JSONDecodeError:
            parsed_details = {"raw": details}
        entry = {
            "timestamp": _utc_now(),
            "action": action,
            "details": parsed_details,
        }
        _append_jsonl(AUDIT_FILE, entry)
        log.info("log_action: %s", action)
        return json.dumps(entry)
