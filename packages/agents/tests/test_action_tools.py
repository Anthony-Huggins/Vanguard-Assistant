"""Tests for the SendGrid + Jira integrations in the action tools.

We mock ``httpx.AsyncClient`` so these tests don't hit the real APIs.
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from apps.mcp_server.tools.action import (
    _create_via_jira,
    _send_via_sendgrid,
)


def _mock_async_client(response: MagicMock):
    """Return an object suitable to patch ``httpx.AsyncClient`` with."""
    instance = MagicMock()
    instance.post = AsyncMock(return_value=response)
    instance.__aenter__ = AsyncMock(return_value=instance)
    instance.__aexit__ = AsyncMock(return_value=False)
    factory = MagicMock(return_value=instance)
    return factory, instance


# ---------------------------------------------------------------------------
# SendGrid
# ---------------------------------------------------------------------------

async def test_sendgrid_returns_none_without_creds(monkeypatch):
    monkeypatch.delenv("SENDGRID_API_KEY", raising=False)
    monkeypatch.delenv("SENDGRID_FROM_EMAIL", raising=False)
    result = await _send_via_sendgrid("a@b.com", "hi", "body")
    assert result is None


async def test_sendgrid_calls_api_when_creds_set(monkeypatch):
    monkeypatch.setenv("SENDGRID_API_KEY", "SG.fake")
    monkeypatch.setenv("SENDGRID_FROM_EMAIL", "alerts@example.com")

    fake_response = MagicMock()
    fake_response.status_code = 202
    fake_response.headers = {"X-Message-Id": "msg-123"}

    factory, instance = _mock_async_client(fake_response)
    with patch("apps.mcp_server.tools.action.httpx.AsyncClient", factory):
        result = await _send_via_sendgrid(
            "user@example.com", "Test subject", "Hello there."
        )

    assert result is not None
    assert result["status"] == "sent"
    assert result["provider"] == "sendgrid"
    assert result["message_id"] == "msg-123"
    assert result["from"] == "alerts@example.com"

    instance.post.assert_awaited_once()
    args, kwargs = instance.post.call_args
    assert args[0] == "https://api.sendgrid.com/v3/mail/send"
    assert kwargs["headers"]["Authorization"] == "Bearer SG.fake"
    payload = kwargs["json"]
    assert payload["from"]["email"] == "alerts@example.com"
    assert payload["personalizations"][0]["to"][0]["email"] == "user@example.com"
    assert payload["subject"] == "Test subject"


async def test_sendgrid_raises_on_api_error(monkeypatch):
    monkeypatch.setenv("SENDGRID_API_KEY", "SG.fake")
    monkeypatch.setenv("SENDGRID_FROM_EMAIL", "alerts@example.com")

    fake_response = MagicMock()
    fake_response.status_code = 401
    fake_response.text = '{"errors":[{"message":"invalid api key"}]}'
    fake_response.headers = {}

    factory, _ = _mock_async_client(fake_response)
    with patch("apps.mcp_server.tools.action.httpx.AsyncClient", factory):
        with pytest.raises(RuntimeError, match="SendGrid 401"):
            await _send_via_sendgrid("user@example.com", "x", "y")


# ---------------------------------------------------------------------------
# Jira
# ---------------------------------------------------------------------------

async def test_jira_returns_none_without_creds(monkeypatch):
    for var in ("JIRA_URL", "JIRA_USER_EMAIL", "JIRA_API_TOKEN"):
        monkeypatch.delenv(var, raising=False)
    result = await _create_via_jira("summary", "description")
    assert result is None


async def test_jira_calls_api_and_returns_issue_url(monkeypatch):
    monkeypatch.setenv("JIRA_URL", "https://example.atlassian.net/")
    monkeypatch.setenv("JIRA_USER_EMAIL", "ops@example.com")
    monkeypatch.setenv("JIRA_API_TOKEN", "tok-xyz")
    monkeypatch.setenv("JIRA_PROJECT_KEY", "VG")

    fake_response = MagicMock()
    fake_response.status_code = 201
    fake_response.json = MagicMock(return_value={"id": "10001", "key": "VG-42"})

    factory, instance = _mock_async_client(fake_response)
    with patch("apps.mcp_server.tools.action.httpx.AsyncClient", factory):
        result = await _create_via_jira(
            "Wire transfer failed", "User Bob can't complete a $10k wire."
        )

    assert result is not None
    assert result["provider"] == "jira"
    assert result["id"] == "VG-42"
    assert result["url"] == "https://example.atlassian.net/browse/VG-42"

    instance.post.assert_awaited_once()
    args, kwargs = instance.post.call_args
    assert args[0] == "https://example.atlassian.net/rest/api/3/issue"
    assert kwargs["auth"] == ("ops@example.com", "tok-xyz")
    payload = kwargs["json"]
    assert payload["fields"]["project"]["key"] == "VG"
    assert payload["fields"]["summary"] == "Wire transfer failed"
    # ADF body shape
    assert payload["fields"]["description"]["type"] == "doc"
    assert (
        payload["fields"]["description"]["content"][0]["content"][0]["text"]
        == "User Bob can't complete a $10k wire."
    )


async def test_jira_raises_on_api_error(monkeypatch):
    monkeypatch.setenv("JIRA_URL", "https://example.atlassian.net")
    monkeypatch.setenv("JIRA_USER_EMAIL", "ops@example.com")
    monkeypatch.setenv("JIRA_API_TOKEN", "tok-xyz")

    fake_response = MagicMock()
    fake_response.status_code = 400
    fake_response.text = '{"errorMessages":["Project VG does not exist"]}'

    factory, _ = _mock_async_client(fake_response)
    with patch("apps.mcp_server.tools.action.httpx.AsyncClient", factory):
        with pytest.raises(RuntimeError, match="Jira 400"):
            await _create_via_jira("x", "y")
