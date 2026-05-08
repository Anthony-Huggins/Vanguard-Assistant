"""Tests for the PII masking utilities in guardrails.py."""

from __future__ import annotations

import logging

import pytest

from vanguard_agents.guardrails import PIIRedactingFilter, mask_pii


# ---------------------------------------------------------------------------
# mask_pii — unit tests for each pattern
# ---------------------------------------------------------------------------

class TestMaskPii:
    def test_ssn_masked(self):
        assert mask_pii("SSN: 123-45-6789") == "SSN: [SSN]"

    def test_ssn_not_partial(self):
        # Partial patterns should NOT be masked (too many false positives).
        text = "account 12345"
        assert mask_pii(text) == text

    def test_email_masked(self):
        assert mask_pii("Contact alice@example.com today") == "Contact [EMAIL] today"

    def test_email_subdomain_masked(self):
        assert "[EMAIL]" in mask_pii("ops@vanguard.financial.com")

    def test_phone_masked(self):
        assert mask_pii("call 555-867-5309 now") == "call [PHONE] now"

    def test_phone_with_area_code_masked(self):
        result = mask_pii("(800) 555-1234")
        assert "[PHONE]" in result

    def test_multiple_pii_in_one_string(self):
        text = "SSN 123-45-6789, email user@foo.com, phone 555-867-5309"
        result = mask_pii(text)
        assert "123-45-6789" not in result
        assert "user@foo.com" not in result
        assert "555-867-5309" not in result
        assert "[SSN]" in result
        assert "[EMAIL]" in result
        assert "[PHONE]" in result

    def test_clean_text_unchanged(self):
        text = "Vanguard total expense ratio is 0.03%."
        assert mask_pii(text) == text

    def test_empty_string(self):
        assert mask_pii("") == ""


# ---------------------------------------------------------------------------
# PIIRedactingFilter — logging integration
# ---------------------------------------------------------------------------

class TestPIIRedactingFilter:
    def _make_record(self, msg: str, args=()) -> logging.LogRecord:
        record = logging.LogRecord(
            name="test", level=logging.INFO,
            pathname="", lineno=0,
            msg=msg, args=args, exc_info=None,
        )
        return record

    def test_filter_redacts_msg(self):
        f = PIIRedactingFilter()
        record = self._make_record("user SSN: 123-45-6789")
        f.filter(record)
        assert "123-45-6789" not in str(record.msg)
        assert "[SSN]" in record.msg

    def test_filter_redacts_positional_args(self):
        f = PIIRedactingFilter()
        record = self._make_record("sending to %s", args=("alice@example.com",))
        f.filter(record)
        assert "alice@example.com" not in str(record.args)

    def test_filter_redacts_dict_args(self):
        # Pass args as a bare dict directly on the record (not via constructor,
        # which would treat it as a mapping-as-single-arg).
        f = PIIRedactingFilter()
        record = self._make_record("info")
        record.args = {"email": "bob@test.org"}
        f.filter(record)
        assert "bob@test.org" not in str(record.args)

    def test_filter_returns_true(self):
        """Filter must return True so the log record is still emitted."""
        f = PIIRedactingFilter()
        record = self._make_record("hello world")
        assert f.filter(record) is True

    def test_filter_no_args(self):
        """Records with no args don't crash."""
        f = PIIRedactingFilter()
        record = self._make_record("plain message")
        record.args = None
        assert f.filter(record) is True
