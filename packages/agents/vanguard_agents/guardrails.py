"""PII-masking utilities for safe logging.

Provides:
  mask_pii(text)        — redacts SSN, phone, and email patterns from a string.
  PIIRedactingFilter    — logging.Filter subclass; attach to the root logger
                          at application startup so ALL log lines are scrubbed
                          before they are written anywhere.

Usage::

    import logging
    from vanguard_agents.guardrails import PIIRedactingFilter

    logging.getLogger().addFilter(PIIRedactingFilter())

This is applied once in ``cli.py`` (and later ``api/main.py``).  Because it
hooks the root logger, every module's ``log = logging.getLogger(__name__)``
call automatically inherits the filter.
"""

from __future__ import annotations

import logging
import re

# ---------------------------------------------------------------------------
# Patterns — intentionally broad so ambiguous strings are masked.
# ---------------------------------------------------------------------------

_SSN = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")

# Covers: 555-867-5309, (555) 867-5309, 555.867.5309, +1 555 867 5309
_PHONE = re.compile(
    r"\b"
    r"(?:\+?1[-.\s]?)?"          # optional country code
    r"(?:\(?\d{3}\)?[-.\s]?)?"   # optional area code
    r"\d{3}[-.\s]?\d{4}"         # 7-digit local number
    r"\b"
)

_EMAIL = re.compile(
    r"\b[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}\b"
)

_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (_SSN,   "[SSN]"),
    (_PHONE, "[PHONE]"),
    (_EMAIL, "[EMAIL]"),
]


def mask_pii(text: str) -> str:
    """Replace any SSN, phone number, or email address in *text* with a
    safe placeholder.  Returns the redacted string."""
    for pattern, placeholder in _PATTERNS:
        text = pattern.sub(placeholder, text)
    return text


class PIIRedactingFilter(logging.Filter):
    """Logging filter that masks PII from every ``LogRecord`` that passes
    through it.

    Handles both positional-args tuples (``log.info("user %s", email)``) and
    keyword-args dicts so that lazy-formatted messages are also redacted.
    """

    def filter(self, record: logging.LogRecord) -> bool:  # noqa: A003
        record.msg = mask_pii(str(record.msg))
        if record.args:
            if isinstance(record.args, dict):
                record.args = {
                    k: mask_pii(str(v)) if isinstance(v, str) else v
                    for k, v in record.args.items()
                }
            elif isinstance(record.args, tuple):
                record.args = tuple(
                    mask_pii(str(a)) if isinstance(a, str) else a
                    for a in record.args
                )
        return True
