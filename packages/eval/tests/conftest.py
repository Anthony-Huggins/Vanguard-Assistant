"""Pytest configuration for packages/eval/tests.

Adds packages/eval/ and vanguard-assistant/ to sys.path so that
ragas_runner and vanguard_agents resolve regardless of where pytest is
invoked from.
"""

from __future__ import annotations

import sys
from pathlib import Path

_EVAL_ROOT = Path(__file__).resolve().parents[1]   # packages/eval/
_REPO_ROOT = _EVAL_ROOT.parents[1]                 # vanguard-assistant/
for _p in (_REPO_ROOT, _EVAL_ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
