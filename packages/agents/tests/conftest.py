"""Shared pytest fixtures + path setup."""

from __future__ import annotations

import sys
from pathlib import Path

# Tests in this folder need to import from ``apps.mcp_server.*`` (the action
# tools) which live outside the ``vanguard_agents`` package tree. Putting the
# project root on sys.path is the simplest way to make that work without
# adding an unnecessary pyproject for ``apps/``.
_REPO_ROOT = Path(__file__).resolve().parents[3]  # vanguard-assistant/
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
