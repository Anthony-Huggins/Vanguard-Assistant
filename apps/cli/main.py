"""Thin shim so `python apps/cli/main.py` works from the repo root.

The real implementation lives in ``vanguard_agents.cli``.
"""

from vanguard_agents.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
