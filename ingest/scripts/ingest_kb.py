"""Ingest the Vanguard knowledge base into the configured vector store.

Usage (from the project root):

    ./.venv/Scripts/python.exe ingest/scripts/ingest_kb.py
    ./.venv/Scripts/python.exe ingest/scripts/ingest_kb.py --reset
    ./.venv/Scripts/python.exe ingest/scripts/ingest_kb.py --data-dir other/folder

The default ``--data-dir`` is ``ingest/data/`` relative to the project root.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from vanguard_agents.ingest import ingest

DEFAULT_DATA_DIR = Path(__file__).resolve().parents[1] / "data"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help=f"Folder containing PDF/MD/HTML docs (default: {DEFAULT_DATA_DIR}).",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Wipe the vector store collection before upserting.",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable debug logging."
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    stats = ingest(args.data_dir, reset=args.reset)
    print(
        f"\nIngest done — files: {stats.files_loaded}, "
        f"raw docs: {stats.documents_loaded}, "
        f"chunks: {stats.chunks_created}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
