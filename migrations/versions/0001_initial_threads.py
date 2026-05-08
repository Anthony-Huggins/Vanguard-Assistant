"""Initial threads table — M9 metadata schema.

Revision ID: 0001_initial_threads
Revises:
Create Date: 2026-05-07
"""

from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op


revision: str = "0001_initial_threads"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "threads",
        sa.Column("thread_id", sa.String(length=128), primary_key=True),
        sa.Column("display_name", sa.String(length=200), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.Column("last_message_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.create_index(
        "ix_threads_last_message_at",
        "threads",
        ["last_message_at"],
    )


def downgrade() -> None:
    op.drop_index("ix_threads_last_message_at", table_name="threads")
    op.drop_table("threads")
