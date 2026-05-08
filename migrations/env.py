"""Alembic environment — reads the DSN from Settings, supports async drivers.

Async-driver detection: SQLAlchemy's ``create_async_engine`` accepts
``sqlite+aiosqlite://`` and ``postgresql+asyncpg://``; both work with
Alembic via the ``run_async`` migration helper.  Plain sync DSNs are
also fine — fall back to ``create_engine`` for those.

The ``target_metadata`` is the API's ``Base.metadata``.  It does NOT
include LangGraph's checkpoint tables — those are owned by the saver
and shouldn't be in this migration graph.
"""

from __future__ import annotations

import asyncio
import sys
from logging.config import fileConfig
from pathlib import Path

from alembic import context
from sqlalchemy import engine_from_config, pool
from sqlalchemy.engine import Connection
from sqlalchemy.ext.asyncio import async_engine_from_config

# Make ``apps`` and ``vanguard_agents`` importable when alembic is invoked
# from the repo root.
_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from apps.api.db import Base  # noqa: E402
from vanguard_agents.settings import get_settings  # noqa: E402


config = context.config

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# Inject the app DB DSN from settings into the alembic config so
# offline mode (--sql) and online mode both pick it up.
config.set_main_option("sqlalchemy.url", get_settings().app_db_dsn)

target_metadata = Base.metadata


def _is_async_dsn(dsn: str) -> bool:
    return "+asyncpg" in dsn or "+aiosqlite" in dsn


def run_migrations_offline() -> None:
    """Render SQL without a live connection (for ``alembic upgrade --sql``)."""
    context.configure(
        url=config.get_main_option("sqlalchemy.url"),
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        compare_type=True,
    )
    with context.begin_transaction():
        context.run_migrations()


def _do_migrate(connection: Connection) -> None:
    context.configure(
        connection=connection,
        target_metadata=target_metadata,
        compare_type=True,
    )
    with context.begin_transaction():
        context.run_migrations()


async def _run_async_migrations() -> None:
    cfg_section = config.get_section(config.config_ini_section) or {}
    cfg_section["sqlalchemy.url"] = config.get_main_option("sqlalchemy.url")
    engine = async_engine_from_config(
        cfg_section,
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )
    async with engine.connect() as connection:
        await connection.run_sync(_do_migrate)
    await engine.dispose()


def run_migrations_online() -> None:
    dsn = config.get_main_option("sqlalchemy.url") or ""
    if _is_async_dsn(dsn):
        asyncio.run(_run_async_migrations())
        return

    # Sync driver — fall back to the standard engine_from_config path.
    cfg_section = config.get_section(config.config_ini_section) or {}
    cfg_section["sqlalchemy.url"] = dsn
    connectable = engine_from_config(
        cfg_section,
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )
    with connectable.connect() as connection:
        _do_migrate(connection)


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
