"""SQLAlchemy 2 ``DeclarativeBase`` + async engine / session factory."""

from __future__ import annotations

from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase


class Base(DeclarativeBase):
    """Common ORM declarative base for every API-owned table."""


def build_engine(dsn: str) -> AsyncEngine:
    """Construct an AsyncEngine.

    Defaults are conservative: ``pool_pre_ping`` so stale Postgres
    connections fail fast and recycle, ``echo=False`` so query logs
    don't drown out the rest of the app's output.
    """
    return create_async_engine(
        dsn,
        echo=False,
        pool_pre_ping=True,
        future=True,
    )


def build_sessionmaker(engine: AsyncEngine) -> async_sessionmaker[AsyncSession]:
    """Return a session factory bound to the given engine."""
    return async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)
