from __future__ import annotations

from sqlalchemy import create_engine
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from app.core.config import settings

engine = create_async_engine(settings.DATABASE_URL, echo=settings.DEBUG, pool_pre_ping=True)
async_session = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

# Sync engine for background workers (avoids event loop conflicts)
_sync_url = settings.DATABASE_URL.replace("postgresql+asyncpg://", "postgresql+psycopg2://")
sync_engine = create_engine(_sync_url, echo=settings.DEBUG, pool_pre_ping=True)
SyncSession = sessionmaker(sync_engine, class_=Session, expire_on_commit=False)


class Base(DeclarativeBase):
    pass


async def get_db() -> AsyncSession:
    async with async_session() as session:
        yield session
