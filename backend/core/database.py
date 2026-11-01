# backend/core/database.py
"""Database configuration and session management."""
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker

from backend.core.config import get_settings

_engine = None
_async_session_maker = None


def get_engine():
    """Get or create the SQLAlchemy engine."""
    global _engine
    if _engine is None:
        settings = get_settings()
        
        # Используем ASYNC_DATABASE_URL если есть, иначе DATABASE_URL
        database_url = getattr(settings, 'ASYNC_DATABASE_URL', None)
        if not database_url:
            database_url = getattr(settings, 'DATABASE_URL', None)
        if not database_url:
            database_url = "sqlite+aiosqlite:///./rung.db"
        
        # Конвертируем postgresql:// в postgresql+asyncpg://
        if database_url.startswith("postgresql://"):
            database_url = database_url.replace(
                "postgresql://", "postgresql+asyncpg://", 1
            )
        
        _engine = create_async_engine(
            database_url,
            echo=settings.DEBUG,
            pool_pre_ping=True,
        )
    return _engine


def get_db_session():
    """Get an async database session."""
    global _async_session_maker
    if _async_session_maker is None:
        engine = get_engine()
        _async_session_maker = async_sessionmaker(
            engine,
            class_=AsyncSession,
            expire_on_commit=False,
        )
    return _async_session_maker()


async def init_db():
    """Initialize database tables."""
    from backend.db.models import Base
    engine = get_engine()
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


async def close_db():
    """Close database connections."""
    global _engine, _async_session_maker
    if _engine:
        await _engine.dispose()
        _engine = None
        _async_session_maker = None