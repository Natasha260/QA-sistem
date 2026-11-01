# backend/db/session.py
from typing import AsyncGenerator

from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from backend.core.config import get_settings

settings = get_settings()

# ---------------------------------------------------------------------------
# Async engine
# ---------------------------------------------------------------------------

async_engine = create_async_engine(
    settings.ASYNC_DATABASE_URL,
    echo=settings.DEBUG,
    pool_size=10,
    max_overflow=20,
    pool_pre_ping=True,
)

AsyncSessionLocal = async_sessionmaker(
    async_engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autoflush=False,
)


async def get_async_session() -> AsyncGenerator[AsyncSession, None]:
    """
    FastAPI dependency that yields an async session.

    This is the single source of truth for async session management
    across the application. Routers and helpers import it from here
    (directly or via ``backend.api.deps``).
    """
    async with AsyncSessionLocal() as session:
        yield session


# Alias with a shorter, familiar name. Both refer to the same callable,
# so ``Depends(get_db)`` and ``Depends(get_async_session)`` are equivalent.
get_db = get_async_session