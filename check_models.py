import asyncio
from sqlalchemy.ext.asyncio import create_async_engine
from backend.db.models import Base

async def main():
    # Используем in-memory SQLite для проверки схемы
    engine = create_async_engine("sqlite+aiosqlite:///:memory:", echo=False, future=True)

    async with engine.begin() as conn:
        # Создаем все таблицы, описанные в Base.metadata
        await conn.run_sync(Base.metadata.create_all)

    # Получаем и сортируем имена таблиц
    tables = sorted(Base.metadata.tables.keys())
    print(tables)

    await engine.dispose()

if __name__ == "__main__":
    asyncio.run(main())
