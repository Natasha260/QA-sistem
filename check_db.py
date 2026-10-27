import asyncio
import logging
from backend.db.session import async_engine
from backend.db.models import Base
from sqlalchemy import text

# Настройка простого лога, чтобы видеть, что происходит
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

async def main():
    logger.info("🚀 Запуск проверки БД...")
    
    # 1. Сначала запускаем логику создания таблиц (как в lifespan)
    # Используем begin(), чтобы транзакции работали корректно
    async with async_engine.begin() as conn:
        logger.info("🗄️ Создаем таблицы (create_all)...")
        await conn.run_sync(Base.metadata.create_all)
        
        logger.info("📜 Применяем дополнительные миграции (индексы)...")
        # Здесь дублируем логику из _POST_CREATE_STATEMENTS из main.py
        statements = [
            "CREATE INDEX IF NOT EXISTS ix_cache_eval_mode ON translation_cache (eval_mode)"
        ]
        for stmt in statements:
            await conn.execute(text(stmt))
        await conn.commit()
        logger.info("✅ Миграции применены.")

    # 2. Теперь проверяем, что таблицы реально появились
    async with async_engine.connect() as conn:
        result = await conn.execute(text("SELECT name FROM sqlite_master WHERE type='table'"))
        tables = sorted([row for row in result])
        
        logger.info("📋 Список таблиц в БД:")
        if not tables:
            logger.error("❌ ОШИБКА: В базе данных нет ни одной таблицы!")
            return
            
        for table in tables:
            logger.info(f"   - {table}")
            
        logger.info(f"🎉 Успешно! Найдено таблиц: {len(tables)}")

    # 3. Корректно закрываем движок
    await async_engine.dispose()
    logger.info("🛑 Соединение с БД закрыто.")

if __name__ == "__main__":
    asyncio.run(main())
