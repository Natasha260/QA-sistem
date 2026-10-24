# backend/api/main.py
"""
Точка входа FastAPI.

Здесь создаётся приложение, подключаются middleware и роутеры.
Uvicorn запускает объект app по пути backend.api.main:app.

На этом этапе подключён только роутер health. Остальные роутеры
(auth, translate, admin, hitl, eval) появятся на этапах 4, 9, 10, 13.
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.api.routes import health
from backend.core.config import get_settings
from backend.utils.logging import setup_logging

# Настраиваем логирование при импорте модуля. Это нужно для случаев,
# когда приложение запускается через `uvicorn backend.api.main:app`
# напрямую (в Docker), а не через `run.py`.
setup_logging()

logger = logging.getLogger(__name__)
settings = get_settings()


# ============================================================================
# Lifespan — код, выполняющийся при старте и остановке приложения
# ============================================================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Управляет ресурсами приложения.

    Всё, что до yield — выполняется при старте.
    Всё, что после yield — при остановке.
    """
    logger.info("=" * 60)
    logger.info("%s starting...", settings.PROJECT_NAME)
    logger.info("Debug mode: %s", settings.DEBUG)
    logger.info("Translation engine: %s", settings.TRANSLATION_ENGINE)
    logger.info("=" * 60)

    # Здесь позже будет:
    # - создание таблиц БД (Base.metadata.create_all) — этап 3
    # - инициализация Qdrant-коллекции — этап 7
    # - прогрев эмбеддингов — этап 7

    yield

    logger.info("=" * 60)
    logger.info("%s stopped", settings.PROJECT_NAME)
    logger.info("=" * 60)


# ============================================================================
# Приложение
# ============================================================================
app = FastAPI(
    title=settings.PROJECT_NAME,
    description="Мультиагентная платформа перевода с RAG",
    version="0.1.0",
    debug=settings.DEBUG,
    lifespan=lifespan,
)


# ============================================================================
# Middleware
# ============================================================================
# CORS — разрешает браузеру обращаться к API с другого домена.
# В DEBUG-режиме разрешаем всё, в проде — только конкретные origin.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"] if settings.DEBUG else [],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================================
# Роутеры
# ============================================================================
app.include_router(health.router)


# ============================================================================
# Базовые эндпоинты
# ============================================================================
@app.get("/", tags=["root"])
async def root():
    """Корневой эндпоинт — простая проверка, что приложение работает."""
    return {
        "message": f"{settings.PROJECT_NAME} is running",
        "docs": "/docs",
        "healthz": "/healthz",
    }