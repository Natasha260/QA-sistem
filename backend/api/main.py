import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from sqlalchemy import text

# 1. ДОБАВЛЕНО: импорт auth вместе с health
from backend.api.routes import auth, health
from backend.core.config import get_settings
from backend.core.constants import LANGUAGES
from backend.core.exceptions import RUNGException
from backend.db.models import Base
from backend.db.session import async_engine

settings = get_settings()

# Idempotent index migrations for deployments on an existing volume.
_POST_CREATE_STATEMENTS = (
    "CREATE INDEX IF NOT EXISTS ix_cache_eval_mode "
    "ON translation_cache (eval_mode)",
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # 1. Инициализация БД при старте (до yield)
    async with async_engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
        for statement in _POST_CREATE_STATEMENTS:
            await conn.execute(text(statement))

    # 2. Приложение работает (между yield и finally)
    yield

    # 3. Очистка ресурсов при остановке (в finally)
    try:
        await async_engine.dispose()
    except Exception as e:
        # В реальном проде лучше использовать logger, но пока выводим в stdout
        print(f"Error disposing DB engine: {e}")


app = FastAPI(
    title=settings.PROJECT_NAME,
    debug=settings.DEBUG,
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"] if settings.DEBUG else settings.cors_origins_list,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(RUNGException)
async def rung_exception_handler(
    request: Request,
    exc: RUNGException,
) -> JSONResponse:
    return JSONResponse(
        status_code=exc.status_code,
        content=exc.to_dict(),
    )


# 2. ДОБАВЛЕНО: Подключение роутера auth сразу после health
app.include_router(health.router)
app.include_router(auth.router)


@app.get("/")
async def root():
    return {"message": f"{settings.PROJECT_NAME} is running!"}


@app.get("/healthz")
async def healthz():
    return {"status": "ok"}


@app.get("/languages")
async def languages():
    return {"languages": LANGUAGES}
