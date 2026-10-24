# backend/api/routes/health.py
"""
Healthcheck endpoints.

/healthz  — простой liveness-проверка, приложение запустилось.
/languages — статический список поддерживаемых языков.
/health   — полная проверка сервисов (появится на этапе 3).
"""

import logging

from fastapi import APIRouter

from backend.core.constants import LANGUAGES, KNOWN_LANGUAGES

logger = logging.getLogger(__name__)
router = APIRouter(tags=["health"])


@router.get("/healthz")
async def healthz():
    """
    Liveness-проверка.

    Возвращает 200, если приложение запустилось.
    Используется Docker и системами мониторинга.
    """
    return {"status": "ok"}


@router.get("/languages")
async def languages():
    """
    Возвращает список поддерживаемых языков.

    Используется UI для построения выпадающих списков.
    """
    return {
        "languages": LANGUAGES,
        "known_languages": KNOWN_LANGUAGES,
    }