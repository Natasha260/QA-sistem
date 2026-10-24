# # backend/api/routes/health.py
import logging

from fastapi import APIRouter, Response, status
from sqlalchemy import text

from backend.core.config import get_settings
from backend.db.session import async_engine

logger = logging.getLogger(__name__)
router = APIRouter(tags=["health"])


@router.get("/health")
async def health(response: Response):
    """
    Liveness / readiness probe.

    Returns HTTP 200 when every dependency is reachable and HTTP 503
    otherwise, so orchestrators (Kubernetes, Docker healthcheck, ...) can
    act on the status code alone. The body carries per-service details.
    """
    settings = get_settings()

    db_ok, db_error = await check_database()

    all_healthy = db_ok
    response.status_code = (
        status.HTTP_200_OK if all_healthy else status.HTTP_503_SERVICE_UNAVAILABLE
    )

    return {
        "status": "healthy" if all_healthy else "unhealthy",
        "project": settings.PROJECT_NAME,
        "debug": settings.DEBUG,
        "services": {
            "database": {"healthy": db_ok, "error": db_error},
        },
    }


async def check_database():
    try:
        async with async_engine.connect() as conn:
            await conn.execute(text("SELECT 1"))
        return True, None
    except Exception as e:
        logger.error("Database health check failed: %s", e)
        return False, str(e)