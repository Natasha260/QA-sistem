# backend/api/routes/health.py
import asyncio
import logging

from fastapi import APIRouter, Response, status
from sqlalchemy import text

from backend.core.config import get_settings
from backend.db.session import async_engine
from backend.utils.http_client import get_ollama_client

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
    redis_ok, redis_error = await check_redis(settings.REDIS_URL)
    qdrant_ok, qdrant_error = await check_qdrant(settings.qdrant_url)
    ollama_ok, ollama_error = await check_ollama(
        settings.ollama_url, settings.LLM_MODEL
    )

    all_healthy = all((db_ok, redis_ok, qdrant_ok, ollama_ok))
    response.status_code = (
        status.HTTP_200_OK if all_healthy else status.HTTP_503_SERVICE_UNAVAILABLE
    )

    return {
        "status": "healthy" if all_healthy else "unhealthy",
        "project": settings.PROJECT_NAME,
        "model": settings.LLM_MODEL,
        "ensemble_models": settings.ensemble_models_list,
        "debug": settings.DEBUG,
        "services": {
            "database": {"healthy": db_ok, "error": db_error},
            "redis": {"healthy": redis_ok, "error": redis_error},
            "qdrant": {"healthy": qdrant_ok, "error": qdrant_error},
            "ollama": {"healthy": ollama_ok, "error": ollama_error},
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


async def check_redis(redis_url: str):
    try:
        import redis.asyncio as redis

        client = redis.from_url(redis_url, socket_connect_timeout=3)
        try:
            await client.ping()
        finally:
            await client.aclose()
        return True, None
    except Exception as e:
        logger.error("Redis health check failed: %s", e)
        return False, str(e)


async def check_qdrant(qdrant_url: str):
    """
    Verify that Qdrant is reachable.

    Runs the synchronous QdrantClient in a worker thread so it does not
    block the event loop. Passes ``QDRANT_API_KEY`` when configured, so
    an authenticated Qdrant reports healthy.
    """
    settings = get_settings()

    def _check() -> None:
        from qdrant_client import QdrantClient

        client_kwargs = {"url": qdrant_url, "timeout": 3}
        if settings.QDRANT_API_KEY:
            client_kwargs["api_key"] = settings.QDRANT_API_KEY

        client = QdrantClient(**client_kwargs)
        try:
            client.get_collections()
        finally:
            try:
                client.close()
            except Exception:
                pass

    try:
        await asyncio.to_thread(_check)
        return True, None
    except Exception as e:
        logger.error("Qdrant health check failed: %s", e)
        return False, str(e)


def _ollama_name_matches(requested: str, available: str) -> bool:
    """
    Exact-match comparison for Ollama model names.

    Substring matching is wrong here: ``"qwen2.5:7b" in "qwen2.5:7b-instruct"``
    is True, so an /health probe would report healthy while the model that
    the agents actually call is not present, and translation silently
    degrades to the Hugging Face fallback. The reverse case
    (``"qwen"`` matching every model that contains "qwen") is equally
    misleading.

    Ollama resolves a bare name ``foo`` to ``foo:latest``, so both forms
    are accepted:

        requested="qwen2.5:7b"  available="qwen2.5:7b"          -> True
        requested="qwen2.5:7b"  available="qwen2.5:7b-instruct" -> False
        requested="qwen2.5"     available="qwen2.5:latest"      -> True
        requested="qwen2.5"     available="qwen2.5"             -> True
        requested="qwen"        available="qwen2.5:7b"          -> False
    """
    if requested == available:
        return True

    # ``name`` and ``name:latest`` refer to the same model in Ollama.
    if ":" not in requested and available == f"{requested}:latest":
        return True
    if ":" not in available and requested == f"{available}:latest":
        return True

    return False


async def check_ollama(ollama_url: str, model_name: str):
    """
    Verify that Ollama is reachable and that ``model_name`` is available.

    Uses the shared per-loop httpx client and overrides its timeout per
    request, so probes stay fast without creating a fresh client on every
    call. The ``ollama/`` prefix is stripped before the comparison.
    """
    normalised = model_name.replace("ollama/", "") if model_name else ""
    try:
        tags_url = ollama_url.replace("/api/generate", "/api/tags")
        client = get_ollama_client()
        response = await client.get(tags_url, timeout=5.0)
        response.raise_for_status()
        data = response.json()
        models = [m["name"] for m in data.get("models", [])]
        if not any(_ollama_name_matches(normalised, m) for m in models):
            return False, f"Model '{normalised}' not found"
        return True, None
    except Exception as e:
        logger.error("Ollama health check failed: %s", e)
        return False, str(e)



