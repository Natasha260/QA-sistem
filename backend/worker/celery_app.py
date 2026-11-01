# backend/worker/celery_app.py
import logging

from celery import Celery
from celery.signals import worker_process_shutdown, worker_shutdown

from backend.core.config import get_settings
from backend.utils.http_client import close_ollama_clients

logger = logging.getLogger(__name__)
settings = get_settings()

celery_app = Celery(
    "rung",
    broker=settings.REDIS_URL,
    backend=settings.REDIS_URL,
    include=["backend.worker.tasks"],
)

celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,
    task_time_limit=30 * 60,
    task_soft_time_limit=25 * 60,
    worker_prefetch_multiplier=1,
    task_acks_late=True,
)

logger.info("Celery app configured with broker: %s", settings.REDIS_URL)


def _shutdown_worker_resources(**kwargs) -> None:
    """
    Release resources held by the process that actually ran tasks.

    Called on both ``worker_process_shutdown`` (fires in each prefork
    child) and ``worker_shutdown`` (fires in the main process, which is
    the executor under the solo pool). The handler is idempotent: if the
    process never created a worker loop, both branches are no-ops.

    Order matters. The httpx clients are bound to the worker's event
    loop, so they must be closed while the loop is still alive.
    ``close_worker_loop`` runs afterwards.
    """
    from backend.worker import tasks

    loop = tasks._worker_loop
    if loop is not None and not loop.is_closed():
        try:
            loop.run_until_complete(close_ollama_clients())
        except Exception as e:
            logger.warning("Failed to close httpx clients on shutdown: %s", e)

    try:
        tasks.close_worker_loop()
    except Exception as e:
        logger.warning("Failed to close worker event loop: %s", e)


worker_process_shutdown.connect(_shutdown_worker_resources)
worker_shutdown.connect(_shutdown_worker_resources)


if __name__ == "__main__":
    celery_app.start()
