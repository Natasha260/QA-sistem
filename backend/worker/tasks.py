# backend/worker/tasks.py
"""Celery-задачи QA-системы."""
import asyncio
import logging
import threading
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

from sqlalchemy import delete

from backend.data.documents import documents_manager
from backend.data.glossary import glossary_manager
from backend.data.verified_qa import verified_qa_manager
from backend.db.models import TranslationCache
from backend.db.session import AsyncSessionLocal
from backend.eval.runner import Evaluator
from backend.services.pipeline import perform_qa
from backend.worker.celery_app import celery_app

logger = logging.getLogger(__name__)

_MAX_CONCURRENT_QUERIES = 4

_worker_loop: Optional[asyncio.AbstractEventLoop] = None
_worker_loop_lock = threading.Lock()


def _get_worker_loop() -> asyncio.AbstractEventLoop:
    global _worker_loop
    if _worker_loop is None or _worker_loop.is_closed():
        with _worker_loop_lock:
            if _worker_loop is None or _worker_loop.is_closed():
                _worker_loop = asyncio.new_event_loop()
                asyncio.set_event_loop(_worker_loop)
    return _worker_loop


def _run_async(coro):
    return _get_worker_loop().run_until_complete(coro)


def close_worker_loop() -> None:
    """
    Close the persistent worker event loop.

    Called from the ``worker_shutdown`` signal in ``celery_app``. Safe to
    call when the loop was never created.
    """
    global _worker_loop
    if _worker_loop is not None and not _worker_loop.is_closed():
        _worker_loop.close()
        logger.info("Worker event loop closed")
    _worker_loop = None


@celery_app.task(bind=True, max_retries=3)
def index_chunks(self, force: bool = False):
    """Полная пересборка BM25-индекса и векторной коллекции чанков."""
    try:
        logger.info("Starting chunk indexing...")
        _run_async(documents_manager.load_all_into_cache())
        logger.info("Chunk indexing completed.")
        return {"status": "success", "message": "Chunks indexed"}
    except Exception as e:
        logger.error("Chunk indexing failed: %s", e, exc_info=True)
        self.retry(exc=e, countdown=60)
        raise


@celery_app.task(bind=True, max_retries=2)
def batch_query(
    self,
    questions: List[str],
    top_k: int = 5,
    eval_mode: str = "full",
    subject: str = "general",
) -> List[Dict[str, Any]]:
    async def _answer_all():
        semaphore = asyncio.Semaphore(_MAX_CONCURRENT_QUERIES)

        async def _bounded(question: str):
            async with semaphore:
                return await perform_qa(
                    question,
                    top_k=top_k,
                    eval_mode=eval_mode,
                    subject=subject,
                )

        tasks = [_bounded(q) for q in questions]
        return await asyncio.gather(*tasks)

    try:
        logger.info(
            "Batch query started: %d questions, mode=%s, subject=%s",
            len(questions), eval_mode, subject,
        )
        results = _run_async(_answer_all())
        logger.info("Batch query completed: %d questions", len(results))
        return results
    except Exception as e:
        logger.error("Batch query failed: %s", e, exc_info=True)
        self.retry(exc=e, countdown=30)
        raise


@celery_app.task
def run_evaluation_async(
    questions: List[str],
    hypotheses: List[str],
    references: List[str],
):
    try:
        evaluator = Evaluator()
        results = evaluator.evaluate(hypotheses, references, questions)
        logger.info("Evaluation completed: %s", results)
        return results
    except Exception as e:
        logger.error("Evaluation failed: %s", e, exc_info=True)
        raise


@celery_app.task
def clear_old_cache(days: int = 30):
    async def _clear():
        async with AsyncSessionLocal() as session:
            cutoff = datetime.now(timezone.utc) - timedelta(days=days)
            stmt = delete(TranslationCache).where(
                TranslationCache.created_at < cutoff
            )
            result = await session.execute(stmt)
            await session.commit()
            deleted = result.rowcount or 0
            logger.info(
                "Cleared %d old cache entries (older than %d days)",
                deleted, days,
            )
            return {"deleted": deleted}

    try:
        return _run_async(_clear())
    except Exception as e:
        logger.error("Failed to clear old cache: %s", e, exc_info=True)
        raise


@celery_app.task
def add_terminology_bulk(entries: List[Dict[str, str]]):
    async def _add():
        added = 0
        for entry_data in entries:
            await glossary_manager.add_entry({
                "subject": entry_data.get("subject") or "general",
                "term": entry_data.get("term") or "",
                "definition": entry_data.get("definition") or "",
            })
            added += 1
        logger.info("Added %d terminology entries", added)
        return {"added": added}

    try:
        return _run_async(_add())
    except Exception as e:
        logger.error("Bulk terminology import failed: %s", e, exc_info=True)
        raise


@celery_app.task
def add_qa_pairs_bulk(entries: List[Dict[str, str]]):
    async def _add():
        added = await verified_qa_manager.add_many(entries)
        logger.info("Added %d QA pairs", added)
        return {"added": added}

    try:
        return _run_async(_add())
    except Exception as e:
        logger.error("Bulk QA import failed: %s", e, exc_info=True)
        raise


@celery_app.task(bind=True, max_retries=2)
def ingest_document(
    self,
    title: str,
    text: str,
    filename: Optional[str] = None,
    description: Optional[str] = None,
):
    """Загрузка учебника и индексация чанков (для больших файлов)."""
    try:
        result = _run_async(
            documents_manager.add_document(
                title=title, text=text,
                filename=filename, description=description,
            )
        )
        logger.info(
            "Document ingested: %s (%d chunks)",
            result["title"], result["chunks"],
        )
        return result
    except Exception as e:
        logger.error("Document ingestion failed: %s", e, exc_info=True)
        self.retry(exc=e, countdown=30)
        raise
