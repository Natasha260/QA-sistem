# backend/hitl/feedback.py
"""Ручное сохранение исправленного ответа в хранилище verified QA."""
import logging
from typing import Any, Dict

from sqlalchemy import select

from backend.data.verified_qa import verified_qa_manager
from backend.db.models import VerifiedQADB
from backend.db.session import AsyncSessionLocal

logger = logging.getLogger(__name__)


class FeedbackHandler:
    @staticmethod
    async def save_correction(
        question: str,
        corrected_answer: str,
    ) -> Dict[str, Any]:
        """
        Persist a human-corrected answer into the verified QA store.

        Deduplication: a read-only lookup here (so the caller can
        distinguish ``"skipped"`` from ``"ok"``) plus the hash-id based
        insert inside ``verified_qa_manager``.
        """
        async with AsyncSessionLocal() as session:
            stmt = select(VerifiedQADB).where(
                VerifiedQADB.source_text == question,
                VerifiedQADB.target_text == corrected_answer,
            ).limit(1)
            existing = (await session.execute(stmt)).scalar_one_or_none()
            if existing is not None:
                logger.info("Correction already exists in verified QA")
                return {"status": "skipped", "message": "Entry already exists"}

        try:
            pair = await verified_qa_manager.add_pair(
                question=question,
                answer=corrected_answer,
            )
        except Exception as e:
            logger.error("Failed to save correction: %s", e, exc_info=True)
            raise

        if pair is not None:
            logger.info(
                "Correction saved to verified QA (%s... -> %s...)",
                question[:40],
                corrected_answer[:40],
            )
        else:
            logger.info(
                "Correction insert skipped as duplicate (race with "
                "concurrent request)"
            )

        return {"status": "ok", "message": "Correction saved"}
