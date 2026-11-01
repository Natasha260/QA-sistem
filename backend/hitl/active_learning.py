# backend/hitl/active_learning.py
"""HITL: очередь неуверенных ответов на проверку редактором."""
import logging
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from sqlalchemy import case, func, select

from backend.core.exceptions import UncertainExampleNotFoundError
from backend.data.verified_qa import verified_qa_manager
from backend.db.models import UncertainExample
from backend.db.session import AsyncSessionLocal

logger = logging.getLogger(__name__)


def _to_dict(e: UncertainExample) -> Dict[str, Any]:
    return {
        "id": e.id,
        "question": e.source_text,
        "draft_answer": e.machine_translation,
        "critic_score": e.critic_score,
        "validation_passed": e.validation_passed,
        "created_at": e.created_at.isoformat(),
    }


class ActiveLearningManager:
    """Manage uncertain QA answers for human review."""

    @staticmethod
    async def add_uncertain_example(
        source_text: str,
        machine_translation: str,
        critic_score: Optional[float] = None,
        validation_passed: bool = False,
        source_lang: str = "ru",
        target_lang: str = "ru",
    ) -> str:
        async with AsyncSessionLocal() as session:
            try:
                stmt = select(UncertainExample).where(
                    UncertainExample.source_text == source_text,
                    UncertainExample.status == "pending",
                ).limit(1)
                existing = (await session.execute(stmt)).scalar_one_or_none()
                if existing is not None:
                    logger.debug(
                        "Pending example already exists: %s...", source_text[:50]
                    )
                    return existing.id

                example = UncertainExample(
                    id=str(uuid.uuid4()),
                    source_text=source_text,
                    source_lang=source_lang,
                    target_lang=target_lang,
                    machine_translation=machine_translation,
                    critic_score=critic_score,
                    validation_passed=validation_passed,
                    status="pending",
                )
                session.add(example)
                await session.commit()
                logger.info("Uncertain example added: %s", example.id)
                return example.id
            except Exception as e:
                logger.error("Failed to add uncertain example: %s", e, exc_info=True)
                await session.rollback()
                raise

    @staticmethod
    async def get_pending_examples(
        limit: int = 10,
        offset: int = 0,
    ) -> List[Dict[str, Any]]:
        async with AsyncSessionLocal() as session:
            try:
                stmt = (
                    select(UncertainExample)
                    .where(UncertainExample.status == "pending")
                    .order_by(UncertainExample.created_at.desc())
                    .offset(offset)
                    .limit(limit)
                )
                examples = (await session.execute(stmt)).scalars().all()
                return [_to_dict(e) for e in examples]
            except Exception as e:
                logger.error(
                    "Failed to fetch pending examples: %s", e, exc_info=True
                )
                return []

    @staticmethod
    async def count_pending() -> int:
        async with AsyncSessionLocal() as session:
            try:
                stmt = select(func.count()).select_from(UncertainExample).where(
                    UncertainExample.status == "pending"
                )
                return (await session.execute(stmt)).scalar_one()
            except Exception as e:
                logger.error(
                    "Failed to count pending examples: %s", e, exc_info=True
                )
                return 0

    @staticmethod
    async def confirm_correction(
        example_id: str,
        corrected_answer: str,
    ) -> bool:
        """
        Confirm a pending example and persist the corrected answer into
        the verified QA store.

        Both writes happen in a single transaction: the INSERT into
        ``verified_qa`` and the UPDATE of the ``uncertain_examples``
        row to ``"confirmed"``. Either both are persisted, or neither.
        """
        inserted = False

        async with AsyncSessionLocal() as session:
            try:
                stmt = select(UncertainExample).where(
                    UncertainExample.id == example_id,
                    UncertainExample.status == "pending",
                )
                example = (await session.execute(stmt)).scalar_one_or_none()
                if example is None:
                    logger.warning(
                        "Pending example %s not found for confirmation", example_id
                    )
                    raise UncertainExampleNotFoundError(
                        "Pending example not found",
                        context={"example_id": example_id},
                    )

                pair = await verified_qa_manager.add_pair(
                    question=example.source_text,
                    answer=corrected_answer,
                    session=session,
                )
                inserted = pair is not None

                example.corrected_translation = corrected_answer
                example.status = "confirmed"
                example.confirmed_at = datetime.now(timezone.utc)

                await session.commit()

            except UncertainExampleNotFoundError:
                raise
            except Exception as e:
                logger.error(
                    "Failed to confirm correction: %s", e, exc_info=True
                )
                await session.rollback()
                raise

        if inserted:
            await verified_qa_manager.load_all_into_cache()

        logger.info(
            "Correction confirmed for example %s (inserted=%s)",
            example_id,
            inserted,
        )
        return inserted

    @staticmethod
    async def reject_example(example_id: str) -> bool:
        """
        Mark a pending example as rejected.

        Raises:
            UncertainExampleNotFoundError: no pending example with that id.
        """
        async with AsyncSessionLocal() as session:
            try:
                stmt = select(UncertainExample).where(
                    UncertainExample.id == example_id,
                    UncertainExample.status == "pending",
                )
                example = (await session.execute(stmt)).scalar_one_or_none()
                if example is None:
                    logger.warning(
                        "Pending example %s not found for rejection", example_id
                    )
                    raise UncertainExampleNotFoundError(
                        "Pending example not found",
                        context={"example_id": example_id},
                    )

                example.status = "rejected"
                await session.commit()
                logger.info("Example %s rejected", example_id)
                return True

            except UncertainExampleNotFoundError:
                raise
            except Exception as e:
                logger.error("Failed to reject example: %s", e, exc_info=True)
                await session.rollback()
                raise

    @staticmethod
    async def get_stats() -> Dict[str, int]:
        async with AsyncSessionLocal() as session:
            try:
                stmt = select(
                    func.sum(
                        case((UncertainExample.status == "pending", 1), else_=0)
                    ).label("pending"),
                    func.sum(
                        case((UncertainExample.status == "confirmed", 1), else_=0)
                    ).label("confirmed"),
                    func.sum(
                        case((UncertainExample.status == "rejected", 1), else_=0)
                    ).label("rejected"),
                )
                row = (await session.execute(stmt)).one()
                return {
                    "pending": row.pending or 0,
                    "confirmed": row.confirmed or 0,
                    "rejected": row.rejected or 0,
                }
            except Exception as e:
                logger.error("Failed to get stats: %s", e, exc_info=True)
                return {"pending": 0, "confirmed": 0, "rejected": 0}
