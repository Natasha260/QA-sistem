# backend/api/routes/hitl.py
"""HITL-эндпоинты: очередь неуверенных ответов."""
import logging

from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel, field_validator

from backend.api.deps import require_editor
from backend.core.exceptions import RUNGException
from backend.hitl.active_learning import ActiveLearningManager
from backend.hitl.feedback import FeedbackHandler

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/hitl", tags=["human-in-the-loop"])


class CorrectionRequest(BaseModel):
    question: str
    corrected_answer: str

    @field_validator("question", "corrected_answer")
    @classmethod
    def not_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("Field cannot be empty")
        return v.strip()


class ConfirmCorrectionRequest(BaseModel):
    corrected_answer: str

    @field_validator("corrected_answer")
    @classmethod
    def not_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("Answer cannot be empty")
        return v.strip()


@router.post("/save-correction", status_code=status.HTTP_201_CREATED)
async def save_correction(
    request: CorrectionRequest,
    _=Depends(require_editor),
):
    try:
        return await FeedbackHandler.save_correction(
            question=request.question,
            corrected_answer=request.corrected_answer,
        )
    except RUNGException:
        raise
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to save correction: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to save correction")


@router.get("/uncertain")
async def get_uncertain_examples(
    limit: int = Query(10, ge=1, le=100),
    offset: int = Query(0, ge=0),
    _=Depends(require_editor),
):
    try:
        examples = await ActiveLearningManager.get_pending_examples(
            limit=limit,
            offset=offset,
        )
        total = await ActiveLearningManager.count_pending()
        stats = await ActiveLearningManager.get_stats()
        return {
            "examples": examples,
            "stats": stats,
            "total": total,
            "limit": limit,
            "offset": offset,
        }
    except RUNGException:
        raise
    except Exception as e:
        logger.error("Failed to get uncertain examples: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to get uncertain examples")


@router.post("/uncertain/{example_id}/confirm", status_code=status.HTTP_200_OK)
async def confirm_uncertain_example(
    example_id: str,
    request: ConfirmCorrectionRequest,
    _=Depends(require_editor),
):
    try:
        inserted = await ActiveLearningManager.confirm_correction(
            example_id,
            request.corrected_answer,
        )
    except RUNGException:
        raise
    except Exception as e:
        logger.error("Failed to confirm correction: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to confirm correction")

    if inserted:
        message = "Correction confirmed and added to verified QA"
    else:
        message = "Correction confirmed; entry already existed in verified QA"

    return {
        "status": "ok",
        "inserted": inserted,
        "message": message,
    }


@router.post("/uncertain/{example_id}/reject", status_code=status.HTTP_200_OK)
async def reject_uncertain_example(
    example_id: str,
    _=Depends(require_editor),
):
    try:
        await ActiveLearningManager.reject_example(example_id)
    except RUNGException:
        raise
    except Exception as e:
        logger.error("Failed to reject example: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to reject example")

    return {"status": "ok", "message": "Example rejected"}


@router.get("/uncertain/stats")
async def get_uncertain_stats(
    _=Depends(require_editor),
):
    try:
        return await ActiveLearningManager.get_stats()
    except RUNGException:
        raise
    except Exception as e:
        logger.error("Failed to get stats: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to get stats")
