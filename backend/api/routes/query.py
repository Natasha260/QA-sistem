# backend/api/routes/query.py
import asyncio
import logging
from typing import List

import httpx
from fastapi import APIRouter, Depends, HTTPException

from backend.agents.critic import Critic
from backend.api.deps import get_current_user
from backend.core.config import get_settings
from backend.core.exceptions import RUNGException
from backend.core.models import (
    EnsembleAnswerVariant, EnsembleRequest, EnsembleResponse,
    QueryBatchRequest, QueryBatchResponse, QueryDetails,
    QueryRequest, QueryResponse,
)
from backend.data.cache import cache_manager
from backend.db.models import User
from backend.services.pipeline import (
    collect_retrieval_context, compute_confidence, perform_qa,
    safe_perform_qa, select_best_variant,
)
from backend.utils.llm_processing import clean_translation

logger = logging.getLogger(__name__)
router = APIRouter()
settings = get_settings()

_MAX_CONCURRENT_QUERIES = 4
_query_semaphore = asyncio.Semaphore(_MAX_CONCURRENT_QUERIES)


def _http_error_from_transport(e: Exception, user_email: str) -> HTTPException:
    if isinstance(e, (httpx.TimeoutException, httpx.ReadTimeout)):
        logger.error("Upstream timeout for user=%s", user_email)
        return HTTPException(status_code=504, detail="Upstream timeout")
    if isinstance(e, httpx.ConnectError):
        logger.error("Upstream connection error for user=%s", user_email)
        return HTTPException(status_code=502, detail="Upstream service unavailable")
    if isinstance(e, httpx.HTTPStatusError):
        logger.error(
            "Upstream HTTP error for user=%s: %s",
            user_email, e.response.status_code,
        )
        return HTTPException(
            status_code=502,
            detail=f"Upstream error: {e.response.status_code}",
        )
    logger.error("Unexpected error for user=%s: %s", user_email, e, exc_info=True)
    return HTTPException(status_code=500, detail=str(e))


def _build_details(result: dict) -> QueryDetails:
    return QueryDetails(
        terminology_used=result["terminology_used"],
        chunks_used=result["chunks_used"],
        verified_used=result["verified_used"],
        critic_score=result["critic_score"],
        validation_passed=result["validation_passed"],
        critic_relevance=result["critic_relevance"],
        critic_faithfulness=result["critic_faithfulness"],
        critic_completeness=result["critic_completeness"],
        critic_suggestions=result["critic_suggestions"],
        critic_missing_terms=result["critic_missing_terms"],
        reasoning_log=result.get("reasoning_log"),
        engine_used=result.get("engine_used"),
        confidence=result.get("confidence"),
    )


async def _bounded_query(*args, **kwargs):
    async with _query_semaphore:
        return await safe_perform_qa(*args, **kwargs)


async def _bounded_answer_with_model(*args, **kwargs):
    from backend.services.pipeline import answer_with_model
    async with _query_semaphore:
        return await answer_with_model(*args, **kwargs)


@router.post("/query", response_model=QueryResponse)
async def query(
    request: QueryRequest,
    current_user: User = Depends(get_current_user),
):
    try:
        result = await perform_qa(
            request.question,
            top_k=request.top_k,
            engine=request.engine,
            eval_mode=request.eval_mode,
            subject=request.subject,
        )
        return QueryResponse(
            question=request.question,
            answer=result["answer"],
            model_used=result["model_used"],
            engine_used=result["engine_used"],
            confidence=result["confidence"],
            terminology_used=result["terminology_used"],
            chunks_used=result["chunks_used"],
            verified_used=result["verified_used"],
            validation_passed=result["validation_passed"],
            critic_score=result["critic_score"],
            iterations=result["iterations"],
            critic_reasoning=result["critic_reasoning"],
            editor_changed=result["editor_changed"],
            critic_relevance=result["critic_relevance"],
            critic_faithfulness=result["critic_faithfulness"],
            critic_completeness=result["critic_completeness"],
            critic_suggestions=result["critic_suggestions"],
            critic_missing_terms=result["critic_missing_terms"],
            reasoning_log=result.get("reasoning_log"),
            cached=result["cached"],
        )
    except RUNGException:
        raise
    except HTTPException:
        raise
    except Exception as e:
        raise _http_error_from_transport(e, current_user.email)


@router.post("/query/batch", response_model=QueryBatchResponse)
async def query_batch(
    request: QueryBatchRequest,
    current_user: User = Depends(get_current_user),
):
    try:
        tasks = [
            _bounded_query(
                question,
                top_k=request.top_k,
                engine=request.engine,
                eval_mode=request.eval_mode,
                subject=request.subject,
            )
            for question in request.questions
        ]
        outcomes = await asyncio.gather(*tasks)

        answers: List[str] = []
        confidences: List[object] = []
        details: List[QueryDetails] = []

        for ok, result in outcomes:
            answers.append(result["answer"])
            confidences.append(result["confidence"])
            details.append(_build_details(result))

        return QueryBatchResponse(
            answers=answers,
            confidences=confidences,
            details=details,
        )
    except RUNGException:
        raise
    except HTTPException:
        raise
    except Exception as e:
        raise _http_error_from_transport(e, current_user.email)


@router.post("/query/ensemble", response_model=EnsembleResponse)
async def query_ensemble(
    request: EnsembleRequest,
    current_user: User = Depends(get_current_user),
):
    models = request.models or settings.ensemble_models_list
    if not models:
        raise HTTPException(status_code=400, detail="No models specified")

    try:
        mode = settings.EVAL_MODE
        terminology, chunks, verified = await collect_retrieval_context(
            request.question, request.subject, mode, request.top_k,
        )

        answer_tasks = [
            _bounded_answer_with_model(
                request.question,
                model,
                chunks,
                terminology,
                verified,
                request.subject,
            )
            for model in models
        ]
        answer_outcomes = await asyncio.gather(
            *answer_tasks, return_exceptions=True
        )

        critic = Critic(engine=settings.AGENT_ENGINE)
        variants: List[EnsembleAnswerVariant] = []
        for model, outcome in zip(models, answer_outcomes):
            if isinstance(outcome, Exception):
                logger.error(
                    "Ensemble: model %s failed: %s", model, outcome, exc_info=True,
                )
                continue

            critic_state = {
                "question": request.question,
                "answer": outcome,
                "chunks_used": chunks,
                "terminology_used": terminology,
                "subject": request.subject,
            }
            critic_state = await critic.process(critic_state)

            variants.append(
                EnsembleAnswerVariant(
                    model=model,
                    answer=outcome,
                    critic_score=critic_state.get("critic_score"),
                    critic_relevance=critic_state.get("critic_relevance"),
                    critic_faithfulness=critic_state.get("critic_faithfulness"),
                    critic_completeness=critic_state.get("critic_completeness"),
                    critic_reasoning=critic_state.get("critic_reasoning"),
                )
            )

        if not variants:
            raise HTTPException(
                status_code=502,
                detail="All ensemble models failed to produce an answer",
            )

        best_variant = select_best_variant(variants)

        best_clean = clean_translation(best_variant.answer)
        best_answer = best_clean if best_clean and len(best_clean) > 1 \
            else best_variant.answer

        confidence = compute_confidence(
            best_variant.critic_score,
            None,
            max((c.score or 0.0) for c in chunks) if chunks else None,
        )

        await cache_manager.save_cache(
            request.question,
            best_answer,
            best_variant.model,
            engine="ollama",
            eval_mode=mode,
        )

        return EnsembleResponse(
            question=request.question,
            best_answer=best_answer,
            best_model=best_variant.model,
            best_critic_score=best_variant.critic_score or 0.0,
            confidence=confidence,
            variants=variants,
        )
    except RUNGException:
        raise
    except HTTPException:
        raise
    except Exception as e:
        raise _http_error_from_transport(e, current_user.email)
