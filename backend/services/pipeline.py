# backend/services/pipeline.py
"""QA-пайплайн: гибридный поиск по учебникам → ответ LLM → оценка уверенности."""
import logging
from typing import Any, Dict, List, Optional, Tuple

from backend.agents.supervisor import Supervisor
from backend.agents.validator import Validator
from backend.core.config import get_settings
from backend.core.constants import (
    CONFIDENCE_THRESHOLD,
    CONFIDENCE_WEIGHTS,
)
from backend.core.models import ChunkHit, EnsembleAnswerVariant
from backend.data.cache import cache_manager
from backend.data.documents import documents_manager
from backend.data.glossary import glossary_manager
from backend.data.verified_qa import verified_qa_manager
from backend.hitl.active_learning import ActiveLearningManager
from backend.retrieval.hybrid_chunks import hybrid_chunk_search
from backend.utils.llm_client import call_ollama_with_model
from backend.utils.llm_processing import clean_translation

logger = logging.getLogger(__name__)

settings = get_settings()

# Оценка релевантности чанка для расчёта уверенности: BM25-скор
# нормализуется неявно через min(max, 1.0).
_MAX_RETRIEVAL_SCORE = 1.0

_MODE_CONFIG: Dict[str, Dict[str, Any]] = {
    "baseline": {
        "use_terminology": False,
        "chunks_backend": None,
        "use_verified": False,
        "use_critic": False,
        "use_editor": False,
    },
    "terminology": {
        "use_terminology": True,
        "chunks_backend": None,
        "use_verified": False,
        "use_critic": False,
        "use_editor": False,
    },
    "chunks_bm25": {
        "use_terminology": True,
        "chunks_backend": "bm25",
        "use_verified": False,
        "use_critic": False,
        "use_editor": False,
    },
    "chunks_hybrid": {
        "use_terminology": True,
        "chunks_backend": "hybrid",
        "use_verified": False,
        "use_critic": False,
        "use_editor": False,
    },
    "agents_only": {
        "use_terminology": False,
        "chunks_backend": None,
        "use_verified": False,
        "use_critic": True,
        "use_editor": True,
    },
    "chunks_hybrid_critic": {
        "use_terminology": True,
        "chunks_backend": "hybrid",
        "use_verified": False,
        "use_critic": True,
        "use_editor": False,
    },
    "chunks_hybrid_editor": {
        "use_terminology": True,
        "chunks_backend": "hybrid",
        "use_verified": False,
        "use_critic": False,
        "use_editor": True,
    },
    "full": {
        "use_terminology": True,
        "chunks_backend": "hybrid",
        "use_verified": True,
        "use_critic": True,
        "use_editor": True,
    },
}

_ALL_MODES: Tuple[str, ...] = tuple(_MODE_CONFIG.keys())


def _mode_config(mode: str) -> Dict[str, Any]:
    if mode not in _MODE_CONFIG:
        from backend.core.exceptions import UnknownEvalModeError
        raise UnknownEvalModeError(
            f"Unknown eval_mode='{mode}'",
            context={
                "mode": mode,
                "supported_modes": list(_ALL_MODES),
            },
        )
    return _MODE_CONFIG[mode]


def build_answer_prompt(
    question: str,
    chunks: List[ChunkHit],
    terminology_entries: List[Any],
    verified_pairs: List[Any],
    subject: str = "general",
) -> str:
    """Собирает промпт для ответа на вопрос по контексту учебника."""
    prompt = (
        "You answer questions from a university textbook. "
        "IMPORTANT: Answer in Russian, using ONLY the context passages below. "
        "Do NOT use knowledge that is absent from the context. "
        "If the context is insufficient, say exactly that: "
        "'В предоставленном контексте нет ответа на этот вопрос.' "
        "Do NOT add phrases like 'Answer:', 'Here is the answer' or meta-remarks.\n"
    )
    if subject and subject != "general":
        prompt += f"Subject: {subject}.\n"

    if terminology_entries:
        prompt += "\nTerminology:\n" + "\n".join(
            f"{e['term']} — {e['definition']}"
            for e in terminology_entries
        )

    if verified_pairs:
        prompt += "\nExamples of verified questions and answers:\n"
        for pair in verified_pairs:
            score = getattr(pair, "score", None)
            suffix = f" (similarity: {score:.2f})" if score is not None else ""
            prompt += f"Q: {pair.question}{suffix}\nA: {pair.answer}\n\n"

    if chunks:
        prompt += "\nContext passages:\n"
        for i, chunk in enumerate(chunks, 1):
            title = getattr(chunk, "title", None) or ""
            label = f"Chunk {i}" + (f" ({title})" if title else "")
            prompt += f"\n{label}:\n{chunk.text}\n"

    prompt += f"\nQuestion: {question}\n"
    prompt += "Answer:"
    return prompt


async def answer_with_model(
    question: str,
    model_name: str,
    chunks: List[ChunkHit],
    terminology_entries: List[Any],
    verified_pairs: List[Any],
    subject: str = "general",
) -> str:
    prompt = build_answer_prompt(
        question, chunks, terminology_entries, verified_pairs, subject
    )
    return await call_ollama_with_model(prompt, model_name)


def compute_confidence(
    critic_score: Optional[float],
    validation_passed: Optional[bool],
    max_retrieval_score: Optional[float],
) -> Optional[float]:
    """Итоговая уверенность 0-1.

    Веса: критик 0.5, релевантность контекста 0.3, валидация 0.2.
    Без критика остаётся валидация + retrieval (в сумме 0.5 —
    нормируем на доступную массу весов).
    """
    parts = []
    weights = []
    if critic_score is not None:
        parts.append(max(0.0, min(float(critic_score), 10.0)) / 10.0)
        weights.append(CONFIDENCE_WEIGHTS["critic"])
    if max_retrieval_score is not None:
        parts.append(max(0.0, min(float(max_retrieval_score), _MAX_RETRIEVAL_SCORE)))
        weights.append(CONFIDENCE_WEIGHTS["retrieval"])
    if validation_passed is not None:
        parts.append(1.0 if validation_passed else 0.0)
        weights.append(CONFIDENCE_WEIGHTS["validation"])

    if not weights:
        return None

    total_weight = sum(weights)
    return round(sum(p * w for p, w in zip(parts, weights)) / total_weight, 2)


def is_low_confidence(confidence: Optional[float]) -> bool:
    return confidence is None or confidence < CONFIDENCE_THRESHOLD


def empty_result(
    engine: Optional[str] = None,
    status: str = "cache",
    model: Optional[str] = None,
) -> Dict[str, Any]:
    return {
        "answer": "",
        "answer_raw": "",
        "answer_after_agents": None,
        "terminology_used": [],
        "chunks_used": [],
        "verified_used": [],
        "validation_passed": None,
        "critic_score": None,
        "critic_reasoning": None,
        "editor_changed": False,
        "iterations": 0,
        "critic_relevance": None,
        "critic_faithfulness": None,
        "critic_completeness": None,
        "critic_suggestions": [],
        "critic_missing_terms": [],
        "cached": False,
        "status": status,
        "engine_used": engine,
        "model_used": model,
        "reasoning_log": [],
        "confidence": None,
    }


async def _fetch_terminology(question: str, subject: str) -> List[Dict[str, str]]:
    """Термины предметной области, встречающиеся в вопросе."""
    try:
        all_entries = await glossary_manager.get_entries(subject)
    except Exception as e:
        logger.warning("Terminology lookup failed: %s", e)
        return []

    lowered = question.lower()
    relevant = [
        {"term": e["term"], "definition": e["definition"]}
        for e in all_entries
        if e["term"] and e["term"].lower() in lowered
    ]
    return relevant


async def _retrieve_context(
    question: str,
    subject: str,
    mode_config: Dict[str, Any],
    top_k: int = 5,
) -> Tuple[List[Dict[str, str]], List[ChunkHit], List[Any]]:
    terminology: List[Dict[str, str]] = []
    chunks: List[ChunkHit] = []
    verified: List[Any] = []

    if mode_config["use_terminology"]:
        terminology = await _fetch_terminology(question, subject)

    backend = mode_config["chunks_backend"]
    if backend == "bm25":
        chunks = await documents_manager.search_bm25(question, top_n=top_k)
    elif backend == "hybrid":
        chunks = await hybrid_chunk_search(question, top_n=top_k)

    if mode_config["use_verified"]:
        verified = await verified_qa_manager.search_similar(question, top_n=3)

    return terminology, chunks, verified


async def perform_qa(
    question: str,
    top_k: int = 5,
    engine: Optional[str] = None,
    eval_mode: Optional[str] = None,
    subject: str = "general",
) -> Dict[str, Any]:
    """Главный пайплайн: вопрос → гибридный поиск → ответ → уверенность."""
    question = (question or "").strip()
    if not question:
        raise ValueError("Question must not be empty")

    mode = eval_mode or settings.EVAL_MODE
    mode_config = _mode_config(mode)

    engine_used = "ollama"
    model_used = settings.LLM_MODEL

    cached = await cache_manager.get_cached(
        question, engine=engine_used, eval_mode=mode, model=model_used,
    )
    if cached is not None:
        cached_answer, cached_engine, cached_model = cached
        result = empty_result(
            engine=cached_engine, status="cache", model=cached_model,
        )
        result["answer"] = cached_answer
        result["critic_reasoning"] = "Cached answer"
        result["cached"] = True
        return result

    terminology, chunks, verified = await _retrieve_context(
        question, subject, mode_config, top_k
    )

    use_agents = mode_config["use_critic"] or mode_config["use_editor"]

    state: Dict[str, Any] = {}

    try:
        prompt = build_answer_prompt(
            question, chunks, terminology, verified, subject,
        )
        answer = await call_ollama_with_model(prompt, settings.LLM_MODEL)
    except Exception as e:
        logger.error("Answer generation error: %s", e, exc_info=True)
        raise

    if use_agents:
        supervisor = Supervisor(max_iterations=3, engine=settings.AGENT_ENGINE)
        state = {
            "question": question,
            "answer": answer,
            "chunks_used": chunks,
            "terminology_used": terminology,
            "verified_used": verified,
            "engine": settings.AGENT_ENGINE,
            "subject": subject,
            "use_critic": mode_config["use_critic"],
            "use_editor": mode_config["use_editor"],
        }
        state = await supervisor.process(state)
        final_answer = state.get("answer", answer)
        validation_passed = state.get("validation_passed")
        critic_score = state.get("critic_score")
        critic_reasoning = state.get("critic_reasoning")
        editor_changed = state.get("editor_changed")
        iterations = state.get("supervisor_iterations")
        reasoning_log = state.get("reasoning_log", [])

        confidence = compute_confidence(
            critic_score, validation_passed,
            max((c.score or 0.0) for c in chunks) if chunks else None,
        )

        if settings.ACTIVE_LEARNING_ENABLED and is_low_confidence(confidence):
            await ActiveLearningManager.add_uncertain_example(
                source_text=question,
                source_lang="ru",
                target_lang="ru",
                machine_translation=final_answer,
                critic_score=critic_score,
                validation_passed=validation_passed,
            )
    else:
        validator = Validator(engine=settings.AGENT_ENGINE)
        validator_state = await validator.process({
            "question": question,
            "answer": answer,
            "chunks_used": chunks,
            "terminology_used": terminology,
        })
        validation_passed = validator_state.get("validation_passed")

        final_answer = answer
        critic_score = None
        critic_reasoning = None
        editor_changed = None
        iterations = 0
        reasoning_log = []

        confidence = compute_confidence(
            None, validation_passed,
            max((c.score or 0.0) for c in chunks) if chunks else None,
        )

    final_clean = clean_translation(final_answer)
    if final_clean and len(final_clean) > 1:
        answer_to_return = final_clean
    else:
        answer_to_return = final_answer

    await cache_manager.save_cache(
        question, answer_to_return, model_used,
        engine=engine_used, eval_mode=mode,
    )

    return {
        "answer": answer_to_return,
        "answer_raw": answer,
        "answer_after_agents": final_answer if use_agents else None,
        "terminology_used": terminology,
        "chunks_used": chunks,
        "verified_used": verified,
        "validation_passed": validation_passed,
        "critic_score": critic_score,
        "critic_reasoning": critic_reasoning,
        "editor_changed": editor_changed,
        "iterations": iterations,
        "critic_relevance": state.get("critic_relevance"),
        "critic_faithfulness": state.get("critic_faithfulness"),
        "critic_completeness": state.get("critic_completeness"),
        "critic_suggestions": state.get("critic_suggestions"),
        "critic_missing_terms": state.get("critic_missing_terms"),
        "cached": False,
        "status": "ok",
        "engine_used": engine_used,
        "model_used": model_used,
        "reasoning_log": reasoning_log,
        "confidence": confidence,
    }


async def safe_perform_qa(
    question: str,
    top_k: int = 5,
    engine: Optional[str] = None,
    eval_mode: Optional[str] = None,
    subject: str = "general",
) -> Tuple[bool, Dict[str, Any]]:
    try:
        result = await perform_qa(
            question, top_k=top_k, engine=engine,
            eval_mode=eval_mode, subject=subject,
        )
        return True, result
    except Exception as e:
        logger.error(
            "Parallel QA failed for '%s...': %s",
            question[:50], e, exc_info=True,
        )
        placeholder = empty_result(status="error")
        placeholder["critic_reasoning"] = f"Answer generation failed: {e}"
        return False, placeholder


async def collect_retrieval_context(
    question: str,
    subject: str = "general",
    mode: Optional[str] = None,
    top_k: int = 5,
) -> Tuple[List[Dict[str, str]], List[ChunkHit], List[Any]]:
    mode_config = _mode_config(mode or settings.EVAL_MODE)
    return await _retrieve_context(question, subject, mode_config, top_k)


def select_best_variant(
    variants: List[EnsembleAnswerVariant],
) -> EnsembleAnswerVariant:
    max_score = max((v.critic_score or 0.0) for v in variants)

    if max_score == 0.0 and all(v.critic_score is None for v in variants):
        logger.warning("Ensemble: no variant was scored by the critic; using first")
        return variants[0]

    tied = [v for v in variants if (v.critic_score or 0.0) == max_score]
    if len(tied) == 1:
        return tied[0]

    logger.info(
        "Ensemble: %d variants tied at score %s; breaking tie by length",
        len(tied),
        max_score,
    )
    return max(tied, key=lambda v: len(v.answer or ""))
