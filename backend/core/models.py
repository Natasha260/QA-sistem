# backend/core/models.py
"""Pydantic-модели данных QA-системы."""
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class TerminologyEntry(BaseModel):
    """Термин предметной области с пояснением."""

    id: Optional[str] = None
    term: str
    definition: str
    subject: Optional[str] = None  # предметная область (physics, history…)


class ChunkHit(BaseModel):
    """Найденный чанк учебника."""

    id: Optional[str] = None
    document_id: Optional[str] = None
    title: Optional[str] = None
    chunk_index: Optional[int] = None
    text: str
    score: Optional[float] = None


class QAPair(BaseModel):
    """Подтверждённая пара «вопрос → ответ»."""

    id: Optional[str] = None
    question: str
    answer: str
    score: Optional[float] = None


class QueryDetails(BaseModel):
    """Диагностические детали одного ответа."""

    terminology_used: List[TerminologyEntry] = Field(default_factory=list)
    chunks_used: List[ChunkHit] = Field(default_factory=list)
    verified_used: List[QAPair] = Field(default_factory=list)
    validation_passed: Optional[bool] = None
    critic_score: Optional[float] = None
    critic_relevance: Optional[float] = None
    critic_faithfulness: Optional[float] = None
    critic_completeness: Optional[float] = None
    critic_suggestions: Optional[List[str]] = None
    critic_missing_terms: Optional[List[str]] = None
    reasoning_log: Optional[List[Dict[str, Any]]] = None
    engine_used: Optional[str] = None
    confidence: Optional[float] = None


# ---------------------------------------------------------------------------
# /query
# ---------------------------------------------------------------------------

class QueryRequest(BaseModel):
    question: str
    top_k: int = Field(default=5, ge=1, le=20)
    # If None, the engine is taken from settings (QA_ENGINE).
    engine: Optional[str] = None
    # Free-form subject hint passed to the Critic.
    subject: str = "general"
    # Ablation mode override (otherwise settings.EVAL_MODE).
    eval_mode: Optional[str] = None


class QueryResponse(BaseModel):
    question: str
    answer: str
    model_used: str
    engine_used: Optional[str] = None
    confidence: Optional[float] = None
    terminology_used: List[TerminologyEntry] = Field(default_factory=list)
    chunks_used: List[ChunkHit] = Field(default_factory=list)
    verified_used: List[QAPair] = Field(default_factory=list)
    validation_passed: Optional[bool] = None
    critic_score: Optional[float] = None
    iterations: Optional[int] = None
    critic_reasoning: Optional[str] = None
    editor_changed: Optional[bool] = None
    critic_relevance: Optional[float] = None
    critic_faithfulness: Optional[float] = None
    critic_completeness: Optional[float] = None
    critic_suggestions: Optional[List[str]] = None
    critic_missing_terms: Optional[List[str]] = None
    reasoning_log: Optional[List[Dict[str, Any]]] = None
    cached: bool = False


# ---------------------------------------------------------------------------
# /query/batch
# ---------------------------------------------------------------------------

class QueryBatchRequest(BaseModel):
    questions: List[str]
    top_k: int = Field(default=5, ge=1, le=20)
    engine: Optional[str] = None
    subject: str = "general"
    eval_mode: Optional[str] = None


class QueryBatchResponse(BaseModel):
    answers: List[str]
    confidences: List[Optional[float]]
    details: Optional[List[QueryDetails]] = None


# ---------------------------------------------------------------------------
# /query/ensemble
# ---------------------------------------------------------------------------

class EnsembleRequest(BaseModel):
    question: str
    top_k: int = Field(default=5, ge=1, le=20)
    models: Optional[List[str]] = None
    subject: str = "general"


class EnsembleAnswerVariant(BaseModel):
    model: str
    answer: str
    critic_score: Optional[float] = None
    critic_relevance: Optional[float] = None
    critic_faithfulness: Optional[float] = None
    critic_completeness: Optional[float] = None
    critic_reasoning: Optional[str] = None


class EnsembleResponse(BaseModel):
    question: str
    best_answer: str
    best_model: str
    best_critic_score: float
    confidence: Optional[float] = None
    variants: List[EnsembleAnswerVariant]


# ---------------------------------------------------------------------------
# /admin/documents
# ---------------------------------------------------------------------------

class DocumentCreate(BaseModel):
    title: str
    text: str
    filename: Optional[str] = None
    description: Optional[str] = None


class DocumentInfo(BaseModel):
    id: str
    title: str
    filename: Optional[str] = None
    description: Optional[str] = None
    chunks: int = 0
    created_at: Optional[str] = None
