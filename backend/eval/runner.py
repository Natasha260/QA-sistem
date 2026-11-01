# backend/eval/runner.py
"""Метрики качества QA: Exact Match (EM) и token-F1 (SQuAD-стиль)."""
import logging
import re
import string
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

_PUNCT_TABLE = str.maketrans("", "", string.punctuation)
_ARTICLES = {"a", "an", "the", "и", "в", "во", "не", "что", "он", "на", "с", "со"}
_ARTICLES_RU = {"и", "в", "во", "не", "что", "он", "на", "с", "со", "как", "а", "то"}


def normalize_answer(text: str) -> str:
    """SQuAD-style normalization: lower, remove punctuation/articles."""
    text = (text or "").lower()
    text = text.translate(_PUNCT_TABLE)
    tokens = [
        t for t in text.split()
        if t not in _ARTICLES and t not in _ARTICLES_RU
    ]
    return " ".join(tokens)


def exact_match(hypothesis: str, reference: str) -> float:
    return 1.0 if normalize_answer(hypothesis) == normalize_answer(reference) else 0.0


def token_f1(hypothesis: str, reference: str) -> float:
    hyp_tokens = normalize_answer(hypothesis).split()
    ref_tokens = normalize_answer(reference).split()
    if not hyp_tokens or not ref_tokens:
        return 0.0 if (hyp_tokens or ref_tokens) else 1.0

    common = set(hyp_tokens) & set(ref_tokens)
    if not common:
        return 0.0

    precision = len(common) / len(hyp_tokens)
    recall = len(common) / len(ref_tokens)
    return 2 * precision * recall / (precision + recall)


class Evaluator:
    """QA-метрики по корпусу: EM и token-F1 (0-100)."""

    def evaluate(
        self,
        hypotheses: List[str],
        references: List[str],
        questions: Optional[List[str]] = None,
    ) -> Dict[str, float]:
        if not hypotheses or not references:
            raise ValueError("hypotheses and references must be non-empty")
        if len(hypotheses) != len(references):
            raise ValueError(
                f"hypotheses ({len(hypotheses)}) and references "
                f"({len(references)}) lengths differ"
            )

        em_values = [
            exact_match(h, r) for h, r in zip(hypotheses, references)
        ]
        f1_values = [
            token_f1(h, r) for h, r in zip(hypotheses, references)
        ]

        results: Dict[str, float] = {
            "exact_match": round(100.0 * sum(em_values) / len(em_values), 2),
            "f1": round(100.0 * sum(f1_values) / len(f1_values), 2),
            "n": len(hypotheses),
        }
        logger.info("Evaluated %d pairs: %s", len(hypotheses), results)
        return results
