"""Rule-based validator for QA answers.

This agent does not call an LLM. It performs deterministic checks:
non-empty answer, echo of the question, refusal phrases without
context, repetitions and leftover prompt markers.
"""

import logging
import re
from typing import Any, Dict, List, Optional

from backend.agents.base import BaseAgent

logger = logging.getLogger(__name__)

_MIN_ANSWER_LEN = 5
_MAX_LEFTOVER_REPORTED = 5

_PLACEHOLDER_RE = re.compile(r"\{[^{}]+\}|%[sdif]|<[^<>]+>|\[\w+\]")
_PROMPT_MARKER_RE = re.compile(
    r"^(?:Answer|Ответ|Context|Контекст|Question|Вопрос)\s*:",
    re.IGNORECASE | re.MULTILINE,
)
_REFUSAL_RE = re.compile(
    r"(?:не\s+(?:может|могу|имею)\s+(?:найти|определить)|"
    r"контекст\s+не\s+содержит|"
    r"(?:i don't know|cannot find|no answer))",
    re.IGNORECASE,
)


def _contains_word(text: str, term: str) -> bool:
    if not term or not text:
        return False
    pattern = r"(?<!\w)" + re.escape(term) + r"(?!\w)"
    return re.search(pattern, text, re.IGNORECASE) is not None


class Validator(BaseAgent):
    """Rule-based validator. Does NOT call an LLM."""

    def __init__(self, engine: Optional[str] = None):
        super().__init__(name="Validator")

    async def process(self, state: Dict[str, Any]) -> Dict[str, Any]:
        logger.info("[%s] Starting...", self.name)

        question = state.get("question", "") or state.get("source_text", "") or ""
        answer = state.get("answer", "") or state.get("translation", "") or ""
        chunks = state.get("chunks_used", []) or []

        errors: List[str] = []
        warnings: List[str] = []

        if not answer.strip():
            errors.append("Answer is empty")
        elif len(answer.strip()) < _MIN_ANSWER_LEN:
            errors.append("Answer is too short to be meaningful")

        errors.extend(self._check_echo(question, answer))
        errors.extend(self._check_prompt_markers(answer))
        errors.extend(self._check_placeholders(answer))
        errors.extend(self._check_grounding(question, answer, chunks))

        warnings.extend(self._check_refusal(answer, chunks))
        warnings.extend(self._check_repetitions(answer))

        state["validator_errors"] = errors
        state["validator_warnings"] = warnings
        state["validation_passed"] = len(errors) == 0

        self._last_result = {
            "passed": state["validation_passed"],
            "errors": errors,
            "warnings": warnings,
        }

        state.setdefault("reasoning_log", []).append(
            {
                "agent": self.name,
                "step": "validation",
                "errors": errors,
                "warnings": warnings,
                "passed": state["validation_passed"],
            }
        )

        logger.info(
            "[%s] Finished: passed=%s, errors=%d, warnings=%d",
            self.name,
            state["validation_passed"],
            len(errors),
            len(warnings),
        )
        return state

    def _check_echo(self, question: str, answer: str) -> List[str]:
        question_clean = re.sub(r"\s+", " ", question.strip()).lower()
        answer_clean = re.sub(r"\s+", " ", answer.strip()).lower()

        if question_clean and question_clean == answer_clean:
            return ["Answer is identical to the question"]
        return []

    def _check_prompt_markers(self, answer: str) -> List[str]:
        if _PROMPT_MARKER_RE.search(answer):
            return ["Answer contains raw prompt markers (Context:/Question:…)"]
        return []

    def _check_placeholders(self, answer: str) -> List[str]:
        placeholders = _PLACEHOLDER_RE.findall(answer)
        if placeholders:
            return [f"Answer contains unfilled placeholders: {', '.join(placeholders[:5])}"]
        return []

    def _check_grounding(
        self,
        question: str,
        answer: str,
        chunks: List[Any],
    ) -> List[str]:
        """Если контекст есть, но ответ не пересекается с ним ни по одному
        значимому слову — это признак галлюцинации."""
        if not chunks or not answer.strip():
            return []

        context_text = " ".join(
            (c.get("text", "") if isinstance(c, dict) else getattr(c, "text", ""))
            for c in chunks
        )
        if not context_text:
            return []

        words = re.findall(r"\w{6,}", answer.lower(), flags=re.UNICODE)
        if not words:
            return []

        overlap = sum(1 for w in set(words) if _contains_word(context_text, w))
        if overlap == 0:
            return [
                "Answer does not overlap with retrieved context "
                "(possible hallucination)"
            ]
        return []

    def _check_refusal(self, answer: str, chunks: List[Any]) -> List[str]:
        if _REFUSAL_RE.search(answer) and chunks:
            return [
                "Answer refuses to reply although context was provided — "
                "check completeness"
            ]
        return []

    def _check_repetitions(self, answer: str) -> List[str]:
        words = re.findall(r"\w+", answer.lower(), flags=re.UNICODE)

        for index in range(len(words) - 2):
            if words[index] == words[index + 1] == words[index + 2]:
                return [f"Repeated word detected: '{words[index]}'"]
        return []
