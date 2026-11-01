# backend/agents/critic.py
"""LLM-критик: оценивает ответ системы на вопрос по учебнику.

Выставляет оценки по трём измерениям (0-10): relevance — насколько
ответ отвечает на вопрос; faithfulness — насколько ответ опирается на
предоставленный контекст (не выдумывает факты); completeness — полнота
ответа. Итоговый overall_score агрегирует измерения и используется
пайплайном как компонент уверенности.
"""
import json
import logging
import re
from typing import Any, Dict, List, Optional

import httpx

from backend.agents.base import Agent
from backend.core.constants import CRITIC_PASS_THRESHOLD

logger = logging.getLogger(__name__)

_EXPECTED_KEYS = frozenset({
    "relevance", "faithfulness", "completeness", "overall_score",
})

# === ПРАВИЛА ОЦЕНКИ ===
_COMMON_RULES = """
1. Ответ, не относящийся к вопросу → relevance ≤ 3.
2. Утверждение, отсутствующее в контексте (галлюцинация) → faithfulness = 0.
3. Если контекст не содержит ответа и модель это честно сообщает —
   это корректный ответ (не штрафуется за полноту).
4. Максимум 8/10, если есть сомнения в фактах.
5. Разные аспекты не могут отличаться больше чем на 2 балла.
6. overall_score ≤ 8, если есть замечания.
7. Пустой ответ или ответ «не знаю» без контекста → overall_score ≤ 2.
"""

_SUBJECT_RULES = {
    "physics": "S. Физика: формулы и единицы измерения должны совпадать с контекстом.",
    "math": "S. Математика: вычисления перепроверяй по контексту.",
    "history": "S. История: даты и имена должны точно соответствовать контексту.",
    "biology": "S. Биология: терминология (виды, органы, процессы) — строго из контекста.",
    "chemistry": "S. Химия: обозначения элементов и реакций — из контекста.",
}

# === FEW-SHOT ПРИМЕРЫ ===
_UNIVERSAL_FEWSHOT = """
Example 1 (Good):
{"relevance": 9, "faithfulness": 9, "completeness": 9, "overall_score": 9, "reasoning": "Direct, grounded answer", "suggestions": [], "missing_terms": []}

Example 2 (Hallucination):
{"relevance": 8, "faithfulness": 2, "completeness": 7, "overall_score": 4, "reasoning": "Answer adds a fact absent from context", "suggestions": ["Remove the claim about X, it is not in the context"], "missing_terms": []}

Example 3 (Incomplete):
{"relevance": 9, "faithfulness": 9, "completeness": 4, "overall_score": 7, "reasoning": "Answers only part of the question", "suggestions": ["Also mention the cause described in the second chunk"], "missing_terms": []}

Example 4 (Off-topic):
{"relevance": 1, "faithfulness": 5, "completeness": 2, "overall_score": 2, "reasoning": "Answer addresses a different question", "suggestions": [], "missing_terms": []}
"""

_PROMPT_TEMPLATE = """You grade an answer to a student's question about a university textbook.

Question: {question}

Context passages retrieved from the textbook:
{context_section}

Proposed answer: {answer}

{terminology_section}
Subject: {subject}

Apply these rules:
{scoring_rules}

Few-shot examples:
{few_shot}

Return ONLY valid JSON with keys: relevance, faithfulness, completeness, overall_score (each 0-10), reasoning, suggestions (list), missing_terms (list).
"""


class Critic(Agent):
    hf_model_setting = "HF_CRITIC_MODEL"
    ollama_model_setting = "CRITIC_MODEL"

    def __init__(self, model_name: Optional[str] = None, engine: Optional[str] = None):
        super().__init__(name="Critic", engine=engine)
        if self.engine == "huggingface":
            self.model_name = model_name or self._hf_model_name()
            from backend.utils.hf_llm import get_hf_llm
            self.hf_llm = get_hf_llm(self.model_name)
        else:
            self.model_name = (
                model_name or self._ollama_model_name() or self.settings.LLM_MODEL
            ).replace("ollama/", "")
            self.hf_llm = None
        self.temperature = 0.0

    async def process(self, state: Dict[str, Any]) -> Dict[str, Any]:
        logger.info("[%s] Starting...", self.name)
        question = state.get("question", "") or state.get("source_text", "")
        answer = state.get("answer", "") or state.get("translation", "")
        chunks = state.get("chunks_used", []) or []
        terminology = state.get("terminology_used", []) or []
        subject = state.get("subject", "general")

        prompt = self._build_prompt(
            question=question,
            answer=answer,
            chunks=chunks,
            terminology_entries=terminology,
            subject=subject,
        )

        failed = False
        try:
            response = await self._call_llm(prompt)
            evaluation = self._parse_json_response(response)
        except httpx.TimeoutException:
            logger.error("[%s] LLM timeout", self.name)
            evaluation = self._get_fallback_evaluation("Timeout error")
            failed = True
        except httpx.HTTPStatusError as e:
            logger.error("[%s] LLM HTTP error: %s", self.name, e.response.status_code)
            evaluation = self._get_fallback_evaluation(f"HTTP error: {e.response.status_code}")
            failed = True
        except Exception as e:
            logger.error("[%s] Unexpected error: %s", self.name, e, exc_info=True)
            evaluation = self._get_fallback_evaluation("Unexpected error")
            failed = True

        if evaluation.get("failed"):
            failed = True

        suggestions = evaluation.get("suggestions") or []
        missing_terms = evaluation.get("missing_terms") or []
        overall_score = evaluation.get("overall_score")

        state["critic_relevance"] = evaluation.get("relevance")
        state["critic_faithfulness"] = evaluation.get("faithfulness")
        state["critic_completeness"] = evaluation.get("completeness")
        state["critic_score"] = overall_score
        state["critic_reasoning"] = evaluation.get("reasoning")
        state["critic_suggestions"] = suggestions
        state["critic_missing_terms"] = missing_terms
        state["critic_failed"] = failed
        state["critic_passed"] = (
            overall_score is not None
            and overall_score >= CRITIC_PASS_THRESHOLD
            and not failed
        )

        self._last_result = {
            "relevance": evaluation.get("relevance"),
            "faithfulness": evaluation.get("faithfulness"),
            "completeness": evaluation.get("completeness"),
            "overall_score": overall_score,
            "reasoning": evaluation.get("reasoning"),
            "suggestions": suggestions,
            "missing_terms": missing_terms,
            "failed": failed,
        }

        state.setdefault("reasoning_log", []).append(self._last_result | {
            "agent": self.name,
            "step": "critique",
        })

        return state

    def _get_fallback_evaluation(self, reason: str) -> Dict[str, Any]:
        return {
            "relevance": None,
            "faithfulness": None,
            "completeness": None,
            "overall_score": None,
            "reasoning": f"Evaluation failed: {reason}",
            "suggestions": [],
            "missing_terms": [],
            "failed": True,
        }

    def _build_prompt(
        self,
        question: str,
        answer: str,
        chunks: List[Any],
        terminology_entries: List[Any],
        subject: str,
    ) -> str:
        context_section = self._format_chunks(chunks)
        terminology_section = self._format_terminology(terminology_entries)
        scoring_rules = self._build_scoring_rules(subject)

        return _PROMPT_TEMPLATE.format(
            question=question,
            answer=answer,
            context_section=context_section,
            terminology_section=terminology_section,
            subject=subject,
            scoring_rules=scoring_rules,
            few_shot=_UNIVERSAL_FEWSHOT,
        )

    def _build_scoring_rules(self, subject: str) -> str:
        rules = [_COMMON_RULES]
        if subject in _SUBJECT_RULES:
            rules.append(_SUBJECT_RULES[subject])
        return "\n".join(rules)

    def _format_chunks(self, chunks: List[Any]) -> str:
        if not chunks:
            return "(no context was retrieved)\n"
        lines = []
        for i, chunk in enumerate(chunks, 1):
            text = chunk.get("text", "") if isinstance(chunk, dict) else getattr(chunk, "text", "")
            title = chunk.get("title", "") if isinstance(chunk, dict) else getattr(chunk, "title", "") or ""
            label = f"Chunk {i}" + (f" ({title})" if title else "")
            lines.append(f"{label}:\n{text}")
        return "\n\n".join(lines) + "\n"

    def _format_terminology(self, entries: List[Any]) -> str:
        if not entries:
            return ""
        lines = ["Terminology (prefer these terms in the answer):"]
        for entry in entries:
            if isinstance(entry, dict):
                term = entry.get("term", "")
                definition = entry.get("definition", "")
            else:
                term = getattr(entry, "term", "")
                definition = getattr(entry, "definition", "")
            if term and definition:
                lines.append(f"  - {term}: {definition}")
        return "\n".join(lines) + "\n"

    @staticmethod
    def _extract_json_spans(text: str) -> List[str]:
        spans: List[str] = []
        i = 0
        n = len(text)
        while i < n:
            if text[i] != "{":
                i += 1
                continue
            depth = 0
            in_string = False
            escaped = False
            j = i
            while j < n:
                ch = text[j]
                if escaped:
                    escaped = False
                elif ch == "\\":
                    escaped = True
                elif ch == '"':
                    in_string = not in_string
                elif not in_string:
                    if ch == "{":
                        depth += 1
                    elif ch == "}":
                        depth -= 1
                        if depth == 0:
                            spans.append(text[i : j + 1])
                            break
                j += 1
            i = j + 1
        return spans

    def _parse_json_response(self, response: str) -> Dict[str, Any]:
        candidates = self._extract_json_spans(response)
        if not candidates:
            logger.error("No JSON object found in response")
            return self._fallback_extract(response)

        parsed_candidates: List[Dict[str, Any]] = []
        for candidate in candidates:
            try:
                parsed = json.loads(candidate)
            except json.JSONDecodeError:
                continue
            if not isinstance(parsed, dict):
                continue
            parsed.pop("failed", None)
            parsed_candidates.append(parsed)

        if not parsed_candidates:
            logger.error("All JSON candidates failed to parse")
            return self._fallback_extract(response)

        for parsed in parsed_candidates:
            if _EXPECTED_KEYS.issubset(parsed.keys()):
                return parsed

        logger.warning("No candidate had all expected keys; using first parsed object")
        return parsed_candidates[0]

    def _fallback_extract(self, text: str) -> Dict[str, Any]:
        score_match = re.search(r"SCORE:\s*([\d.]+)", text, re.IGNORECASE)
        relevance_match = re.search(r"RELEVANCE:\s*([\d.]+)", text, re.IGNORECASE)
        faithfulness_match = re.search(r"FAITHFULNESS:\s*([\d.]+)", text, re.IGNORECASE)
        completeness_match = re.search(r"COMPLETENESS:\s*([\d.]+)", text, re.IGNORECASE)

        return {
            "overall_score": float(score_match.group(1)) if score_match else None,
            "relevance": float(relevance_match.group(1)) if relevance_match else None,
            "faithfulness": float(faithfulness_match.group(1)) if faithfulness_match else None,
            "completeness": float(completeness_match.group(1)) if completeness_match else None,
            "reasoning": text[:200],
            "suggestions": [],
            "missing_terms": [],
            "failed": True,
        }
