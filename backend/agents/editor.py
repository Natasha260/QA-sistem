# backend/agents/editor.py
"""LLM-редактор: улучшает ответ с опорой на контекст из учебника."""
import logging
from typing import Any, Dict, List, Optional

from backend.agents.base import Agent
from backend.utils.llm_processing import clean_translation

logger = logging.getLogger(__name__)

_PROMPT_TEMPLATE = """You are an academic answer editor. Improve the answer using ONLY the context below.

Question: {question}

Context passages from the textbook:
{context}

Current answer: {answer}

Issues to fix:
{issues}

Instructions:
- Keep every fact grounded in the context. Do NOT add facts absent from it.
- Fix completeness and clarity; use the textbook terminology.
- If the answer is wrong, correct it using the context.
- Return ONLY the corrected answer text (no JSON, no quotes, no explanations).
- NO-CHANGE is better than a broken change. If nothing needs fixing, return the original answer.

Corrected answer:"""


class Editor(Agent):
    hf_model_setting = "HF_EDITOR_MODEL"
    ollama_model_setting = "EDITOR_MODEL"

    def __init__(self, model_name: Optional[str] = None, engine: Optional[str] = None):
        super().__init__(name="Editor", engine=engine)
        if self.engine == "huggingface":
            self.model_name = model_name or self._hf_model_name()
            from backend.utils.hf_llm import get_hf_llm
            self.hf_llm = get_hf_llm(self.model_name)
        else:
            self.model_name = (
                model_name or self._ollama_model_name() or self.settings.LLM_MODEL
            ).replace("ollama/", "")
            self.hf_llm = None
        self.temperature = 0.3

    async def process(self, state: Dict[str, Any]) -> Dict[str, Any]:
        logger.info("[%s] Starting...", self.name)

        # Пропуск если Critic упал
        if state.get("critic_failed"):
            logger.info("[%s] Critic failed, skipping edit.", self.name)
            state["editor_changed"] = False
            state["editor_failed"] = False
            state.setdefault("reasoning_log", []).append({
                "agent": self.name,
                "step": "editing",
                "changed": False,
                "note": "Skipped: critic_failed",
            })
            return state

        answer = state.get("answer", "") or state.get("translation", "")
        errors = state.get("validator_errors", []) or []
        warnings = state.get("validator_warnings", []) or []
        suggestions = state.get("critic_suggestions", []) or []
        question = state.get("question", "") or state.get("source_text", "")
        chunks = state.get("chunks_used", []) or []

        # Пропуск если нет проблем
        if not errors and not warnings and not suggestions:
            state["editor_changed"] = False
            state["editor_failed"] = False
            state.setdefault("reasoning_log", []).append({
                "agent": self.name,
                "step": "editing",
                "changed": False,
                "note": "No issues to fix",
            })
            return state

        prompt = self._build_prompt(
            question=question,
            answer=answer,
            chunks=chunks,
            errors=errors,
            warnings=warnings,
            suggestions=suggestions,
        )

        editor_failed = False
        try:
            raw_corrected = await self._call_llm(prompt)
            corrected = clean_translation(raw_corrected)
        except Exception as e:
            logger.error("[%s] LLM call failed: %s", self.name, e, exc_info=True)
            corrected = answer
            editor_failed = True

        if not corrected or not corrected.strip():
            logger.warning("[%s] Empty correction, keeping original.", self.name)
            corrected = answer
            editor_failed = True

        changed = corrected != answer

        state["editor_failed"] = editor_failed
        state["editor_changed"] = changed
        state["editor_original"] = answer
        state["editor_corrected"] = corrected
        state["editor_iterations"] = state.get("editor_iterations", 0) + 1

        if changed:
            state["answer"] = corrected

        self._last_result = {
            "changed": changed,
            "failed": editor_failed,
            "original": answer,
            "corrected": corrected,
        }

        reasoning_entry = {
            "agent": self.name,
            "step": "editing",
            "original": answer,
            "corrected": corrected,
            "changed": changed,
            "failed": editor_failed,
        }

        state.setdefault("reasoning_log", []).append(reasoning_entry)

        return state

    @staticmethod
    def _build_prompt(
        question: str,
        answer: str,
        chunks: List[Any],
        errors: List[str],
        warnings: List[str],
        suggestions: List[str],
    ) -> str:
        issues = Editor._format_issues(errors, warnings, suggestions)
        context = Editor._format_context(chunks)
        return _PROMPT_TEMPLATE.format(
            question=question,
            answer=answer,
            context=context,
            issues=issues,
        )

    @staticmethod
    def _format_context(chunks: List[Any]) -> str:
        if not chunks:
            return "(no context)"
        lines = []
        for i, chunk in enumerate(chunks, 1):
            text = chunk.get("text", "") if isinstance(chunk, dict) else getattr(chunk, "text", "")
            lines.append(f"Chunk {i}:\n{text}")
        return "\n\n".join(lines)

    @staticmethod
    def _format_issues(
        errors: List[str],
        warnings: List[str],
        suggestions: List[str],
    ) -> str:
        lines = []
        if errors:
            lines.append("ERRORS (must fix):")
            for i, e in enumerate(errors, 1):
                lines.append(f"  {i}. {e}")
        if warnings:
            lines.append("WARNINGS (should fix):")
            for i, w in enumerate(warnings, 1):
                lines.append(f"  {i}. {w}")
        if suggestions:
            lines.append("SUGGESTIONS (apply if improves quality):")
            for i, s in enumerate(suggestions, 1):
                lines.append(f"  {i}. {s}")
        if not lines:
            return "No issues found."
        return "\n".join(lines)
