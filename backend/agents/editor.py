# backend/agents/editor.py
import logging
from typing import Any, Dict, List, Optional

from backend.agents.base import Agent
from backend.core.config import get_settings
from backend.utils.llm_processing import clean_translation

logger = logging.getLogger(__name__)

_PROMPT_TEMPLATE = """You are a professional translation editor. Improve the translation based on the issues below.

Source ({source_lang}): {source_text}
Current translation ({target_lang}): {translation}

Issues to fix:
{issues}

Instructions:
- Apply suggestions literally when possible.
- Fix glossary terms, grammar, fluency, and terminology errors.
- Keep the meaning identical to the source.
- Return ONLY the corrected translation text (no JSON, no quotes, no explanations).
- NO-CHANGE is better than a broken change. If nothing needs fixing, return the original translation.

Examples of Incorrect Edits:
- Adding words not in source: "He has a car" → "У него есть машина" (added "у него") — WRONG.
- Changing tense: "She went" → "Она идёт" (past → present) — WRONG.
- Breaking grammar: "The book is on the table" → "Книга на стол" (missing verb) — WRONG.

Corrected translation:"""


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

        translation = state.get("translation", "")
        errors = state.get("validator_errors", []) or []
        warnings = state.get("validator_warnings", []) or []
        suggestions = state.get("critic_suggestions", []) or []
        source_text = state.get("source_text", "")
        source_lang = state.get("source_lang", "en")
        target_lang = state.get("target_lang", "ru")

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
            source_text=source_text,
            translation=translation,
            source_lang=source_lang,
            target_lang=target_lang,
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
            corrected = translation
            editor_failed = True

        if not corrected or not corrected.strip():
            logger.warning("[%s] Empty correction, keeping original.", self.name)
            corrected = translation
            editor_failed = True

        changed = corrected != translation

        state["editor_failed"] = editor_failed
        state["editor_changed"] = changed
        state["editor_original"] = translation
        state["editor_corrected"] = corrected
        state["editor_iterations"] = state.get("editor_iterations", 0) + 1

        if changed:
            state["translation"] = corrected

        self._last_result = {
            "changed": changed,
            "failed": editor_failed,
            "original": translation,
            "corrected": corrected,
        }

        reasoning_entry = {
            "agent": self.name,
            "step": "editing",
            "original": translation,
            "corrected": corrected,
            "changed": changed,
            "failed": editor_failed,
        }

        state.setdefault("reasoning_log", []).append(reasoning_entry)

        return state

    @staticmethod
    def _build_prompt(
        source_text: str,
        translation: str,
        source_lang: str,
        target_lang: str,
        errors: List[str],
        warnings: List[str],
        suggestions: List[str],
    ) -> str:
        issues = Editor._format_issues(errors, warnings, suggestions)
        return _PROMPT_TEMPLATE.format(
            source_lang=source_lang,
            source_text=source_text,
            target_lang=target_lang,
            translation=translation,
            issues=issues,
        )

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

    # === ВРЕМЕННАЯ ЗАГЛУШКА (удалить в Этапе 8) ===
    async def _call_llm(self, prompt: str) -> str:
        # Возвращает перевод без изменений (имитация "no_change")
        translation_match = prompt.split("Current translation")[1].split("\n")[0] if "Current translation" in prompt else ""
        return translation_match.strip() if translation_match else "Заглушка: перевод не изменён"