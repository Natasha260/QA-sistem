# backend/agents/critic.py
import json
import logging
import re
from typing import Any, Dict, List, Optional

import httpx

from backend.agents.base import Agent
from backend.core.config import get_settings
from backend.core.constants import CRITIC_PASS_THRESHOLD

logger = logging.getLogger(__name__)

_EXPECTED_KEYS = frozenset({"adequacy", "fluency", "terminology", "overall_score"})
_BULLET_PREFIX_RE = r"^\s*(?:[-\*•]|\d+[\.\)])(?:\s+|$)"

# === ПРАВИЛА ОЦЕНКИ ===
_COMMON_RULES = """
1. Кальки и дословный перевод → fluency ≤ 5.
2. Терминология: точное соответствие глоссарию обязательно.
3. Максимум 8/10, если есть сомнения в адекватности.
4. Разные аспекты не могут отличаться больше чем на 2 балла.
5. overall_score ≤ 8, если есть замечания.
6. fluency ≤ 5 при грамматических ошибках.
7. fluency = 0 при утечке CJK, meta-тексте, пустом ответе.
"""

_DOMAIN_RULES = {
    "medical": "8. Медицинские термины требуют точности — ошибка в дозировке = adequacy = 0.",
}

_LANGUAGE_RULES = {
    "ru": "A. Кириллица обязательна. Латиница в словах → fluency = 0.",
    "tt": "A. Татарская кириллица (ә, ө, ү, җ, ң, һ). Замена на русские буквы → terminology ≤ 5.",
    "ba": "A. Башкирская кириллица (ә, ө, ү, ҫ, ң, һ). Проверка суффиксов -лар/-ләр.",
    "tg": "A. Таджикская кириллица. Проверка и→ӣ, у→ӯ, қ, ғ, ҷ, ҳ.",
    "fa": "A. Персидский алфавит. Проверка ی/ک vs Arabic ي/ك.",
    "tr": "A. Турецкая латиница. Проверка ı/i, ğ, ş, ç, ö, ü.",
    "kk": "A. Казахская кириллица. Проверка ә, ғ, қ, ң, ө, ұ, ү, һ, і.",
    "ky": "A. Киргизская кириллица. Проверка ө, ү, ң, җ.",
    "udm": "A. Удмуртская кириллица. Проверка ӟ, ӝ, ӥ, ӧ, ӱ.",
    "os": "A. Осетинская кириллица. Проверка ӕ, ӕ.",
    "av": "A. Аварская кириллица. Проверка Ӏ, ӏ.",
    "ar": "A. Арабский алфавит. Проверка точек и диакритики.",
    "bg": "A. Болгарская кириллица. Проверка ъ, ь, ю, я.",
    "id": "A. Индонезийская латиница. Проверка эй vs ай.",
    "ms": "A. Малайская латиница. Проверка окончаний -kan/-i.",
    "uz": "A. Узбекская латиница. Проверка o' vs ó, g' vs ғ.",
}

# === FEW-SHOT ПРИМЕРЫ ===
_UNIVERSAL_FEWSHOT = """
Example 1 (Good):
{"adequacy": 9, "fluency": 9, "terminology": 9, "overall_score": 9, "reasoning": "Excellent translation", "suggestions": [], "missing_terms": []}

Example 2 (Terminology error):
{"adequacy": 7, "fluency": 9, "terminology": 4, "overall_score": 6, "reasoning": "Term not from glossary", "suggestions": ["Use 'артериальное давление' not 'давление крови'"], "missing_terms": ["blood pressure"]}

Example 3 (Fluency error):
{"adequacy": 9, "fluency": 4, "terminology": 9, "overall_score": 6, "reasoning": "Word order is unnatural", "suggestions": ["Reorder: 'быстрый коричневый лис' → 'быстрая коричневая лиса'"], "missing_terms": []}

Example 4 (CJK leak):
{"adequacy": 0, "fluency": 0, "terminology": 0, "overall_score": 0, "reasoning": "Chinese characters detected", "suggestions": [], "missing_terms": []}
"""

_LANGUAGE_FEWSHOT = {
    "ru": "Example (Russian): Check Cyrillic only. Latin in words = fluency 0.",
    "tt": "Example (Tatar): Check ә, ө, ү, җ, ң, һ. Russian letters = terminology 5.",
}

_PROMPT_TEMPLATE = """Evaluate the translation from {source_lang} to {target_lang}.

Source: {source}
Translation: {translation}
{glossary_section}
{reference_section}
Domain: {domain}
Audience: {audience}

Apply these rules:
{scoring_rules}

Few-shot examples:
{few_shot}

Return ONLY valid JSON with keys: adequacy, fluency, terminology, overall_score (each 0-10), reasoning, suggestions (list), missing_terms (list).
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
        source_text = state.get("source_text", "")
        translation = state.get("translation", "")
        source_lang = state.get("source_lang", "en")
        target_lang = state.get("target_lang", "ru")
        glossary_entries = state.get("glossary_used", []) or []
        reference = state.get("reference")
        domain = state.get("domain", "general")
        audience = state.get("audience", "general")

        prompt = self._build_prompt(
            source=source_text,
            translation=translation,
            source_lang=source_lang,
            target_lang=target_lang,
            glossary_entries=glossary_entries,
            reference=reference,
            domain=domain,
            audience=audience,
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

        state["critic_adequacy"] = evaluation.get("adequacy")
        state["critic_fluency"] = evaluation.get("fluency")
        state["critic_terminology"] = evaluation.get("terminology")
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
            "adequacy": evaluation.get("adequacy"),
            "fluency": evaluation.get("fluency"),
            "terminology": evaluation.get("terminology"),
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
            "adequacy": None,
            "fluency": None,
            "terminology": None,
            "overall_score": None,
            "reasoning": f"Evaluation failed: {reason}",
            "suggestions": [],
            "missing_terms": [],
            "failed": True,
        }

    def _build_prompt(
        self,
        source: str,
        translation: str,
        source_lang: str,
        target_lang: str,
        glossary_entries: List[Any],
        reference: Optional[str],
        domain: str,
        audience: str,
    ) -> str:
        glossary_section = self._format_glossary(glossary_entries)
        reference_section = f"Reference translation: {reference}\n" if reference else ""
        scoring_rules = self._build_scoring_rules(domain, target_lang)
        few_shot = self._build_few_shot(target_lang, domain)

        return _PROMPT_TEMPLATE.format(
            source_lang=source_lang,
            target_lang=target_lang,
            source=source,
            translation=translation,
            glossary_section=glossary_section,
            reference_section=reference_section,
            domain=domain,
            audience=audience,
            scoring_rules=scoring_rules,
            few_shot=few_shot,
        )

    def _build_scoring_rules(self, domain: str, target_lang: str) -> str:
        rules = [_COMMON_RULES]
        if domain in _DOMAIN_RULES:
            rules.append(_DOMAIN_RULES[domain])
        if target_lang in _LANGUAGE_RULES:
            rules.append(_LANGUAGE_RULES[target_lang])
        return "\n".join(rules)

    def _build_few_shot(self, target_lang: str, domain: str) -> str:
        shots = [_UNIVERSAL_FEWSHOT]
        if target_lang in _LANGUAGE_FEWSHOT:
            shots.append(_LANGUAGE_FEWSHOT[target_lang])
        return "\n\n".join(shots)

    def _format_glossary(self, glossary_entries: List[Any]) -> str:
        if not glossary_entries:
            return "Glossary: None\n"
        lines = ["Glossary terms (must be used):"]
        for entry in glossary_entries:
            if isinstance(entry, dict):
                src = entry.get("en", "")
                tgt = entry.get("tgt", "")
            else:
                src = getattr(entry, "en", "")
                tgt = getattr(entry, "tgt", "")
            if src and tgt:
                lines.append(f"  - {src} → {tgt}")
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
        adequacy_match = re.search(r"ADEQUACY:\s*([\d.]+)", text, re.IGNORECASE)
        fluency_match = re.search(r"FLUENCY:\s*([\d.]+)", text, re.IGNORECASE)
        terminology_match = re.search(r"TERMINOLOGY:\s*([\d.]+)", text, re.IGNORECASE)

        return {
            "overall_score": float(score_match.group(1)) if score_match else None,
            "adequacy": float(adequacy_match.group(1)) if adequacy_match else None,
            "fluency": float(fluency_match.group(1)) if fluency_match else None,
            "terminology": float(terminology_match.group(1)) if terminology_match else None,
            "reasoning": text[:200],
            "suggestions": [],
            "missing_terms": [],
            "failed": True,
        }

    # === ВРЕМЕННАЯ ЗАГЛУШКА (удалить в Этапе 8) ===
    async def _call_llm(self, prompt: str) -> str:
        return '{"adequacy": 5.0, "fluency": 5.0, "terminology": 5.0, "overall_score": 5.0, "reasoning": "Stub LLM: real call in Этап 8.", "suggestions": [], "missing_terms": []}'