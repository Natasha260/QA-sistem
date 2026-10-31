"""Rule-based validator for translation results.

This agent does not call an LLM. It performs deterministic checks:
glossary usage, echo translation, placeholders, target-script match,
repetitions and leftover source text.
"""

import logging
import re
from typing import Any, Dict, List, Optional, Set

from backend.agents.base import BaseAgent
from backend.data.embeddings import embedding_model

logger = logging.getLogger(__name__)

MIN_TERM_LEN = 2
MIN_TERM_LEN_FOR_SEMANTIC = 5
MIN_CONTENT_WORD_LEN = 4
GLOSSARY_SEMANTIC_THRESHOLD = 0.7
_MAX_LEFTOVER_REPORTED = 5
_MIN_TARGET_SCRIPT_RATIO = 0.5

_COMMON_STOPWORDS: Set[str] = {
    # English
    "the", "and", "for", "are", "but", "not", "you", "all", "any", "can",
    "had", "her", "was", "one", "our", "out", "day", "get", "has", "him",
    "his", "how", "its", "new", "now", "old", "see", "two", "way", "who",
    "did", "she", "use", "that", "this", "with", "have", "from", "they",
    "will", "would", "there", "their", "what", "about", "which", "when",
    "your", "said", "each", "make", "like", "just", "over", "such", "take",
    "than", "them", "well", "were",
    # Russian
    "и", "в", "во", "не", "что", "он", "на", "я", "с", "со", "как", "а",
    "то", "все", "она", "так", "его", "но", "да", "ты", "к", "у", "же",
    "вы", "за", "бы", "по", "только", "ее", "мне", "было", "вот", "от",
    "меня", "еще", "нет", "о", "из", "ему", "теперь", "когда", "даже",
    "ну", "вдруг", "ли", "если", "уже", "или", "ни", "быть", "был", "него",
    "до", "вас", "нибудь", "опять", "уж", "вам", "ведь", "там", "потом",
    "себя", "ничего", "ей", "они", "тут", "где", "есть", "надо", "ней",
    "для", "мы", "тебя", "их", "чем", "была", "сам", "чтоб", "без",
    "будто", "чего", "раз", "тоже", "себе", "под", "будет", "ж", "тогда",
    "кто", "этот", "того", "потому", "этого", "какой", "совсем", "ним",
    "здесь", "этом", "один", "почти", "мой", "тем", "чтобы", "нее",
    "сейчас", "были", "куда", "зачем", "всех", "никогда", "можно", "при",
    "наконец", "два", "об", "другой", "хоть", "после", "над", "больше",
    "тот", "через", "эти", "нас", "про", "всего", "них", "какая", "много",
    "разве", "три", "эту", "моя", "впрочем", "хорошо", "свою", "этой",
    "перед", "иногда", "лучше", "чуть", "том", "нельзя", "такой", "им",
    "более", "всегда", "конечно", "всю", "между",
}

_SCRIPT_PATTERNS = {
    "cyrillic": re.compile(r"[\u0400-\u04FF]"),
    "latin": re.compile(r"[A-Za-z]"),
    "arabic": re.compile(r"[\u0600-\u06FF]"),
}

_LANG_SCRIPT = {
    "ru": "cyrillic",
    "uk": "cyrillic",
    "be": "cyrillic",
    "bg": "cyrillic",
    "sr": "cyrillic",
    "mk": "cyrillic",
    "en": "latin",
    "de": "latin",
    "fr": "latin",
    "es": "latin",
    "it": "latin",
    "pt": "latin",
    "nl": "latin",
    "pl": "latin",
    "tr": "latin",
    "ar": "arabic",
    "fa": "arabic",
    "ur": "arabic",
}

_CJK_RE = re.compile(r"[\u4E00-\u9FFF\u3040-\u30FF\uAC00-\uD7AF]")
_PLACEHOLDER_RE = re.compile(r"\{[^{}]+\}|%[sdif]|<[^<>]+>|\[\w+\]")
_LETTER_RE = re.compile(r"[^\W\d_]+", re.UNICODE)


def _contains_word(text: str, term: str) -> bool:
    """Whole-word, case-insensitive match of ``term`` inside ``text``."""
    if not term or not text:
        return False
    pattern = r"(?<!\w)" + re.escape(term) + r"(?!\w)"
    return re.search(pattern, text, re.IGNORECASE) is not None


def _extract_content_words(text: str) -> Set[str]:
    """Return meaningful lowercase words, excluding stopwords/short words."""
    if not text:
        return set()
    words = re.findall(r"\w+", text.lower(), flags=re.UNICODE)
    return {
        word
        for word in words
        if len(word) >= MIN_CONTENT_WORD_LEN and word not in _COMMON_STOPWORDS
    }


def _glossary_pair(entry: Any) -> tuple[str, str]:
    """Extract ``(source_term, target_term)`` from a glossary entry."""
    if isinstance(entry, dict):
        source_term = entry.get("en") or entry.get("source") or ""
        target_term = entry.get("tgt") or entry.get("target") or ""
    else:
        source_term = getattr(entry, "en", "") or getattr(entry, "source", "") or ""
        target_term = getattr(entry, "tgt", "") or getattr(entry, "target", "") or ""
    return str(source_term).strip(), str(target_term).strip()


def _semantic_match(
    term: str,
    translation: str,
    threshold: float = GLOSSARY_SEMANTIC_THRESHOLD,
) -> bool:
    """Check semantic similarity between a glossary term and translation."""
    if not term or not translation:
        return False

    try:
        vectors = embedding_model.encode([term, translation])
        term_vec = vectors[0]
        translation_vec = vectors[1]

        dot_product = sum(a * b for a, b in zip(term_vec, translation_vec))
        term_norm = sum(a * a for a in term_vec) ** 0.5
        translation_norm = sum(b * b for b in translation_vec) ** 0.5

        if term_norm == 0 or translation_norm == 0:
            return False

        cosine = dot_product / (term_norm * translation_norm)
        return cosine >= threshold
    except Exception as exc:
        logger.warning("Semantic match failed for term '%s': %s", term, exc)
        return False


class Validator(BaseAgent):
    """Rule-based validator. Does NOT call an LLM."""

    def __init__(self, engine: Optional[str] = None):
        super().__init__(name="Validator")

    async def process(self, state: Dict[str, Any]) -> Dict[str, Any]:
        logger.info("[%s] Starting...", self.name)

        source_text = state.get("source_text", "") or ""
        translation = state.get("translation", "") or ""
        target_lang = state.get("target_lang", "") or ""
        glossary_entries = state.get("glossary_used", []) or []

        errors: List[str] = []
        warnings: List[str] = []

        errors.extend(
            self._check_glossary(source_text, translation, glossary_entries)
        )

        if not translation.strip():
            errors.append("Translation is empty")

        errors.extend(self._check_echo(source_text, translation))
        errors.extend(self._check_placeholders(source_text, translation))
        errors.extend(self._check_script(target_lang, translation))

        warnings.extend(self._check_repetitions(translation))
        warnings.extend(self._check_leftover_source(source_text, translation))

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

    def _check_glossary(
        self,
        source_text: str,
        translation: str,
        glossary_entries: List[Any],
    ) -> List[str]:
        errors: List[str] = []

        for entry in glossary_entries:
            source_term, target_term = _glossary_pair(entry)

            if not source_term or not target_term:
                continue
            if len(source_term) < MIN_TERM_LEN:
                continue

            source_in_text = _contains_word(source_text, source_term)
            if not source_in_text:
                continue

            if _contains_word(translation, target_term):
                continue

            if (
                len(source_term) >= MIN_TERM_LEN_FOR_SEMANTIC
                and _semantic_match(source_term, translation)
            ):
                continue

            errors.append(
                f"Glossary term '{source_term}' was not translated "
                f"as '{target_term}'"
            )

        return errors

    def _check_echo(self, source_text: str, translation: str) -> List[str]:
        source_clean = re.sub(r"\s+", " ", source_text.strip()).lower()
        translation_clean = re.sub(r"\s+", " ", translation.strip()).lower()

        if source_clean and source_clean == translation_clean:
            return ["Translation is identical to the source text"]
        return []

    def _check_placeholders(
        self, source_text: str, translation: str
    ) -> List[str]:
        source_placeholders = _PLACEHOLDER_RE.findall(source_text)
        translation_placeholders = _PLACEHOLDER_RE.findall(translation)

        missing = [
            placeholder
            for placeholder in source_placeholders
            if placeholder not in translation_placeholders
        ]

        if missing:
            return [
                f"Missing placeholders in translation: {', '.join(missing)}"
            ]
        return []

    def _check_script(
        self, target_lang: str, translation: str
    ) -> List[str]:
        expected_script = _LANG_SCRIPT.get(target_lang.lower())
        if not expected_script or not translation.strip():
            return []

        letters = _LETTER_RE.findall(translation)
        if not letters:
            return []

        target_pattern = _SCRIPT_PATTERNS[expected_script]
        matched = sum(bool(target_pattern.search(word)) for word in letters)
        ratio = matched / len(letters)

        if ratio < _MIN_TARGET_SCRIPT_RATIO:
            return [
                f"Translation script does not match target language "
                f"'{target_lang}'"
            ]
        return []

    def _check_repetitions(self, translation: str) -> List[str]:
        warnings: List[str] = []
        words = re.findall(r"\w+", translation.lower(), flags=re.UNICODE)

        for index in range(len(words) - 2):
            if words[index] == words[index + 1] == words[index + 2]:
                warnings.append(
                    f"Repeated word detected: '{words[index]}'"
                )
                break

        return warnings

    def _check_leftover_source(
        self, source_text: str, translation: str
    ) -> List[str]:
        warnings: List[str] = []
        source_words = sorted(
            _extract_content_words(source_text),
            key=len,
            reverse=True,
        )

        leftovers = [
            word
            for word in source_words
            if len(word) >= MIN_TERM_LEN_FOR_SEMANTIC
            and _contains_word(translation, word)
        ]

        if leftovers:
            shown = leftovers[:_MAX_LEFTOVER_REPORTED]
            warnings.append(
                "Possible untranslated source words: "
                + ", ".join(shown)
            )

        return warnings