# backend/utils/text_normalization.py
import re

# Arabic diacritics (harakat) and Quranic annotation marks.
_ARABIC_DIACRITICS_RE = re.compile(
    r"[\u064B-\u0652\u0670\u06D6-\u06ED\u08F0-\u08F3]"
)


def strip_arabic_diacritics(text: str) -> str:
    """Remove Arabic diacritics from text."""
    if not text:
        return text
    return _ARABIC_DIACRITICS_RE.sub("", text)
