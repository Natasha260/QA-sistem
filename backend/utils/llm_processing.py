# backend/utils/llm_processing.py
import json
import logging
import re

logger = logging.getLogger(__name__)

_TOKENIZER_ARTEFACT_RE = re.compile(
    r"(?:LBLBLB|YYZBB|CCWCC|XXZXX)[A-Z_]*",
    re.IGNORECASE,
)

_CJK_RE = re.compile(r"[\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff]")

_CJK_PUNCT_RE = re.compile(r"[\u3000-\u303F\uFF00-\uFFEF]")


def clean_translation(raw: str) -> str:
    if not raw:
        return ""

    original = raw

    # 1. Single JSON object
    try:
        data = json.loads(raw)
        if isinstance(data, dict):
            extracted = data.get("translation", data.get("text", raw))
            if isinstance(extracted, str):
                raw = extracted.strip()
            else:
                logger.warning("JSON value is not a string, keeping original response")
        else:
            logger.debug("JSON is not an object, keeping original response")
    except (json.JSONDecodeError, AttributeError):
        logger.debug("Response is not a single JSON object, trying other formats")

    # 2. ```json ... ``` fenced block
    json_match = re.search(r"```json\s*(\{.*?\})\s*```", raw, re.DOTALL)
    if json_match:
        try:
            data = json.loads(json_match.group(1))
            extracted = data.get("translation", data.get("text", raw))
            if isinstance(extracted, str):
                raw = extracted.strip()
        except (json.JSONDecodeError, AttributeError) as e:
            logger.debug("Failed to parse JSON block: %s", e)

    # 3. XML wrapper
    xml_match = re.search(r"<[^>]+>(.*?)</[^>]+>", raw, re.DOTALL)
    if xml_match:
        raw = xml_match.group(1).strip()

    # 4. Leading "Translation:" / "Перевод:" style prefixes
    prefixes = [
        r"^Translation\s*\([^)]*\)\s*:\s*",
        r"^Translation\s*:\s*",
        r"^Перевод\s*:\s*",
        r"^Ответ\s*:\s*",
        r"^Here is the translation\s*:\s*",
        r"^Result\s*:\s*",
    ]
    for p in prefixes:
        raw = re.sub(p, "", raw, flags=re.IGNORECASE)

    # 5. Meta lines ("Here is...", "Sure,...", "As an AI,...")
    lines = raw.split('\n')
    filtered_lines = []
    for line in lines:
        if re.match(
            r'^(I am|Note:|Here is|Let me|Sure|Of course|As an AI|I\'ll|I will|My response|The translation is)',
            line.strip(),
            re.IGNORECASE
        ):
            continue
        filtered_lines.append(line)
    raw = '\n'.join(filtered_lines).strip()

    # 6. Surrounding quotes
    raw = raw.strip(" \"'")

    # 7. Tokenizer artefacts (LBLBLB_YYZBB_... chains)
    if _TOKENIZER_ARTEFACT_RE.search(raw):
        logger.warning("Tokenizer artefact detected in output")
        raw = _TOKENIZER_ARTEFACT_RE.sub("", raw).strip()

    # 8. Leaked CJK characters in non-CJK text
    if _CJK_RE.search(raw):
        logger.warning("CJK characters detected in output")
        raw = _CJK_RE.sub("", raw).strip()

    # 9. Leaked CJK punctuation in non-CJK text
    if _CJK_PUNCT_RE.search(raw):
        logger.warning("CJK punctuation detected in output")
        raw = _CJK_PUNCT_RE.sub("", raw).strip()

    if not raw:
        logger.warning("Clean translation resulted in empty string, returning raw")
        return original.strip()

    return raw
