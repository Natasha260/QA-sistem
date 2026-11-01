# backend/utils/llm_client.py
import asyncio
import logging

import httpx

from backend.core.config import get_settings
from backend.utils.http_client import get_ollama_client
from backend.utils.llm_processing import clean_translation

logger = logging.getLogger(__name__)
settings = get_settings()

_DEFAULT_SEED = 42


async def call_ollama_with_model(
    prompt: str,
    model_name: str,
    temperature: float = 0.3,
    max_retries: int = 3,
    retry_delay: float = 5.0,
    clean: bool = True,
    seed: int = _DEFAULT_SEED,
) -> str:
    if model_name.startswith("ollama/"):
        model_name = model_name.replace("ollama/", "")

    options = {"temperature": temperature}
    if seed is not None:
        options["seed"] = seed

    payload = {
        "model": model_name,
        "prompt": prompt,
        "stream": False,
        "options": options,
    }

    client = get_ollama_client()

    for attempt in range(max_retries):
        try:
            response = await client.post(settings.ollama_url, json=payload)
            response.raise_for_status()
            data = response.json()
            raw = data.get("response", "")
            if not raw:
                logger.warning("Empty response from Ollama for model %s", model_name)
                return ""
            result = clean_translation(raw) if clean else raw
            logger.debug(
                "Ollama response from %s: '%s...'", model_name, result[:50]
            )
            return result
        except (
            httpx.TimeoutException,
            httpx.HTTPStatusError,
            httpx.ConnectError,
        ) as e:
            if attempt < max_retries - 1:
                logger.warning(
                    "Ollama error (attempt %d/%d) for model %s: %s. "
                    "Retrying in %ss...",
                    attempt + 1,
                    max_retries,
                    model_name,
                    e,
                    retry_delay,
                )
                await asyncio.sleep(retry_delay)
            else:
                logger.error(
                    "All %d attempts failed for model %s", max_retries, model_name
                )
                raise

    raise RuntimeError(
        f"Failed to get response from Ollama for model {model_name}"
    )
