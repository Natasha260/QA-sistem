# backend/agents/base.py
import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

from backend.core.config import get_settings
from backend.core.exceptions import ModelNotLoadedError
from backend.utils.async_utils import retry
from backend.utils.http_client import get_ollama_client

logger = logging.getLogger(__name__)

_DEFAULT_SEED = 42


class BaseAgent(ABC):
    """
    Abstract base for every agent in the pipeline.

    Exposes the two methods the Supervisor drives every stage through:
    ``process(state)`` and ``get_last_result()``.
    """

    def __init__(self, name: Optional[str] = None):
        self.name = name or self.__class__.__name__
        self._last_result: Any = None

    @abstractmethod
    async def process(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Process the current state and return an updated state."""

    def get_last_result(self) -> Any:
        return self._last_result


class Agent(BaseAgent):
    """
    Base for LLM-backed agents.

    Subclasses declare which Settings fields hold their HF- and
    Ollama-model names via ``hf_model_setting`` / ``ollama_model_setting``.
    Setting either to ``None`` means the agent does not support the
    corresponding engine.

    The engine is resolved once in ``__init__`` and always normalizes to
    a concrete value: ``"ollama"`` or ``"huggingface"``.

    Every call to ``_call_llm`` pins ``seed=42``. The Editor runs at
    temperature 0.3, so without a fixed seed the same input produces a
    different correction on every run and eval modes that include the
    Editor are not reproducible.
    """

    hf_model_setting: Optional[str] = None
    ollama_model_setting: Optional[str] = None

    model_name: Optional[str] = None
    hf_llm: Any = None
    temperature: float = 0.0
    max_new_tokens: int = 512

    def __init__(self, name: Optional[str] = None, engine: Optional[str] = None):
        super().__init__(name=name)
        self.settings = get_settings()
        requested = engine or self.settings.AGENT_ENGINE
        self.engine = self._resolve_engine(requested)

    def _hf_model_name(self) -> Optional[str]:
        if not self.hf_model_setting:
            return None
        value = getattr(self.settings, self.hf_model_setting, None)
        if not value or not str(value).strip():
            return None
        return str(value).replace("ollama/", "")

    def _ollama_model_name(self) -> Optional[str]:
        if not self.ollama_model_setting:
            return None
        value = getattr(self.settings, self.ollama_model_setting, None)
        if not value or not str(value).strip():
            return None
        return str(value).replace("ollama/", "")

    def _resolve_engine(self, requested: str) -> str:
        requested = (requested or "").lower()

        if requested == "huggingface":
            if self._hf_model_name():
                return "huggingface"
            logger.warning(
                "[%s] engine=huggingface but %s is empty — falling back to ollama",
                self.name,
                self.hf_model_setting,
            )
            return "ollama"

        if requested == "auto":
            return "huggingface" if self._hf_model_name() else "ollama"

        if requested in ("", "ollama"):
            return "ollama"

        logger.warning("[%s] unknown engine='%s', using ollama", self.name, requested)
        return "ollama"

    @retry(max_attempts=3, delay=1.0)
    async def _call_llm(self, prompt: str) -> str:
        if self.engine == "huggingface":
            if self.hf_llm is None:
                raise ModelNotLoadedError(
                    f"[{self.name}] engine='huggingface' but hf_llm is not "
                    "initialized. Subclasses must set self.hf_llm in __init__.",
                    context={
                        "agent": self.name,
                        "engine": self.engine,
                        "hf_model_setting": self.hf_model_setting,
                    },
                )
            return await self.hf_llm.generate(
                prompt,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                seed=_DEFAULT_SEED,
            )

        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": self.temperature,
                "seed": _DEFAULT_SEED,
            },
        }
        client = get_ollama_client()
        response = await client.post(self.settings.ollama_url, json=payload)
        response.raise_for_status()
        data = response.json()
        return data.get("response", "").strip()
