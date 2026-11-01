# backend/utils/hf_llm.py
import asyncio
import logging
from typing import Dict, Optional

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
)

from backend.core.config import get_settings

logger = logging.getLogger(__name__)

_DEFAULT_SEED = 42


class HuggingFaceLLM:
    """Async wrapper around a Hugging Face model for text generation."""

    def __init__(self, model_name: str):
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self.model_type: Optional[str] = None
        self.device = self._pick_device()
        self._load_lock = asyncio.Lock()
        self._loaded = False

    @staticmethod
    def _pick_device() -> str:
        if torch.cuda.is_available():
            return "cuda"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def _pick_dtype(self) -> torch.dtype:
        if self.device == "cuda":
            return torch.float16
        return torch.float32

    async def _load_model(self) -> None:
        if self._loaded:
            return

        async with self._load_lock:
            if self._loaded:
                return

            settings = get_settings()
            logger.info(
                "Loading Hugging Face model '%s' on device=%s ...",
                self.model_name, self.device,
            )
            loop = asyncio.get_running_loop()

            def _load():
                tokenizer = AutoTokenizer.from_pretrained(
                    self.model_name,
                    token=settings.HF_TOKEN,
                    trust_remote_code=False,
                )

                try:
                    model = AutoModelForCausalLM.from_pretrained(
                        self.model_name,
                        token=settings.HF_TOKEN,
                        trust_remote_code=False,
                        torch_dtype=self._pick_dtype(),
                    )
                    return tokenizer, model, "causal"
                except (ValueError, OSError) as e:
                    logger.info(
                        "'%s' is not loadable as CausalLM "
                        "(%s: %s); trying Seq2Seq",
                        self.model_name, type(e).__name__, e,
                    )
                    model = AutoModelForSeq2SeqLM.from_pretrained(
                        self.model_name,
                        token=settings.HF_TOKEN,
                        trust_remote_code=False,
                        torch_dtype=self._pick_dtype(),
                    )
                    return tokenizer, model, "seq2seq"

            try:
                tokenizer, model, model_type = await loop.run_in_executor(
                    None, _load
                )
            except Exception:
                logger.exception(
                    "Failed to load model '%s'", self.model_name
                )
                raise

            model.eval()
            try:
                model.to(self.device)
            except Exception as e:
                logger.warning(
                    "Could not move '%s' to %s (%s); staying on CPU",
                    self.model_name, self.device, e,
                )
                self.device = "cpu"

            self.tokenizer = tokenizer
            self.model = model
            self.model_type = model_type
            self._loaded = True
            logger.info(
                "Model '%s' loaded (type=%s, device=%s)",
                self.model_name, model_type, self.device,
            )

    @staticmethod
    def _seed_everything(seed: int) -> None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    async def generate(
        self,
        prompt: str,
        max_new_tokens: int = 512,
        temperature: float = 0.0,
        seed: int = _DEFAULT_SEED,
    ) -> str:
        await self._load_model()
        loop = asyncio.get_running_loop()

        def _generate() -> str:
            if seed is not None:
                self._seed_everything(seed)

            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=1024,
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            generate_kwargs = {
                "max_new_tokens": max_new_tokens,
                "do_sample": temperature > 0.0,
                "pad_token_id": self.tokenizer.pad_token_id,
            }
            if temperature > 0.0:
                generate_kwargs["temperature"] = temperature

            with torch.inference_mode():
                outputs = self.model.generate(**inputs, **generate_kwargs)

            result = self.tokenizer.decode(
                outputs[0], skip_special_tokens=True
            )

            if self.model_type == "causal" and prompt in result:
                result = result.split(prompt)[-1].strip()

            return result

        return await loop.run_in_executor(None, _generate)


_instances: Dict[str, HuggingFaceLLM] = {}


def get_hf_llm(model_name: str) -> HuggingFaceLLM:
    instance = _instances.get(model_name)
    if instance is None:
        instance = HuggingFaceLLM(model_name)
        _instances[model_name] = instance
        logger.debug("Created HuggingFaceLLM singleton for '%s'", model_name)
    return instance


def unload_hf_llm(model_name: Optional[str] = None) -> None:
    global _instances
    if model_name is None:
        _instances.clear()
        logger.info("All HuggingFaceLLM instances cleared")
    else:
        _instances.pop(model_name, None)
        logger.info("HuggingFaceLLM instance for '%s' cleared", model_name)
