# tests/unit/test_llm_client.py
"""Tests for call_ollama_with_model."""
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

import backend.utils.llm_client as lc
from backend.utils.llm_client import call_ollama_with_model


def make_client(response=None, exc=None):
    client = MagicMock()
    if exc is not None:
        client.post = AsyncMock(side_effect=exc)
    else:
        resp = MagicMock()
        resp.raise_for_status = MagicMock()
        resp.json = MagicMock(return_value=response)
        client.post = AsyncMock(return_value=resp)
    return client


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch):
    monkeypatch.setattr(lc.asyncio, "sleep", AsyncMock())


@pytest.mark.asyncio
async def test_success_returns_cleaned(monkeypatch):
    client = make_client(response={"response": "привет"})
    monkeypatch.setattr(lc, "get_ollama_client", lambda: client)
    result = await call_ollama_with_model("p", "qwen2.5:7b")
    assert result == "привет"
    payload = client.post.call_args.kwargs["json"]
    assert payload["model"] == "qwen2.5:7b"
    assert payload["options"]["seed"] == 42


@pytest.mark.asyncio
async def test_retry_then_success(monkeypatch):
    ok = MagicMock()
    ok.raise_for_status = MagicMock()
    ok.json = MagicMock(return_value={"response": "ok"})
    client = MagicMock()
    client.post = AsyncMock(
        side_effect=[httpx.ConnectError("down"), ok]
    )
    monkeypatch.setattr(lc, "get_ollama_client", lambda: client)
    result = await call_ollama_with_model("p", "m", max_retries=3, retry_delay=5.0)
    assert result == "ok"
    assert client.post.call_count == 2


@pytest.mark.asyncio
async def test_all_attempts_timeout_raises(monkeypatch):
    client = MagicMock()
    client.post = AsyncMock(side_effect=httpx.TimeoutException("t"))
    monkeypatch.setattr(lc, "get_ollama_client", lambda: client)
    with pytest.raises(httpx.TimeoutException):
        await call_ollama_with_model("p", "m", max_retries=3, retry_delay=5.0)
    assert client.post.call_count == 3


@pytest.mark.asyncio
async def test_http_status_error_raises(monkeypatch):
    client = MagicMock()
    client.post = AsyncMock(
        side_effect=httpx.HTTPStatusError(
            "500", request=MagicMock(), response=MagicMock()
        )
    )
    monkeypatch.setattr(lc, "get_ollama_client", lambda: client)
    with pytest.raises(httpx.HTTPStatusError):
        await call_ollama_with_model("p", "m", max_retries=3, retry_delay=5.0)


@pytest.mark.asyncio
async def test_empty_response_returns_empty(monkeypatch):
    client = make_client(response={"response": ""})
    monkeypatch.setattr(lc, "get_ollama_client", lambda: client)
    result = await call_ollama_with_model("p", "m")
    assert result == ""


@pytest.mark.asyncio
async def test_ollama_prefix_stripped(monkeypatch):
    client = make_client(response={"response": "привет"})
    monkeypatch.setattr(lc, "get_ollama_client", lambda: client)
    await call_ollama_with_model("p", "ollama/qwen2.5:7b")
    payload = client.post.call_args.kwargs["json"]
    assert payload["model"] == "qwen2.5:7b"


@pytest.mark.asyncio
async def test_seed_none_omitted(monkeypatch):
    client = make_client(response={"response": "x"})
    monkeypatch.setattr(lc, "get_ollama_client", lambda: client)
    await call_ollama_with_model("p", "m", seed=None)
    payload = client.post.call_args.kwargs["json"]
    assert "seed" not in payload["options"]


@pytest.mark.asyncio
async def test_clean_false_returns_raw(monkeypatch):
    client = make_client(response={"response": 'Translation: "привет"'})
    monkeypatch.setattr(lc, "get_ollama_client", lambda: client)
    result = await call_ollama_with_model("p", "m", clean=False)
    assert result == 'Translation: "привет"'
