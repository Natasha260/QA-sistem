# tests/unit/test_health_ollama.py
"""Tests for _ollama_name_matches and check_ollama."""
from unittest.mock import AsyncMock, MagicMock

import pytest

from backend.api.routes.health import _ollama_name_matches, check_ollama


@pytest.mark.parametrize(
    "requested,available,expected",
    [
        ("qwen2.5:7b", "qwen2.5:7b", True),
        ("qwen2.5:7b", "qwen2.5:7b-instruct", False),
        ("qwen2.5", "qwen2.5:latest", True),
        ("qwen2.5:latest", "qwen2.5", True),
        ("qwen", "qwen2.5:7b", False),
        ("qwen2.5:7b", "qwen2.5", False),
    ],
)
def test_name_matches(requested, available, expected):
    assert _ollama_name_matches(requested, available) is expected


def _make_client(response=None, exc=None):
    client = MagicMock()
    if exc is not None:
        client.get = AsyncMock(side_effect=exc)
    else:
        resp = MagicMock()
        resp.raise_for_status = MagicMock()
        resp.json = MagicMock(return_value=response)
        client.get = AsyncMock(return_value=resp)
    return client


@pytest.mark.asyncio
async def test_check_ollama_model_present(monkeypatch):
    client = _make_client(
        response={"models": [{"name": "qwen2.5:7b"}, {"name": "llama3.1:8b"}]}
    )
    monkeypatch.setattr("backend.api.routes.health.get_ollama_client", lambda: client)
    ok, err = await check_ollama("http://x/api/generate", "qwen2.5:7b")
    assert ok is True and err is None


@pytest.mark.asyncio
async def test_check_ollama_model_missing(monkeypatch):
    client = _make_client(response={"models": [{"name": "llama3.1:8b"}]})
    monkeypatch.setattr("backend.api.routes.health.get_ollama_client", lambda: client)
    ok, err = await check_ollama("http://x/api/generate", "qwen2.5:7b")
    assert ok is False
    assert "not found" in err


@pytest.mark.asyncio
async def test_check_ollama_connect_error(monkeypatch):
    import httpx
    client = _make_client(exc=httpx.ConnectError("down"))
    monkeypatch.setattr("backend.api.routes.health.get_ollama_client", lambda: client)
    ok, err = await check_ollama("http://x/api/generate", "qwen2.5:7b")
    assert ok is False and err


@pytest.mark.asyncio
async def test_check_ollama_prefix_stripped(monkeypatch):
    client = _make_client(response={"models": [{"name": "qwen2.5:7b"}]})
    monkeypatch.setattr("backend.api.routes.health.get_ollama_client", lambda: client)
    ok, _ = await check_ollama("http://x/api/generate", "ollama/qwen2.5:7b")
    assert ok is True
