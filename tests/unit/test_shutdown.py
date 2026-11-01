# tests/unit/test_shutdown.py
"""Tests for the Celery shutdown handler."""
import asyncio

import pytest


@pytest.fixture(autouse=True)
def clean_loop():
    from backend.worker import tasks
    tasks.close_worker_loop()
    yield
    tasks.close_worker_loop()


def test_shutdown_noop_when_no_loop(monkeypatch):
    from backend.worker import tasks, celery_app
    calls = []
    monkeypatch.setattr(celery_app, "close_ollama_clients", lambda: calls.append("clients"))
    celery_app._shutdown_worker_resources()
    assert calls == []
    assert tasks._worker_loop is None


def test_shutdown_closes_clients_then_loop(monkeypatch):
    from backend.worker import tasks, celery_app

    tasks._get_worker_loop()
    calls = []

    async def mock_close():
        calls.append("clients")

    monkeypatch.setattr(celery_app, "close_ollama_clients", mock_close)
    orig_close = tasks.close_worker_loop
    monkeypatch.setattr(
        tasks, "close_worker_loop", lambda: (calls.append("loop"), orig_close())
    )
    celery_app._shutdown_worker_resources()
    assert calls == ["clients", "loop"]
    assert tasks._worker_loop is None


def test_shutdown_closed_loop_noop(monkeypatch):
    from backend.worker import tasks, celery_app
    calls = []

    async def mock_close():
        calls.append("clients")

    monkeypatch.setattr(celery_app, "close_ollama_clients", mock_close)
    loop = asyncio.new_event_loop()
    loop.close()
    tasks._worker_loop = loop
    celery_app._shutdown_worker_resources()
    assert calls == []
    assert tasks._worker_loop is None
