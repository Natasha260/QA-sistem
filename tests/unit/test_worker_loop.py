# tests/unit/test_worker_loop.py
"""Tests for the persistent worker event loop."""
import pytest


@pytest.fixture(autouse=True)
def clean_loop():
    from backend.worker import tasks
    tasks.close_worker_loop()
    yield
    tasks.close_worker_loop()


def test_persistent_loop_same_instance():
    from backend.worker import tasks
    loop1 = tasks._get_worker_loop()
    loop2 = tasks._get_worker_loop()
    assert loop1 is loop2
    assert not loop1.is_closed()


def test_close_creates_new_loop():
    from backend.worker import tasks
    loop1 = tasks._get_worker_loop()
    tasks.close_worker_loop()
    assert tasks._worker_loop is None
    loop2 = tasks._get_worker_loop()
    assert loop2 is not loop1


def test_close_idempotent():
    from backend.worker import tasks
    tasks.close_worker_loop()
    tasks.close_worker_loop()
    assert tasks._worker_loop is None
