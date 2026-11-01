# tests/unit/test_tasks.py
import pytest
from unittest.mock import patch

from backend.worker.celery_app import celery_app


@pytest.fixture(autouse=True)
def eager_mode():
    """Run tasks synchronously, no Redis needed."""
    celery_app.conf.task_always_eager = True
    celery_app.conf.task_eager_propagates = True
    yield
    celery_app.conf.task_always_eager = False


def test_index_chunks_calls_load(monkeypatch):
    from backend.worker import tasks
    called = []

    async def mock_load():
        called.append(True)

    monkeypatch.setattr(tasks.documents_manager, "load_all_into_cache", mock_load)
    result = tasks.index_chunks.apply().get()
    assert result["status"] == "success"
    assert called == [True]


def test_add_terminology_bulk(monkeypatch):
    from backend.worker import tasks
    seen = []

    async def mock_add(entry):
        seen.append(entry["term"])

    monkeypatch.setattr(tasks.glossary_manager, "add_entry", mock_add)
    result = tasks.add_terminology_bulk.apply(args=[[
        {"term": "сила", "definition": "векторная величина", "subject": "physics"},
        {"term": "скорость", "definition": "производная координаты", "subject": "physics"},
    ]]).get()
    assert result == {"added": 2}
    assert seen == ["сила", "скорость"]


def test_add_qa_pairs_bulk(monkeypatch):
    from backend.worker import tasks

    async def mock_add_many(entries):
        return len(entries)

    monkeypatch.setattr(tasks.verified_qa_manager, "add_many", mock_add_many)
    result = tasks.add_qa_pairs_bulk.apply(args=[[
        {"question": "Что такое сила?", "answer": "Мера взаимодействия тел."},
    ]]).get()
    assert result["added"] == 1


def test_run_evaluation_async():
    from backend.worker.tasks import run_evaluation_async
    result = run_evaluation_async.apply(args=[
        ["Что такое сила?", "Что такое сила?"],
        ["Мера взаимодействия тел.", "Мера взаимодействия тел."],
        ["Мера взаимодействия тел.", "Мера взаимодействия тел."],
    ]).get()
    assert result["n"] == 2
    assert result["exact_match"] == 100.0
    assert result["f1"] == 100.0
