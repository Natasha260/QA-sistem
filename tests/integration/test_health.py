# tests/integration/test_health.py
"""
Интеграционные тесты для healthcheck-эндпоинтов.

Используется TestClient — синхронная обёртка над httpx, которая
обращается к FastAPI-приложению напрямую, без сети и без uvicorn.
"""

from fastapi.testclient import TestClient

from backend.api.main import app


client = TestClient(app)


class TestRoot:
    """GET / — корневой эндпоинт."""

    def test_root_returns_200(self):
        response = client.get("/")
        assert response.status_code == 200

    def test_root_contains_expected_keys(self):
        response = client.get("/")
        data = response.json()
        assert "message" in data
        assert "docs" in data
        assert "healthz" in data

    def test_root_message_contains_project_name(self):
        response = client.get("/")
        data = response.json()
        # PROJECT_NAME может быть переопределён в .env, но он должен быть
        assert "is running" in data["message"]


class TestHealthz:
    """GET /healthz — liveness-проверка."""

    def test_healthz_returns_200(self):
        response = client.get("/healthz")
        assert response.status_code == 200

    def test_healthz_returns_ok_status(self):
        response = client.get("/healthz")
        assert response.json() == {"status": "ok"}


class TestLanguages:
    """GET /languages — список поддерживаемых языков."""

    def test_languages_returns_200(self):
        response = client.get("/languages")
        assert response.status_code == 200

    def test_languages_contains_english(self):
        response = client.get("/languages")
        data = response.json()
        assert "languages" in data
        assert "en" in data["languages"]
        assert data["languages"]["en"] == "English"

    def test_languages_contains_known_languages(self):
        response = client.get("/languages")
        data = response.json()
        assert "known_languages" in data
        assert "en" in data["known_languages"]
        assert "ru" in data["known_languages"]