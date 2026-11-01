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


class TestModes:
    """GET /docs — OpenAPI-схема должна содержать QA-эндпоинты."""

    def test_openapi_has_query_endpoints(self):
        response = client.get("/openapi.json")
        assert response.status_code == 200
        paths = response.json()["paths"]
        assert "/query" in paths
        assert "/query/batch" in paths
        assert "/query/ensemble" in paths
        assert "/admin/documents" in paths
        assert "/hitl/uncertain" in paths

    def test_no_translate_endpoints_left(self):
        response = client.get("/openapi.json")
        paths = response.json()["paths"]
        assert not any(p.startswith("/translate") for p in paths)