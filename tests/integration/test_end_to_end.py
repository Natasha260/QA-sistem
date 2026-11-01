# tests/integration/test_end_to_end.py
"""
End-to-end test: register, login, ask a question, cache hit.

Runs only against a live environment (Ollama + services). Set
RUNG_E2E=1 to enable; otherwise the test is skipped because the
pipeline requires a reachable Ollama server.
"""
import os

import pytest
from fastapi.testclient import TestClient

from backend.api.main import app

pytestmark = pytest.mark.skipif(
    os.environ.get("RUNG_E2E") != "1",
    reason="E2E requires a live environment; set RUNG_E2E=1 to enable",
)


@pytest.fixture
def client():
    return TestClient(app)


def test_full_user_journey(client):
    # 1. Register
    reg = client.post("/auth/register", json={
        "email": "e2e@test.local",
        "username": "e2e_user",
        "password": "TestPass1234",
    })
    assert reg.status_code in (200, 201)
    token = reg.json()["access_token"]
    headers = {"Authorization": f"Bearer {token}"}

    # 2. Login (same credentials)
    login = client.post("/auth/login", json={
        "email": "e2e@test.local",
        "password": "TestPass1234",
    })
    assert login.status_code == 200

    # 3. Ask a question (full mode: hybrid retrieval + agents)
    r1 = client.post("/query", json={
        "question": "Что такое сила?",
    }, headers=headers)
    assert r1.status_code == 200
    body1 = r1.json()
    assert body1["answer"]
    assert body1.get("confidence") is not None

    # 4. Second identical question: should hit cache
    r2 = client.post("/query", json={
        "question": "Что такое сила?",
    }, headers=headers)
    assert r2.status_code == 200
    body2 = r2.json()
    assert body2.get("cached") is True
    assert body2["answer"] == body1["answer"]

    # 5. HITL queue reachable (editor/admin role required; 403 for user role)
    uncertain = client.get("/hitl/uncertain", headers=headers)
    assert uncertain.status_code in (200, 403)
