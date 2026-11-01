# tests/integration/test_auth.py
"""
Интеграционные тесты для /auth/register, /auth/login, /auth/me (Этап 4.6).

Инфраструктура
--------------
* Каждый тест получает СВОЮ пустую SQLite in-memory базу.
* Приложение настоящее (backend.api.main.app), но:
  - зависимость get_async_session подменена через app.dependency_overrides
    на сессию к тестовой базе;
  - async_engine в main.py подменён на тестовый движок, поэтому lifespan
    при старте TestClient сам создаёт таблицы (create_all) именно в
    тестовой базе, а не в вашем rung.db.

Почему одной подмены достаточно и для deps.py: в session.py написано
``get_db = get_async_session`` — это ОДНА и та же функция, а
dependency_overrides ищет по объекту функции. Подменили одну —
подменились обе.

Почему StaticPool: у SQLite ":memory:" каждая новая connection — это
новая пустая база. StaticPool держит ровно одно соединение и отдаёт его
всем сессиям, поэтому таблицы, созданные в lifespan, видны в запросах.

Почему ``with TestClient(app)``: только в режиме контекст-менеджера
TestClient запускает lifespan и держит ОДИН event loop на весь тест.
Это важно, потому что aiosqlite-соединение привязано к своему loop.

Все тесты синхронные — асинхронность прячется внутри TestClient,
поэтому @pytest.mark.asyncio здесь не нужен (asyncio_mode = "strict"
нас не касается).
"""

import time

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import delete, update
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import StaticPool

import backend.api.main as main_module
from backend.api.main import app
from backend.db.models import User
from backend.db.session import get_async_session


TEST_DATABASE_URL = "sqlite+aiosqlite:///:memory:"

ALICE = {
    "email": "alice@example.com",
    "username": "alice",
    "password": "MyPass123",
}


# ============================================================================
# Фикстуры
# ============================================================================
@pytest.fixture
def test_sessionmaker(monkeypatch):
    """Свежая in-memory база на каждый тест."""
    engine = create_async_engine(
        TEST_DATABASE_URL,
        poolclass=StaticPool,
        connect_args={"check_same_thread": False},
    )
    # lifespan в main.py возьмёт этот движок для create_all и dispose
    monkeypatch.setattr(main_module, "async_engine", engine)
    return async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)


@pytest.fixture
def client(test_sessionmaker):
    async def override_get_async_session():
        async with test_sessionmaker() as session:
            yield session

    app.dependency_overrides[get_async_session] = override_get_async_session
    with TestClient(app) as c:  # здесь отрабатывает lifespan → create_all
        yield c
    app.dependency_overrides.clear()


@pytest.fixture
def run_sql(client, test_sessionmaker):
    """
    Выполнить SQLAlchemy-запрос к тестовой базе напрямую, минуя API.
    Нужно для сценариев "деактивировать юзера" и "удалить юзера".

    client.portal — это event loop, в котором живёт приложение внутри
    TestClient. Выполняем запрос в нём же, чтобы не трогать
    aiosqlite-соединение из чужого loop.
    """

    def _run(statement):
        async def _exec():
            async with test_sessionmaker() as session:
                await session.execute(statement)
                await session.commit()

        client.portal.call(_exec)

    return _run


def register(client, **overrides):
    payload = {**ALICE, **overrides}
    return client.post("/auth/register", json=payload)


def login(client, email=ALICE["email"], password=ALICE["password"]):
    return client.post("/auth/login", json={"email": email, "password": password})


def auth_header(token):
    return {"Authorization": f"Bearer {token}"}


# ============================================================================
# /auth/register
# ============================================================================
class TestRegister:
    def test_register_returns_201_and_token(self, client):
        response = register(client)
        assert response.status_code == 201
        data = response.json()
        assert data["access_token"]
        assert data["token_type"] == "bearer"
        assert data["role"] == "user"
        assert data["email"] == ALICE["email"]
        assert data["username"] == ALICE["username"]

    def test_register_duplicate_email_returns_409(self, client):
        register(client)
        response = register(client, username="alice2")
        assert response.status_code == 409
        assert response.json()["detail"] == "Email already registered"

    def test_register_duplicate_username_returns_409(self, client):
        register(client)
        response = register(client, email="alice2@example.com")
        assert response.status_code == 409
        assert response.json()["detail"] == "Username already taken"

    @pytest.mark.parametrize(
        "password, reason",
        [
            ("Short1", "короче 8 символов"),
            ("NoDigitsHere", "нет цифры"),
            ("nouppercase1", "нет заглавной"),
        ],
        ids=["too_short", "no_digit", "no_uppercase"],
    )
    def test_register_weak_password_returns_422(self, client, password, reason):
        response = register(client, password=password)
        assert response.status_code == 422, reason

    def test_register_invalid_email_returns_422(self, client):
        response = register(client, email="alice.example.com")
        assert response.status_code == 422


# ============================================================================
# /auth/login
# ============================================================================
class TestLogin:
    def test_login_returns_200_and_new_token(self, client):
        reg_token = register(client).json()["access_token"]

        # iat/exp в JWT хранятся с точностью до СЕКУНДЫ. Если логин случится
        # в ту же секунду, что и регистрация, токены совпадут байт в байт.
        time.sleep(1.1)

        response = login(client)
        assert response.status_code == 200
        login_token = response.json()["access_token"]
        assert login_token != reg_token

    def test_login_wrong_password_returns_401(self, client):
        register(client)
        response = login(client, password="WrongPass123")
        assert response.status_code == 401
        assert response.json()["detail"] == "Invalid credentials"

    def test_login_unknown_email_returns_401(self, client):
        response = login(client, email="nobody@example.com")
        assert response.status_code == 401
        assert response.json()["detail"] == "Invalid credentials"

    def test_login_inactive_user_returns_403(self, client, run_sql):
        register(client)
        run_sql(
            update(User).where(User.email == ALICE["email"]).values(is_active=False)
        )
        response = login(client)
        assert response.status_code == 403
        assert response.json()["detail"] == "Account is disabled"


# ============================================================================
# /auth/me
# ============================================================================
class TestMe:
    def test_me_without_token_returns_401(self, client):
        response = client.get("/auth/me")
        assert response.status_code == 401
        assert response.json()["detail"] == "Not authenticated"

    def test_me_with_valid_token_returns_profile(self, client):
        token = register(client).json()["access_token"]
        response = client.get("/auth/me", headers=auth_header(token))
        assert response.status_code == 200

        data = response.json()
        for field in ("id", "email", "username", "full_name", "role", "is_active"):
            assert field in data
        assert data["email"] == ALICE["email"]
        assert data["is_active"] is True
        # хеш пароля наружу не отдаётся ни под каким видом
        assert "hashed_password" not in data
        assert "password" not in data

    def test_me_for_deleted_user_returns_404(self, client, run_sql):
        token = register(client).json()["access_token"]
        run_sql(delete(User).where(User.email == ALICE["email"]))
        response = client.get("/auth/me", headers=auth_header(token))
        assert response.status_code == 404

    def test_me_for_inactive_user_returns_404(self, client, run_sql):
        # get_current_user ищет только is_active=True,
        # поэтому деактивированный юзер для /me "не найден"
        token = register(client).json()["access_token"]
        run_sql(
            update(User).where(User.email == ALICE["email"]).values(is_active=False)
        )
        response = client.get("/auth/me", headers=auth_header(token))
        assert response.status_code == 404
