# tests/unit/test_security.py
"""
Unit-тесты для backend/utils/security.py (Этап 4.6).

Проверяем две независимые части:
  1. Хеширование паролей (bcrypt) — get_password_hash / verify_password.
  2. JWT — create_access_token / decode_token.

Все тесты синхронные: в security.py нет async-функций, поэтому
pytest-asyncio и @pytest.mark.asyncio здесь не нужны.
"""

import importlib
import time
from datetime import timedelta

import pytest
from fastapi import HTTPException

import backend.utils.security as security
from backend.core.config import get_settings
from backend.utils.security import (
    create_access_token,
    decode_token,
    get_password_hash,
    verify_password,
)


PASSWORD = "MyPass123"


# ============================================================================
# 1. Хеширование паролей
# ============================================================================
class TestPasswordHashing:
    def test_hash_has_bcrypt_prefix(self):
        # $2b$ — идентификатор алгоритма bcrypt в строке хеша
        assert get_password_hash(PASSWORD).startswith("$2b$")

    def test_hash_does_not_contain_plain_password(self):
        assert PASSWORD not in get_password_hash(PASSWORD)

    def test_verify_correct_password(self):
        hashed = get_password_hash(PASSWORD)
        assert verify_password(PASSWORD, hashed) is True

    def test_verify_wrong_password(self):
        hashed = get_password_hash(PASSWORD)
        assert verify_password("WrongPass123", hashed) is False

    def test_verify_against_hash_of_other_password(self):
        # Хеш вполне валидный, но от другого пароля → False, без исключения
        other_hash = get_password_hash("OtherPass456")
        assert verify_password(PASSWORD, other_hash) is False

    def test_verify_with_corrupted_hash_returns_false(self):
        # bcrypt.checkpw бросает ValueError("Invalid salt"),
        # verify_password обязан его поймать и вернуть False
        assert verify_password(PASSWORD, "not-a-bcrypt-hash") is False

    def test_hash_73_bytes_raises(self):
        with pytest.raises(ValueError):
            get_password_hash("a" * 73)

    def test_hash_72_bytes_ok(self):
        hashed = get_password_hash("a" * 72)
        assert verify_password("a" * 72, hashed) is True

    def test_hash_cyrillic_36_chars_72_bytes_ok(self):
        # Кириллица в UTF-8 — 2 байта на символ: 36 * 2 = 72 байта
        password = "п" * 36
        assert len(password.encode("utf-8")) == 72
        hashed = get_password_hash(password)
        assert verify_password(password, hashed) is True

    def test_hash_cyrillic_37_chars_74_bytes_raises(self):
        # 37 символов, но 74 байта — лимит bcrypt считается в БАЙТАХ, не в символах
        password = "п" * 37
        assert len(password.encode("utf-8")) == 74
        with pytest.raises(ValueError):
            get_password_hash(password)

    def test_same_password_gives_different_hashes(self):
        # bcrypt генерирует новую соль на каждый вызов
        assert get_password_hash(PASSWORD) != get_password_hash(PASSWORD)


# ============================================================================
# 2. JWT
# ============================================================================
class TestJWT:
    def test_token_is_three_part_string(self):
        token = create_access_token({"sub": "x", "role": "user"})
        assert isinstance(token, str)
        # header.payload.signature
        assert token.count(".") == 2

    def test_decode_returns_original_claims_plus_iat_exp(self):
        token = create_access_token({"sub": "x", "role": "user"})
        payload = decode_token(token)
        assert payload["sub"] == "x"
        assert payload["role"] == "user"
        assert "iat" in payload
        assert "exp" in payload
        assert payload["exp"] > payload["iat"]

    def test_expired_token_raises_401(self):
        token = create_access_token(
            {"sub": "x", "role": "user"},
            expires_delta=timedelta(seconds=1),
        )
        time.sleep(2)
        with pytest.raises(HTTPException) as exc_info:
            decode_token(token)
        assert exc_info.value.status_code == 401
        assert exc_info.value.detail == "Token expired"

    def test_claim_of_wrong_type_raises_type_error(self):
        with pytest.raises(TypeError):
            create_access_token({"sub": ["list"]})

    def test_decode_garbage_raises_401(self):
        with pytest.raises(HTTPException) as exc_info:
            decode_token("not.a.jwt")
        assert exc_info.value.status_code == 401
        assert exc_info.value.detail == "Invalid token"


# ============================================================================
# 3. Токен, подписанный другим секретом
# ============================================================================
@pytest.fixture
def reload_security(monkeypatch):
    """
    ЛОВУШКА, из-за которой нужна эта фикстура.

    SECRET_KEY в security.py — константа уровня модуля. Она вычисляется
    ОДИН раз, при первом импорте модуля. Поэтому просто
    monkeypatch.setenv("JWT_SECRET_KEY", ...) ничего не меняет.

    Причём перечитать надо ДВА слоя кеша:
      1. get_settings() обёрнута в @lru_cache → без cache_clear()
         вернётся старый объект Settings со старым секретом;
      2. сам модуль security → importlib.reload() заново выполняет его код
         и пересчитывает SECRET_KEY.

    После теста обязательно возвращаем всё как было, иначе остальные
    тесты (и deps.py, который импортировал decode_token) поедут
    с "чужим" секретом.
    """

    def _reload_with_secret(secret: str):
        monkeypatch.setenv("JWT_SECRET_KEY", secret)
        get_settings.cache_clear()
        return importlib.reload(security)

    yield _reload_with_secret

    # Откат: monkeypatch уже вернул переменную окружения,
    # сбрасываем кеш и перезагружаем модуль с исходным секретом.
    monkeypatch.undo()
    get_settings.cache_clear()
    importlib.reload(security)


class TestForeignSecret:
    def test_monkeypatch_setenv_alone_does_not_change_secret(self, monkeypatch):
        # Демонстрация ловушки: без reload константа остаётся прежней
        old_secret = security.SECRET_KEY
        monkeypatch.setenv("JWT_SECRET_KEY", "something-else")
        assert security.SECRET_KEY == old_secret

    def test_token_signed_with_other_secret_raises_401(self, reload_security):
        original_secret = security.SECRET_KEY

        # 1. Выпускаем токен "чужим" секретом
        foreign = reload_security(original_secret + "-foreign")
        foreign_token = foreign.create_access_token({"sub": "x", "role": "user"})

        # 2. Возвращаем исходный секрет и пытаемся проверить токен
        restored = reload_security(original_secret)
        assert restored.SECRET_KEY == original_secret

        with pytest.raises(HTTPException) as exc_info:
            restored.decode_token(foreign_token)
        assert exc_info.value.status_code == 401
        assert exc_info.value.detail == "Invalid token"
