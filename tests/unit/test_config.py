# tests/unit/test_config.py
"""
Юнит-тесты для backend.core.config.

ВАЖНО: каждый тест создаёт Settings с _env_file=None, чтобы изолировать
тесты от содержимого .env. Иначе тест упадёт, если студент забыл вернуть
PROJECT_NAME=RUNG после эксперимента в разделе 2.2.
"""

import pytest

from backend.core.config import Settings, get_settings


# ============================================================================
# Хелпер: чистый Settings без чтения .env
# ============================================================================
def make_settings(**overrides) -> Settings:
    """Создаёт Settings с отключённым .env и переопределениями."""
    return Settings(_env_file=None, **overrides)


class TestSettingsDefaults:
    """Проверка значений по умолчанию (без .env)."""

    def test_project_name_default(self):
        s = make_settings()
        assert s.PROJECT_NAME == "RUNG"

    def test_debug_default(self):
        s = make_settings()
        assert isinstance(s.DEBUG, bool)

    def test_translation_engine_default(self):
        s = make_settings()
        assert s.TRANSLATION_ENGINE in {"auto", "ollama", "huggingface"}

    def test_eval_mode_default(self):
        s = make_settings()
        assert s.EVAL_MODE in {"baseline", "glossary", "tm", "full"}


class TestAsyncDatabaseUrl:
    """Проверка конвертации DATABASE_URL в async-формат."""

    def test_sqlite_to_aiosqlite(self):
        s = make_settings(DATABASE_URL="sqlite:///./test.db")
        assert s.ASYNC_DATABASE_URL == "sqlite+aiosqlite:///./test.db"

    def test_postgresql_to_asyncpg(self):
        s = make_settings(DATABASE_URL="postgresql://user:pass@host:5432/db")
        assert s.ASYNC_DATABASE_URL == "postgresql+asyncpg://user:pass@host:5432/db"

    def test_already_async_unchanged(self):
        """Если URL уже с +asyncpg, ничего не меняем."""
        s = make_settings(DATABASE_URL="postgresql+asyncpg://user:pass@host:5432/db")
        assert s.ASYNC_DATABASE_URL == "postgresql+asyncpg://user:pass@host:5432/db"


class TestUrls:
    """Проверка сборки URL из компонентов."""

    def test_ollama_url(self):
        s = make_settings(OLLAMA_HOST="myhost", OLLAMA_PORT=9999)
        assert s.ollama_url == "http://myhost:9999/api/generate"

    def test_qdrant_url(self):
        s = make_settings(QDRANT_HOST="qdrant.local", QDRANT_PORT=6334)
        assert s.qdrant_url == "http://qdrant.local:6334"


class TestEnsembleModels:
    """Проверка парсинга ENSEMBLE_MODELS."""

    def test_two_models(self):
        s = make_settings(ENSEMBLE_MODELS="model1,model2")
        assert s.ensemble_models_list == ["model1", "model2"]

    def test_with_spaces(self):
        s = make_settings(ENSEMBLE_MODELS="model1 , model2 , model3")
        assert s.ensemble_models_list == ["model1", "model2", "model3"]

    def test_empty(self):
        s = make_settings(ENSEMBLE_MODELS="")
        assert s.ensemble_models_list == []


class TestValidation:
    """Проверка валидаторов."""

    def test_invalid_translation_engine_raises(self):
        with pytest.raises(ValueError, match="TRANSLATION_ENGINE"):
            make_settings(TRANSLATION_ENGINE="invalid-engine")

    def test_invalid_eval_mode_raises(self):
        with pytest.raises(ValueError, match="EVAL_MODE"):
            make_settings(EVAL_MODE="invalid-mode")

    def test_valid_translation_engine(self):
        s = make_settings(TRANSLATION_ENGINE="ollama")
        assert s.TRANSLATION_ENGINE == "ollama"

    def test_valid_eval_mode(self):
        s = make_settings(EVAL_MODE="baseline")
        assert s.EVAL_MODE == "baseline"


class TestGetSettingsSingleton:
    """Проверка, что get_settings кэширует результат."""

    def test_returns_same_object(self):
        s1 = get_settings()
        s2 = get_settings()
        assert s1 is s2

    def test_returns_settings_instance(self):
        assert isinstance(get_settings(), Settings)