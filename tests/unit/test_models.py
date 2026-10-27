import asyncio
import uuid

import pytest
import pytest_asyncio  # <-- Добавлено
from sqlalchemy import select
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from backend.db.models import GlossaryDB, TMDB, TranslationCache, UncertainExample, User, Base


# ВАЖНО: Используем pytest_asyncio.fixture вместо pytest.fixture для async функций
@pytest_asyncio.fixture(scope="function")
async def engine():
    """Fresh in-memory engine per test function."""
    url = "sqlite+aiosqlite:///:memory:"
    eng = create_async_engine(url, echo=False, future=True)

    # Создаём таблицы ДО yield, чтобы они были готовы к моменту использования
    async with eng.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    yield eng
    
    # Очистка после теста
    await eng.dispose()


@pytest_asyncio.fixture(scope="function")
async def session(engine):
    """Yield a fresh session for each test."""
    async_session = async_sessionmaker(engine, expire_on_commit=False)
    async with async_session() as s:
        yield s


class TestTablesCreated:
    # Эти тесты синхронные и НЕ используют фикстуру session — их можно оставить как есть
    def test_all_five_tables(self):
        expected = {
            "glossaries",
            "translation_memories",
            "uncertain_examples",
            "translation_cache",
            "users",
        }
        assert set(Base.metadata.tables.keys()) == expected

    def test_glossary_has_langs_index(self):
        idx_names = {i.name for i in GlossaryDB.__table__.indexes}
        assert "ix_glossaries_langs" in idx_names

    def test_tm_has_langs_index(self):
        idx_names = {i.name for i in TMDB.__table__.indexes}
        assert "ix_tm_langs" in idx_names

    def test_cache_has_eval_mode_index(self):
        idx_names = {i.name for i in TranslationCache.__table__.indexes}
        assert "ix_cache_eval_mode" in idx_names


class TestUserDefaults:
    @pytest.mark.asyncio
    async def test_user_insert(self, session):
        u = User(
            id=str(uuid.uuid4()),
            email="a@b.com",
            username="alice",
            hashed_password="hash",
        )
        session.add(u)
        await session.commit()

        found = (await session.execute(
            select(User).where(User.email == "a@b.com")
        )).scalar_one()
        assert found.username == "alice"
        assert found.role == "user"
        assert found.is_active is True


class TestTMDBRoundTrip:
    @pytest.mark.asyncio
    async def test_insert_and_select(self, session):
        entry = TMDB(
            id="en|ru|abcdef",
            source_lang="en",
            target_lang="ru",
            source_text="hello",
            target_text="привет",
        )
        session.add(entry)
        await session.commit()

        found = (await session.execute(
            select(TMDB).where(TMDB.id == "en|ru|abcdef")
        )).scalar_one()
        assert found.source_text == "hello"
        assert found.target_text == "привет"

    @pytest.mark.asyncio
    async def test_deterministic_id_is_primary_key(self, session):
        """Two inserts with the same id — the second fails on PK conflict."""
        first = TMDB(
            id="en|ru|same",
            source_lang="en", target_lang="ru",
            source_text="hello", target_text="привет",
        )
        session.add(first)
        await session.commit()

        second = TMDB(
            id="en|ru|same",
            source_lang="en", target_lang="ru",
            source_text="hi", target_text="здравствуй",
        )
        session.add(second)
        with pytest.raises(Exception):  # IntegrityError
            await session.commit()
        await session.rollback()


class TestCacheSentinel:
    @pytest.mark.asyncio
    async def test_cache_defaults_to_unknown(self, session):
        c = TranslationCache(
            source_text="hi",
            source_lang="en", target_lang="ru",
            translation="привет",
        )
        session.add(c)
        await session.commit()
        await session.refresh(c)

        assert c.engine_used == "unknown"
        assert c.eval_mode == "unknown"
        assert c.model_used == "unknown"
        assert c.hits == 1


class TestUncertainExample:
    @pytest.mark.asyncio
    async def test_pending_by_default(self, session):
        ex = UncertainExample(
            source_text="hi", source_lang="en", target_lang="ru",
            machine_translation="привет",
            critic_score=5.0,
            validation_passed=False,
        )  
        session.add(ex)
        await session.commit()
        await session.refresh(ex)

        assert ex.status == "pending"
        assert ex.corrected_translation is None
        assert ex.confirmed_at is None