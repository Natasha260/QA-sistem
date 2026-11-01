import asyncio
import uuid

import pytest
import pytest_asyncio
from sqlalchemy import select
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from backend.db.models import (
    ChunkDB, DocumentDB, GlossaryDB, TranslationCache,
    UncertainExample, User, VerifiedQADB, Base,
)


# ВАЖНО: Используем pytest_asyncio.fixture вместо pytest.fixture для async функций
@pytest_asyncio.fixture(scope="function")
async def engine():
    """Fresh in-memory engine per test function."""
    url = "sqlite+aiosqlite:///:memory:"
    eng = create_async_engine(url, echo=False, future=True)

    async with eng.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    yield eng

    await eng.dispose()


@pytest_asyncio.fixture(scope="function")
async def session(engine):
    """Yield a fresh session for each test."""
    async_session = async_sessionmaker(engine, expire_on_commit=False)
    async with async_session() as s:
        yield s


class TestTablesCreated:
    def test_all_tables(self):
        expected = {
            "terminology",
            "verified_qa",
            "documents",
            "chunks",
            "uncertain_examples",
            "query_cache",
            "users",
        }
        assert set(Base.metadata.tables.keys()) == expected

    def test_terminology_has_subject_index(self):
        idx_names = {i.name for i in GlossaryDB.__table__.indexes}
        assert "ix_glossaries_langs" in idx_names

    def test_verified_qa_has_subject_index(self):
        idx_names = {i.name for i in VerifiedQADB.__table__.indexes}
        assert "ix_verified_qa_langs" in idx_names

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


class TestDocumentAndChunks:
    @pytest.mark.asyncio
    async def test_document_with_chunks(self, session):
        doc = DocumentDB(id="d1", title="Физика. Глава 1")
        session.add(doc)
        session.add_all([
            ChunkDB(id="c1", document_id="d1", chunk_index=0, text="Механика…"),
            ChunkDB(id="c2", document_id="d1", chunk_index=1, text="Термодинамика…"),
        ])
        await session.commit()

        found = (await session.execute(
            select(ChunkDB).where(ChunkDB.document_id == "d1")
            .order_by(ChunkDB.chunk_index)
        )).scalars().all()
        assert len(found) == 2
        assert found[0].text == "Механика…"


class TestVerifiedQA:
    @pytest.mark.asyncio
    async def test_round_trip(self, session):
        pair = VerifiedQADB(
            id="qa|abc",
            source_lang="ru", target_lang="ru",
            source_text="Что такое ускорение?",
            target_text="Быстрота изменения скорости.",
        )
        session.add(pair)
        await session.commit()

        found = (await session.execute(
            select(VerifiedQADB).where(VerifiedQADB.id == "qa|abc")
        )).scalar_one()
        assert found.source_text == "Что такое ускорение?"
        assert found.target_text == "Быстрота изменения скорости."


class TestCacheSentinel:
    @pytest.mark.asyncio
    async def test_cache_defaults_to_unknown(self, session):
        c = TranslationCache(
            source_text="Что такое сила?",
            translation="Мера взаимодействия тел.",
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
            source_text="Что такое сила?",
            machine_translation="Не знаю.",
            critic_score=5.0,
            validation_passed=False,
        )
        session.add(ex)
        await session.commit()
        await session.refresh(ex)

        assert ex.status == "pending"
        assert ex.corrected_translation is None
        assert ex.confirmed_at is None
