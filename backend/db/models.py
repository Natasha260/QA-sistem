# backend/db/models.py
import uuid

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Float,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.orm import declarative_base

Base = declarative_base()


class GlossaryDB(Base):
    __tablename__ = "terminology"

    id = Column(String, primary_key=True)
    source_lang = Column(String, nullable=False, default="ru")
    target_lang = Column(String, nullable=False, default="ru")
    source_text = Column(Text, nullable=False)  # термин
    target_text = Column(Text, nullable=False)  # определение / пояснение

    created_at = Column(
        DateTime(timezone=True),
        server_default=func.now(),
    )

    # Only (source_lang, target_lang) is indexed. ``source_text`` used to
    # be part of the index, but its length can exceed the Postgres B-tree
    # key limit (2704 bytes) and made INSERT fail on long terms. It did
    # not help any query either:
    #   - get_entries searches with ILIKE '%...%' (a B-tree is useless);
    #   - list_all filters by source_lang/target_lang.
    __table_args__ = (
        Index(
            "ix_glossaries_langs",
            "source_lang",
            "target_lang",
        ),
    )


class VerifiedQADB(Base):
    """Подтверждённая пара «вопрос → ответ».

    Сюда попадают исправления, подтверждённые редактором через HITL,
    а также пары, загруженные вручную. Используются как few-shot
    примеры в промпте при поиске похожих вопросов.
    """

    __tablename__ = "verified_qa"

    id = Column(String, primary_key=True)
    source_lang = Column(String, nullable=False, default="ru")
    target_lang = Column(String, nullable=False, default="ru")
    source_text = Column(Text, nullable=False)  # вопрос
    target_text = Column(Text, nullable=False)  # ответ

    created_at = Column(
        DateTime(timezone=True),
        server_default=func.now(),
    )

    __table_args__ = (
        Index(
            "ix_verified_qa_langs",
            "source_lang",
            "target_lang",
        ),
    )


class DocumentDB(Base):
    """Учебник / документ, загруженный администратором."""

    __tablename__ = "documents"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    title = Column(String, nullable=False)
    filename = Column(String, nullable=True)
    description = Column(Text, nullable=True)

    created_at = Column(
        DateTime(timezone=True),
        server_default=func.now(),
    )


class ChunkDB(Base):
    """Чанк документа после разбиения учебника."""

    __tablename__ = "chunks"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    document_id = Column(String, nullable=False, index=True)
    chunk_index = Column(Integer, nullable=False, default=0)
    text = Column(Text, nullable=False)

    created_at = Column(
        DateTime(timezone=True),
        server_default=func.now(),
    )


class UncertainExample(Base):
    """Неуверенный ответ — кандидат на проверку редактором (HITL).

    ``source_text`` — вопрос пользователя, ``machine_translation`` —
    черновой ответ системы, ``corrected_translation`` — подтверждённая
    правка. Имена колонок оставлены для совместимости миграций.
    """

    __tablename__ = "uncertain_examples"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    source_text = Column(Text, nullable=False)          # вопрос
    source_lang = Column(String, nullable=False, default="ru")
    target_lang = Column(String, nullable=False, default="ru")
    machine_translation = Column(Text, nullable=False)  # черновой ответ
    corrected_translation = Column(Text, nullable=True)  # подтверждённый ответ
    critic_score = Column(Float, nullable=True)
    validation_passed = Column(Boolean, default=False)
    status = Column(String, default="pending")

    created_at = Column(
        DateTime(timezone=True),
        server_default=func.now(),
    )
    confirmed_at = Column(
        DateTime(timezone=True),
        nullable=True,
    )

    __table_args__ = (
        # get_pending_examples / count_pending:
        #   WHERE status='pending' ORDER BY created_at DESC
        Index(
            "ix_uncertain_status_created_at",
            "status",
            "created_at",
        ),
        # Same queries with an optional (source_lang, target_lang) filter.
        Index(
            "ix_uncertain_langs_status_created_at",
            "source_lang",
            "target_lang",
            "status",
            "created_at",
        ),
    )


class TranslationCache(Base):
    """
    Persistent query cache.

    Cache key:
        (engine_used, eval_mode, model_used, question)

    - ``eval_mode`` separates baseline / chunks_hybrid / full and so on.
    - ``model_used`` separates runs that use different LLM models, so
      switching ``LLM_MODEL`` does not return stale answers.
    - All key columns except ``question`` are NOT NULL with the sentinel
      ``"unknown"``. SQL treats NULL as distinct in UNIQUE constraints,
      so a NULL-able key column would silently allow duplicate rows.
    """

    __tablename__ = "query_cache"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    source_text = Column(Text, nullable=False)  # вопрос
    source_lang = Column(String, nullable=False, default="unknown")
    target_lang = Column(String, nullable=False, default="unknown")
    translation = Column(Text, nullable=False)  # ответ
    model_used = Column(
        String,
        nullable=False,
        default="unknown",
        server_default="unknown",
    )
    engine_used = Column(
        String,
        nullable=False,
        default="unknown",
        server_default="unknown",
    )
    eval_mode = Column(
        String,
        nullable=False,
        default="unknown",
        server_default="unknown",
    )
    hits = Column(Integer, default=1, nullable=False)

    created_at = Column(
        DateTime(timezone=True),
        server_default=func.now(),
    )

    __table_args__ = (
        UniqueConstraint(
            "engine_used",
            "eval_mode",
            "model_used",
            "source_text",
            name="unique_query_engine_mode_model",
        ),
        # Standalone index on eval_mode. clear_cache() can filter by
        # eval_mode alone, and get_cached() always constrains on it.
        Index("ix_cache_eval_mode", "eval_mode"),
    )


class User(Base):
    __tablename__ = "users"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    email = Column(String, unique=True, nullable=False)
    username = Column(String, unique=True, nullable=False)
    hashed_password = Column(String, nullable=False)
    full_name = Column(String, nullable=True)
    role = Column(String, default="user")
    is_active = Column(Boolean, default=True)

    created_at = Column(
        DateTime(timezone=True),
        server_default=func.now(),
    )
    updated_at = Column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
    )

    __table_args__ = (
        Index("ix_users_email", "email"),
        Index("ix_users_username", "username"),
        Index("ix_users_role", "role"),
    )

    def __repr__(self):
        return f"<User {self.email} ({self.role})>"