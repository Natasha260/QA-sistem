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
    __tablename__ = "glossaries"

    id = Column(String, primary_key=True)
    source_lang = Column(String, nullable=False)
    target_lang = Column(String, nullable=False)
    source_text = Column(Text, nullable=False)
    target_text = Column(Text, nullable=False)

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


class TMDB(Base):
    __tablename__ = "translation_memories"

    id = Column(String, primary_key=True)
    source_lang = Column(String, nullable=False)
    target_lang = Column(String, nullable=False)
    source_text = Column(Text, nullable=False)
    target_text = Column(Text, nullable=False)

    created_at = Column(
        DateTime(timezone=True),
        server_default=func.now(),
    )

    # See the comment above. ``source_text`` is not part of the index
    # because it can hold tens of kilobytes (hadiths, book chapters) and
    # would make INSERT fail with:
    #   index row size 4320 exceeds btree version 4 maximum 2704
    # for index "ix_tm_langs_text"
    __table_args__ = (
        Index(
            "ix_tm_langs",
            "source_lang",
            "target_lang",
        ),
    )


class UncertainExample(Base):
    __tablename__ = "uncertain_examples"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    source_text = Column(Text, nullable=False)
    source_lang = Column(String, nullable=False)
    target_lang = Column(String, nullable=False)
    machine_translation = Column(Text, nullable=False)
    corrected_translation = Column(Text, nullable=True)
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
    Persistent translation cache.

    Cache key:
        (source_lang, target_lang, engine_used, eval_mode,
         model_used, source_text)

    - ``eval_mode`` separates baseline / glossary / tm / tm_hybrid / full.
    - ``model_used`` separates runs that use different LLM or HF models,
      so switching ``LLM_MODEL`` / ``HF_TRANSLATION_MODEL`` does not
      return stale translations.
    - All key columns except ``source_text`` are NOT NULL with the
      sentinel ``"unknown"``. SQL treats NULL as distinct in UNIQUE
      constraints, so a NULL-able key column would silently allow
      duplicate rows.
    """

    __tablename__ = "translation_cache"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    source_text = Column(Text, nullable=False)
    source_lang = Column(String, nullable=False)
    target_lang = Column(String, nullable=False)
    translation = Column(Text, nullable=False)
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
            "source_lang",
            "target_lang",
            "engine_used",
            "eval_mode",
            "model_used",
            "source_text",
            name="unique_translation_engine_mode_model",
        ),
        # Standalone index on eval_mode. clear_cache() can filter by
        # eval_mode alone (no language filter), and get_cached() always
        # constrains on it. The UniqueConstraint above has eval_mode in
        # the middle, so a prefix scan cannot use it.
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