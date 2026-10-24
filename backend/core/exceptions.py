# backend/core/exceptions.py
"""
Собственные исключения проекта RUNG.

Иерархия:
    RUNGException (базовое)
    ├── TranslationError      — ошибка при переводе
    ├── RetrievalError        — ошибка при поиске (BM25/Qdrant)
    ├── ValidationError       — ошибка формальной проверки
    ├── LLMError              — ошибка при обращении к LLM
    └── ConfigurationError    — ошибка конфигурации
"""


class RUNGException(Exception):
    """Базовое исключение проекта. Все остальные наследуются от него."""
    pass


class TranslationError(RUNGException):
    """Ошибка при выполнении перевода (LLM вернула пустой результат, timeout, и т.п.)."""
    pass


class RetrievalError(RUNGException):
    """Ошибка при поиске по базе знаний (BM25 или Qdrant недоступны, индекс не построен)."""
    pass


class ValidationError(RUNGException):
    """Ошибка формальной проверки результата (минимальная длина, отсутствие термина и т.п.)."""
    pass


class LLMError(RUNGException):
    """Ошибка при обращении к LLM (Ollama или HuggingFace)."""
    pass


class ConfigurationError(RUNGException):
    """Ошибка конфигурации (не задан HF_TOKEN, некорректный DATABASE_URL и т.п.)."""
    pass