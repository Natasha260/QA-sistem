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
    """Base exception for RUNG service."""
    def __init__(self, message: str, status_code: int = 500, context: dict = None):
        super().__init__(message)
        self.status_code = status_code
        self.context = context or {}


class BaseAPIError(RUNGException):
    """Base exception for API errors."""
    pass


class ValidationError(BaseAPIError):
    """Raised when request validation fails."""
    def __init__(self, message: str, context: dict = None):
        super().__init__(message, status_code=400, context=context)


class AuthenticationError(BaseAPIError):
    """Raised when authentication fails."""
    def __init__(self, message: str, context: dict = None):
        super().__init__(message, status_code=401, context=context)


class PermissionDeniedError(BaseAPIError):
    """Raised when user lacks permission."""
    def __init__(self, message: str, context: dict = None):
        super().__init__(message, status_code=403, context=context)


class NotFoundError(BaseAPIError):
    """Raised when resource is not found."""
    def __init__(self, message: str, context: dict = None):
        super().__init__(message, status_code=404, context=context)


class ModelNotLoadedError(Exception):
    """Raised when a model is required but not loaded."""
    def __init__(self, message: str, context: dict = None):
        super().__init__(message)
        self.context = context or {}


class TranslationError(BaseAPIError):
    """Raised when translation fails."""
    def __init__(self, message: str, context: dict = None):
        super().__init__(message, status_code=500, context=context)


class RateLimitError(BaseAPIError):
    """Raised when rate limit is exceeded."""
    def __init__(self, message: str, context: dict = None):
        super().__init__(message, status_code=429, context=context)


class ExternalServiceError(BaseAPIError):
    """Raised when external service fails."""
    def __init__(self, message: str, service_name: str = None, context: dict = None):
        super().__init__(message, status_code=502, context=context)
        self.service_name = service_name