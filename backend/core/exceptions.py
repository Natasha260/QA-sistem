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
        self.code = type(self).__name__
        self.context = context or {}

    def to_dict(self) -> dict:
        return {
            "error": {
                "code": self.code,
                "message": str(self),
                "context": self.context,
            }
        }


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

class EngineError(ExternalServiceError):
    """Raised when a translation engine (Ollama / HuggingFace) fails."""
    def __init__(self, message: str, context: dict = None):
        super().__init__(message, service_name="engine", context=context)


class EngineUnavailableError(EngineError):
    """Raised when the requested engine is unavailable and fallback is disabled."""
    def __init__(self, message: str, context: dict = None):
        super().__init__(message, context=context)


class UnknownEvalModeError(ValidationError):
    """Raised when an unknown eval_mode is requested."""

    def __init__(self, message: str, context: dict = None):
        super().__init__(message, context=context)


class EngineFallbackTriggeredError(ExternalServiceError):
    """Raised when a strict engine request silently degraded to a fallback."""

    def __init__(self, message: str, context: dict = None):
        super().__init__(message, service_name="engine", context=context)


class ImportFormatError(ValidationError):
    """Raised when an uploaded file has an unsupported format."""

    def __init__(self, message: str, context: dict = None):
        super().__init__(message, context=context)


class UncertainExampleNotFoundError(NotFoundError):
    """Raised when a pending uncertain example is not found."""

    def __init__(self, message: str, context: dict = None):
        super().__init__(message, context=context)
