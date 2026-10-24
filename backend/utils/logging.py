# backend/utils/logging.py
"""
Настройка логирования для проекта RUNG.

Особенности:
- цветной вывод в консоль (если поддерживается TTY);
- запись в файл с ротацией (10 МБ × 10 файлов);
- отдельный файл для ошибок уровня ERROR и выше;
- приглушение логов сторонних библиотек;
- опциональный JSON-формат через переменную LOG_JSON=true.
"""

import logging
import logging.config
import os
import sys
from datetime import datetime
from pathlib import Path

from backend.core.config import get_settings


# ============================================================================
# Директория и имена файлов
# ============================================================================
LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)

_DATE = datetime.now().strftime("%Y%m%d")
LOG_FILE = LOG_DIR / f"rung_{_DATE}.log"
ERROR_FILE = LOG_DIR / f"errors_{_DATE}.log"


# ============================================================================
# Настройки по умолчанию
# ============================================================================
settings = get_settings()
DEFAULT_LEVEL = logging.DEBUG if settings.DEBUG else logging.INFO


# ============================================================================
# Кастомный форматтер с цветами
# ============================================================================
class CustomFormatter(logging.Formatter):
    """
    Форматтер с ANSI-цветами для консоли.

    Цвет выбирается по уровню логирования:
    - DEBUG / INFO  — серый
    - WARNING       — жёлтый
    - ERROR         — красный
    - CRITICAL      — жирный красный

    Если вывод идёт не в TTY (например, в файл), цвета не добавляются.
    """

    grey = "\x1b[38;20m"
    yellow = "\x1b[33;20m"
    red = "\x1b[31;20m"
    bold_red = "\x1b[31;1m"
    reset = "\x1b[0m"

    def __init__(self, fmt: str = None, datefmt: str = None, use_colors: bool = True):
        # dictConfig передаёт "format" по старому API, а logging.Formatter
        # в Python 3.10+ хочет "fmt". Ловим оба варианта.
        super().__init__(fmt=fmt, datefmt=datefmt)
        self.use_colors = use_colors and sys.stdout.isatty()

    def format(self, record: logging.LogRecord) -> str:
        if self.use_colors:
            levelno = record.levelno
            if levelno >= logging.CRITICAL:
                prefix = self.bold_red
            elif levelno >= logging.ERROR:
                prefix = self.red
            elif levelno >= logging.WARNING:
                prefix = self.yellow
            else:
                prefix = self.grey

            original_msg = super().format(record)
            return f"{prefix}{original_msg}{self.reset}"
        return super().format(record)


# ============================================================================
# Конфигурация logging через dictConfig
# ============================================================================
LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        # Формат для консоли: время — имя логгера — уровень — сообщение
        "default": {
            "()": CustomFormatter,
            "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            "datefmt": "%Y-%m-%d %H:%M:%S",
        },
        # Формат для файла: JSON (используется, если LOG_JSON=true)
        "json": {
            "()": "pythonjsonlogger.json.JsonFormatter",
            "format": (
                "%(asctime)s %(name)s %(levelname)s %(message)s "
                "%(module)s %(funcName)s %(lineno)d "
                "%(process)d %(thread)d"
            ),
            "datefmt": "%Y-%m-%dT%H:%M:%S%z", 
         },
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "level": "DEBUG",
            "formatter": "default",
            "stream": sys.stdout,
        },
        "file": {
            "class": "logging.handlers.RotatingFileHandler",
            "level": "DEBUG",
            "formatter": "json" if os.getenv("LOG_JSON", "false").lower() == "true" else "default",
            "filename": str(LOG_FILE),
            "maxBytes": 10_485_760,  # 10 МБ
            "backupCount": 10,
            "encoding": "utf-8",
        },
        "error_file": {
            "class": "logging.handlers.RotatingFileHandler",
            "level": "ERROR",
            "formatter": "json" if os.getenv("LOG_JSON", "false").lower() == "true" else "default",
            "filename": str(ERROR_FILE),
            "maxBytes": 10_485_760,
            "backupCount": 5,
            "encoding": "utf-8",
        },
    },
    "loggers": {
        # Приглушаем шумные библиотеки
        "sqlalchemy.engine": {"level": "WARNING", "propagate": False},
        "httpx": {"level": "WARNING", "propagate": False},
        "httpcore": {"level": "WARNING", "propagate": False},
        "urllib3": {"level": "WARNING", "propagate": False},
        "transformers": {"level": "WARNING", "propagate": False},
        "sentence_transformers": {"level": "WARNING", "propagate": False},
        "huggingface_hub": {"level": "WARNING", "propagate": False},
        "passlib": {"level": "WARNING", "propagate": False},
    },
    "root": {
        "level": "DEBUG",
        "handlers": ["console", "file", "error_file"],
    },
}


# ============================================================================
# Публичное API
# ============================================================================
def setup_logging(level: int = None) -> logging.Logger:
    """
    Инициализирует логирование.

    Вызывается один раз при старте приложения.
    Если level не задан, используется DEFAULT_LEVEL (DEBUG или INFO).
    """
    if level is None:
        level = DEFAULT_LEVEL

    logging.config.dictConfig(LOGGING_CONFIG)
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    return root_logger


def get_logger(name: str) -> logging.Logger:
    """Возвращает логгер с указанным именем."""
    return logging.getLogger(name)


# Инициализация при импорте модуля
logger = setup_logging()

__all__ = ["setup_logging", "get_logger", "logger"]