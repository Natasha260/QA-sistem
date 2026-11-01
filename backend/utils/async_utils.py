# backend/utils/async_utils.py
import asyncio
import logging
from functools import wraps
from typing import Any, Callable

logger = logging.getLogger(__name__)


def retry(max_attempts: int = 3, delay: float = 1.0):
    """Decorator to retry an async function on exceptions."""
    if max_attempts < 1:
        raise ValueError("retry: max_attempts must be >= 1")

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs) -> Any:
            for attempt in range(max_attempts):
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    if attempt < max_attempts - 1:
                        logger.warning(
                            "Attempt %d/%d for %s failed: %s. "
                            "Retrying in %ss...",
                            attempt + 1,
                            max_attempts,
                            func.__name__,
                            e,
                            delay,
                        )
                        await asyncio.sleep(delay)
                    else:
                        logger.error(
                            "All %d attempts failed for %s",
                            max_attempts,
                            func.__name__,
                        )
                        raise

        return wrapper

    return decorator
