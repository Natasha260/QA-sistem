# backend/utils/http_client.py
import asyncio
import logging
import threading
import weakref

import httpx

logger = logging.getLogger(__name__)

_DEFAULT_TIMEOUT = 300.0

_clients: "weakref.WeakKeyDictionary[asyncio.AbstractEventLoop, httpx.AsyncClient]" = (
    weakref.WeakKeyDictionary()
)
_lock = threading.Lock()


def get_ollama_client() -> httpx.AsyncClient:
    """
    Return an ``httpx.AsyncClient`` bound to the current event loop.

    One client per loop keeps the connection pool reusable within a
    single loop while avoiding sharing sockets across loops — FastAPI
    and the Celery worker each run their own.
    """
    loop = asyncio.get_running_loop()
    with _lock:
        client = _clients.get(loop)
        if client is None:
            client = httpx.AsyncClient(timeout=_DEFAULT_TIMEOUT)
            _clients[loop] = client
            logger.debug("Created httpx.AsyncClient for loop id=%s", id(loop))
        return client


async def close_ollama_clients() -> None:
    with _lock:
        clients = list(_clients.values())
        _clients.clear()
    for client in clients:
        try:
            await client.aclose()
        except Exception as e:
            logger.warning("Error closing httpx.AsyncClient: %s", e)
