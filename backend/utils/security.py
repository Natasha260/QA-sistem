# backend/utils/security.py
import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Optional

import bcrypt
import jwt
from fastapi import HTTPException

from backend.core.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

SECRET_KEY = settings.JWT_SECRET_KEY
ALGORITHM = settings.JWT_ALGORITHM
ACCESS_TOKEN_EXPIRE_MINUTES = settings.JWT_EXPIRATION_MINUTES

_BCRYPT_MAX_BYTES = 72
_CLAIM_TYPES = (str, int, bool)


def _encode_password(password: str) -> bytes:
    encoded = password.encode("utf-8")
    if len(encoded) > _BCRYPT_MAX_BYTES:
        raise ValueError(
            f"Password exceeds {_BCRYPT_MAX_BYTES} bytes when UTF-8 encoded"
        )
    return encoded


def verify_password(plain_password: str, hashed_password: str) -> bool:
    try:
        return bcrypt.checkpw(
            _encode_password(plain_password),
            hashed_password.encode("utf-8"),
        )
    except (ValueError, TypeError) as e:
        logger.warning("Password verification failed: %s", e)
        return False


def get_password_hash(password: str) -> str:
    hashed = bcrypt.hashpw(_encode_password(password), bcrypt.gensalt())
    return hashed.decode("utf-8")


def create_access_token(
    data: Dict[str, Any],
    expires_delta: Optional[timedelta] = None,
) -> str:
    to_encode = dict(data)

    for key, value in to_encode.items():
        if not isinstance(value, _CLAIM_TYPES):
            raise TypeError(
                f"JWT claim '{key}' has unsupported type "
                f"{type(value).__name__}; expected str, int or bool"
            )

    now = datetime.now(timezone.utc)
    to_encode.setdefault("iat", now)
    to_encode["exp"] = now + (
        expires_delta or timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    )

    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)


def decode_token(token: str) -> Dict[str, Any]:
    try:
        return jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
    except jwt.ExpiredSignatureError:
        logger.warning("JWT token expired")
        raise HTTPException(status_code=401, detail="Token expired")
    except jwt.InvalidTokenError:
        logger.warning("JWT decode failed")
        raise HTTPException(status_code=401, detail="Invalid token")