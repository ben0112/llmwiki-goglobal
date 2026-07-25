"""Small, backend-neutral structured telemetry contract."""

from __future__ import annotations

import json
import logging
import os
from datetime import date, datetime
from enum import Enum
from uuid import UUID

_FORBIDDEN_FIELDS = frozenset({"object_url", "payload", "s3_key", "secret", "token"})


def _safe_value(value: object) -> str | int | float | bool | None:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Enum):
        return str(value.value)
    if isinstance(value, (UUID, date, datetime)):
        return str(value)
    raise TypeError(f"unsupported telemetry value type: {type(value).__name__}")


def emit(logger: logging.Logger, event: str, /, **fields: object) -> None:
    """Emit one stable JSON object with an intentionally narrow value surface."""
    if not isinstance(event, str) or not event:
        raise ValueError("event must be a non-empty string")
    forbidden = _FORBIDDEN_FIELDS.intersection(fields)
    if forbidden:
        raise ValueError(f"forbidden telemetry fields: {', '.join(sorted(forbidden))}")
    body = {"event": event, **{name: _safe_value(value) for name, value in fields.items()}}
    logger.info(json.dumps(body, ensure_ascii=True, separators=(",", ":"), sort_keys=True))


def replica_role(default: str) -> str:
    """Return the explicit process role when deployment supplies one."""
    return os.getenv("LLMWIKI_REPLICA_ROLE", default)
