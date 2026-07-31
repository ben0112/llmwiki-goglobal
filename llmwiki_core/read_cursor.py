"""Dependency-free, opaque cursor values for bounded read models."""

from __future__ import annotations

import base64
import binascii
import json
from dataclasses import dataclass
from typing import Any

_CURSOR_KEYS = frozenset({"v", "s", "r", "o", "d", "k"})
_MAX_ENCODED_LENGTH = 2048


class CursorError(ValueError):
    """A cursor could not be encoded or decoded safely."""


@dataclass(frozen=True, slots=True)
class ReadCursor:
    scope: str
    revision: int
    sort: str
    direction: str
    key: tuple[str, ...]
    version: int = 1


def encode_cursor(cursor: ReadCursor) -> str:
    """Validate and encode a cursor as unpadded canonical Base64URL."""

    validated = _validated_cursor(cursor)
    payload = {
        "v": validated.version,
        "s": validated.scope,
        "r": validated.revision,
        "o": validated.sort,
        "d": validated.direction,
        "k": list(validated.key),
    }
    raw = json.dumps(
        payload,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    encoded = base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")
    if len(encoded) > _MAX_ENCODED_LENGTH:
        raise CursorError("cursor length exceeds limit")
    return encoded


def decode_cursor(encoded: str, *, expected_scope: str) -> ReadCursor:
    """Decode a cursor and bind it to the endpoint that will consume it."""

    if not isinstance(encoded, str) or not encoded:
        raise CursorError("invalid cursor payload")
    if len(encoded) > _MAX_ENCODED_LENGTH:
        raise CursorError("cursor length exceeds limit")
    if not isinstance(expected_scope, str) or not expected_scope:
        raise CursorError("invalid cursor scope")

    try:
        raw = base64.b64decode(
            encoded + "=" * (-len(encoded) % 4),
            altchars=b"-_",
            validate=True,
        )
        payload = json.loads(raw.decode("utf-8"), object_pairs_hook=_strict_object)
    except CursorError:
        raise
    except (binascii.Error, UnicodeDecodeError, json.JSONDecodeError, ValueError, TypeError):
        raise CursorError("invalid cursor payload") from None

    if not isinstance(payload, dict) or set(payload) != _CURSOR_KEYS:
        raise CursorError("invalid cursor payload keys")

    cursor = _validated_cursor(
        ReadCursor(
            scope=payload["s"],
            revision=payload["r"],
            sort=payload["o"],
            direction=payload["d"],
            key=payload["k"],
            version=payload["v"],
        )
    )
    if cursor.scope != expected_scope:
        raise CursorError("cursor scope does not match endpoint")
    return cursor


def _validated_cursor(cursor: object) -> ReadCursor:
    if not isinstance(cursor, ReadCursor):
        raise CursorError("invalid cursor payload")
    if type(cursor.version) is not int or cursor.version != 1:
        raise CursorError("unsupported cursor version")
    if not isinstance(cursor.scope, str) or not cursor.scope:
        raise CursorError("invalid cursor scope")
    if type(cursor.revision) is not int or cursor.revision <= 0:
        raise CursorError("invalid cursor revision")
    if not isinstance(cursor.sort, str) or not cursor.sort:
        raise CursorError("invalid cursor sort")
    if not isinstance(cursor.direction, str) or cursor.direction not in {"asc", "desc"}:
        raise CursorError("invalid cursor direction")
    if not isinstance(cursor.key, (tuple, list)) or not cursor.key:
        raise CursorError("invalid cursor key")
    if any(not isinstance(part, str) or not part for part in cursor.key):
        raise CursorError("invalid cursor key")
    return ReadCursor(
        scope=cursor.scope,
        revision=cursor.revision,
        sort=cursor.sort,
        direction=cursor.direction,
        key=tuple(cursor.key),
        version=cursor.version,
    )


def _strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise CursorError("invalid cursor payload keys")
        result[key] = value
    return result


__all__ = ["CursorError", "ReadCursor", "decode_cursor", "encode_cursor"]
