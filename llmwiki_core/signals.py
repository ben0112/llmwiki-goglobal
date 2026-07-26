"""Sanitized process-control signal selection across adapter boundaries."""

from __future__ import annotations

import asyncio


def _fresh_control_signal(failure: BaseException) -> tuple[int, BaseException] | None:
    if isinstance(failure, KeyboardInterrupt):
        return 0, KeyboardInterrupt()
    if isinstance(failure, SystemExit):
        code = failure.code
        safe_code = int(code) if isinstance(code, bool) else code if type(code) is int else 1
        return 1, SystemExit(safe_code)
    if isinstance(failure, asyncio.CancelledError):
        return 2, asyncio.CancelledError()
    if isinstance(failure, GeneratorExit):
        return 3, GeneratorExit()
    return None


def sanitized_boundary_signal_or_unknown(
    *failures: BaseException,
) -> BaseException | None:
    """Classify boundary failures without exposing provider-owned exceptions."""
    seen: set[int] = set()
    pending = list(reversed(failures))
    selected: tuple[int, BaseException] | None = None
    has_unknown = False
    while pending:
        current = pending.pop()
        identity = id(current)
        if identity in seen:
            continue
        seen.add(identity)
        control = _fresh_control_signal(current)
        if control is not None and (selected is None or control[0] < selected[0]):
            selected = control
        if isinstance(current, BaseExceptionGroup):
            pending.extend(reversed(current.exceptions))
        elif not isinstance(current, Exception):
            has_unknown = True
        pending.extend(
            linked
            for linked in reversed((current.__cause__, current.__context__))
            if linked is not None
        )
    if selected is not None:
        return selected[1]
    if has_unknown:
        return BaseException()
    return None


__all__ = ["sanitized_boundary_signal_or_unknown"]
