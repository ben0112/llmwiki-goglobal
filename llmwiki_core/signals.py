"""Sanitized process-control signal selection across adapter boundaries."""

from __future__ import annotations

import asyncio


def sanitized_process_signal(*failures: BaseException) -> BaseException | None:
    """Select a fresh signal using KI > SystemExit > cancellation priority."""
    seen: set[int] = set()
    pending = list(reversed(failures))
    system_exit_code = None
    has_system_exit = False
    has_cancellation = False
    while pending:
        current = pending.pop()
        identity = id(current)
        if identity in seen:
            continue
        seen.add(identity)
        if isinstance(current, KeyboardInterrupt):
            return KeyboardInterrupt()
        if isinstance(current, SystemExit):
            if not has_system_exit:
                code = current.code
                system_exit_code = (
                    int(code)
                    if isinstance(code, bool)
                    else code
                    if type(code) is int
                    else 1
                )
                has_system_exit = True
        elif isinstance(current, asyncio.CancelledError):
            has_cancellation = True
        if isinstance(current, BaseExceptionGroup):
            pending.extend(reversed(current.exceptions))
        pending.extend(
            linked
            for linked in reversed((current.__cause__, current.__context__))
            if linked is not None
        )
    if has_system_exit:
        return SystemExit(system_exit_code)
    if has_cancellation:
        return asyncio.CancelledError()
    return None


__all__ = ["sanitized_process_signal"]
