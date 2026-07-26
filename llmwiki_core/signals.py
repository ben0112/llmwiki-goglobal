"""Sanitized process-control signal selection across adapter boundaries."""

from __future__ import annotations

import asyncio

_MAX_EXCEPTION_GRAPH_NODES = 4_096


def _fresh_control_signal(failure: BaseException) -> tuple[int, BaseException] | None:
    if isinstance(failure, KeyboardInterrupt):
        return 0, KeyboardInterrupt()
    if isinstance(failure, SystemExit):
        try:
            code = failure.code
        except BaseException:  # noqa: BLE001 - hostile signal subclasses fail closed.
            code = 1
        safe_code = int(code) if isinstance(code, bool) else code if type(code) is int else 1
        return 1, SystemExit(safe_code)
    if isinstance(failure, asyncio.CancelledError):
        return 2, asyncio.CancelledError()
    if isinstance(failure, GeneratorExit):
        return 3, GeneratorExit()
    return None


def _boundary_link(failure: BaseException, name: str) -> tuple[BaseException | None, bool]:
    try:
        linked = getattr(failure, name)
    except BaseException:  # noqa: BLE001 - exception graphs are provider-controlled.
        return None, False
    if linked is not None and not isinstance(linked, BaseException):
        return None, False
    return linked, True


def _boundary_group_children(
    failure: BaseExceptionGroup,
) -> tuple[tuple[object, ...], bool]:
    try:
        children = failure.exceptions
    except BaseException:  # noqa: BLE001 - exception groups are provider-controlled.
        return (), False
    if type(children) is not tuple:
        return (), False
    return children, True


def sanitized_boundary_signal_or_unknown(  # noqa: C901 - bounded defensive graph traversal.
    *failures: BaseException,
) -> BaseException | None:
    """Classify boundary failures without exposing provider-owned exceptions."""
    if len(failures) > _MAX_EXCEPTION_GRAPH_NODES:
        return None
    seen: set[int] = set()
    pending = list(reversed(failures))
    selected: tuple[int, BaseException] | None = None
    has_unknown = False
    while pending:
        if len(seen) >= _MAX_EXCEPTION_GRAPH_NODES:
            return None
        current = pending.pop()
        identity = id(current)
        if identity in seen:
            continue
        seen.add(identity)
        if not isinstance(current, BaseException):
            has_unknown = True
            continue
        control = _fresh_control_signal(current)
        if control is not None and (selected is None or control[0] < selected[0]):
            selected = control
        if isinstance(current, BaseExceptionGroup):
            children, valid = _boundary_group_children(current)
            if valid:
                available = _MAX_EXCEPTION_GRAPH_NODES - len(seen) - len(pending)
                if len(children) > available:
                    return None
                pending.extend(reversed(children))
            else:
                has_unknown = True
        elif not isinstance(current, Exception):
            has_unknown = True
        cause, valid_cause = _boundary_link(current, "__cause__")
        context, valid_context = _boundary_link(current, "__context__")
        if not valid_cause or not valid_context:
            has_unknown = True
        for linked in (context, cause):
            if linked is not None:
                if len(seen) + len(pending) >= _MAX_EXCEPTION_GRAPH_NODES:
                    return None
                pending.append(linked)
    if selected is not None:
        return selected[1]
    if has_unknown:
        return BaseException()
    return None


__all__ = ["sanitized_boundary_signal_or_unknown"]
