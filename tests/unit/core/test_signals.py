import asyncio

import pytest

from llmwiki_core import signals as signals_module
from llmwiki_core.signals import sanitized_boundary_signal_or_unknown


class _UnknownBoundaryFailure(BaseException):
    pass


class _HostileGraphAccess(RuntimeError):
    def __getattribute__(self, name):
        if name in {"__cause__", "__context__"}:
            raise RuntimeError("private graph accessor secret")
        return super().__getattribute__(name)


class _HostileGroupAccess(BaseExceptionGroup):
    def __getattribute__(self, name):
        if name == "exceptions":
            raise RuntimeError("private group accessor secret")
        return super().__getattribute__(name)


def _linked(signal, *, relationship):
    wrapper = RuntimeError("private wrapper")
    if relationship == "cause":
        wrapper.__cause__ = signal
    else:
        wrapper.__context__ = signal
    return wrapper


def _unknown_failure(shape: str) -> BaseException:
    unknown = _UnknownBoundaryFailure("private unknown")
    if shape == "direct":
        return unknown
    wrapper = RuntimeError("private wrapper")
    if shape == "cause":
        wrapper.__cause__ = unknown
        return wrapper
    if shape == "context":
        wrapper.__context__ = unknown
        return wrapper
    if shape == "nested-group":
        return BaseExceptionGroup(
            "private outer",
            [RuntimeError("ordinary"), BaseExceptionGroup("private inner", [unknown])],
        )
    if shape == "mixed":
        return BaseExceptionGroup(
            "private mixed",
            [ValueError("ordinary"), unknown, RuntimeError("ordinary two")],
        )
    if shape == "cycle":
        wrapper.__cause__ = wrapper
        wrapper.__context__ = unknown
        return wrapper
    raise AssertionError(f"unsupported shape: {shape}")


@pytest.mark.parametrize(
    "shape",
    ["direct", "cause", "context", "nested-group", "mixed", "cycle"],
)
def test_boundary_classifier_maps_unknown_failures_to_fresh_empty_base_exception(shape):
    signal = sanitized_boundary_signal_or_unknown(_unknown_failure(shape))

    assert type(signal) is BaseException
    assert signal.args == ()
    assert signal.__cause__ is None and signal.__context__ is None


@pytest.mark.parametrize(
    "failure",
    [
        RuntimeError("ordinary"),
        ExceptionGroup("ordinary group", [RuntimeError("one"), ValueError("two")]),
        _linked(RuntimeError("ordinary cause"), relationship="cause"),
    ],
)
def test_boundary_classifier_returns_none_for_ordinary_exception_graphs(failure):
    assert sanitized_boundary_signal_or_unknown(failure) is None


@pytest.mark.parametrize(
    "failure",
    [
        _HostileGraphAccess("private"),
        _HostileGroupAccess("private", [RuntimeError("private child")]),
    ],
)
def test_boundary_classifier_fails_closed_for_hostile_graph_access(failure):
    signal = sanitized_boundary_signal_or_unknown(failure)

    assert type(signal) is BaseException
    assert signal.args == ()
    assert signal.__cause__ is signal.__context__ is None


def test_boundary_classifier_accepts_an_exactly_bounded_complete_group():
    limit = signals_module._MAX_EXCEPTION_GRAPH_NODES
    failure = BaseExceptionGroup(
        "private exact limit",
        [asyncio.CancelledError("private cancellation")]
        + [RuntimeError("ordinary") for _ in range(limit - 3)]
        + [KeyboardInterrupt("private keyboard")],
    )

    signal = sanitized_boundary_signal_or_unknown(failure)

    assert type(signal) is KeyboardInterrupt
    assert signal.args == ()


def test_boundary_classifier_returns_none_without_partial_control_on_limit_plus_one():
    limit = signals_module._MAX_EXCEPTION_GRAPH_NODES
    failure = BaseExceptionGroup(
        "private over limit",
        [asyncio.CancelledError("private cancellation")] + [RuntimeError("ordinary") for _ in range(limit - 1)],
    )

    assert sanitized_boundary_signal_or_unknown(failure) is None


def test_boundary_classifier_does_not_select_front_control_before_unvisited_tail():
    limit = signals_module._MAX_EXCEPTION_GRAPH_NODES
    failure = BaseExceptionGroup(
        "private incomplete priority",
        [asyncio.CancelledError("private cancellation")]
        + [RuntimeError("ordinary") for _ in range(limit)]
        + [KeyboardInterrupt("private keyboard")],
    )

    assert sanitized_boundary_signal_or_unknown(failure) is None


def test_boundary_classifier_rejects_huge_groups_without_traversing_partial_results():
    failure = BaseExceptionGroup(
        "private huge group",
        [RuntimeError("ordinary") for _ in range(100_000)],
    )

    assert sanitized_boundary_signal_or_unknown(failure) is None


@pytest.mark.parametrize(
    ("failure", "expected", "expected_args"),
    [
        (KeyboardInterrupt("private"), KeyboardInterrupt, ()),
        (SystemExit("private"), SystemExit, (1,)),
        (asyncio.CancelledError("private"), asyncio.CancelledError, ()),
        (GeneratorExit("private"), GeneratorExit, ()),
        (
            BaseExceptionGroup(
                "private priority",
                [
                    _UnknownBoundaryFailure("private unknown"),
                    GeneratorExit("private generator"),
                    asyncio.CancelledError("private cancellation"),
                    SystemExit(23),
                    KeyboardInterrupt("private keyboard"),
                ],
            ),
            KeyboardInterrupt,
            (),
        ),
    ],
)
def test_boundary_classifier_preserves_control_priority_with_fresh_signals(
    failure,
    expected,
    expected_args,
):
    signal = sanitized_boundary_signal_or_unknown(failure)

    assert type(signal) is expected
    assert signal.args == expected_args
    assert signal.__cause__ is None and signal.__context__ is None


@pytest.mark.parametrize(
    "failure",
    [
        GeneratorExit("private direct"),
        _linked(GeneratorExit("private cause"), relationship="cause"),
        _linked(GeneratorExit("private context"), relationship="context"),
        BaseExceptionGroup(
            "private outer",
            [
                RuntimeError("ordinary"),
                BaseExceptionGroup(
                    "private inner",
                    [GeneratorExit("private nested")],
                ),
            ],
        ),
    ],
)
def test_generator_exit_is_recursively_selected_as_a_fresh_sanitized_signal(failure):
    signal = sanitized_boundary_signal_or_unknown(failure)

    assert type(signal) is GeneratorExit
    assert signal.args == ()
    assert signal.__cause__ is None and signal.__context__ is None


@pytest.mark.parametrize(
    ("higher_priority", "expected", "exit_code"),
    [
        (KeyboardInterrupt("private"), KeyboardInterrupt, None),
        (SystemExit(23), SystemExit, 23),
        (asyncio.CancelledError("private"), asyncio.CancelledError, None),
    ],
)
def test_generator_exit_keeps_existing_control_priority(
    higher_priority,
    expected,
    exit_code,
):
    failure = BaseExceptionGroup(
        "private mixed",
        [GeneratorExit("private generator"), higher_priority, RuntimeError("ordinary")],
    )

    signal = sanitized_boundary_signal_or_unknown(failure)

    assert type(signal) is expected
    assert signal.args in ((), (exit_code,))
    assert signal.__cause__ is None and signal.__context__ is None


def test_generator_exit_traversal_is_cycle_safe():
    wrapper = RuntimeError("private cycle")
    wrapper.__cause__ = wrapper
    wrapper.__context__ = GeneratorExit("private generator")

    signal = sanitized_boundary_signal_or_unknown(wrapper)

    assert type(signal) is GeneratorExit
    assert signal.args == ()
    assert signal.__cause__ is None and signal.__context__ is None
