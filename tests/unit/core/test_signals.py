import asyncio

import pytest

from llmwiki_core.signals import sanitized_boundary_signal_or_unknown


class _UnknownBoundaryFailure(BaseException):
    pass


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
