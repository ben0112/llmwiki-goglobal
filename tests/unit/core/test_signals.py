import asyncio

import pytest

from llmwiki_core.signals import sanitized_process_signal


def _linked(signal, *, relationship):
    wrapper = RuntimeError("private wrapper")
    if relationship == "cause":
        wrapper.__cause__ = signal
    else:
        wrapper.__context__ = signal
    return wrapper


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
    signal = sanitized_process_signal(failure)

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

    signal = sanitized_process_signal(failure)

    assert type(signal) is expected
    assert signal.args in ((), (exit_code,))
    assert signal.__cause__ is None and signal.__context__ is None


def test_generator_exit_traversal_is_cycle_safe():
    wrapper = RuntimeError("private cycle")
    wrapper.__cause__ = wrapper
    wrapper.__context__ = GeneratorExit("private generator")

    signal = sanitized_process_signal(wrapper)

    assert type(signal) is GeneratorExit
    assert signal.args == ()
    assert signal.__cause__ is None and signal.__context__ is None
