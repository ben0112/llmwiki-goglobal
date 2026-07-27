"""REST-only server RAG CLI contracts."""

from __future__ import annotations

import json
from dataclasses import replace
from uuid import UUID

import pytest
from scripts import rag as rag_cli

KB_ID = UUID("11111111-1111-1111-1111-111111111111")
RUN_ID = UUID("22222222-2222-2222-2222-222222222222")
JOB_ID = UUID("33333333-3333-3333-3333-333333333333")
TOKEN = "private-llmwiki-access-token"
GOAL = "PRIVATE launch goal"


def _accepted() -> rag_cli.ApiResponse:
    return rag_cli.ApiResponse(
        status_code=202,
        payload={
            "run_id": str(RUN_ID),
            "job_id": str(JOB_ID),
            "state": "queued",
            "run_url": f"/v1/rag/runs/{RUN_ID}",
            "job_url": f"/v1/jobs/{JOB_ID}",
        },
    )


@pytest.fixture(autouse=True)
def _environment(monkeypatch):
    monkeypatch.setenv("LLMWIKI_API_URL", "https://wiki.example.test/")
    monkeypatch.setenv("LLMWIKI_ACCESS_TOKEN", TOKEN)


def test_build_wiki_sends_exact_rest_request(monkeypatch, capsys):
    sent = []
    monkeypatch.setattr(rag_cli, "_send", lambda request: sent.append(request) or _accepted())

    code = rag_cli.main(
        [
            "build-wiki",
            "--knowledge-base",
            str(KB_ID),
            "--goal",
            GOAL,
            "--target-prefix",
            "/wiki/launch/",
            "--model-profile",
            "primary",
            "--idempotency-key",
            "create-1",
            "--json",
        ]
    )

    assert code == 0
    assert len(sent) == 1
    request = sent[0]
    assert request.method == "POST"
    assert request.api_url == "https://wiki.example.test"
    assert request.path == "/v1/rag/build-wiki"
    assert request.headers == {
        "Accept": "application/json",
        "Authorization": f"Bearer {TOKEN}",
        "Content-Type": "application/json",
        "Idempotency-Key": "create-1",
    }
    assert request.body == {
        "knowledge_base_id": str(KB_ID),
        "goal": GOAL,
        "target_path_prefix": "/wiki/launch/",
        "model_profile": "primary",
        "retrieval_profile": "lexical",
        "dry_run": False,
        "budget": {},
    }
    assert json.loads(capsys.readouterr().out)["state"] == "queued"
    assert TOKEN not in repr(request)
    assert GOAL not in repr(request)


def test_build_wiki_translates_optional_controls(monkeypatch):
    sent = []
    monkeypatch.setattr(rag_cli, "_send", lambda request: sent.append(request) or _accepted())

    code = rag_cli.main(
        [
            "build-wiki",
            "--knowledge-base",
            str(KB_ID),
            "--goal",
            "Build launch",
            "--target-prefix",
            "/wiki/launch/",
            "--model-profile",
            "primary",
            "--retrieval-profile",
            "hybrid",
            "--dry-run",
            "--max-pages",
            "2",
            "--max-model-tokens",
            "1000",
            "--idempotency-key",
            "create-2",
        ]
    )

    assert code == 0
    assert sent[0].body["retrieval_profile"] == "hybrid"
    assert sent[0].body["dry_run"] is True
    assert sent[0].body["budget"] == {"max_pages": 2, "max_model_tokens": 1000}


@pytest.mark.parametrize(
    ("argv", "method", "path", "body"),
    [
        (["status", str(RUN_ID)], "GET", f"/v1/rag/runs/{RUN_ID}", None),
        (
            ["steps", str(RUN_ID), "--after", "7", "--limit", "25"],
            "GET",
            f"/v1/rag/runs/{RUN_ID}/steps?after=7&limit=25",
            None,
        ),
        (
            [
                "resume",
                str(RUN_ID),
                "--idempotency-key",
                "resume-1",
                "--max-pages",
                "4",
                "--max-steps",
                "20",
            ],
            "POST",
            f"/v1/rag/runs/{RUN_ID}/resume",
            {"budget": {"max_pages": 4, "max_steps": 20}},
        ),
    ],
)
def test_observation_and_resume_commands_use_only_rest(monkeypatch, argv, method, path, body):
    sent = []
    monkeypatch.setattr(rag_cli, "_send", lambda request: sent.append(request) or _accepted())

    assert rag_cli.main(argv) == 0
    assert [(request.method, request.path, request.body) for request in sent] == [(method, path, body)]
    if argv[0] == "resume":
        assert sent[0].headers["Idempotency-Key"] == "resume-1"
    assert not any("provider" in key.lower() for key in sent[0].headers)


@pytest.mark.parametrize(
    "argv",
    [
        ["status", "not-a-uuid"],
        ["steps", str(RUN_ID), "--limit", "0"],
        ["steps", str(RUN_ID), "--after", "-1"],
        ["resume", str(RUN_ID), "--idempotency-key", "resume-1", "--max-pages", "0"],
        ["build-wiki", "--provider-key", "private"],
    ],
)
def test_invalid_arguments_fail_before_network_without_echo(monkeypatch, capsys, argv):
    monkeypatch.setattr(rag_cli, "_send", lambda request: pytest.fail(f"network called: {request!r}"))

    assert rag_cli.main(argv) == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert json.loads(captured.err) == {"category": "arguments", "code": "invalid_arguments"}
    assert TOKEN not in captured.err
    assert "private" not in captured.err


@pytest.mark.parametrize("missing", ["LLMWIKI_API_URL", "LLMWIKI_ACCESS_TOKEN"])
def test_missing_or_unsafe_configuration_is_exit_two(monkeypatch, capsys, missing):
    monkeypatch.delenv(missing)
    assert rag_cli.main(["status", str(RUN_ID), "--json"]) == 2
    assert json.loads(capsys.readouterr().err) == {
        "category": "configuration",
        "code": "invalid_configuration",
    }


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("LLMWIKI_ACCESS_TOKEN", "non-ascii-token-密钥"),
        ("LLMWIKI_ACCESS_TOKEN", "token\nwith-control"),
        ("LLMWIKI_API_URL", "https://user:private@wiki.example.test"),
        ("LLMWIKI_API_URL", "https://wiki.example.test/private-path"),
        ("LLMWIKI_API_URL", "https://wiki.example.test\n.evil.invalid"),
        ("LLMWIKI_API_URL", "https://wiki.example.test:99999"),
    ],
)
def test_unsafe_configuration_fails_before_network(monkeypatch, capsys, name, value):
    monkeypatch.setenv(name, value)
    monkeypatch.setattr(rag_cli, "_send", lambda request: pytest.fail(f"network called: {request!r}"))

    assert rag_cli.main(["status", str(RUN_ID)]) == 2
    assert json.loads(capsys.readouterr().err) == {
        "category": "configuration",
        "code": "invalid_configuration",
    }


@pytest.mark.parametrize("key", ["resume\nprivate", "resume-密钥"])
def test_unsafe_idempotency_key_fails_before_network(monkeypatch, capsys, key):
    monkeypatch.setattr(rag_cli, "_send", lambda request: pytest.fail(f"network called: {request!r}"))

    assert (
        rag_cli.main(
            [
                "resume",
                str(RUN_ID),
                "--idempotency-key",
                key,
            ]
        )
        == 2
    )
    assert json.loads(capsys.readouterr().err) == {
        "category": "arguments",
        "code": "invalid_arguments",
    }


def test_api_error_uses_only_allowlisted_code_and_generic_category(monkeypatch, capsys):
    response = rag_cli.ApiResponse(
        status_code=409,
        payload={
            "detail": {
                "code": "rag_idempotency_conflict",
                "message": "PRIVATE response body",
                "provider_endpoint": "https://provider.invalid",
            }
        },
    )
    monkeypatch.setattr(rag_cli, "_send", lambda request: response)

    assert rag_cli.main(["status", str(RUN_ID)]) == 3
    captured = capsys.readouterr()
    assert captured.out == ""
    assert json.loads(captured.err) == {"category": "api", "code": "rag_idempotency_conflict"}
    assert "PRIVATE" not in captured.err
    assert "provider.invalid" not in captured.err


def test_unknown_api_error_code_is_not_repeated(monkeypatch, capsys):
    monkeypatch.setattr(
        rag_cli,
        "_send",
        lambda request: rag_cli.ApiResponse(500, {"detail": {"code": "PRIVATE_INTERNAL_CODE"}}),
    )

    assert rag_cli.main(["status", str(RUN_ID)]) == 3
    assert json.loads(capsys.readouterr().err) == {"category": "api", "code": "rag_api_error"}


def test_transport_exception_and_linked_context_are_never_rendered(monkeypatch, capsys):
    private_context = RuntimeError(f"{TOKEN} {GOAL} https://provider.invalid")
    failure = RuntimeError("PRIVATE response body")
    failure.__cause__ = private_context

    def fail(_request):
        raise failure

    monkeypatch.setattr(rag_cli, "_send", fail)

    assert rag_cli.main(["status", str(RUN_ID)]) == 3
    captured = capsys.readouterr()
    assert json.loads(captured.err) == {"category": "transport", "code": "request_failed"}
    assert TOKEN not in captured.err
    assert GOAL not in captured.err
    assert "provider.invalid" not in captured.err
    assert "PRIVATE" not in captured.err


def test_output_failure_is_exit_four_and_sanitized(monkeypatch, capsys):
    monkeypatch.setattr(rag_cli, "_send", lambda request: _accepted())
    failure = OSError(f"{TOKEN} {GOAL}")
    monkeypatch.setattr(rag_cli, "_write_output", lambda payload, as_json: (_ for _ in ()).throw(failure))

    assert rag_cli.main(["status", str(RUN_ID)]) == 4
    captured = capsys.readouterr()
    assert json.loads(captured.err) == {"category": "output", "code": "output_failed"}
    assert TOKEN not in captured.err
    assert GOAL not in captured.err


def test_json_decoder_rejects_duplicate_nonfinite_and_oversized_payloads():
    with pytest.raises(ValueError):
        rag_cli._decode_json(b'{"state":"queued","state":"failed"}')
    with pytest.raises(ValueError):
        rag_cli._decode_json(b'{"latency":NaN}')
    with pytest.raises(ValueError):
        rag_cli._decode_json(b'{"latency":1e999}')
    with pytest.raises(ValueError):
        rag_cli._decode_json(b" " * (rag_cli.MAX_RESPONSE_BYTES + 1))


def test_send_uses_bounded_hardened_http_client(monkeypatch):
    calls = []

    class FakeResponse:
        status_code = 200

        def iter_bytes(self):
            yield b'{"state":'
            yield b'"queued"}'

    class Stream:
        def __enter__(self):
            return FakeResponse()

        def __exit__(self, *_args):
            return False

    class Client:
        def __init__(self, **kwargs):
            calls.append(("client", kwargs))

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def stream(self, method, url, **kwargs):
            calls.append(("stream", method, url, kwargs))
            return Stream()

    monkeypatch.setattr(rag_cli.httpx, "Client", Client)
    request = rag_cli.Request(
        method="GET",
        api_url="https://wiki.example.test",
        path=f"/v1/rag/runs/{RUN_ID}",
        headers={"Accept": "application/json", "Authorization": f"Bearer {TOKEN}"},
        body=None,
    )

    response = rag_cli._send(request)

    assert response == rag_cli.ApiResponse(200, {"state": "queued"})
    assert calls[0] == (
        "client",
        {"follow_redirects": False, "trust_env": False, "timeout": 30.0},
    )
    assert calls[1][0:3] == (
        "stream",
        "GET",
        f"https://wiki.example.test/v1/rag/runs/{RUN_ID}",
    )
    assert calls[1][3]["headers"]["Authorization"] == f"Bearer {TOKEN}"
    assert calls[1][3]["content"] is None


def test_response_dataclasses_have_secret_safe_repr():
    request = rag_cli.Request(
        method="POST",
        api_url="https://wiki.example.test",
        path="/v1/rag/build-wiki",
        headers={"Authorization": f"Bearer {TOKEN}"},
        body={"goal": GOAL},
    )
    assert TOKEN not in repr(request)
    assert GOAL not in repr(request)
    assert replace(_accepted(), status_code=200).status_code == 200
