from dataclasses import replace
from math import nan
from uuid import uuid4

import pytest
from jobs.handlers import (
    TerminalJobError,
    WorkerContext,
    _embedding_request_batches,
    _enqueue_embedding_after_extraction,
    _validate_embedding_job_shape,
    _validate_embedding_vectors,
)
from jobs.models import JobCreate, JobRecord, JobType

from llmwiki_core.models import EmbeddingProfile

PROFILE = EmbeddingProfile("openai_compatible", "embed-v1", 3)


def _payload(doc_id, **changes):
    payload = {
        "document_id": str(doc_id),
        "document_version": 1,
        "provider": PROFILE.provider,
        "model": PROFILE.model,
        "dimensions": PROFILE.dimensions,
    }
    payload.update(changes)
    return payload


def _record(**changes):
    document_id = changes.pop("document_id", uuid4())
    values = {
        "id": uuid4(),
        "job_type": JobType.DOCUMENT_EMBED,
        "user_id": uuid4(),
        "knowledge_base_id": uuid4(),
        "document_id": document_id,
        "payload": _payload(document_id),
    }
    values.update(changes)
    return JobRecord(**values)


def test_document_embed_job_type_and_exact_non_secret_contract():
    record = _record()

    assert JobType.DOCUMENT_EMBED.value == "document.embed"
    assert _validate_embedding_job_shape(record) == (1, PROFILE)


@pytest.mark.parametrize(
    ("change", "value"),
    [
        ("document_id", "not-a-uuid"),
        ("document_version", True),
        ("document_version", 0),
        ("document_version", 2_147_483_648),
        ("provider", ""),
        ("provider", "x" * 101),
        ("model", " "),
        ("model", "x" * 201),
        ("dimensions", True),
        ("dimensions", 0),
        ("dimensions", 4097),
    ],
)
def test_document_embed_payload_rejects_invalid_bounded_fields(change, value):
    record = _record()
    bad = replace(record, payload=_payload(record.document_id, **{change: value}))

    with pytest.raises(TerminalJobError, match="invalid") as raised:
        _validate_embedding_job_shape(bad)

    assert raised.value.error_code == "invalid_embedding_job"


@pytest.mark.parametrize(
    "secret_field",
    ["api_key", "token", "base_url", "content", "vector", "dsn", "authorization"],
)
def test_document_embed_payload_rejects_extra_secret_fields_without_disclosure(secret_field):
    secret = "postgres://private-token"
    document_id = uuid4()
    payload = _payload(document_id, **{secret_field: secret})

    with pytest.raises(ValueError) as raised:
        JobCreate(
            job_type=JobType.DOCUMENT_EMBED,
            user_id=uuid4(),
            knowledge_base_id=uuid4(),
            document_id=document_id,
            payload=payload,
        )

    assert secret not in str(raised.value)
    assert secret not in repr(raised.value)


def test_document_embed_command_repr_does_not_render_payload_mapping():
    document_id = uuid4()
    command = JobCreate(
        job_type=JobType.DOCUMENT_EMBED,
        user_id=uuid4(),
        knowledge_base_id=uuid4(),
        document_id=document_id,
        payload=_payload(document_id),
        idempotency_key=f"embed:{document_id}:1:{PROFILE.provider}:{PROFILE.model}:3",
    )

    assert "payload=" not in repr(command)


def test_document_embed_job_requires_matching_top_level_scope_and_canonical_uuid():
    record = _record()
    for bad in (
        replace(record, document_id=None),
        replace(record, knowledge_base_id=None),
        replace(record, payload=_payload(uuid4())),
        replace(record, payload=_payload(record.document_id).copy() | {"document_id": str(record.document_id).upper()}),
    ):
        with pytest.raises(TerminalJobError) as raised:
            _validate_embedding_job_shape(bad)
        assert raised.value.error_code == "invalid_embedding_job"


@pytest.mark.parametrize(
    "vectors",
    [
        (((1.0, 0.0, 0.0),),),
        ((1.0, 0.0),),
        ((1.0, 0.0, nan),),
        ((1.0, 0.0, True),),
        ((0.0, 0.0, 0.0),),
    ],
)
def test_embedding_handler_rejects_partial_wrong_dimension_nan_bool_and_zero_vectors(vectors):
    with pytest.raises(TerminalJobError) as raised:
        _validate_embedding_vectors(vectors, expected_count=2, dimensions=3)

    assert raised.value.error_code == "invalid_embedding_response"
    assert "nan" not in raised.value.error_message.lower()


def test_embedding_handler_preserves_ordered_chunk_indexes():
    assert _validate_embedding_vectors(
        ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
        expected_count=2,
        dimensions=3,
    ) == ((0, (1.0, 0.0, 0.0)), (1, (0.0, 1.0, 0.0)))


def test_single_oversized_chunk_is_stable_terminal_input_error():
    with pytest.raises(TerminalJobError) as raised:
        _embedding_request_batches(("x" * 200_001,))

    assert raised.value.error_code == "invalid_embedding_input"
    assert "x" * 100 not in str(raised.value)


@pytest.mark.asyncio
async def test_successful_extraction_enqueues_exact_current_profile_job(monkeypatch):
    job = _record(job_type=JobType.DOCUMENT_EXTRACT, payload={"document_id": str(uuid4())})
    job = replace(job, payload={"document_id": str(job.document_id)})
    captured = {}

    async def ensure(self, **kwargs):
        captured.update(kwargs)

    monkeypatch.setattr("jobs.handlers._configured_embedding_profile", lambda: PROFILE)
    monkeypatch.setattr("jobs.service.JobService.ensure_document_embedding", ensure)

    await _enqueue_embedding_after_extraction(
        job,
        7,
        WorkerContext(pool=object(), s3=None, converter_url="", converter_secret=""),
    )

    assert captured == {
        "document_id": job.document_id,
        "document_version": 7,
        "user_id": job.user_id,
        "knowledge_base_id": job.knowledge_base_id,
        "profile": PROFILE,
    }


@pytest.mark.asyncio
async def test_embedding_enqueue_gap_does_not_fail_lexical_extraction_or_log_secret(monkeypatch, caplog):
    job = _record(job_type=JobType.DOCUMENT_EXTRACT, payload={"document_id": str(uuid4())})
    job = replace(job, payload={"document_id": str(job.document_id)})

    async def fail(self, **kwargs):
        raise RuntimeError("api_key=private-token")

    monkeypatch.setattr("jobs.handlers._configured_embedding_profile", lambda: PROFILE)
    monkeypatch.setattr("jobs.service.JobService.ensure_document_embedding", fail)

    await _enqueue_embedding_after_extraction(
        job,
        1,
        WorkerContext(pool=object(), s3=None, converter_url="", converter_secret=""),
    )

    assert "private-token" not in caplog.text
    assert "embedding_enqueue_failed" in caplog.text


def test_missing_reconciliation_cli_has_stable_success_output(monkeypatch, capsys):
    from scripts import enqueue_embeddings

    async def run(page_size):
        assert page_size == 7
        return {"scanned": 4, "enqueued": 2}

    monkeypatch.setattr(enqueue_embeddings, "_run", run)

    assert enqueue_embeddings.main(["--missing", "--page-size", "7"]) == 0
    assert capsys.readouterr().out == '{"enqueued":2,"scanned":4}\n'


def test_missing_reconciliation_cli_requires_explicit_mode(capsys):
    from scripts.enqueue_embeddings import main

    assert main([]) == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == '{"error":"invalid_arguments"}\n'


@pytest.mark.parametrize(
    "argv",
    [
        ["--missing", "--page-size", "postgres://private-token"],
        ["--missing", "--page-size", "9" * 100_000],
        ["--missing", "--page-size", "１２"],
        ["--missing", "--private-path=/Users/private-token/repo"],
    ],
)
def test_missing_reconciliation_cli_argument_errors_never_echo_private_argv(argv, capsys):
    from scripts.enqueue_embeddings import main

    assert main(argv) == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == '{"error":"invalid_arguments"}\n'
    assert "private-token" not in captured.err


def test_missing_reconciliation_cli_help_remains_normal(capsys):
    from scripts.enqueue_embeddings import main

    assert main(["--help"]) == 0
    captured = capsys.readouterr()
    assert "--missing" in captured.out
    assert captured.err == ""


@pytest.mark.parametrize(
    ("failure", "code", "public_error"),
    [
        (ValueError("api_key=private-token"), 3, "invalid_configuration"),
        (RuntimeError("dsn=postgres://private-token"), 1, "reconciliation_failed"),
    ],
)
def test_missing_reconciliation_cli_sanitizes_failures(
    monkeypatch,
    capsys,
    failure,
    code,
    public_error,
):
    from scripts import enqueue_embeddings

    async def fail(_page_size):
        raise failure

    monkeypatch.setattr(enqueue_embeddings, "_run", fail)

    assert enqueue_embeddings.main(["--missing"]) == code
    output = capsys.readouterr().out
    assert output == f'{{"error": "{public_error}"}}\n'
    assert "private-token" not in output
