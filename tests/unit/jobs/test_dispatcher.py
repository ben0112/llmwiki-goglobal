from __future__ import annotations

import asyncio
import hashlib
import inspect
import logging
from contextlib import asynccontextmanager
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import MappingProxyType, SimpleNamespace
from uuid import UUID, uuid4

import asyncpg
import pytest

from tests.helpers.telemetry_contract import assert_telemetry_event

DISPATCH_RUN_AFTER = datetime(2026, 1, 1, tzinfo=UTC)


def _runtime_settings(**changes):
    values = {
        "MODE": "hosted",
        "DURABLE_JOBS_ENABLED": True,
        "REDIS_URL": "redis://redis.internal:6380/4",
        "DATABASE_URL": "postgresql://database.internal/jobs",
        "S3_BUCKET": None,
        "AWS_ACCESS_KEY_ID": None,
        "AWS_SECRET_ACCESS_KEY": None,
        "CONVERTER_URL": "https://converter.invalid",
        "CONVERTER_SECRET": "converter-secret",
        "JOB_LEASE_SECONDS": 60,
        "JOB_HEARTBEAT_SECONDS": 15,
        "JOB_DISPATCH_BATCH_SIZE": 100,
        "JOB_REDELIVER_SECONDS": 30,
        "SERVER_RAG_ENABLED": False,
    }
    values.update(changes)
    return SimpleNamespace(**values)


class FakeConnection:
    def __init__(self, pool, selected=(), *, mark_results=None, mark_errors=None):
        self.pool = pool
        self.selected = list(selected)
        self.mark_results = mark_results or {}
        self.mark_errors = mark_errors or {}
        self.fetch_calls = []
        self.fetchval_calls = []

    @asynccontextmanager
    async def transaction(self):
        self.pool.active_transactions += 1
        try:
            yield
        finally:
            self.pool.active_transactions -= 1

    async def fetch(self, query, *args):
        self.fetch_calls.append((query, args))
        return [{"id": job_id, "run_after": DISPATCH_RUN_AFTER} for job_id in self.selected]

    async def fetchval(self, query, job_id, selected_run_after):
        self.fetchval_calls.append((query, job_id, selected_run_after))
        if job_id in self.mark_errors:
            raise self.mark_errors[job_id]
        return self.mark_results.get(job_id, True)


class FakePool:
    def __init__(self, selected=(), *, mark_results=None, mark_errors=None):
        self.active_connections = 0
        self.active_transactions = 0
        self.connection = FakeConnection(
            self,
            selected,
            mark_results=mark_results,
            mark_errors=mark_errors,
        )

    @asynccontextmanager
    async def acquire(self):
        self.active_connections += 1
        try:
            yield self.connection
        finally:
            self.active_connections -= 1


class FakeRedis:
    def __init__(self, pool, outcomes):
        self.pool = pool
        self.outcomes = list(outcomes)
        self.calls = []

    async def enqueue_job(self, *args, **kwargs):
        assert self.pool.active_connections == 0
        assert self.pool.active_transactions == 0
        self.calls.append((args, kwargs))
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


@pytest.mark.asyncio
@pytest.mark.parametrize("field", ["batch_size", "redeliver_seconds"])
@pytest.mark.parametrize("value", [True, False, 0, -1, 1.5, "1", None])
async def test_select_due_jobs_rejects_non_positive_integral_values(field, value):
    from jobs.dispatcher import select_due_jobs

    values = {"batch_size": 10, "redeliver_seconds": 30, field: value}
    with pytest.raises(ValueError, match=field):
        await select_due_jobs(FakePool().connection, **values)


@pytest.mark.asyncio
async def test_select_due_jobs_returns_generation_tokens_with_ordered_skip_locked_query():
    from jobs.dispatcher import DispatchCandidate, select_due_jobs

    selected = [uuid4(), uuid4()]
    connection = FakePool(selected).connection

    assert await select_due_jobs(connection, batch_size=2, redeliver_seconds=30) == [
        DispatchCandidate(job_id=job_id, run_after=DISPATCH_RUN_AFTER) for job_id in selected
    ]
    query, args = connection.fetch_calls[0]
    normalized = " ".join(query.split()).lower()
    assert args == (30, 2)
    assert "clock_timestamp()" in normalized
    assert "select job.id, job.run_after" in normalized
    assert "job.last_dispatched_at < dispatch_clock.checked_at" in normalized
    assert "job.last_dispatched_at < job.run_after" in normalized
    assert "order by job.run_after, job.created_at, job.id" in normalized
    assert "for update of job skip locked" in normalized
    assert "limit $2" in normalized


def test_delivery_transport_id_is_stable_within_generation_and_changes_for_retry():
    from jobs.dispatcher import DispatchCandidate, delivery_transport_id

    job_id = uuid4()
    same_in_utc = DISPATCH_RUN_AFTER.astimezone(UTC)
    retry_run_after = DISPATCH_RUN_AFTER + timedelta(seconds=1)

    first = delivery_transport_id(DispatchCandidate(job_id, DISPATCH_RUN_AFTER))

    assert first == delivery_transport_id(DispatchCandidate(job_id, same_in_utc))
    assert first != delivery_transport_id(DispatchCandidate(job_id, retry_run_after))
    assert first.startswith(f"{job_id}:")


@pytest.mark.asyncio
async def test_dispatch_sends_only_opaque_uuid_and_marks_new_and_duplicate_delivery():
    from jobs.dispatcher import DispatchCandidate, DispatchSummary, delivery_transport_id, dispatch_due_jobs

    job_ids = [uuid4(), uuid4()]
    pool = FakePool(job_ids)
    redis = FakeRedis(pool, [object(), None])

    summary = await dispatch_due_jobs(pool, redis, batch_size=10, redeliver_seconds=30)

    assert summary == DispatchSummary(
        selected=2,
        enqueued=1,
        already_present=1,
        marked=2,
        enqueue_failed=0,
        mark_failed=0,
    )
    assert redis.calls == [
        (
            ("run_job", str(job_ids[0])),
            {"_job_id": delivery_transport_id(DispatchCandidate(job_ids[0], DISPATCH_RUN_AFTER))},
        ),
        (
            ("run_job", str(job_ids[1])),
            {"_job_id": delivery_transport_id(DispatchCandidate(job_ids[1], DISPATCH_RUN_AFTER))},
        ),
    ]
    assert [(call[1], call[2]) for call in pool.connection.fetchval_calls] == [
        (job_id, DISPATCH_RUN_AFTER) for job_id in job_ids
    ]


@pytest.mark.asyncio
async def test_dispatch_continues_after_enqueue_and_mark_failures():
    from jobs.dispatcher import DispatchSummary, dispatch_due_jobs

    enqueue_failed, mark_failed, delivered = uuid4(), uuid4(), uuid4()
    pool = FakePool(
        [enqueue_failed, mark_failed, delivered],
        mark_errors={mark_failed: RuntimeError("database unavailable")},
    )
    redis = FakeRedis(pool, [ConnectionError("redis unavailable"), object(), None])

    summary = await dispatch_due_jobs(pool, redis, batch_size=3, redeliver_seconds=10)

    assert summary == DispatchSummary(
        selected=3,
        enqueued=1,
        already_present=1,
        marked=1,
        enqueue_failed=1,
        mark_failed=1,
    )
    assert [(call[1], call[2]) for call in pool.connection.fetchval_calls] == [
        (mark_failed, DISPATCH_RUN_AFTER),
        (delivered, DISPATCH_RUN_AFTER),
    ]


@pytest.mark.asyncio
async def test_dispatch_counts_terminal_mark_mismatch_as_mark_failure():
    from jobs.dispatcher import dispatch_due_jobs

    job_id = uuid4()
    pool = FakePool([job_id], mark_results={job_id: False})

    summary = await dispatch_due_jobs(
        pool,
        FakeRedis(pool, [None]),
        batch_size=1,
        redeliver_seconds=1,
    )

    assert summary.already_present == 1
    assert summary.marked == 0
    assert summary.mark_failed == 1


@pytest.mark.asyncio
async def test_dispatch_cron_reads_only_worker_context_resources(monkeypatch):
    from jobs import dispatcher

    calls = []

    async def fake_dispatch(pool, redis, *, batch_size, redeliver_seconds):
        calls.append((pool, redis, batch_size, redeliver_seconds))

    monkeypatch.setattr(dispatcher, "dispatch_due_jobs", fake_dispatch)
    ctx = {
        "pool": object(),
        "redis": object(),
        "dispatch_batch_size": 17,
        "redeliver_seconds": 41,
    }

    assert await dispatcher.dispatch_cron(ctx) is None
    assert calls == [(ctx["pool"], ctx["redis"], 17, 41)]


def test_handler_registry_is_complete_read_only_and_transport_neutral():
    from jobs.handlers import (
        HANDLERS,
        handle_document_embed,
        handle_document_extract,
        handle_graph_rebuild,
        handle_upload_cleanup,
    )
    from jobs.models import JobType

    assert isinstance(HANDLERS, MappingProxyType)
    assert set(HANDLERS) == set(JobType)
    assert HANDLERS[JobType.DOCUMENT_EXTRACT] is handle_document_extract
    assert HANDLERS[JobType.DOCUMENT_EMBED] is handle_document_embed
    assert HANDLERS[JobType.GRAPH_REBUILD] is handle_graph_rebuild
    assert HANDLERS[JobType.UPLOAD_CLEANUP] is handle_upload_cleanup
    assert HANDLERS[JobType.BUILD_WIKI].__name__ == "handle_build_wiki"
    with pytest.raises(TypeError):
        HANDLERS[JobType.DOCUMENT_EXTRACT] = object()
    with pytest.raises(TypeError):
        HANDLERS[JobType.BUILD_WIKI] = object()

    handlers_module = inspect.getmodule(next(iter(HANDLERS.values())))
    assert handlers_module is not None
    source = Path(inspect.getfile(handlers_module)).read_text()
    assert "import arq" not in source
    assert "from arq" not in source
    assert "redis" not in source.lower()


def test_upload_cleanup_handler_is_registered_as_concrete_business_work():
    from jobs.handlers import HANDLERS, handle_upload_cleanup
    from jobs.models import JobType

    assert HANDLERS[JobType.UPLOAD_CLEANUP] is handle_upload_cleanup


def test_handler_errors_persist_only_vetted_utf8_postgres_safe_messages():
    from jobs.handlers import RetryableJobError, TerminalJobError

    error = RetryableJobError(
        "converter_timeout",
        "TOP_SECRET_TOKEN from raw converter exception",
    )
    assert error.error_code == "converter_timeout"
    assert error.error_message == "Converter timed out."
    assert str(error) == "Converter timed out."

    for unsafe in ("TOP_SECRET_TOKEN\x00db unsafe", "TOP_SECRET_TOKEN\ud800"):
        unsafe_error = TerminalJobError("invalid_document", unsafe)
        assert unsafe_error.error_message == "Document is invalid."
        unsafe_error.error_message.encode("utf-8")
        assert "\x00" not in unsafe_error.error_message

    with pytest.raises(TypeError, match="message"):
        TerminalJobError("stable", b"not utf-8")

    with pytest.raises(ValueError, match="error_code"):
        TerminalJobError("Not Stable!", "message")
    with pytest.raises(ValueError, match="message"):
        TerminalJobError("stable", " \n\t ")


def test_arq_imports_are_confined_to_adapter_modules():
    jobs_path = Path(__file__).parents[3] / "api" / "jobs"
    offenders = []
    for path in jobs_path.glob("*.py"):
        source = path.read_text()
        if "import arq" in source or "from arq" in source:
            offenders.append(path.name)
    assert sorted(offenders) == ["dispatcher.py", "worker.py"]


def test_worker_settings_disable_arq_retry_and_results_with_unique_safe_crons():
    from jobs import worker

    built = worker.build_worker_settings(_runtime_settings())
    assert built["functions"] == [worker.run_job]
    assert built["max_tries"] == 1
    assert built["retry_jobs"] is False
    assert built["keep_result"] == 0
    assert built["job_timeout"] == 3600
    assert built["max_jobs"] == worker.WORKER_MAX_JOBS
    assert worker.WorkerSettings.max_jobs == worker.WORKER_MAX_JOBS
    assert built["on_startup"] is worker.startup
    assert built["on_shutdown"] is worker.shutdown
    assert len(built["cron_jobs"]) == 3

    dispatch, reap, cleanup = built["cron_jobs"]
    assert dispatch.coroutine is worker.dispatch_cron
    assert dispatch.second == {0, 10, 20, 30, 40, 50}
    assert reap.coroutine is worker.reap_cron
    assert reap.second == {5, 35}
    assert cleanup.coroutine is worker.upload_cleanup_cron
    assert cleanup.second == {15}
    assert {dispatch.name, reap.name, cleanup.name} == {
        "durable_job_dispatch",
        "durable_job_reaper",
        "durable_upload_cleanup_scan",
    }
    assert dispatch.job_id is None
    assert reap.job_id is None
    assert cleanup.job_id is None
    for job in built["cron_jobs"]:
        assert job.run_at_startup is True
        assert job.unique is True
        assert job.max_tries == 1
        assert job.keep_result_s == 0


@pytest.mark.asyncio
async def test_worker_pool_reserves_connections_beyond_handler_concurrency(monkeypatch):
    from jobs import worker

    captured = {}

    async def create_pool(database_url, *, min_size, max_size):
        captured.update(database_url=database_url, min_size=min_size, max_size=max_size)
        return object()

    monkeypatch.setattr(worker.asyncpg, "create_pool", create_pool)

    await worker._create_pool("postgresql://worker.test/jobs")

    assert captured == {
        "database_url": "postgresql://worker.test/jobs",
        "min_size": 2,
        "max_size": worker.WORKER_MAX_JOBS + worker.WORKER_POOL_RESERVED_CONNECTIONS,
    }
    assert worker.WORKER_POOL_RESERVED_CONNECTIONS >= 2

    capacity = asyncio.Semaphore(captured["max_size"])
    handlers = []
    for _ in range(worker.WORKER_MAX_JOBS):
        await capacity.acquire()
        handlers.append(object())
    try:
        async with asyncio.timeout(0.1):
            await capacity.acquire()  # heartbeat
            await capacity.acquire()  # dispatcher/reaper cron
        assert capacity.locked()
    finally:
        for _ in range(len(handlers) + 2):
            capacity.release()


class CronRecordingRedis:
    def __init__(self, seen):
        self.seen = seen
        self.attempts = []
        self.enqueued = []

    async def enqueue_job(self, function, *args, **kwargs):
        del args
        job_id = kwargs["_job_id"]
        self.attempts.append((function, job_id))
        if job_id in self.seen:
            return None
        self.seen.add(job_id)
        self.enqueued.append((function, job_id))
        return object()


@pytest.mark.asyncio
async def test_real_arq_cron_ids_change_by_schedule_and_dedupe_across_replicas():
    from jobs import worker

    seen = set()
    first = worker.create_durable_worker(_runtime_settings(), handle_signals=False)
    second = worker.create_durable_worker(_runtime_settings(), handle_signals=False)
    first_redis = CronRecordingRedis(seen)
    second_redis = CronRecordingRedis(seen)
    first._pool = first_redis
    second._pool = second_redis
    scheduled = datetime(2026, 1, 1, tzinfo=UTC)

    await asyncio.gather(
        first.run_cron(scheduled, delay=0.1),
        second.run_cron(scheduled, delay=0.1),
    )

    assert sorted(first_redis.attempts) == sorted(second_redis.attempts)
    assert len(first_redis.attempts) == 3
    assert len(first_redis.enqueued) + len(second_redis.enqueued) == 3

    first_dispatch_id = next(job_id for function, job_id in first_redis.attempts if function == "durable_job_dispatch")
    await first.run_cron(scheduled + timedelta(seconds=10), delay=0.1)
    dispatch_ids = [job_id for function, job_id in first_redis.attempts if function == "durable_job_dispatch"]
    assert len(dispatch_ids) == 2
    assert dispatch_ids[0] == first_dispatch_id
    assert dispatch_ids[1] != first_dispatch_id


def test_worker_module_exposes_safe_arq_cli_worker_settings():
    from arq.utils import import_string
    from jobs import worker

    imported = import_string("jobs.worker.WorkerSettings")
    assert imported is worker.WorkerSettings
    assert imported.functions == [worker.run_job]
    assert imported.on_startup is worker.startup
    assert imported.on_shutdown is worker.shutdown
    assert imported.ctx == {"runtime_settings": worker.settings}


@pytest.mark.parametrize(
    "runtime_settings,match",
    [
        (_runtime_settings(MODE="local"), "MODE=hosted"),
        (_runtime_settings(DURABLE_JOBS_ENABLED=False), "DURABLE_JOBS_ENABLED"),
        (_runtime_settings(REDIS_URL=None), "REDIS_URL"),
        (_runtime_settings(REDIS_URL="  "), "REDIS_URL"),
        (_runtime_settings(DATABASE_URL=""), "DATABASE_URL"),
        (_runtime_settings(DATABASE_URL="  "), "DATABASE_URL"),
    ],
)
def test_worker_factories_validate_before_arq_construction(
    monkeypatch,
    runtime_settings,
    match,
):
    from jobs import worker

    calls = []

    def create_worker(*args, **kwargs):
        calls.append(("create", args, kwargs))

    def run_worker(*args, **kwargs):
        calls.append(("run", args, kwargs))

    monkeypatch.setattr(worker, "arq_create_worker", create_worker)
    monkeypatch.setattr(worker, "arq_run_worker", run_worker)

    with pytest.raises(RuntimeError, match=match):
        worker.create_durable_worker(runtime_settings)
    with pytest.raises(RuntimeError, match=match):
        worker.run_durable_worker(runtime_settings)
    assert calls == []


def test_build_worker_settings_preserves_safety_and_parses_validated_redis_url():
    from jobs import worker

    runtime_settings = _runtime_settings()
    built = worker.build_worker_settings(runtime_settings)

    assert built["functions"] == [worker.run_job]
    assert len(built["cron_jobs"]) == 3
    assert built["max_tries"] == 1
    assert built["retry_jobs"] is False
    assert built["keep_result"] == 0
    assert built["job_timeout"] == 3600
    assert built["on_startup"] is worker.startup
    assert built["on_shutdown"] is worker.shutdown
    assert built["ctx"]["runtime_settings"] is runtime_settings
    redis_settings = built["redis_settings"]
    assert redis_settings.host == "redis.internal"
    assert redis_settings.port == 6380
    assert redis_settings.database == 4


def test_create_durable_worker_constructs_configured_unconnected_arq_worker():
    from jobs import worker

    runtime_settings = _runtime_settings()
    arq_worker = worker.create_durable_worker(
        runtime_settings,
        handle_signals=False,
        ctx={"trace_id": "opaque"},
    )

    assert arq_worker.redis_settings.host == "redis.internal"
    assert arq_worker.redis_settings.port == 6380
    assert arq_worker.redis_settings.database == 4
    assert arq_worker.redis_settings.host != "localhost"
    assert arq_worker._pool is None
    assert arq_worker.ctx["runtime_settings"] is runtime_settings
    assert arq_worker.ctx["trace_id"] == "opaque"

    none_ctx_worker = worker.create_durable_worker(
        runtime_settings,
        handle_signals=False,
        ctx=None,
    )
    assert none_ctx_worker.ctx["runtime_settings"] is runtime_settings


def test_run_durable_worker_passes_validated_dynamic_settings_to_arq(monkeypatch):
    from jobs import worker

    expected = object()
    calls = []

    def run_worker(settings_object, **kwargs):
        calls.append((settings_object, kwargs))
        return expected

    monkeypatch.setattr(worker, "arq_run_worker", run_worker)

    result = worker.run_durable_worker(_runtime_settings(), burst=True)

    assert result is expected
    assert calls[0][1] == {"burst": True}
    assert calls[0][0]["redis_settings"].host == "redis.internal"
    assert calls[0][0]["redis_settings"].port == 6380
    assert calls[0][0]["redis_settings"].database == 4


def test_worker_module_main_uses_preflight_launcher(monkeypatch):
    from jobs import worker

    calls = []
    monkeypatch.setattr(worker, "run_durable_worker", lambda: calls.append("run"))

    assert worker.main() is None
    assert calls == ["run"]


def _job(job_type=None, **changes):
    from jobs.models import JobRecord, JobType

    record = JobRecord(
        id=uuid4(),
        job_type=job_type or JobType.DOCUMENT_EXTRACT,
        user_id=uuid4(),
    )
    return replace(record, **changes)


def _rag_run_for_job(job, **changes):
    from rag.records import RagRunRecord

    from llmwiki_core.rag import RagBudget, RagUsage

    run_id = uuid4()
    if set(job.payload) == {"run_id"}:
        run_id = UUID(job.payload["run_id"])
    record = RagRunRecord(
        id=run_id,
        job_id=job.id,
        root_run_id=run_id,
        parent_run_id=None,
        user_id=job.user_id,
        knowledge_base_id=job.knowledge_base_id,
        goal="private goal",
        goal_digest=hashlib.sha256(b"private goal").hexdigest(),
        target_path_prefix="/wiki/private/",
        model_profile="primary",
        model_profile_version="primary-v1",
        retrieval_profile="lexical",
        dry_run=False,
        budget=RagBudget(),
        usage=RagUsage(),
        idempotency_key="private-key",
        request_digest="a" * 64,
        completion_reason=None,
        last_committed_ordinal=-1,
        created_at=datetime(2026, 7, 27, tzinfo=UTC),
        updated_at=datetime(2026, 7, 27, tzinfo=UTC),
    )
    return replace(record, **changes)


def _failed_transition(job, error_code: str):
    from jobs.models import JobState

    return replace(job, state=JobState.FAILED, attempt_count=1, error_code=error_code)


def _worker_context(pool):
    from jobs.handlers import WorkerContext

    return WorkerContext(
        pool=pool,
        s3=None,
        converter_url="https://converter.invalid",
        converter_secret="secret",
    )


class FakeWorkerPool:
    def __init__(self):
        self.connection = object()
        self.acquires = 0
        self.transactions = 0
        self.close_calls = 0
        self.active_connections = 0
        self.active_transactions = 0
        self.postgres_result_bytes = None
        self.postgres_validation_error = None
        self.result_validation_calls = []
        self.events = []

    @asynccontextmanager
    async def acquire(self):
        self.acquires += 1
        self.active_connections += 1
        try:
            yield self.connection
        finally:
            self.active_connections -= 1

    @asynccontextmanager
    async def transaction(self):
        self.transactions += 1
        self.active_transactions += 1
        try:
            yield
        finally:
            self.active_transactions -= 1

    async def close(self):
        self.close_calls += 1

    async def fetchval(self, query, serialized_json=None):
        if query == "SELECT 1" and serialized_json is None:
            return 1
        self.result_validation_calls.append((query, serialized_json))
        self.events.append("postgres-validate")
        if self.postgres_validation_error is not None:
            raise self.postgres_validation_error
        if self.postgres_result_bytes is not None:
            return self.postgres_result_bytes
        return len(serialized_json.encode("utf-8"))


class PoolWithConnectionTransaction(FakeWorkerPool):
    def __init__(self):
        super().__init__()
        self.connection = self


def _install_legacy_reaper_mock(monkeypatch, worker, reap_expired):
    """Adapt pre-decision reaper tests to the lock-then-transition protocol."""
    from jobs.models import JobState

    state = {}

    async def lock_expired(conn, *, limit):
        transitions = await reap_expired(conn, limit=limit, include_transitions=True)
        state["transitions"] = transitions
        return [
            replace(
                transition,
                state=JobState.RUNNING,
                result=None,
                error_code=None,
                error_message=None,
                lease_owner="expired-worker",
            )
            for transition in transitions
        ]

    async def reap_locked(_conn, jobs, *, include_transitions):
        transitions = state.get("transitions", [])
        assert {job.id for job in jobs} == {transition.id for transition in transitions}
        return transitions if include_transitions else [transition.id for transition in transitions]

    async def unfinished_rag(*_args):
        return None

    monkeypatch.setattr(worker.repository, "lock_expired_for_reap", lock_expired)
    monkeypatch.setattr(worker.repository, "reap_locked", reap_locked)
    monkeypatch.setattr(worker, "_recover_terminal_rag_success", unfinished_rag)


class FakeLease:
    instances = []

    def __init__(self, pool, job_id, owner, lease_seconds, heartbeat_seconds):
        self.args = (pool, job_id, owner, lease_seconds, heartbeat_seconds)
        self.entered = False
        self.__class__.instances.append(self)

    async def __aenter__(self):
        self.entered = True
        return self

    async def __aexit__(self, *_args):
        self.entered = False


def _ctx(pool, handlers):
    return {
        "pool": pool,
        "worker_id": "worker-test",
        "lease_seconds": 60,
        "heartbeat_seconds": 15,
        "handlers": handlers,
        "worker_context": _worker_context(pool),
    }


@pytest.mark.asyncio
async def test_run_job_rejects_invalid_uuid_without_touching_database():
    from jobs.worker import run_job

    pool = PoolWithConnectionTransaction()
    assert await run_job(_ctx(pool, {}), "payload-not-a-uuid") == {"status": "invalid_job_id"}
    assert pool.acquires == 0


@pytest.mark.asyncio
async def test_run_job_claims_from_postgres_and_returns_duplicate_without_handler(monkeypatch):
    from jobs import worker

    pool = PoolWithConnectionTransaction()
    calls = []

    async def claim(conn, job_id, owner, lease_seconds):
        calls.append((conn, job_id, owner, lease_seconds))

    monkeypatch.setattr(worker.repository, "claim", claim)
    job_id = uuid4()

    outcome = await worker.run_job(_ctx(pool, {}), str(job_id))

    assert outcome == {"status": "duplicate", "job_id": str(job_id)}
    assert calls == [(pool.connection, job_id, "worker-test", 60)]
    assert pool.acquires == 1
    assert pool.transactions == 1


@pytest.mark.asyncio
async def test_run_job_executes_persisted_handler_under_lease_and_succeeds(monkeypatch, caplog):
    from jobs import worker
    from jobs.models import JobState

    pool = PoolWithConnectionTransaction()
    job = _job()
    calls = pool.events

    async def claim(*_args):
        return job

    async def handler(record, lease, context):
        assert record is job
        assert lease.entered
        assert context is ctx["worker_context"]
        calls.append("handler")
        return {"ok": True, "nested": [1, None]}

    async def succeed(conn, job_id, owner, result):
        assert FakeLease.instances[-1].entered
        calls.append((conn, job_id, owner, result))
        return replace(job, state=JobState.SUCCEEDED, attempt_count=1)

    monkeypatch.setattr(worker.repository, "claim", claim)
    monkeypatch.setattr(worker.repository, "succeed", succeed)
    monkeypatch.setattr(worker, "JobLease", FakeLease)
    ctx = _ctx(pool, {job.job_type: handler})

    with caplog.at_level(logging.INFO, logger="jobs.worker"):
        outcome = await worker.run_job(ctx, str(job.id))

    assert outcome == {"status": "succeeded", "job_id": str(job.id)}
    assert calls == [
        "handler",
        "postgres-validate",
        (pool.connection, job.id, "worker-test", {"ok": True, "nested": [1, None]}),
    ]
    assert FakeLease.instances[-1].args == (pool, job.id, "worker-test", 60, 15)
    assert pool.acquires == 2
    assert pool.transactions == 2
    assert_telemetry_event(
        caplog,
        "durable_job_finished",
        expected={
            "job_id": str(job.id),
            "job_type": job.job_type.value,
            "attempt": 1,
            "state": "succeeded",
            "lease_owner": "worker-test",
            "error_code": None,
            "replica_role": "worker",
        },
        sensitive=("converter-secret", "RAW_RESULT_TOKEN", "postgresql://private.invalid"),
    )


def test_prepare_job_result_returns_strict_mapping_and_safe_json_input():
    from jobs.worker import _prepare_job_result

    result, serialized = _prepare_job_result({"x": "你", "nested": [1, None]})

    assert result == {"x": "你", "nested": [1, None]}
    assert serialized == '{"x": "你", "nested": [1, null]}'
    assert "\\u4f60" not in serialized


@pytest.mark.parametrize(
    "result",
    [
        ["not", "a", "mapping"],
        {"nested": {"not_finite": float("nan")}},
        {"nested": {"bytes": b"TOP_SECRET_RESULT"}},
        {"nested": {"object": object()}},
    ],
)
def test_prepare_job_result_rejects_nonmapping_and_nested_non_json_values(result):
    from jobs.handlers import TerminalJobError
    from jobs.worker import _prepare_job_result

    with pytest.raises(TerminalJobError) as raised:
        _prepare_job_result(result)
    assert raised.value.error_code == "invalid_job_result"
    assert raised.value.error_message == "The job produced an invalid result."


@pytest.mark.parametrize(
    "result",
    [
        {"x": "a" * 16_385},
        {"a" * 16_385: "x"},
    ],
)
def test_prepare_job_result_rejects_obviously_oversized_single_strings(result):
    from jobs.handlers import TerminalJobError
    from jobs.worker import _prepare_job_result

    with pytest.raises(TerminalJobError) as raised:
        _prepare_job_result(result)
    assert raised.value.error_code == "invalid_job_result"


def test_prepare_job_result_rejects_cumulative_small_string_bytes_over_limit():
    from jobs.handlers import TerminalJobError
    from jobs.worker import _prepare_job_result

    result = {f"key-{index:04d}": "v" * 100 for index in range(160)}
    assert all(len(key.encode("utf-8")) <= 16_384 for key in result)
    assert all(len(value.encode("utf-8")) <= 16_384 for value in result.values())
    assert sum(len(key.encode("utf-8")) + len(value.encode("utf-8")) for key, value in result.items()) > 16_384

    with pytest.raises(TerminalJobError) as raised:
        _prepare_job_result(result)
    assert raised.value.error_code == "invalid_job_result"


@pytest.mark.asyncio
async def test_postgres_result_validation_uses_parameterized_canonical_byte_count():
    from jobs.worker import _validate_result_in_postgres

    pool = PoolWithConnectionTransaction()
    pool.postgres_result_bytes = 16_384

    assert await _validate_result_in_postgres(pool.connection, '{"x": "value"}') is None
    assert len(pool.result_validation_calls) == 1
    query, serialized = pool.result_validation_calls[0]
    assert " ".join(query.split()).lower() == "select octet_length($1::jsonb::text)"
    assert serialized == '{"x": "value"}'


@pytest.mark.asyncio
async def test_postgres_result_validation_rejects_canonical_byte_count_over_limit():
    from jobs.handlers import TerminalJobError
    from jobs.worker import _validate_result_in_postgres

    pool = PoolWithConnectionTransaction()
    pool.postgres_result_bytes = 16_385

    with pytest.raises(TerminalJobError) as raised:
        await _validate_result_in_postgres(pool.connection, '{"x": "value"}')
    assert raised.value.error_code == "invalid_job_result"
    assert raised.value.error_message == "The job produced an invalid result."


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "validation_error,expected_failure",
    [
        (
            asyncpg.DataError("TOP_SECRET_DB_ERROR RAW_RESULT_TOKEN"),
            {
                "error_code": "invalid_job_result",
                "error_message": "The job produced an invalid result.",
                "retryable": False,
            },
        ),
        (
            asyncpg.ConnectionDoesNotExistError("TOP_SECRET_DB_ERROR RAW_RESULT_TOKEN"),
            {
                "error_code": "unhandled_worker_error",
                "error_message": "The job encountered an unexpected error.",
                "retryable": True,
            },
        ),
    ],
)
async def test_postgres_result_validation_classifies_data_and_operational_errors(
    monkeypatch,
    caplog,
    validation_error,
    expected_failure,
):
    from jobs import worker

    pool = PoolWithConnectionTransaction()
    pool.postgres_validation_error = validation_error
    job = _job()
    succeed_calls = []
    recorded = []

    async def claim(*_args):
        return job

    async def handler(*_args):
        return {"result": "RAW_RESULT_TOKEN"}

    async def succeed(*args):
        succeed_calls.append(args)

    async def fail_or_retry(_conn, _job_id, _owner, **kwargs):
        assert pool.active_connections == 1
        assert pool.active_transactions == 1
        pool.events.append("fail")
        recorded.append(kwargs)
        return _failed_transition(job, kwargs["error_code"])

    monkeypatch.setattr(worker.repository, "claim", claim)
    monkeypatch.setattr(worker.repository, "succeed", succeed)
    monkeypatch.setattr(worker.repository, "fail_or_retry", fail_or_retry)
    monkeypatch.setattr(worker, "JobLease", FakeLease)
    caplog.set_level(logging.ERROR, logger="jobs.worker")

    outcome = await worker.run_job(_ctx(pool, {job.job_type: handler}), str(job.id))

    assert outcome == {"status": "failed", "job_id": str(job.id)}
    assert succeed_calls == []
    assert recorded == [expected_failure]
    assert pool.events == ["postgres-validate", "fail"]
    assert pool.transactions == 3
    assert pool.active_connections == 0
    assert pool.active_transactions == 0
    assert "TOP_SECRET_DB_ERROR" not in caplog.text
    assert "RAW_RESULT_TOKEN" not in caplog.text
    assert "RAW_RESULT_TOKEN" not in repr(recorded)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "invalid_result",
    [
        ["not", "a", "mapping"],
        {"not_finite": float("nan")},
        {"nested": {"bytes": b"TOP_SECRET_RESULT"}},
        {"nested": {"object": object()}},
        {"x": "a" * 16_376},
    ],
)
async def test_run_job_records_invalid_result_as_terminal_without_succeed(
    monkeypatch,
    invalid_result,
):
    from jobs import worker

    pool = PoolWithConnectionTransaction()
    job = _job()
    succeeded = []
    recorded = []

    async def claim(*_args):
        return job

    async def handler(*_args):
        return invalid_result

    async def succeed(*args):
        succeeded.append(args)

    async def fail_or_retry(_conn, _job_id, _owner, **kwargs):
        recorded.append(kwargs)
        return _failed_transition(job, kwargs["error_code"])

    monkeypatch.setattr(worker.repository, "claim", claim)
    monkeypatch.setattr(worker.repository, "succeed", succeed)
    monkeypatch.setattr(worker.repository, "fail_or_retry", fail_or_retry)
    monkeypatch.setattr(worker, "JobLease", FakeLease)

    outcome = await worker.run_job(_ctx(pool, {job.job_type: handler}), str(job.id))

    assert outcome == {"status": "failed", "job_id": str(job.id)}
    assert succeeded == []
    assert recorded == [
        {
            "error_code": "invalid_job_result",
            "error_message": "The job produced an invalid result.",
            "retryable": False,
        }
    ]
    assert "TOP_SECRET_RESULT" not in repr(recorded)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "failure,retryable,code,message",
    [
        ("retryable", True, "converter_timeout", "Converter timed out."),
        ("terminal", False, "invalid_document", "Document is invalid."),
        ("cancelled", False, "cancelled", "Job cancellation was requested."),
        ("unexpected", True, "unhandled_worker_error", "The job encountered an unexpected error."),
    ],
)
async def test_run_job_records_sanitized_handler_failures(
    monkeypatch,
    failure,
    retryable,
    code,
    message,
    caplog,
):
    from jobs import worker
    from jobs.handlers import RetryableJobError, TerminalJobError
    from jobs.models import JobCancelled, JobState

    pool = PoolWithConnectionTransaction()
    job = _job()
    recorded = []

    async def claim(*_args):
        return job

    async def handler(*_args):
        if failure == "retryable":
            raise RetryableJobError("converter_timeout", "Converter timed out.")
        if failure == "terminal":
            raise TerminalJobError("invalid_document", "Document is invalid.")
        if failure == "cancelled":
            raise JobCancelled("raw cancellation detail")
        raise RuntimeError("raw secret must not be persisted")

    persisted = {
        "retryable": ("retry_wait", "converter_timeout"),
        "terminal": ("failed", "invalid_document"),
        "cancelled": ("cancelled", None),
        "unexpected": ("failed", "attempts_exhausted"),
    }[failure]

    async def fail_or_retry(conn, job_id, owner, **kwargs):
        assert FakeLease.instances[-1].entered
        recorded.append((conn, job_id, owner, kwargs))
        return replace(
            job,
            state=JobState(persisted[0]),
            attempt_count=2,
            error_code=persisted[1],
        )

    monkeypatch.setattr(worker.repository, "claim", claim)
    monkeypatch.setattr(worker.repository, "fail_or_retry", fail_or_retry)
    monkeypatch.setattr(worker, "JobLease", FakeLease)

    with caplog.at_level(logging.INFO, logger="jobs.worker"):
        outcome = await worker.run_job(_ctx(pool, {job.job_type: handler}), str(job.id))

    assert outcome == {"status": "failed", "job_id": str(job.id)}
    assert recorded == [
        (
            pool.connection,
            job.id,
            "worker-test",
            {"error_code": code, "error_message": message, "retryable": retryable},
        )
    ]
    assert "raw secret" not in repr(recorded)
    assert "raw cancellation" not in repr(recorded)
    assert_telemetry_event(
        caplog,
        "durable_job_finished",
        expected={
            "job_id": str(job.id),
            "attempt": 2,
            "state": persisted[0],
            "error_code": persisted[1],
            "replica_role": "worker",
        },
    )


@pytest.mark.asyncio
async def test_unexpected_handler_failure_logs_only_stable_fields(monkeypatch, caplog):
    from jobs import worker

    pool = PoolWithConnectionTransaction()
    job = _job()
    recorded = []

    async def claim(*_args):
        return job

    async def handler(*_args):
        raise RuntimeError("TOP_SECRET_TOKEN")

    async def fail_or_retry(_conn, _job_id, _owner, **kwargs):
        recorded.append(kwargs)
        return _failed_transition(job, kwargs["error_code"])

    monkeypatch.setattr(worker.repository, "claim", claim)
    monkeypatch.setattr(worker.repository, "fail_or_retry", fail_or_retry)
    monkeypatch.setattr(worker, "JobLease", FakeLease)
    caplog.set_level(logging.ERROR, logger="jobs.worker")

    outcome = await worker.run_job(_ctx(pool, {job.job_type: handler}), str(job.id))

    assert outcome == {"status": "failed", "job_id": str(job.id)}
    assert recorded == [
        {
            "error_code": "unhandled_worker_error",
            "error_message": "The job encountered an unexpected error.",
            "retryable": True,
        }
    ]
    assert str(job.id) in caplog.text
    assert "RuntimeError" in caplog.text
    assert "TOP_SECRET_TOKEN" not in caplog.text
    assert all(record.exc_info is None for record in caplog.records)


@pytest.mark.asyncio
async def test_persistence_failure_never_logs_or_persists_failing_result(monkeypatch, caplog):
    from jobs import worker

    class FakeCheckViolationError(Exception):
        pass

    pool = PoolWithConnectionTransaction()
    job = _job()
    recorded = []

    async def claim(*_args):
        return job

    async def handler(*_args):
        return {"result": "RAW_RESULT_TOKEN"}

    async def succeed(*_args):
        raise FakeCheckViolationError("TOP_SECRET_TOKEN row contains RAW_RESULT_TOKEN")

    async def fail_or_retry(_conn, _job_id, _owner, **kwargs):
        recorded.append(kwargs)
        return _failed_transition(job, kwargs["error_code"])

    monkeypatch.setattr(worker.repository, "claim", claim)
    monkeypatch.setattr(worker.repository, "succeed", succeed)
    monkeypatch.setattr(worker.repository, "fail_or_retry", fail_or_retry)
    monkeypatch.setattr(worker, "JobLease", FakeLease)
    caplog.set_level(logging.ERROR, logger="jobs.worker")

    outcome = await worker.run_job(_ctx(pool, {job.job_type: handler}), str(job.id))

    assert outcome == {"status": "failed", "job_id": str(job.id)}
    assert recorded == [
        {
            "error_code": "unhandled_worker_error",
            "error_message": "The job encountered an unexpected error.",
            "retryable": True,
        }
    ]
    assert "FakeCheckViolationError" in caplog.text
    assert "TOP_SECRET_TOKEN" not in caplog.text
    assert "RAW_RESULT_TOKEN" not in caplog.text
    assert "RAW_RESULT_TOKEN" not in repr(recorded)
    assert all(record.exc_info is None for record in caplog.records)


@pytest.mark.asyncio
async def test_run_job_missing_handler_records_terminal_unsupported(monkeypatch):
    from jobs import worker

    pool = PoolWithConnectionTransaction()
    job = _job()
    recorded = []

    async def claim(*_args):
        return job

    async def fail_or_retry(*_args, **kwargs):
        recorded.append(kwargs)
        return _failed_transition(job, kwargs["error_code"])

    monkeypatch.setattr(worker.repository, "claim", claim)
    monkeypatch.setattr(worker.repository, "fail_or_retry", fail_or_retry)
    monkeypatch.setattr(worker, "JobLease", FakeLease)

    outcome = await worker.run_job(_ctx(pool, {}), str(job.id))

    assert outcome["status"] == "failed"
    assert recorded == [
        {
            "error_code": "unsupported_job_type",
            "error_message": "This job type is not supported.",
            "retryable": False,
        }
    ]


@pytest.mark.asyncio
async def test_run_job_never_mutates_after_lease_loss(monkeypatch, caplog):
    from jobs import worker
    from jobs.models import LeaseLost

    pool = PoolWithConnectionTransaction()
    job = _job()
    fail_calls = []

    async def claim(*_args):
        return job

    async def handler(*_args):
        raise LeaseLost("lost")

    async def fail_or_retry(*args, **kwargs):
        fail_calls.append((args, kwargs))

    monkeypatch.setattr(worker.repository, "claim", claim)
    monkeypatch.setattr(worker.repository, "fail_or_retry", fail_or_retry)
    monkeypatch.setattr(worker, "JobLease", FakeLease)

    with caplog.at_level(logging.INFO, logger="jobs.worker"):
        outcome = await worker.run_job(_ctx(pool, {job.job_type: handler}), str(job.id))

    assert outcome == {"status": "lease_lost", "job_id": str(job.id)}
    assert fail_calls == []
    assert_telemetry_event(caplog, "durable_job_finished", count=0)


@pytest.mark.asyncio
async def test_run_job_returns_lost_if_failure_transition_loses_lease(monkeypatch, caplog):
    from jobs import worker
    from jobs.handlers import RetryableJobError
    from jobs.models import LeaseLost

    pool = PoolWithConnectionTransaction()
    job = _job()

    async def claim(*_args):
        return job

    async def handler(*_args):
        raise RetryableJobError("transient", "Try again.")

    async def fail_or_retry(*_args, **_kwargs):
        raise LeaseLost("lost")

    monkeypatch.setattr(worker.repository, "claim", claim)
    monkeypatch.setattr(worker.repository, "fail_or_retry", fail_or_retry)
    monkeypatch.setattr(worker, "JobLease", FakeLease)

    with caplog.at_level(logging.INFO, logger="jobs.worker"):
        outcome = await worker.run_job(_ctx(pool, {job.job_type: handler}), str(job.id))
    assert outcome == {"status": "lease_lost", "job_id": str(job.id)}
    assert_telemetry_event(caplog, "durable_job_finished", count=0)


@pytest.mark.asyncio
async def test_disabled_build_wiki_job_fails_terminal_without_retry_wait(monkeypatch):
    from jobs import worker
    from jobs.handlers import HANDLERS, WorkerContext
    from jobs.models import JobState, JobType
    from rag import handler as rag_handler

    pool = PoolWithConnectionTransaction()
    run_id = uuid4()
    job = _job(
        job_type=JobType.BUILD_WIKI,
        state=JobState.RUNNING,
        knowledge_base_id=uuid4(),
        payload={"run_id": str(run_id)},
    )
    run = _rag_run_for_job(job)
    recorded = []

    async def claim(*_args):
        return job

    async def get_for_worker(*_args):
        return run

    async def fail_or_retry(_conn, _job_id, _owner, **kwargs):
        recorded.append(kwargs)
        state = JobState.RETRY_WAIT if kwargs["retryable"] else JobState.FAILED
        return replace(job, state=state, error_code=kwargs["error_code"], attempt_count=1)

    async def record_terminal(_conn, _transition):
        return run

    monkeypatch.setattr(worker.repository, "claim", claim)
    monkeypatch.setattr(worker.repository, "fail_or_retry", fail_or_retry)
    monkeypatch.setattr(worker, "_record_terminal_rag_run", record_terminal)
    monkeypatch.setattr(rag_handler.repository, "get_for_worker", get_for_worker)
    monkeypatch.setattr(worker, "JobLease", FakeLease)
    context = WorkerContext(
        pool=pool,
        s3=None,
        converter_url="https://converter.invalid",
        converter_secret="secret",
        rag_orchestrator_factory=None,
    )
    ctx = _ctx(pool, HANDLERS)
    ctx["worker_context"] = context

    outcome = await worker.run_job(ctx, str(job.id))

    assert outcome == {"status": "failed", "job_id": str(job.id)}
    assert recorded == [
        {
            "error_code": "rag_disabled",
            "error_message": "Server-side RAG is disabled.",
            "retryable": False,
        }
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("state", "error_code", "expected_completion"),
    [
        ("retry_wait", "rag_model_unavailable", None),
        ("cancelled", None, None),
        ("failed", "rag_budget_exhausted", "budget_exhausted"),
        ("failed", "rag_invalid_plan", "partial_failure"),
    ],
)
async def test_rag_failure_transition_updates_run_only_for_terminal_failure_in_same_transaction(
    monkeypatch,
    state,
    error_code,
    expected_completion,
):
    from jobs import worker
    from jobs.models import JobState, JobType

    pool = PoolWithConnectionTransaction()
    run_id = uuid4()
    job = _job(
        job_type=JobType.BUILD_WIKI,
        state=JobState.RUNNING,
        knowledge_base_id=uuid4(),
        payload={"run_id": str(run_id)},
    )
    transition = replace(job, state=JobState(state), error_code=error_code)
    calls = []

    async def fail_or_retry(conn, *_args, **_kwargs):
        assert pool.active_transactions == 1
        calls.append(("job", conn))
        return transition

    async def record_terminal(conn, record):
        assert pool.active_transactions == 1
        calls.append(("run", conn, record))

    monkeypatch.setattr(worker.repository, "fail_or_retry", fail_or_retry)
    monkeypatch.setattr(worker, "_record_terminal_rag_run", record_terminal)

    recorded = await worker._record_failure(
        pool,
        job.id,
        "worker-safe",
        error_code=error_code or "cancelled",
        error_message="safe",
        retryable=state == "retry_wait",
    )

    assert recorded is transition
    expected = [("job", pool.connection)]
    if expected_completion is not None:
        expected.append(("run", pool.connection, transition))
    assert calls == expected
    assert pool.transactions == 1


@pytest.mark.asyncio
async def test_terminal_rag_failure_emits_once_after_commit_with_authoritative_counters(
    monkeypatch,
    caplog,
):
    from jobs import worker
    from jobs.models import JobState, JobType
    from rag import handler as rag_handler

    from llmwiki_core.rag import RagCompletionReason, RagUsage

    pool = PoolWithConnectionTransaction()
    run_id = uuid4()
    job = _job(
        job_type=JobType.BUILD_WIKI,
        state=JobState.RUNNING,
        knowledge_base_id=uuid4(),
        payload={"run_id": str(run_id)},
    )
    transition = replace(
        job,
        state=JobState.FAILED,
        attempt_count=3,
        error_code="attempts_exhausted",
    )
    authoritative = _rag_run_for_job(
        job,
        completion_reason=RagCompletionReason.PARTIAL_FAILURE,
        usage=RagUsage(steps=11, model_tokens=29),
        last_committed_ordinal=2,
    )
    order = []

    async def fail_or_retry(*_args, **_kwargs):
        assert pool.active_transactions == 1
        order.append("job")
        return transition

    async def record_terminal(*_args):
        assert pool.active_transactions == 1
        order.append("run")
        return authoritative

    original_emit = rag_handler.emit_rag_run_failed

    def post_commit_emit(selected_transition, selected_run):
        assert pool.active_transactions == 0
        order.append("event")
        original_emit(selected_transition, selected_run)

    monkeypatch.setattr(worker.repository, "fail_or_retry", fail_or_retry)
    monkeypatch.setattr(worker, "_record_terminal_rag_run", record_terminal)
    monkeypatch.setattr(rag_handler, "emit_rag_run_failed", post_commit_emit)

    with caplog.at_level(logging.INFO, logger="rag.handler"):
        recorded = await worker._record_failure(
            pool,
            job.id,
            "worker-safe",
            error_code="rag_model_unavailable",
            error_message="safe",
            retryable=True,
        )

    assert recorded is transition
    assert order == ["job", "run", "event"]
    assert_telemetry_event(
        caplog,
        "rag_run_failed",
        count=1,
        expected={
            "run_id": str(run_id),
            "job_id": str(job.id),
            "page_count": 3,
            "step_count": 11,
            "model_token_count": 29,
            "error_code": "attempts_exhausted",
        },
        sensitive=("private goal", "/wiki/private/", "private-key"),
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("state", "requested_code"),
    [
        ("retry_wait", "rag_model_unavailable"),
        ("cancelled", "cancelled"),
        ("failed", "worker_shutdown"),
    ],
)
async def test_nonterminal_and_shutdown_rag_failures_emit_no_failed_event(
    monkeypatch,
    state,
    requested_code,
):
    from jobs import worker
    from jobs.models import JobState, JobType
    from rag import handler as rag_handler

    pool = PoolWithConnectionTransaction()
    run_id = uuid4()
    job = _job(
        job_type=JobType.BUILD_WIKI,
        state=JobState.RUNNING,
        knowledge_base_id=uuid4(),
        payload={"run_id": str(run_id)},
    )
    transition = replace(job, state=JobState(state), error_code=requested_code)
    events = []

    async def fail_or_retry(*_args, **_kwargs):
        return transition

    monkeypatch.setattr(worker.repository, "fail_or_retry", fail_or_retry)
    monkeypatch.setattr(rag_handler, "emit_rag_run_failed", lambda *_args: events.append("event"))
    if state == "failed":

        async def record_terminal(*_args):
            return _rag_run_for_job(job)

        monkeypatch.setattr(worker, "_record_terminal_rag_run", record_terminal)

    await worker._record_failure(
        pool,
        job.id,
        "worker-safe",
        error_code=requested_code,
        error_message="safe",
        retryable=state == "retry_wait",
    )

    assert events == []


@pytest.mark.asyncio
@pytest.mark.parametrize("failure_point", ["terminal_sync", "transaction_commit"])
async def test_rolled_back_rag_failure_transition_emits_no_failed_event(
    monkeypatch,
    failure_point,
):
    from jobs import worker
    from jobs.models import JobState, JobType
    from rag import handler as rag_handler

    class CommitFailingPool(PoolWithConnectionTransaction):
        @asynccontextmanager
        async def transaction(self):
            self.transactions += 1
            self.active_transactions += 1
            try:
                yield
            finally:
                self.active_transactions -= 1
            raise RuntimeError("TOP_SECRET_COMMIT_FAILURE")

    pool = CommitFailingPool() if failure_point == "transaction_commit" else PoolWithConnectionTransaction()
    run_id = uuid4()
    job = _job(
        job_type=JobType.BUILD_WIKI,
        state=JobState.RUNNING,
        knowledge_base_id=uuid4(),
        payload={"run_id": str(run_id)},
    )
    transition = replace(job, state=JobState.FAILED, error_code="rag_invalid_plan")
    events = []

    async def fail_or_retry(*_args, **_kwargs):
        return transition

    async def record_terminal(*_args):
        if failure_point == "terminal_sync":
            raise RuntimeError("TOP_SECRET_SYNC_FAILURE")
        return _rag_run_for_job(job)

    monkeypatch.setattr(worker.repository, "fail_or_retry", fail_or_retry)
    monkeypatch.setattr(worker, "_record_terminal_rag_run", record_terminal)
    monkeypatch.setattr(rag_handler, "emit_rag_run_failed", lambda *_args: events.append("event"))

    with pytest.raises(RuntimeError):
        await worker._record_failure(
            pool,
            job.id,
            "worker-safe",
            error_code="rag_invalid_plan",
            error_message="safe",
            retryable=False,
        )

    assert events == []


@pytest.mark.asyncio
async def test_post_commit_rag_telemetry_failure_does_not_change_committed_transition(monkeypatch, caplog):
    from jobs import worker
    from jobs.models import JobState, JobType
    from rag import handler as rag_handler

    pool = PoolWithConnectionTransaction()
    run_id = uuid4()
    job = _job(
        job_type=JobType.BUILD_WIKI,
        state=JobState.RUNNING,
        knowledge_base_id=uuid4(),
        payload={"run_id": str(run_id)},
    )
    transition = replace(job, state=JobState.FAILED, error_code="rag_invalid_plan")

    async def fail_or_retry(*_args, **_kwargs):
        return transition

    async def record_terminal(*_args):
        return _rag_run_for_job(job)

    def fail_telemetry(*_args):
        assert pool.active_transactions == 0
        raise RuntimeError("TOP_SECRET_TELEMETRY_FAILURE")

    monkeypatch.setattr(worker.repository, "fail_or_retry", fail_or_retry)
    monkeypatch.setattr(worker, "_record_terminal_rag_run", record_terminal)
    monkeypatch.setattr(rag_handler, "emit_rag_run_failed", fail_telemetry)

    with caplog.at_level(logging.ERROR, logger="jobs.worker"):
        assert (
            await worker._record_failure(
                pool,
                job.id,
                "worker-safe",
                error_code="rag_invalid_plan",
                error_message="safe",
                retryable=False,
            )
            is transition
        )
    assert "TOP_SECRET_TELEMETRY_FAILURE" not in caplog.text


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("error_code", "completion"),
    [
        ("rag_budget_exhausted", "budget_exhausted"),
        ("rag_invalid_draft", "partial_failure"),
    ],
)
async def test_terminal_rag_run_completion_uses_exact_payload_binding(monkeypatch, error_code, completion):
    from jobs import worker
    from jobs.models import JobState, JobType
    from rag import repository as rag_repository

    run_id = uuid4()
    transition = _job(
        job_type=JobType.BUILD_WIKI,
        state=JobState.FAILED,
        knowledge_base_id=uuid4(),
        payload={"run_id": str(run_id)},
        error_code=error_code,
    )
    from llmwiki_core.rag import RagCompletionReason

    run = _rag_run_for_job(transition)
    terminal_run = replace(run, completion_reason=RagCompletionReason(completion))
    calls = []

    async def get_for_worker(conn, selected_run_id, selected_job_id):
        assert conn is connection
        assert selected_run_id == run_id
        assert selected_job_id == transition.id
        return run

    async def record_terminal_job_state(conn, **kwargs):
        calls.append((conn, kwargs))
        return terminal_run

    connection = object()
    monkeypatch.setattr(rag_repository, "get_for_worker", get_for_worker)
    monkeypatch.setattr(rag_repository, "record_terminal_job_state", record_terminal_job_state)
    recorded = await worker._record_terminal_rag_run(connection, transition)

    assert recorded is terminal_run
    assert calls[0][0] is connection
    assert calls[0][1]["run_id"] == run_id
    assert calls[0][1]["completion_reason"].value == completion


@pytest.mark.asyncio
@pytest.mark.parametrize("completion", ["budget_exhausted", "partial_failure"])
async def test_terminal_rag_run_reuses_existing_authoritative_failure_completion(monkeypatch, completion):
    from jobs import worker
    from jobs.models import JobState, JobType
    from rag import repository as rag_repository

    from llmwiki_core.rag import RagCompletionReason

    run_id = uuid4()
    transition = _job(
        job_type=JobType.BUILD_WIKI,
        state=JobState.FAILED,
        knowledge_base_id=uuid4(),
        payload={"run_id": str(run_id)},
        error_code="attempts_exhausted",
    )
    terminal_run = _rag_run_for_job(
        transition,
        completion_reason=RagCompletionReason(completion),
    )

    async def get_for_worker(*_args):
        return terminal_run

    async def record_terminal_job_state(*_args, **_kwargs):
        raise AssertionError("an already terminal failure run must not be rewritten")

    monkeypatch.setattr(rag_repository, "get_for_worker", get_for_worker)
    monkeypatch.setattr(rag_repository, "record_terminal_job_state", record_terminal_job_state)

    assert await worker._record_terminal_rag_run(object(), transition) is terminal_run


@pytest.mark.asyncio
async def test_run_job_propagates_worker_cancellation(monkeypatch, caplog):
    from jobs import worker
    from jobs.models import JobState

    pool = PoolWithConnectionTransaction()
    job = _job()

    async def claim(*_args):
        return job

    async def handler(*_args):
        raise asyncio.CancelledError

    recorded = []

    async def record_failure(*args, **kwargs):
        recorded.append((args, kwargs))
        return replace(
            job,
            state=JobState.RETRY_WAIT,
            attempt_count=1,
            error_code="worker_shutdown",
        )

    monkeypatch.setattr(worker.repository, "claim", claim)
    monkeypatch.setattr(worker, "JobLease", FakeLease)
    monkeypatch.setattr(worker, "_record_failure", record_failure)

    with caplog.at_level(logging.INFO, logger="jobs.worker"), pytest.raises(asyncio.CancelledError):
        await worker.run_job(_ctx(pool, {job.job_type: handler}), str(job.id))

    assert recorded[0][1] == {
        "error_code": "worker_shutdown",
        "error_message": "Worker shutdown interrupted the job.",
        "retryable": True,
    }
    assert_telemetry_event(
        caplog,
        "durable_job_finished",
        expected={
            "job_id": str(job.id),
            "attempt": 1,
            "state": "retry_wait",
            "error_code": "worker_shutdown",
            "replica_role": "worker",
        },
    )


@pytest.mark.asyncio
async def test_run_job_propagates_custom_base_exception_after_lease_cleanup(monkeypatch):
    from jobs import worker

    class WorkerShutdownSignal(BaseException):
        pass

    pool = PoolWithConnectionTransaction()
    job = _job()

    async def claim(*_args):
        return job

    async def handler(*_args):
        raise WorkerShutdownSignal("stop worker")

    monkeypatch.setattr(worker.repository, "claim", claim)
    monkeypatch.setattr(worker, "JobLease", FakeLease)

    with pytest.raises(WorkerShutdownSignal):
        await worker.run_job(_ctx(pool, {job.job_type: handler}), str(job.id))

    lease = FakeLease.instances[-1]
    assert lease.entered is False
    assert pool.active_connections == 0
    assert pool.active_transactions == 0


@pytest.mark.asyncio
async def test_reap_cron_uses_short_transaction_and_propagates_failures(monkeypatch):
    from jobs import worker

    pool = PoolWithConnectionTransaction()
    failure = RuntimeError("reaper database unavailable")

    async def reap_expired(conn, *, limit, include_transitions):
        assert conn is pool.connection
        assert limit == 23
        assert include_transitions is True
        raise failure

    _install_legacy_reaper_mock(monkeypatch, worker, reap_expired)
    with pytest.raises(RuntimeError, match="reaper database unavailable") as raised:
        await worker.reap_cron({"pool": pool, "reap_batch_size": 23})
    assert raised.value is failure
    assert pool.transactions == 1
    assert pool.active_connections == 0
    assert pool.active_transactions == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("reason", ["completed", "no_work", "dry_run"])
async def test_reap_cron_recovers_terminal_successful_rag_run_without_failure_event(
    monkeypatch,
    caplog,
    reason,
):
    from jobs import worker
    from jobs.models import JobState, JobType
    from rag import handler as rag_handler

    from llmwiki_core.rag import RagCompletionReason

    pool = PoolWithConnectionTransaction()
    run_id = uuid4()
    candidate = _job(
        job_type=JobType.BUILD_WIKI,
        state=JobState.RUNNING,
        knowledge_base_id=uuid4(),
        payload={"run_id": str(run_id)},
        attempt_count=3,
        max_attempts=3,
        lease_owner="dead-worker",
    )
    canonical = {
        "run_id": str(run_id),
        "completion_reason": reason,
        "pages_committed": 1 if reason == "completed" else 0,
    }
    if reason == "dry_run":
        canonical["pages_dry_run"] = 1
    succeeded = replace(
        candidate,
        state=JobState.SUCCEEDED,
        result=canonical,
        lease_owner=None,
    )
    terminal_run = _rag_run_for_job(
        candidate,
        dry_run=reason == "dry_run",
        completion_reason=RagCompletionReason(reason),
        last_committed_ordinal=-1 if reason == "no_work" else 0,
    )
    calls = []

    async def lock_expired(conn, *, limit):
        calls.append(("lock", conn, limit))
        return [candidate]

    async def recover(conn, job):
        calls.append(("recover", conn, job.id))
        return succeeded, terminal_run

    async def reap_locked(conn, jobs, *, include_transitions):
        calls.append(("reap", conn, tuple(job.id for job in jobs), include_transitions))
        return []

    failure_events = []
    monkeypatch.setattr(worker.repository, "lock_expired_for_reap", lock_expired, raising=False)
    monkeypatch.setattr(worker.repository, "reap_locked", reap_locked, raising=False)
    monkeypatch.setattr(worker, "_recover_terminal_rag_success", recover, raising=False)
    monkeypatch.setattr(rag_handler, "emit_rag_run_failed", lambda *_args: failure_events.append("failed"))

    with caplog.at_level(logging.INFO, logger="jobs.worker"):
        await worker.reap_cron({"pool": pool, "reap_batch_size": 7})

    assert calls == [
        ("lock", pool.connection, 7),
        ("recover", pool.connection, candidate.id),
        ("reap", pool.connection, (), True),
    ]
    assert failure_events == []
    assert_telemetry_event(
        caplog,
        "durable_job_lease_reaped",
        expected={"job_id": str(candidate.id), "state": "succeeded", "error_code": None},
    )


@pytest.mark.asyncio
async def test_reap_cron_pending_cancel_wins_over_terminal_rag_success_recovery(monkeypatch, caplog):
    from jobs import worker
    from jobs.models import JobState, JobType
    from rag import handler as rag_handler

    pool = PoolWithConnectionTransaction()
    run_id = uuid4()
    candidate = _job(
        job_type=JobType.BUILD_WIKI,
        state=JobState.RUNNING,
        knowledge_base_id=uuid4(),
        payload={"run_id": str(run_id)},
        attempt_count=3,
        max_attempts=3,
        lease_owner="dead-worker",
        cancel_requested_at=datetime(2026, 7, 27, tzinfo=UTC),
    )
    cancelled = replace(
        candidate,
        state=JobState.CANCELLED,
        lease_owner=None,
        error_code=None,
    )
    calls = []

    async def lock_expired(*_args, **_kwargs):
        return [candidate]

    async def recover(*_args):
        raise AssertionError("pending cancellation must bypass terminal success recovery")

    async def reap_locked(_conn, jobs, *, include_transitions):
        calls.append(tuple(job.id for job in jobs))
        assert include_transitions is True
        return [cancelled]

    failure_events = []
    monkeypatch.setattr(worker.repository, "lock_expired_for_reap", lock_expired)
    monkeypatch.setattr(worker.repository, "reap_locked", reap_locked)
    monkeypatch.setattr(worker, "_recover_terminal_rag_success", recover)
    monkeypatch.setattr(rag_handler, "emit_rag_run_failed", lambda *_args: failure_events.append("failed"))

    with caplog.at_level(logging.INFO, logger="jobs.worker"):
        await worker.reap_cron({"pool": pool, "reap_batch_size": 1})

    assert calls == [(candidate.id,)]
    assert failure_events == []
    assert_telemetry_event(
        caplog,
        "durable_job_lease_reaped",
        expected={"job_id": str(candidate.id), "state": "cancelled", "error_code": None},
    )


@pytest.mark.asyncio
async def test_reap_cron_malformed_rag_candidate_fails_closed_without_blocking_batch(monkeypatch, caplog):
    from jobs import worker
    from jobs.handlers import TerminalJobError
    from jobs.models import JobState, JobType

    pool = PoolWithConnectionTransaction()
    malformed_run_id = uuid4()
    malformed = _job(
        job_type=JobType.BUILD_WIKI,
        state=JobState.RUNNING,
        knowledge_base_id=uuid4(),
        payload={"run_id": str(malformed_run_id)},
        attempt_count=3,
        max_attempts=3,
    )
    ordinary = _job(state=JobState.RUNNING, attempt_count=3, max_attempts=3)
    failed_malformed = replace(malformed, state=JobState.FAILED, error_code="attempts_exhausted")
    failed_ordinary = replace(ordinary, state=JobState.FAILED, error_code="attempts_exhausted")
    synced = []

    async def lock_expired(*_args, **_kwargs):
        return [malformed, ordinary]

    async def recover(*_args):
        raise TerminalJobError("rag_job_binding_invalid", "The RAG job binding is invalid.")

    async def reap_locked(_conn, jobs, *, include_transitions):
        assert tuple(job.id for job in jobs) == (malformed.id, ordinary.id)
        assert include_transitions is True
        return [failed_malformed, failed_ordinary]

    async def record_terminal(_conn, transition):
        synced.append(transition.id)
        raise AssertionError("malformed RAG run must not be failure-synchronized")

    monkeypatch.setattr(worker.repository, "lock_expired_for_reap", lock_expired, raising=False)
    monkeypatch.setattr(worker.repository, "reap_locked", reap_locked, raising=False)
    monkeypatch.setattr(worker, "_recover_terminal_rag_success", recover, raising=False)
    monkeypatch.setattr(worker, "_record_terminal_rag_run", record_terminal)

    with caplog.at_level(logging.INFO, logger="jobs.worker"):
        await worker.reap_cron({"pool": pool, "reap_batch_size": 2})

    assert synced == []
    assert_telemetry_event(caplog, "durable_job_lease_reaped", count=2)


@pytest.mark.asyncio
async def test_terminal_rag_success_recovery_maps_malformed_repository_snapshot_fail_closed(monkeypatch):
    from jobs import worker
    from jobs.handlers import TerminalJobError
    from jobs.models import JobState, JobType
    from rag import repository as rag_repository

    from llmwiki_core.rag import RagDomainError

    run_id = uuid4()
    candidate = _job(
        job_type=JobType.BUILD_WIKI,
        state=JobState.RUNNING,
        knowledge_base_id=uuid4(),
        payload={"run_id": str(run_id)},
    )

    async def malformed_snapshot(*_args, **_kwargs):
        raise RagDomainError("rag_job_binding_invalid", "TOP_SECRET_MALFORMED")

    monkeypatch.setattr(rag_repository, "get_terminal_snapshot_for_worker", malformed_snapshot)

    with pytest.raises(TerminalJobError) as raised:
        await worker._recover_terminal_rag_success(object(), candidate)

    assert (raised.value.error_code, raised.value.error_message) == (
        "rag_job_binding_invalid",
        "The RAG job binding is invalid.",
    )
    assert raised.value.__cause__ is None
    assert raised.value.__context__ is None
    assert "TOP_SECRET" not in repr(raised.value)


@pytest.mark.asyncio
async def test_reap_cron_atomically_syncs_failed_rag_batch_then_emits_authoritative_runs(
    monkeypatch,
    caplog,
):
    from jobs import worker
    from jobs.models import JobState, JobType
    from rag import handler as rag_handler

    from llmwiki_core.rag import RagCompletionReason, RagUsage

    class StrictPool(PoolWithConnectionTransaction):
        @asynccontextmanager
        async def transaction(self):
            self.transactions += 1
            self.active_transactions += 1
            self.events.append("begin")
            try:
                yield
            except BaseException:
                self.events.append("rollback")
                raise
            else:
                self.events.append("commit")
            finally:
                self.active_transactions -= 1

    def rag_transition(state, *, error_code=None):
        run_id = uuid4()
        return _job(
            job_type=JobType.BUILD_WIKI,
            state=JobState(state),
            knowledge_base_id=uuid4(),
            payload={"run_id": str(run_id)},
            error_code=error_code,
        )

    pool = StrictPool()
    first = rag_transition("failed", error_code="attempts_exhausted")
    retry = rag_transition("retry_wait", error_code="lease_expired")
    non_rag = _job(state=JobState.FAILED, error_code="attempts_exhausted")
    cancelled = rag_transition("cancelled")
    second = rag_transition("failed", error_code="attempts_exhausted")
    transitions = [first, retry, non_rag, cancelled, second]
    terminal_runs = {
        first.id: _rag_run_for_job(
            first,
            completion_reason=RagCompletionReason.PARTIAL_FAILURE,
            usage=RagUsage(steps=5, model_tokens=13),
            last_committed_ordinal=0,
        ),
        second.id: _rag_run_for_job(
            second,
            completion_reason=RagCompletionReason.PARTIAL_FAILURE,
            usage=RagUsage(steps=9, model_tokens=31),
            last_committed_ordinal=2,
        ),
    }

    async def reap_expired(conn, *, limit, include_transitions):
        assert conn is pool.connection
        assert limit == 10
        assert include_transitions is True
        assert pool.active_transactions == 1
        pool.events.append("reap")
        return transitions

    async def record_terminal(conn, transition):
        assert conn is pool.connection
        assert pool.active_transactions == 1
        pool.events.append(("sync", transition.id))
        return terminal_runs[transition.id]

    original_emit = rag_handler.emit_rag_run_failed

    def emit_failed(transition, run):
        assert pool.active_transactions == 0
        pool.events.append(("rag_event", transition.id))
        original_emit(transition, run)

    _install_legacy_reaper_mock(monkeypatch, worker, reap_expired)
    monkeypatch.setattr(worker, "_record_terminal_rag_run", record_terminal)
    monkeypatch.setattr(rag_handler, "emit_rag_run_failed", emit_failed)

    with caplog.at_level(logging.INFO):
        await worker.reap_cron({"pool": pool, "reap_batch_size": 10})

    assert pool.events[:5] == [
        "begin",
        "reap",
        ("sync", first.id),
        ("sync", second.id),
        "commit",
    ]
    assert pool.events[5:] == [
        ("rag_event", first.id),
        ("rag_event", second.id),
    ]
    assert retry.id not in terminal_runs
    assert cancelled.id not in terminal_runs
    assert_telemetry_event(
        caplog,
        "rag_run_failed",
        count=2,
        expected={
            "run_id": str(terminal_runs[second.id].id),
            "job_id": str(second.id),
            "page_count": 3,
            "step_count": 9,
            "model_token_count": 31,
            "error_code": "attempts_exhausted",
        },
    )
    assert_telemetry_event(caplog, "durable_job_lease_reaped", count=len(transitions))


@pytest.mark.asyncio
async def test_reap_cron_terminal_rag_sync_failure_rolls_back_batch_and_emits_nothing(
    monkeypatch,
):
    from jobs import worker
    from jobs.models import JobState, JobType
    from rag import handler as rag_handler

    class StrictPool(PoolWithConnectionTransaction):
        @asynccontextmanager
        async def transaction(self):
            self.transactions += 1
            self.active_transactions += 1
            self.events.append("begin")
            try:
                yield
            except BaseException:
                self.events.append("rollback")
                raise
            else:
                self.events.append("commit")
            finally:
                self.active_transactions -= 1

    def failed_rag():
        run_id = uuid4()
        return _job(
            job_type=JobType.BUILD_WIKI,
            state=JobState.FAILED,
            knowledge_base_id=uuid4(),
            payload={"run_id": str(run_id)},
            error_code="attempts_exhausted",
        )

    pool = StrictPool()
    first, second = failed_rag(), failed_rag()
    events = []

    async def reap_expired(*_args, **_kwargs):
        return [first, second]

    async def record_terminal(_conn, transition):
        pool.events.append(("sync", transition.id))
        if transition is second:
            raise RuntimeError("TOP_SECRET_TERMINAL_SYNC")
        return _rag_run_for_job(first)

    _install_legacy_reaper_mock(monkeypatch, worker, reap_expired)
    monkeypatch.setattr(worker, "_record_terminal_rag_run", record_terminal)
    monkeypatch.setattr(rag_handler, "emit_rag_run_failed", lambda *_args: events.append("event"))

    with pytest.raises(RuntimeError, match="TOP_SECRET_TERMINAL_SYNC"):
        await worker.reap_cron({"pool": pool, "reap_batch_size": 10})

    assert pool.events == [
        "begin",
        ("sync", first.id),
        ("sync", second.id),
        "rollback",
    ]
    assert events == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "sink_failure",
    [
        RuntimeError("TOP_SECRET_TELEMETRY"),
        KeyboardInterrupt("TOP_SECRET_TELEMETRY"),
        asyncio.CancelledError("TOP_SECRET_TELEMETRY"),
    ],
)
async def test_reap_cron_isolates_ordinary_rag_sink_failure_and_sanitizes_controls(
    monkeypatch,
    caplog,
    sink_failure,
):
    from jobs import worker
    from jobs.models import JobState, JobType
    from rag import handler as rag_handler

    from llmwiki_core.rag import RagCompletionReason

    pool = PoolWithConnectionTransaction()
    run_id = uuid4()
    transition = _job(
        job_type=JobType.BUILD_WIKI,
        state=JobState.FAILED,
        knowledge_base_id=uuid4(),
        payload={"run_id": str(run_id)},
        error_code="attempts_exhausted",
    )
    terminal_run = _rag_run_for_job(
        transition,
        completion_reason=RagCompletionReason.PARTIAL_FAILURE,
    )

    async def reap_expired(*_args, **_kwargs):
        return [transition]

    async def record_terminal(*_args):
        return terminal_run

    def fail_sink(*_args):
        assert pool.active_transactions == 0
        raise sink_failure

    _install_legacy_reaper_mock(monkeypatch, worker, reap_expired)
    monkeypatch.setattr(worker, "_record_terminal_rag_run", record_terminal)
    monkeypatch.setattr(rag_handler, "emit_rag_run_failed", fail_sink)

    with caplog.at_level(logging.INFO):
        if isinstance(sink_failure, Exception):
            await worker.reap_cron({"pool": pool, "reap_batch_size": 10})
            assert_telemetry_event(caplog, "durable_job_lease_reaped", count=1)
        else:
            with pytest.raises(type(sink_failure)) as raised:
                await worker.reap_cron({"pool": pool, "reap_batch_size": 10})
            assert raised.value is not sink_failure
            assert str(raised.value) == ""
            assert raised.value.__cause__ is None
            assert raised.value.__context__ is None
            assert_telemetry_event(caplog, "durable_job_lease_reaped", count=0)

    assert pool.active_transactions == 0
    assert "TOP_SECRET_TELEMETRY" not in caplog.text


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "state,error_code",
    [
        ("retry_wait", "lease_expired"),
        ("failed", "attempts_exhausted"),
        ("cancelled", None),
    ],
)
async def test_reap_cron_emits_persisted_transition_for_each_expired_lease(
    monkeypatch,
    caplog,
    state,
    error_code,
):
    from jobs import worker
    from jobs.models import JobState

    pool = PoolWithConnectionTransaction()
    transition = _job(
        state=JobState(state),
        attempt_count=3,
        error_code=error_code,
    )

    async def reap_expired(conn, *, limit, include_transitions):
        assert conn is pool.connection
        assert limit == 23
        assert include_transitions is True
        return [transition]

    _install_legacy_reaper_mock(monkeypatch, worker, reap_expired)
    with caplog.at_level(logging.INFO, logger="jobs.worker"):
        await worker.reap_cron({"pool": pool, "reap_batch_size": 23})

    assert_telemetry_event(
        caplog,
        "durable_job_lease_reaped",
        expected={
            "job_id": str(transition.id),
            "state": state,
            "error_code": error_code,
            "replica_role": "worker",
        },
        sensitive=("Worker lease expired raw text", "postgresql://private.invalid"),
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("state", "transition_error", "outcome", "event_error"),
    [
        ("retry_wait", "lease_expired", "retry", "lease_expired"),
        ("failed", "attempts_exhausted", "terminal", "attempts_exhausted"),
        ("cancelled", None, "terminal", "cancelled"),
    ],
)
async def test_reap_cron_emits_embedding_outcome_once_after_transaction(
    monkeypatch,
    caplog,
    state,
    transition_error,
    outcome,
    event_error,
):
    from jobs import worker
    from jobs.models import JobState, JobType

    pool = PoolWithConnectionTransaction()
    document_id = uuid4()
    transition = _job(
        job_type=JobType.DOCUMENT_EMBED,
        state=JobState(state),
        knowledge_base_id=uuid4(),
        document_id=document_id,
        payload={
            "document_id": str(document_id),
            "document_version": 1,
            "provider": "openai_compatible",
            "model": "embed-v1",
            "dimensions": 3,
        },
        attempt_count=2,
        error_code=transition_error,
    )
    calls = 0

    async def reap_expired(conn, *, limit, include_transitions):
        nonlocal calls
        assert conn is pool.connection
        assert include_transitions is True
        calls += 1
        return [transition] if calls == 1 else []

    emit_embedding_finished = worker._emit_embedding_finished

    def emit_after_commit(record, *, duration_ms):
        assert pool.active_connections == 0
        assert pool.active_transactions == 0
        emit_embedding_finished(record, duration_ms=duration_ms)

    _install_legacy_reaper_mock(monkeypatch, worker, reap_expired)
    monkeypatch.setattr(worker, "_emit_embedding_finished", emit_after_commit)
    with caplog.at_level(logging.INFO, logger="jobs.worker"):
        await worker.reap_cron({"pool": pool, "reap_batch_size": 23})
        await worker.reap_cron({"pool": pool, "reap_batch_size": 23})

    assert pool.active_transactions == 0
    assert_telemetry_event(
        caplog,
        "embedding_finished",
        expected={
            "job_id": str(transition.id),
            "attempt": 2,
            "outcome": outcome,
            "error_code": event_error,
            "provider": "openai_compatible",
            "model": "embed-v1",
            "dimensions": 3,
            "chunk_count": 0,
            "duration_ms": 0,
            "replica_role": "worker",
        },
    )


@pytest.mark.asyncio
async def test_reap_cron_embedding_sink_failure_never_masks_committed_transition(
    monkeypatch,
):
    from jobs import worker
    from jobs.models import JobState, JobType

    pool = PoolWithConnectionTransaction()
    document_id = uuid4()
    transition = _job(
        job_type=JobType.DOCUMENT_EMBED,
        state=JobState.RETRY_WAIT,
        knowledge_base_id=uuid4(),
        document_id=document_id,
        payload={
            "document_id": str(document_id),
            "document_version": 1,
            "provider": "openai_compatible",
            "model": "embed-v1",
            "dimensions": 3,
        },
        attempt_count=1,
        error_code="lease_expired",
    )

    async def reap_expired(*_args, **_kwargs):
        return [transition]

    class FailingEmbeddingSink:
        calls = []

        def info(self, serialized):
            self.calls.append(serialized)
            if '"event":"embedding_finished"' in serialized:
                raise RuntimeError("private sink")

    sink = FailingEmbeddingSink()
    _install_legacy_reaper_mock(monkeypatch, worker, reap_expired)
    monkeypatch.setattr(worker, "logger", sink)

    await worker.reap_cron({"pool": pool, "reap_batch_size": 1})

    assert pool.active_transactions == 0
    assert len(sink.calls) == 2


def _linked_reaper_control(signal):
    failure = RuntimeError("private wrapper")
    failure.__cause__ = signal
    return failure


def _context_reaper_control(signal):
    failure = RuntimeError("private wrapper")
    failure.__context__ = signal
    return failure


class _UnknownReaperSignal(BaseException):
    pass


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("signal", "expected", "exit_code"),
    [
        (KeyboardInterrupt("private"), KeyboardInterrupt, None),
        (SystemExit("private"), SystemExit, 1),
        (asyncio.CancelledError("private"), asyncio.CancelledError, None),
        (GeneratorExit("private"), GeneratorExit, None),
        (_linked_reaper_control(asyncio.CancelledError("private")), asyncio.CancelledError, None),
        (_linked_reaper_control(GeneratorExit("private")), GeneratorExit, None),
        (_context_reaper_control(GeneratorExit("private")), GeneratorExit, None),
        (
            BaseExceptionGroup(
                "private outer",
                [BaseExceptionGroup("private inner", [GeneratorExit("private")])],
            ),
            GeneratorExit,
            None,
        ),
        (
            BaseExceptionGroup(
                "private priority",
                [GeneratorExit("private"), asyncio.CancelledError("private")],
            ),
            asyncio.CancelledError,
            None,
        ),
        (_UnknownReaperSignal("private"), BaseException, None),
        (
            BaseExceptionGroup(
                "private unknown",
                [RuntimeError("ordinary"), _UnknownReaperSignal("private")],
            ),
            BaseException,
            None,
        ),
        (
            BaseExceptionGroup("private", [RuntimeError("ordinary"), KeyboardInterrupt("private")]),
            KeyboardInterrupt,
            None,
        ),
    ],
)
async def test_reap_cron_embedding_sink_controls_propagate_sanitized(
    monkeypatch,
    signal,
    expected,
    exit_code,
):
    from jobs import worker
    from jobs.models import JobState, JobType

    pool = PoolWithConnectionTransaction()
    document_id = uuid4()
    transition = _job(
        job_type=JobType.DOCUMENT_EMBED,
        state=JobState.RETRY_WAIT,
        knowledge_base_id=uuid4(),
        document_id=document_id,
        payload={
            "document_id": str(document_id),
            "document_version": 1,
            "provider": "openai_compatible",
            "model": "embed-v1",
            "dimensions": 3,
        },
        attempt_count=1,
        error_code="lease_expired",
    )

    async def reap_expired(*_args, **_kwargs):
        return [transition]

    class SignalEmbeddingSink:
        def info(self, serialized):
            if '"event":"embedding_finished"' in serialized:
                raise signal

    _install_legacy_reaper_mock(monkeypatch, worker, reap_expired)
    monkeypatch.setattr(worker, "logger", SignalEmbeddingSink())

    with pytest.raises(expected) as raised:
        await worker.reap_cron({"pool": pool, "reap_batch_size": 1})

    assert raised.value.args in ((), (exit_code,))
    if expected is BaseException:
        assert type(raised.value) is BaseException
    assert "private" not in str(raised.value)
    assert raised.value.__cause__ is None and raised.value.__context__ is None
    assert pool.active_transactions == 0


@pytest.mark.asyncio
async def test_startup_builds_only_durable_worker_resources_and_shutdown_preserves_redis(monkeypatch):
    from jobs import worker

    pool = PoolWithConnectionTransaction()

    class Redis:
        async def ping(self):
            return True

    class S3:
        async def head_bucket(self):
            return None

    redis = Redis()
    s3 = S3()

    async def create_pool(database_url):
        assert database_url == "postgresql://worker.test/jobs"
        return pool

    async def check_readiness(**dependencies):
        assert dependencies == {
            "pool": pool,
            "redis": redis,
            "s3": s3,
            "converter_url": "https://converter.test",
        }

    monkeypatch.setattr(worker, "_create_pool", create_pool)
    monkeypatch.setattr(worker, "_create_s3_service", lambda: s3)
    monkeypatch.setattr(worker, "_check_worker_readiness", check_readiness)
    monkeypatch.setattr(worker, "_make_worker_id", lambda: "host:42:opaque")
    runtime_settings = _runtime_settings(
        REDIS_URL="redis://worker.test/0",
        DATABASE_URL="postgresql://worker.test/jobs",
        AWS_ACCESS_KEY_ID="access",
        AWS_SECRET_ACCESS_KEY="secret",
        S3_BUCKET="bucket",
        CONVERTER_URL="https://converter.test",
        CONVERTER_SECRET="converter-secret",
    )
    monkeypatch.setattr(worker.settings, "MODE", "local")

    ctx = {"redis": redis, "runtime_settings": runtime_settings}
    await worker.startup(ctx)

    assert ctx["pool"] is pool
    assert ctx["redis"] is redis
    assert ctx["s3"] is s3
    assert ctx["worker_id"] == "host:42:opaque"
    assert ctx["worker_context"].pool is pool
    assert ctx["worker_context"].converter_secret == "converter-secret"
    assert ctx["handlers"] is worker.HANDLERS
    assert ctx["runtime_settings"] is runtime_settings
    assert "websocket" not in " ".join(ctx).lower()

    await worker.shutdown(ctx)
    await worker.shutdown(ctx)
    assert pool.close_calls == 1
    assert ctx["redis"] is redis
    assert "pool" not in ctx
    assert "worker_context" not in ctx


@pytest.mark.asyncio
@pytest.mark.parametrize("enabled", [False, True])
async def test_startup_constructs_rag_factory_only_when_enabled(monkeypatch, enabled):
    from jobs import worker

    pool = PoolWithConnectionTransaction()

    class Redis:
        async def ping(self):
            return True

    factory = object()
    calls = []

    async def create_pool(_database_url):
        return pool

    def build_factory(shared_pool, runtime_settings):
        calls.append((shared_pool, runtime_settings))
        return factory

    async def readiness(**_kwargs):
        return None

    monkeypatch.setattr(worker, "_create_pool", create_pool)
    monkeypatch.setattr(worker, "_build_rag_orchestrator_factory", build_factory)
    monkeypatch.setattr(worker, "_check_worker_readiness", readiness)
    runtime_settings = _runtime_settings(SERVER_RAG_ENABLED=enabled)
    ctx = {"redis": Redis(), "runtime_settings": runtime_settings}

    await worker.startup(ctx)

    assert calls == ([(pool, runtime_settings)] if enabled else [])
    assert ctx["worker_context"].rag_orchestrator_factory is (factory if enabled else None)
    await worker.shutdown(ctx)


@pytest.mark.asyncio
async def test_startup_disabled_does_not_import_rag_adapters_or_resolve_profiles(monkeypatch):
    import builtins

    from jobs import worker

    pool = PoolWithConnectionTransaction()

    class Redis:
        async def ping(self):
            return True

    async def create_pool(_database_url):
        return pool

    async def readiness(**_kwargs):
        return None

    imported = []
    real_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name == "rag" or name.startswith("rag."):
            imported.append(name)
            raise AssertionError("disabled startup imported a RAG adapter")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(worker, "_create_pool", create_pool)
    monkeypatch.setattr(worker, "_check_worker_readiness", readiness)
    monkeypatch.setattr(builtins, "__import__", guarded_import)
    runtime_settings = _runtime_settings(SERVER_RAG_ENABLED=False)
    ctx = {"redis": Redis(), "runtime_settings": runtime_settings}

    await worker.startup(ctx)

    assert imported == []
    assert ctx["worker_context"].rag_orchestrator_factory is None
    await worker.shutdown(ctx)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "mode,durable,redis_url,has_ctx_redis,match",
    [
        ("local", True, "redis://configured", True, "MODE=hosted"),
        ("hosted", False, "redis://configured", True, "DURABLE_JOBS_ENABLED"),
        ("hosted", True, None, True, "REDIS_URL"),
        ("hosted", True, "redis://configured", False, "ctx.*redis"),
    ],
)
async def test_startup_rejects_non_durable_context_before_creating_pool(
    monkeypatch,
    mode,
    durable,
    redis_url,
    has_ctx_redis,
    match,
):
    from jobs import worker

    calls = []

    async def create_pool(_url):
        calls.append("pool")
        return PoolWithConnectionTransaction()

    monkeypatch.setattr(worker, "_create_pool", create_pool)
    runtime_settings = _runtime_settings(
        MODE=mode,
        DURABLE_JOBS_ENABLED=durable,
        REDIS_URL=redis_url,
        DATABASE_URL="postgresql://configured",
    )
    ctx = {"runtime_settings": runtime_settings}
    if has_ctx_redis:
        ctx["redis"] = object()

    with pytest.raises(RuntimeError, match=match):
        await worker.startup(ctx)
    assert calls == []


@pytest.mark.asyncio
async def test_startup_closes_pool_when_later_resource_creation_fails(monkeypatch):
    from jobs import worker

    pool = PoolWithConnectionTransaction()

    async def create_pool(_url):
        return pool

    def fail_s3():
        raise RuntimeError("s3 construction failed")

    monkeypatch.setattr(worker, "_create_pool", create_pool)
    monkeypatch.setattr(worker, "_create_s3_service", fail_s3)
    with pytest.raises(RuntimeError, match="s3 construction failed"):
        await worker.startup(
            {
                "redis": object(),
                "runtime_settings": _runtime_settings(
                    AWS_ACCESS_KEY_ID="access",
                    AWS_SECRET_ACCESS_KEY="secret",
                    S3_BUCKET="bucket",
                ),
            }
        )
    assert pool.close_calls == 1


@pytest.mark.asyncio
async def test_startup_rejects_caller_resources_before_pool_creation(monkeypatch):
    from jobs import worker

    external_pool = PoolWithConnectionTransaction()

    class ExternalS3:
        def __init__(self):
            self.close_calls = 0

        async def close(self):
            self.close_calls += 1

    external_s3 = ExternalS3()

    create_pool_calls = []

    async def create_pool(_url):
        create_pool_calls.append(_url)
        raise AssertionError("reserved ctx keys must fail before pool creation")

    monkeypatch.setattr(worker, "_create_pool", create_pool)
    ctx = {
        "redis": object(),
        "runtime_settings": _runtime_settings(),
        "pool": external_pool,
        "s3": external_s3,
    }

    with pytest.raises(RuntimeError, match="reserved keys: pool, s3"):
        await worker.startup(ctx)
    await worker.shutdown(ctx)

    assert create_pool_calls == []
    assert external_pool.close_calls == 0
    assert external_s3.close_calls == 0
    assert ctx["pool"] is external_pool
    assert ctx["s3"] is external_s3


@pytest.mark.asyncio
async def test_startup_closes_tracked_s3_and_pool_when_context_construction_fails(monkeypatch):
    from jobs import worker

    pool = PoolWithConnectionTransaction()

    class CloseableS3:
        def __init__(self):
            self.close_calls = 0

        async def close(self):
            self.close_calls += 1

    s3 = CloseableS3()

    async def create_pool(_url):
        return pool

    def fail_worker_context(**_kwargs):
        raise RuntimeError("worker context construction failed")

    monkeypatch.setattr(worker, "_create_pool", create_pool)
    monkeypatch.setattr(worker, "_create_s3_service", lambda: s3)
    monkeypatch.setattr(worker, "WorkerContext", fail_worker_context)
    ctx = {
        "redis": object(),
        "runtime_settings": _runtime_settings(
            AWS_ACCESS_KEY_ID="access",
            AWS_SECRET_ACCESS_KEY="secret",
            S3_BUCKET="bucket",
        ),
    }

    with pytest.raises(RuntimeError, match="worker context construction failed"):
        await worker.startup(ctx)

    assert s3.close_calls == 1
    assert pool.close_calls == 1
    assert "s3" not in ctx
    assert "pool" not in ctx
