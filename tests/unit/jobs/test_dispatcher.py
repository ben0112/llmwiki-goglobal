from __future__ import annotations

import asyncio
import inspect
import logging
from contextlib import asynccontextmanager
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import MappingProxyType, SimpleNamespace
from uuid import uuid4

import asyncpg
import pytest

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
    from jobs.handlers import HANDLERS
    from jobs.models import JobType

    assert isinstance(HANDLERS, MappingProxyType)
    assert set(HANDLERS) == set(JobType)
    with pytest.raises(TypeError):
        HANDLERS[JobType.DOCUMENT_EXTRACT] = object()

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


def _job(job_type=None):
    from jobs.models import JobRecord, JobType

    return JobRecord(
        id=uuid4(),
        job_type=job_type or JobType.DOCUMENT_EXTRACT,
        user_id=uuid4(),
    )


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
async def test_run_job_executes_persisted_handler_under_lease_and_succeeds(monkeypatch):
    from jobs import worker

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

    monkeypatch.setattr(worker.repository, "claim", claim)
    monkeypatch.setattr(worker.repository, "succeed", succeed)
    monkeypatch.setattr(worker, "JobLease", FakeLease)
    ctx = _ctx(pool, {job.job_type: handler})

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
):
    from jobs import worker
    from jobs.handlers import RetryableJobError, TerminalJobError
    from jobs.models import JobCancelled

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

    async def fail_or_retry(conn, job_id, owner, **kwargs):
        assert FakeLease.instances[-1].entered
        recorded.append((conn, job_id, owner, kwargs))

    monkeypatch.setattr(worker.repository, "claim", claim)
    monkeypatch.setattr(worker.repository, "fail_or_retry", fail_or_retry)
    monkeypatch.setattr(worker, "JobLease", FakeLease)

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
async def test_run_job_never_mutates_after_lease_loss(monkeypatch):
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

    outcome = await worker.run_job(_ctx(pool, {job.job_type: handler}), str(job.id))

    assert outcome == {"status": "lease_lost", "job_id": str(job.id)}
    assert fail_calls == []


@pytest.mark.asyncio
async def test_run_job_returns_lost_if_failure_transition_loses_lease(monkeypatch):
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

    outcome = await worker.run_job(_ctx(pool, {job.job_type: handler}), str(job.id))
    assert outcome == {"status": "lease_lost", "job_id": str(job.id)}


@pytest.mark.asyncio
async def test_run_job_propagates_worker_cancellation(monkeypatch):
    from jobs import worker

    pool = PoolWithConnectionTransaction()
    job = _job()

    async def claim(*_args):
        return job

    async def handler(*_args):
        raise asyncio.CancelledError

    recorded = []

    async def record_failure(*args, **kwargs):
        recorded.append((args, kwargs))
        return True

    monkeypatch.setattr(worker.repository, "claim", claim)
    monkeypatch.setattr(worker, "JobLease", FakeLease)
    monkeypatch.setattr(worker, "_record_failure", record_failure)

    with pytest.raises(asyncio.CancelledError):
        await worker.run_job(_ctx(pool, {job.job_type: handler}), str(job.id))

    assert recorded[0][1] == {
        "error_code": "worker_shutdown",
        "error_message": "Worker shutdown interrupted the job.",
        "retryable": True,
    }


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

    async def reap_expired(conn, *, limit):
        assert conn is pool.connection
        assert limit == 23
        raise failure

    monkeypatch.setattr(worker.repository, "reap_expired", reap_expired)
    with pytest.raises(RuntimeError, match="reaper database unavailable") as raised:
        await worker.reap_cron({"pool": pool, "reap_batch_size": 23})
    assert raised.value is failure
    assert pool.transactions == 1
    assert pool.active_connections == 0
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
