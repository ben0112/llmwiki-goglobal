from __future__ import annotations

import asyncio
import inspect
from contextlib import asynccontextmanager
from pathlib import Path
from types import MappingProxyType
from uuid import uuid4

import pytest


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
        return [{"id": job_id} for job_id in self.selected]

    async def fetchval(self, query, job_id):
        self.fetchval_calls.append((query, job_id))
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
async def test_select_due_job_ids_rejects_non_positive_integral_values(field, value):
    from jobs.dispatcher import select_due_job_ids

    values = {"batch_size": 10, "redeliver_seconds": 30, field: value}
    with pytest.raises(ValueError, match=field):
        await select_due_job_ids(FakePool().connection, **values)


@pytest.mark.asyncio
async def test_select_due_job_ids_uses_parameterized_ordered_skip_locked_query():
    from jobs.dispatcher import select_due_job_ids

    selected = [uuid4(), uuid4()]
    connection = FakePool(selected).connection

    assert await select_due_job_ids(connection, batch_size=2, redeliver_seconds=30) == selected
    query, args = connection.fetch_calls[0]
    normalized = " ".join(query.split()).lower()
    assert args == (30, 2)
    assert "clock_timestamp()" in normalized
    assert "job.last_dispatched_at < dispatch_clock.checked_at" in normalized
    assert "order by job.run_after, job.created_at, job.id" in normalized
    assert "for update of job skip locked" in normalized
    assert "limit $2" in normalized


@pytest.mark.asyncio
async def test_dispatch_sends_only_opaque_uuid_and_marks_new_and_duplicate_delivery():
    from jobs.dispatcher import DispatchSummary, dispatch_due_jobs

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
        (("run_job", str(job_ids[0])), {"_job_id": str(job_ids[0])}),
        (("run_job", str(job_ids[1])), {"_job_id": str(job_ids[1])}),
    ]
    assert [call[1] for call in pool.connection.fetchval_calls] == job_ids


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
    assert [call[1] for call in pool.connection.fetchval_calls] == [mark_failed, delivered]


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


@pytest.mark.asyncio
async def test_all_initial_handlers_explicitly_reject_unsupported_business_work():
    from jobs.handlers import HANDLERS, UnsupportedJobHandler, WorkerContext
    from jobs.models import JobRecord

    context = WorkerContext(pool=object(), s3=None, converter_url="", converter_secret="")
    lease = object()
    for job_type, handler in HANDLERS.items():
        record = JobRecord(id=uuid4(), job_type=job_type, user_id=uuid4())
        with pytest.raises(UnsupportedJobHandler) as raised:
            await handler(record, lease, context)
        assert raised.value.error_code == "unsupported_job_type"
        assert "not supported" in raised.value.error_message.lower()


def test_handler_errors_validate_stable_codes_and_sanitize_messages():
    from jobs.handlers import RetryableJobError, TerminalJobError

    error = RetryableJobError("converter_timeout", "  converter\n timed\tout  ")
    assert error.error_code == "converter_timeout"
    assert error.error_message == "converter timed out"
    assert str(error) == "converter timed out"

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
    from jobs.worker import WorkerSettings, dispatch_cron, reap_cron, run_job

    assert WorkerSettings.functions == [run_job]
    assert WorkerSettings.max_tries == 1
    assert WorkerSettings.retry_jobs is False
    assert WorkerSettings.keep_result == 0
    assert WorkerSettings.job_timeout == 3600
    assert WorkerSettings.on_startup.__name__ == "startup"
    assert WorkerSettings.on_shutdown.__name__ == "shutdown"
    assert len(WorkerSettings.cron_jobs) == 2

    dispatch, reap = WorkerSettings.cron_jobs
    assert dispatch.coroutine is dispatch_cron
    assert dispatch.second == {0, 10, 20, 30, 40, 50}
    assert reap.coroutine is reap_cron
    assert reap.second == {5, 35}
    assert {dispatch.name, reap.name} == {"durable_job_dispatch", "durable_job_reaper"}
    assert {dispatch.job_id, reap.job_id} == {
        "durable-job-dispatch-cron",
        "durable-job-reaper-cron",
    }
    for job in WorkerSettings.cron_jobs:
        assert job.run_at_startup is True
        assert job.unique is True
        assert job.max_tries == 1
        assert job.keep_result_s == 0


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

    @asynccontextmanager
    async def acquire(self):
        self.acquires += 1
        yield self.connection

    @asynccontextmanager
    async def transaction(self):
        self.transactions += 1
        yield

    async def close(self):
        self.close_calls += 1


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
    calls = []

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
    assert calls == ["handler", (pool.connection, job.id, "worker-test", {"ok": True, "nested": [1, None]})]
    assert FakeLease.instances[-1].args == (pool, job.id, "worker-test", 60, 15)
    assert pool.acquires == 2
    assert pool.transactions == 2


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

    monkeypatch.setattr(worker.repository, "claim", claim)
    monkeypatch.setattr(worker, "JobLease", FakeLease)

    with pytest.raises(asyncio.CancelledError):
        await worker.run_job(_ctx(pool, {job.job_type: handler}), str(job.id))


@pytest.mark.asyncio
async def test_reap_cron_uses_short_transaction_and_propagates_failures(monkeypatch):
    from jobs import worker

    pool = PoolWithConnectionTransaction()
    calls = []

    async def reap_expired(conn, *, limit):
        calls.append((conn, limit))
        return [uuid4()]

    monkeypatch.setattr(worker.repository, "reap_expired", reap_expired)
    assert await worker.reap_cron({"pool": pool, "reap_batch_size": 23}) is None
    assert calls == [(pool.connection, 23)]
    assert pool.transactions == 1


@pytest.mark.asyncio
async def test_startup_builds_only_durable_worker_resources_and_shutdown_preserves_redis(monkeypatch):
    from jobs import worker

    pool = PoolWithConnectionTransaction()
    redis = object()
    s3 = object()

    async def create_pool(database_url):
        assert database_url == "postgresql://worker.test/jobs"
        return pool

    monkeypatch.setattr(worker, "_create_pool", create_pool)
    monkeypatch.setattr(worker, "_create_s3_service", lambda: s3)
    monkeypatch.setattr(worker, "_make_worker_id", lambda: "host:42:opaque")
    monkeypatch.setattr(worker.settings, "MODE", "hosted")
    monkeypatch.setattr(worker.settings, "DURABLE_JOBS_ENABLED", True)
    monkeypatch.setattr(worker.settings, "REDIS_URL", "redis://worker.test/0")
    monkeypatch.setattr(worker.settings, "DATABASE_URL", "postgresql://worker.test/jobs")
    monkeypatch.setattr(worker.settings, "AWS_ACCESS_KEY_ID", "access")
    monkeypatch.setattr(worker.settings, "AWS_SECRET_ACCESS_KEY", "secret")
    monkeypatch.setattr(worker.settings, "S3_BUCKET", "bucket")
    monkeypatch.setattr(worker.settings, "CONVERTER_URL", "https://converter.test")
    monkeypatch.setattr(worker.settings, "CONVERTER_SECRET", "converter-secret")

    ctx = {"redis": redis}
    await worker.startup(ctx)

    assert ctx["pool"] is pool
    assert ctx["redis"] is redis
    assert ctx["s3"] is s3
    assert ctx["worker_id"] == "host:42:opaque"
    assert ctx["worker_context"].pool is pool
    assert ctx["worker_context"].converter_secret == "converter-secret"
    assert ctx["handlers"] is worker.HANDLERS
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
    monkeypatch.setattr(worker.settings, "MODE", mode)
    monkeypatch.setattr(worker.settings, "DURABLE_JOBS_ENABLED", durable)
    monkeypatch.setattr(worker.settings, "REDIS_URL", redis_url)
    monkeypatch.setattr(worker.settings, "DATABASE_URL", "postgresql://configured")
    ctx = {"redis": object()} if has_ctx_redis else {}

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
    monkeypatch.setattr(worker.settings, "MODE", "hosted")
    monkeypatch.setattr(worker.settings, "DURABLE_JOBS_ENABLED", True)
    monkeypatch.setattr(worker.settings, "REDIS_URL", "redis://configured")
    monkeypatch.setattr(worker.settings, "DATABASE_URL", "postgresql://configured")
    monkeypatch.setattr(worker.settings, "AWS_ACCESS_KEY_ID", "access")
    monkeypatch.setattr(worker.settings, "AWS_SECRET_ACCESS_KEY", "secret")
    monkeypatch.setattr(worker.settings, "S3_BUCKET", "bucket")

    with pytest.raises(RuntimeError, match="s3 construction failed"):
        await worker.startup({"redis": object()})
    assert pool.close_calls == 1
