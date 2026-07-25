from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest


@pytest.mark.asyncio
async def test_listener_is_not_ready_until_subscribed_and_clears_before_close(monkeypatch):
    from routes import ws

    subscribe = asyncio.Event()
    ready = asyncio.Event()

    class Connection:
        def __init__(self):
            self.closed = False

        async def add_listener(self, channel, callback):
            assert channel == "document_changes"
            assert callback is ws._on_notify
            await subscribe.wait()

        async def execute(self, query):
            assert query == "SELECT 1"
            await asyncio.Future()

        def is_closed(self):
            return self.closed

        async def close(self):
            assert not ready.is_set()
            self.closed = True

    connection = Connection()

    async def connect(database_url):
        assert database_url == "postgresql://listener.test/database"
        return connection

    monkeypatch.setattr(ws.asyncpg, "connect", connect)
    monkeypatch.setattr(ws, "KEEPALIVE_SECONDS", 0)
    task = asyncio.create_task(ws._listen_until_closed("postgresql://listener.test/database", ready))

    await asyncio.sleep(0)
    assert not ready.is_set()
    subscribe.set()
    await asyncio.wait_for(ready.wait(), timeout=1)

    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert not ready.is_set()
    assert connection.closed


@pytest.mark.asyncio
async def test_listener_supervisor_clears_readiness_between_reconnects(monkeypatch):
    from routes import ws

    ready = asyncio.Event()
    second_listen = asyncio.Event()
    calls = 0

    async def listen(_database_url, current_ready):
        nonlocal calls
        calls += 1
        assert current_ready is ready
        assert not ready.is_set()
        ready.set()
        if calls == 1:
            raise ConnectionError("socket dropped")
        second_listen.set()
        await asyncio.Future()

    async def no_delay(_seconds):
        return None

    monkeypatch.setattr(ws, "_listen_until_closed", listen)
    monkeypatch.setattr(ws.asyncio, "sleep", no_delay)
    task = asyncio.create_task(ws._supervise_listener("postgresql://listener.test/database", ready))

    await asyncio.wait_for(second_listen.wait(), timeout=1)
    assert calls == 2
    assert ready.is_set()

    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert not ready.is_set()


@pytest.mark.asyncio
async def test_initial_listener_subscription_failure_never_reports_ready(monkeypatch):
    from routes import ws

    ready = asyncio.Event()
    failed = asyncio.Event()

    async def fail_subscription(_database_url, current_ready):
        assert current_ready is ready
        failed.set()
        raise ConnectionError("subscription failed")

    async def wait_before_retry(_seconds):
        await asyncio.Future()

    monkeypatch.setattr(ws, "_listen_until_closed", fail_subscription)
    monkeypatch.setattr(ws.asyncio, "sleep", wait_before_retry)
    task = asyncio.create_task(ws._supervise_listener("postgresql://listener.test/database", ready))

    await asyncio.wait_for(failed.wait(), timeout=1)
    assert not ready.is_set()

    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task


@pytest.mark.asyncio
async def test_listener_handle_times_out_initial_subscription_and_closes_task(monkeypatch):
    from routes import ws

    async def never_subscribes(_database_url, current_ready):
        assert not current_ready.is_set()
        await asyncio.Future()

    monkeypatch.setattr(ws, "_supervise_listener", never_subscribes)
    handle = await ws.setup_listener("postgresql://listener.test/database")

    with pytest.raises(TimeoutError):
        await handle.wait_ready(timeout_seconds=0.01)
    assert not handle.ready.is_set()

    await handle.close()
    assert handle.task.done()


@pytest.mark.asyncio
async def test_hosted_listener_startup_waits_for_subscription_and_cleans_timeout(monkeypatch):
    import main
    from routes import ws

    ready = asyncio.Event()

    async def idle():
        await asyncio.Future()

    class Handle:
        def __init__(self):
            self.task = asyncio.create_task(idle())
            self.ready = ready
            self.closed = 0

        async def wait_ready(self, *, timeout_seconds):
            async with asyncio.timeout(timeout_seconds):
                await ready.wait()

        async def close(self):
            self.closed += 1
            self.task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await self.task

    handle = Handle()

    async def setup(database_url):
        assert database_url == main.settings.listen_database_url
        return handle

    monkeypatch.setattr(ws, "setup_listener", setup)
    monkeypatch.setattr(main, "HOSTED_LISTENER_STARTUP_TIMEOUT_SECONDS", 0.01)
    app = SimpleNamespace(state=SimpleNamespace())

    with pytest.raises(RuntimeError, match="listener subscription timed out"):
        await main._start_hosted_listener(app)

    assert app.state.listener_ready is ready
    assert not app.state.listener_ready.is_set()
    assert handle.closed == 1
