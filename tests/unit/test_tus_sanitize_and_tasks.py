"""TUS 路径清洗与后台任务封装的单元测试。

清洗:旧实现用单轮 re.sub 剥 `/../`,不回扫,`../..` 之类输入清洗后
仍残留 `/../`;新实现逐段过滤,任何形式的穿越段都无法存活。
"""

import asyncio
import logging

import pytest

from infra.tasks import spawn_logged
from infra.tus import sanitize_upload_path


@pytest.mark.parametrize("raw, expected", [
    ("/reports/2026/", "/reports/2026/"),   # 合法路径原样保留
    ("reports", "/reports/"),
    ("../..", "/"),                          # 旧实现的绕过输入
    ("/../../etc/passwd", "/etc/passwd/"),
    ("/a/../b/", "/a/b/"),
    ("a\\..\\b", "/a/b/"),                   # Windows 分隔符
    ("//a//b//", "/a/b/"),
    ("/./.././", "/"),
    ("", "/"),
    ("   ", "/"),
])
def test_sanitize_upload_path(raw, expected):
    assert sanitize_upload_path(raw) == expected


def test_sanitize_never_leaves_traversal():
    for raw in ("..", "../", "/..", "../../..", "a/../../b", "..\\..\\x"):
        cleaned = sanitize_upload_path(raw)
        assert ".." not in cleaned and cleaned.startswith("/") and cleaned.endswith("/")


@pytest.mark.asyncio
async def test_spawn_logged_reports_exception(caplog):
    async def boom():
        raise RuntimeError("炸了")

    with caplog.at_level(logging.ERROR, logger="infra.tasks"):
        task = spawn_logged(boom(), "test-boom")
        with pytest.raises(RuntimeError):
            await task
        await asyncio.sleep(0)  # 让 done callback 跑完
    assert any("test-boom" in r.message for r in caplog.records)


@pytest.mark.asyncio
async def test_spawn_logged_silent_on_cancel_and_success(caplog):
    async def ok():
        return 42

    async def forever():
        await asyncio.Event().wait()

    with caplog.at_level(logging.ERROR, logger="infra.tasks"):
        assert await spawn_logged(ok(), "test-ok") == 42
        t = spawn_logged(forever(), "test-cancel")
        await asyncio.sleep(0)
        t.cancel()
        with pytest.raises(asyncio.CancelledError):
            await t
        await asyncio.sleep(0)
    assert not caplog.records
