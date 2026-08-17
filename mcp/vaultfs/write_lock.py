"""跨进程 SQLite 写入口:.llmwiki/db-write.lock 上的 flock 咨询锁。

API 与 MCP 是同一容器里的两个进程,进程内的 asyncio 写闸门互相不可见,
大事务写撞车时只能靠 busy_timeout 硬等(交互式写最坏排队几十秒)。
两边的重量级写统一先取这把文件锁 —— 写入口从"两个"变"一个"。

- 获取在 to_thread 中轮询(LOCK_NB + 50ms),不冻结事件循环;
- 超时(60s)后降级为"无锁继续"并告警:退回 busy_timeout 兜底行为,
  绝不因对端进程卡死而饿死自己;
- 未 configure(测试/托管模式)时为空操作,行为与从前完全一致。

与 api/infra/write_lock.py 保持逐字一致(两包独立部署,不共享导入)。
"""

import asyncio
import fcntl
import logging
import time
from contextlib import asynccontextmanager
from pathlib import Path

logger = logging.getLogger(__name__)

ACQUIRE_TIMEOUT_SECONDS = 60.0
_lock_path: Path | None = None


def configure(workspace: Path) -> None:
    global _lock_path
    _lock_path = workspace / ".llmwiki" / "db-write.lock"


def _acquire_blocking(fh, timeout: float) -> bool:
    deadline = time.monotonic() + timeout
    while True:
        try:
            fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
            return True
        except OSError:
            if time.monotonic() >= deadline:
                return False
            time.sleep(0.05)


@asynccontextmanager
async def cross_process_write_lock(timeout: float = ACQUIRE_TIMEOUT_SECONDS):
    if _lock_path is None:
        yield
        return
    fh = open(_lock_path, "a+")
    try:
        got = await asyncio.to_thread(_acquire_blocking, fh, timeout)
        if not got:
            logger.warning(
                "Cross-process write lock timed out after %.0fs — proceeding "
                "on busy_timeout only", timeout)
        try:
            yield
        finally:
            if got:
                fcntl.flock(fh, fcntl.LOCK_UN)
    finally:
        fh.close()
