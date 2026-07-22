"""Fire-and-forget 后台任务封装:异常写日志,而非在 GC 时无声丢失。

适用于"发射后不管"的任务(文档处理、恢复、维基重生成)。被持有并在
关停时 cancel/await 的长驻任务无需使用。
"""

import asyncio
import logging
from typing import Coroutine

logger = logging.getLogger(__name__)


def spawn_logged(coro: Coroutine, label: str) -> asyncio.Task:
    task = asyncio.create_task(coro)

    def _done(t: asyncio.Task) -> None:
        if t.cancelled():
            return
        exc = t.exception()
        if exc is not None:
            logger.error("后台任务失败 [%s]: %r", label, exc, exc_info=exc)

    task.add_done_callback(_done)
    return task
