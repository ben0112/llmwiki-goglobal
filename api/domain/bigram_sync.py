"""chunks_fts_bi 后台同步:消费 chunks_bi_dirty 脏队列,补算 bigram 文本。

二字中文查询(税务/备案)的伴生索引。触发器只记 rowid(纯 SQL,任何
写入方都安全),真正的 CJK 二元组切词在这里完成 —— 老库首次升级的
全量回填与日常增量走同一条路径,逐批消费不阻塞启动;查询侧只在队列
清空(索引追平)时启用 bigram,回填期间维持 LIKE,正确性零回归。
"""

import asyncio
import logging

import aiosqlite

from services.cjk_bigram import bigramize

logger = logging.getLogger(__name__)

BATCH = 500
INTERVAL_SECONDS = 15


async def drain_once(db: aiosqlite.Connection) -> int:
    """消费一批脏行:删旧索引行、按当前 chunk 内容重建、清队列。"""
    cursor = await db.execute(
        "SELECT chunk_rowid FROM chunks_bi_dirty LIMIT ?", (BATCH,))
    rowids = [r[0] for r in await cursor.fetchall()]
    if not rowids:
        return 0

    ph = ",".join("?" for _ in rowids)
    cursor = await db.execute(
        f"SELECT rowid, content FROM document_chunks WHERE rowid IN ({ph})", rowids)
    live = {r[0]: (r[1] or "") for r in await cursor.fetchall()}
    rows = [(rid, bigramize(live[rid])) for rid in rowids if rid in live]  # CPU 段在闸门外

    from domain.local_processor import _gated_write
    async with _gated_write(db):
        await db.executemany(
            "DELETE FROM chunks_fts_bi WHERE rowid = ?", [(r,) for r in rowids])
        if rows:
            await db.executemany(
                "INSERT INTO chunks_fts_bi(rowid, content) VALUES (?, ?)", rows)
        await db.executemany(
            "DELETE FROM chunks_bi_dirty WHERE chunk_rowid = ?", [(r,) for r in rowids])
        await db.commit()
    return len(rowids)


async def bigram_sync_loop(db: aiosqlite.Connection) -> None:
    """常驻同步循环:有积压连续排空(批间小憩让路),追平后按周期轮询。"""
    synced = 0
    while True:
        try:
            n = await drain_once(db)
            if n:
                synced += n
                if n >= BATCH:
                    await asyncio.sleep(0.05)   # 回填期给其他写入者让路
                    continue
                logger.info("Bigram index caught up (+%d chunks)", synced)
                synced = 0
        except Exception:
            logger.exception("Bigram sync failed")
        await asyncio.sleep(INTERVAL_SECONDS)
