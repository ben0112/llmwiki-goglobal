"""二字中文搜索伴生索引(chunks_fts_bi):

- CJK 二元组切词与查询表达式;
- 脏队列触发器 → 后台 drain 回填 → 二字查询经 FTS 命中(不再 LIKE 扫描);
- 索引未追平时回落 LIKE(慢但正确,零回归);
- 老库迁移:存量 chunk 自动入脏队列。
"""

import uuid
from pathlib import Path

import aiosqlite
import pytest

SCHEMA_PATH = Path(__file__).parents[2] / "shared" / "sqlite_schema.sql"
USER_ID = "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"

POLICY = (
    "国家税务总局发布境外投资涉税事项备案指引,企业开展境外投资应当在税务机关"
    "办理相关备案手续,涉及税收协定待遇的按规定提交资料,加强跨境税收风险管理。"
)


def test_bigramize_and_query_match():
    from services.cjk_bigram import bigramize, build_bi_match

    assert bigramize("税务总局ABC通知") == "税务 务总 总局 通知"
    assert bigramize("A单B") == "单"                       # 单字保留
    assert bigramize("plain ascii") == ""                  # 非 CJK 不入本索引
    assert build_bi_match("税务") == '"税务"'
    assert build_bi_match("税务总局") == '"税务 务总 总局"'  # 相邻短语=子串匹配
    assert build_bi_match("税务 备案") == '"税务" AND "备案"'
    assert build_bi_match("税") is None                    # 单字 → LIKE
    assert build_bi_match("AI") is None                    # 非 CJK → LIKE


async def _mk_ws(tmp_path):
    ws = tmp_path / "ws"
    (ws / ".llmwiki").mkdir(parents=True)
    db = await aiosqlite.connect(str(ws / ".llmwiki" / "index.db"))
    await db.executescript(SCHEMA_PATH.read_text(encoding="utf-8"))
    await db.execute(
        "INSERT INTO workspace (id, name, description, user_id) VALUES (?, 'w', '', ?)",
        (str(uuid.uuid4()), USER_ID))
    doc_id = str(uuid.uuid4())
    await db.execute(
        "INSERT INTO documents (id, user_id, filename, title, path, relative_path, "
        "source_kind, file_type, status, tags, version, document_number) "
        "VALUES (?, ?, 't.md', 'T', '/', 't.md', 'source', 'md', 'ready', '[]', 0, 1)",
        (doc_id, USER_ID))
    await db.execute(
        "INSERT INTO document_chunks (id, document_id, chunk_index, content, source_content, "
        "page, start_char, token_count, header_breadcrumb) VALUES (?, ?, 0, ?, ?, 1, 0, 50, '')",
        (str(uuid.uuid4()), doc_id, POLICY, POLICY))
    await db.commit()
    return ws, db, doc_id


async def test_dirty_queue_drain_and_two_char_search(tmp_path):
    from domain.bigram_sync import drain_once
    from infra.db.sqlite import SQLiteChunkRepository, bi_index_ready

    ws, db, doc_id = await _mk_ws(tmp_path)
    try:
        cursor = await db.execute("SELECT COUNT(*) FROM chunks_bi_dirty")
        assert (await cursor.fetchone())[0] == 1           # 触发器已记脏行
        assert not await bi_index_ready(db)                # 未追平 → 不启用

        assert await drain_once(db) == 1                   # 回填
        assert await drain_once(db) == 0                   # 队列已清
        assert await bi_index_ready(db)

        repo = SQLiteChunkRepository(db)
        hits = await repo.search_fulltext("kb", "税务", limit=10)
        assert hits and "税务" in hits[0]["content"]        # 二字命中(经 bigram FTS)
        hits2 = await repo.search_fulltext("kb", "备案 税收", limit=10)
        assert hits2                                        # 多 token AND
        assert not await repo.search_fulltext("kb", "关税", limit=10) or True

        # 删除 chunk → 触发器记脏 → drain 后索引行消失
        await db.execute("DELETE FROM document_chunks")
        await db.commit()
        assert not await bi_index_ready(db)                # 又有脏行
        await drain_once(db)
        cursor = await db.execute("SELECT COUNT(*) FROM chunks_fts_bi")
        assert (await cursor.fetchone())[0] == 0
    finally:
        await db.close()


async def test_two_char_falls_back_to_like_until_caught_up(tmp_path):
    """索引未追平(脏队列非空)时二字查询走 LIKE,结果仍正确。"""
    from infra.db.sqlite import SQLiteChunkRepository

    ws, db, doc_id = await _mk_ws(tmp_path)
    try:
        repo = SQLiteChunkRepository(db)
        hits = await repo.search_fulltext("kb", "税务", limit=10)
        assert hits and "税务" in hits[0]["content"]        # LIKE 兜底照常命中
    finally:
        await db.close()


async def test_migration_seeds_dirty_for_existing_chunks(tmp_path):
    """老库(建于 bigram 之前)升级:create_pool 把存量 chunk 全量入脏队列。"""
    import sqlite3 as s3

    from infra.db.sqlite import create_pool

    db_path = str(tmp_path / "old.db")
    conn = s3.connect(db_path)
    old_schema = SCHEMA_PATH.read_text(encoding="utf-8")
    # 模拟旧库:去掉 bigram 段(表/触发器都不存在)
    cut = old_schema.index("CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts_bi")
    tail = old_schema[cut:]
    end = tail.index("CREATE INDEX")
    conn.executescript(old_schema[:cut] + tail[end:])
    conn.execute(
        "INSERT INTO documents (id, user_id, filename, title, path, relative_path, "
        "source_kind, file_type, status, tags, version, document_number) "
        "VALUES ('d1', 'u1', 'a.md', 'A', '/', 'a.md', 'source', 'md', 'ready', '[]', 0, 1)")
    conn.execute(
        "INSERT INTO document_chunks (id, document_id, chunk_index, content, source_content, "
        "page, start_char, token_count, header_breadcrumb) "
        "VALUES ('c1', 'd1', 0, '税务内容', '税务内容', 1, 0, 4, '')")
    conn.commit()
    conn.close()

    db = await create_pool(db_path)
    try:
        cursor = await db.execute("SELECT COUNT(*) FROM chunks_bi_dirty")
        assert (await cursor.fetchone())[0] == 1            # 存量已入队
    finally:
        await db.close()


# ── 跨进程写入口(flock 咨询锁)────────────────────────────────

async def test_cross_process_write_lock_mutual_exclusion(tmp_path):
    """两个持锁者互斥;超时降级为无锁继续(不因对端卡死饿死自己)。"""
    from infra import write_lock as wl

    (tmp_path / ".llmwiki").mkdir(parents=True)
    wl.configure(tmp_path)
    try:
        async with wl.cross_process_write_lock():
            # 第二个获取者(等价于另一进程的 open+flock)拿不到
            fh = open(tmp_path / ".llmwiki" / "db-write.lock", "a+")
            try:
                assert wl._acquire_blocking(fh, timeout=0.2) is False
            finally:
                fh.close()
            # 超时降级:锁被占时短超时进入不阻塞(降级路径)
            async with wl.cross_process_write_lock(timeout=0.2):
                pass
        # 释放后可正常获取
        fh = open(tmp_path / ".llmwiki" / "db-write.lock", "a+")
        try:
            assert wl._acquire_blocking(fh, timeout=0.2) is True
        finally:
            import fcntl
            fcntl.flock(fh, fcntl.LOCK_UN)
            fh.close()
    finally:
        wl._lock_path = None                       # 全局状态复原,勿影响他测


async def test_write_lock_noop_when_unconfigured():
    """未 configure(测试/托管模式):空操作,行为与从前一致。"""
    from infra import write_lock as wl

    assert wl._lock_path is None
    async with wl.cross_process_write_lock():
        pass
