"""磁盘↔索引对账(watcher.sweep_workspace)的单元测试。

场景:Docker Desktop 宿主侧拷入 bind mount 不产生容器内 inotify 事件、
容器停机期间的增删 —— sweep 在启动与定期兜底补录/清理。
"""

import uuid
from pathlib import Path

import aiosqlite
import pytest

from domain import watcher

SCHEMA_PATH = Path(__file__).parents[2] / "shared" / "sqlite_schema.sql"
USER_ID = "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"


async def _init_db(workspace: Path) -> aiosqlite.Connection:
    (workspace / ".llmwiki").mkdir(parents=True, exist_ok=True)
    db = await aiosqlite.connect(str(workspace / ".llmwiki" / "index.db"))
    db.row_factory = aiosqlite.Row
    await db.executescript(SCHEMA_PATH.read_text(encoding="utf-8"))
    await db.execute(
        "INSERT INTO workspace (id, name, description, user_id) VALUES (?, 'ws', '', ?)",
        (str(uuid.uuid4()), USER_ID),
    )
    await db.commit()
    return db


async def _paths(db: aiosqlite.Connection) -> set[str]:
    cursor = await db.execute("SELECT relative_path FROM documents")
    return {r[0] for r in await cursor.fetchall()}


async def test_sweep_indexes_offline_added_files(tmp_path):
    ws = tmp_path / "ws"
    db = await _init_db(ws)
    try:
        # 模拟停机/无 inotify 时拷入:直接落盘,索引一无所知
        (ws / "区域报告").mkdir()
        (ws / "区域报告" / "沙特市场准入.md").write_text("# 沙特\n准入要点。", encoding="utf-8")
        (ws / "散件.txt").write_text("散件内容", encoding="utf-8")

        swept, removed = await watcher.sweep_workspace(db, ws)
        assert swept == 2 and removed == 0
        assert {"区域报告/沙特市场准入.md", "散件.txt"} <= await _paths(db)

        # 幂等:内容与 mtime 未变,第二轮零动作
        assert await watcher.sweep_workspace(db, ws) == (0, 0)
    finally:
        await db.close()


async def test_sweep_skips_ignored_and_hidden(tmp_path):
    ws = tmp_path / "ws"
    db = await _init_db(ws)
    try:
        (ws / ".llmwiki" / "tmp").mkdir(parents=True, exist_ok=True)
        (ws / ".llmwiki" / "tmp" / "x.part").write_bytes(b"x")
        (ws / ".隐藏目录").mkdir()
        (ws / ".隐藏目录" / "秘密.txt").write_text("x", encoding="utf-8")
        (ws / "正常.txt").write_text("ok", encoding="utf-8")

        swept, _ = await watcher.sweep_workspace(db, ws)
        assert swept == 1
        assert await _paths(db) == {"正常.txt"}
    finally:
        await db.close()


async def test_sweep_removes_rows_for_deleted_files(tmp_path):
    ws = tmp_path / "ws"
    db = await _init_db(ws)
    try:
        f = ws / "将被删.txt"
        f.write_text("内容", encoding="utf-8")
        await watcher.sweep_workspace(db, ws)
        assert "将被删.txt" in await _paths(db)

        f.unlink()   # 离线删除(无事件)
        swept, removed = await watcher.sweep_workspace(db, ws)
        assert removed == 1 and "将被删.txt" not in await _paths(db)
    finally:
        await db.close()


async def test_sweep_reindexes_offline_modified(tmp_path):
    ws = tmp_path / "ws"
    db = await _init_db(ws)
    try:
        f = ws / "改动.txt"
        f.write_text("旧内容", encoding="utf-8")
        await watcher.sweep_workspace(db, ws)

        f.write_text("新内容", encoding="utf-8")
        import os
        os.utime(f, ns=(f.stat().st_atime_ns, f.stat().st_mtime_ns + 10**9))
        swept, _ = await watcher.sweep_workspace(db, ws)
        assert swept == 1
        cursor = await db.execute(
            "SELECT content FROM documents WHERE relative_path = '改动.txt'")
        assert (await cursor.fetchone())[0] == "新内容"
    finally:
        await db.close()


async def test_sweep_same_content_new_mtime_converges(tmp_path):
    """重拷同内容文件(mtime 变、哈希不变):刷新 mtime 后下一轮零动作。"""
    ws = tmp_path / "ws"
    db = await _init_db(ws)
    try:
        f = ws / "重拷.txt"
        f.write_text("同样内容", encoding="utf-8")
        await watcher.sweep_workspace(db, ws)

        import os
        os.utime(f, ns=(f.stat().st_atime_ns, f.stat().st_mtime_ns + 10**9))
        await watcher.sweep_workspace(db, ws)              # 刷新 mtime
        assert await watcher.sweep_workspace(db, ws) == (0, 0)   # 已收敛
    finally:
        await db.close()
