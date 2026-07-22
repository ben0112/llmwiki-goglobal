"""Filesystem watcher for local mode.

Watches the workspace for file changes and updates the SQLite index.
Uses watchfiles for efficient cross-platform filesystem monitoring.

Key design rules:
- App-initiated writes register in _recently_written to prevent re-indexing loops
- Hidden dirs (.llmwiki, .git, node_modules, etc.) are ignored
- File identity is by path — rename = delete old + create new
"""

import asyncio
import hashlib
import json
import logging
import os
import time
import uuid
from pathlib import Path

import aiosqlite

from domain.file_types import SIMPLE_TEXT_TYPES

logger = logging.getLogger(__name__)

IGNORE_DIRS = frozenset({
    ".llmwiki", ".git", "node_modules", "__pycache__", ".venv", "venv",
    ".idea", ".vscode", ".DS_Store",
})

COOLDOWN_SECONDS = 2.0

_ignore_patterns: list[str] | None = None


def _load_ignore_patterns(workspace: Path) -> list[str]:
    """Load ignore patterns from .llmwikiignore, falling back to .gitignore."""
    global _ignore_patterns
    if _ignore_patterns is not None:
        return _ignore_patterns

    patterns = []
    for ignore_file in (".llmwikiignore", ".gitignore"):
        p = workspace / ignore_file
        if p.is_file():
            for line in p.read_text().splitlines():
                line = line.strip()
                if line and not line.startswith("#"):
                    patterns.append(line)
            break  # Use first found, don't merge
    _ignore_patterns = patterns
    return patterns


def _matches_ignore_pattern(relative: str, patterns: list[str]) -> bool:
    """Simple gitignore-style matching (directory and glob patterns)."""
    from fnmatch import fnmatch
    for pattern in patterns:
        pattern = pattern.rstrip("/")
        if fnmatch(relative, pattern):
            return True
        if fnmatch(relative, f"**/{pattern}"):
            return True
        # Check if any path component matches a directory pattern
        for part in relative.split("/"):
            if fnmatch(part, pattern):
                return True
    return False

# Paths written by the app — skip re-indexing for these
_recently_written: dict[str, float] = {}


def mark_written(path: str) -> None:
    """Mark a path as recently written by the app. Watcher will skip it."""
    _recently_written[str(Path(path).resolve())] = time.monotonic()


def _is_recently_written(path: str) -> bool:
    resolved = str(Path(path).resolve())
    ts = _recently_written.get(resolved)
    if ts and (time.monotonic() - ts) < COOLDOWN_SECONDS:
        return True
    _recently_written.pop(resolved, None)
    return False


def _should_ignore(path: Path, workspace: Path) -> bool:
    """Check if a path should be ignored based on directory rules + ignore files."""
    try:
        relative = path.relative_to(workspace)
    except ValueError:
        return True

    relative_str = str(relative)
    parts = relative.parts

    # Built-in ignores
    for part in parts:
        if part in IGNORE_DIRS or part.startswith("."):
            return True

    # User-configured ignore patterns
    patterns = _load_ignore_patterns(workspace)
    if patterns and _matches_ignore_pattern(relative_str, patterns):
        return True

    return False


def _get_source_kind(relative_path: str) -> str:
    if relative_path.startswith("wiki/"):
        return "wiki"
    return "source"


async def _index_file(db: aiosqlite.Connection, workspace: Path, file_path: Path) -> None:
    """Index or re-index a single file."""
    relative = str(file_path.relative_to(workspace))
    filename = file_path.name
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""

    parts = relative.split("/")
    if len(parts) > 1:
        dir_path = "/" + "/".join(parts[:-1]) + "/"
    else:
        dir_path = "/"

    source_kind = _get_source_kind(relative)
    stat = file_path.stat()

    # Derive title
    stem = filename.rsplit(".", 1)[0] if "." in filename else filename
    title = stem.replace("-", " ").replace("_", " ").strip().title()

    # Read content for text files
    content = None
    if ext in SIMPLE_TEXT_TYPES:
        try:
            content = file_path.read_text(encoding="utf-8", errors="replace")
        except Exception:
            pass

    content_hash = None
    if stat.st_size < 100_000_000:
        try:
            content_hash = hashlib.sha256(file_path.read_bytes()).hexdigest()
        except Exception:
            pass

    # Check if document already exists at this path
    cursor = await db.execute(
        "SELECT id, content_hash, mtime_ns FROM documents WHERE relative_path = ?",
        (relative,),
    )
    existing = await cursor.fetchone()

    if existing:
        doc_id, old_hash, old_mtime = existing
        if old_hash == content_hash:
            # 内容未变:仅刷新 mtime,让 sweep 的 mtime 快筛下一轮不再重查
            if old_mtime != int(stat.st_mtime_ns):
                await db.execute(
                    "UPDATE documents SET mtime_ns = ? WHERE id = ?",
                    (int(stat.st_mtime_ns), doc_id),
                )
                await db.commit()
            return  # No change
        # Update existing
        await db.execute(
            "UPDATE documents SET content = ?, file_size = ?, content_hash = ?, "
            "mtime_ns = ?, last_indexed_at = datetime('now'), "
            "updated_at = datetime('now'), version = version + 1 "
            "WHERE id = ?",
            (content, stat.st_size, content_hash, int(stat.st_mtime_ns), doc_id),
        )
        await db.commit()
        logger.info("Re-indexed (modified): %s", relative)
        if ext not in SIMPLE_TEXT_TYPES and ext:
            await db.execute(
                "UPDATE documents SET status = 'pending', parser = NULL, error_message = NULL, "
                "updated_at = datetime('now') WHERE id = ?",
                (doc_id,),
            )
            await db.commit()
            from domain.local_processor import process_document_isolated
            from infra.tasks import spawn_logged
            spawn_logged(process_document_isolated(workspace, doc_id), f"reindex:{doc_id[:8]}")
        elif content is not None:
            from domain.local_processor import chunk_text_document
            await chunk_text_document(db, doc_id, content)
        return
    else:
        # Create new
        doc_id = str(uuid.uuid4())
        cursor = await db.execute(
            "SELECT COALESCE(MAX(document_number), 0) + 1 FROM documents",
        )
        row = await cursor.fetchone()
        doc_number = row[0]

        status = "ready" if content is not None else "pending"
        await db.execute(
            "INSERT INTO documents (id, user_id, filename, title, path, relative_path, "
            "source_kind, file_type, file_size, status, content, tags, version, "
            "content_hash, mtime_ns, last_indexed_at, document_number) "
            "VALUES (?, (SELECT user_id FROM workspace LIMIT 1), ?, ?, ?, ?, ?, ?, ?, "
            "?, ?, '[]', 0, ?, ?, datetime('now'), ?)",
            (doc_id, filename, title, dir_path, relative, source_kind,
             ext or "bin", stat.st_size, status, content, content_hash,
             int(stat.st_mtime_ns), doc_number),
        )
        logger.info("Indexed (new): %s", relative)
        if status == "pending":
            from domain.local_processor import process_document_isolated
            from infra.tasks import spawn_logged
            spawn_logged(process_document_isolated(workspace, doc_id), f"index:{doc_id[:8]}")

    await db.commit()

    if status == "ready" and content is not None:
        from domain.local_processor import chunk_text_document
        await chunk_text_document(db, doc_id, content)


async def _remove_file(db: aiosqlite.Connection, workspace: Path, file_path: Path) -> None:
    """Remove a file from the index."""
    try:
        relative = str(file_path.relative_to(workspace))
    except ValueError:
        return

    cursor = await db.execute(
        "DELETE FROM documents WHERE relative_path = ?", (relative,),
    )
    if cursor.rowcount > 0:
        await db.commit()
        logger.info("Removed from index: %s", relative)


async def watch_workspace(db: aiosqlite.Connection, workspace: Path) -> None:
    """Watch the workspace for file changes and update the SQLite index.

    Runs indefinitely as an async task. Cancel to stop.
    """
    from watchfiles import awatch, Change

    logger.info("File watcher started: %s", workspace)

    async for changes in awatch(
        str(workspace),
        watch_filter=lambda change, path: not _should_ignore(Path(path), workspace),
        debounce=500,
        step=200,
    ):
        for change_type, path_str in changes:
            path = Path(path_str)

            if _should_ignore(path, workspace):
                continue

            if _is_recently_written(path_str):
                continue

            try:
                if change_type == Change.added or change_type == Change.modified:
                    if path.is_file():
                        await _index_file(db, workspace, path)
                elif change_type == Change.deleted:
                    await _remove_file(db, workspace, path)
            except Exception as e:
                logger.warning("Watcher error for %s: %s", path_str, e)

        # Clean up expired entries from _recently_written
        now = time.monotonic()
        expired = [k for k, v in _recently_written.items() if now - v > COOLDOWN_SECONDS * 2]
        for k in expired:
            _recently_written.pop(k, None)


SWEEP_INTERVAL_SECONDS = 60


async def sweep_workspace(db: aiosqlite.Connection, workspace: Path) -> tuple[int, int]:
    """磁盘 ↔ 索引对账一轮:补录新增/离线修改的文件,清理已消失文件的索引行。

    watcher 依赖 inotify,但 Docker Desktop(macOS/Windows)宿主侧拷入
    bind mount 的文件常常不产生容器内事件;容器停机期间的增删也没有事件。
    启动时与每 SWEEP_INTERVAL_SECONDS 各跑一轮兜底,让「直接把文件拷进
    工作区目录」始终可用。mtime 快筛避免每轮重读文件内容。
    """
    disk: dict[str, tuple[Path, int]] = {}

    def _walk() -> None:
        for p in workspace.rglob("*"):
            try:
                if not p.is_file() or _should_ignore(p, workspace):
                    continue
                disk[str(p.relative_to(workspace))] = (p, p.stat().st_mtime_ns)
            except OSError:
                continue

    await asyncio.to_thread(_walk)

    cursor = await db.execute("SELECT relative_path, mtime_ns FROM documents")
    indexed = {r[0]: r[1] for r in await cursor.fetchall()}

    swept = 0
    for relative, (path, mtime) in disk.items():
        if _is_recently_written(str(path)):
            continue
        if indexed.get(relative) == mtime:
            continue
        try:
            await _index_file(db, workspace, path)
            swept += 1
        except Exception as e:
            logger.warning("Sweep index error for %s: %s", relative, e)

    removed = 0
    for relative in set(indexed) - set(disk):
        path = workspace / relative
        if _is_recently_written(str(path)) or path.exists():
            continue
        try:
            await _remove_file(db, workspace, path)
            removed += 1
        except Exception as e:
            logger.warning("Sweep remove error for %s: %s", relative, e)

    if swept or removed:
        logger.info("Workspace sweep: %d indexed/updated, %d removed", swept, removed)
    return swept, removed


async def sweep_loop(db: aiosqlite.Connection, workspace: Path) -> None:
    """启动立即对账一轮(接住停机期间拷入的文件),之后定期兜底。"""
    while True:
        try:
            await sweep_workspace(db, workspace)
        except Exception:
            logger.exception("Workspace sweep failed")
        await asyncio.sleep(SWEEP_INTERVAL_SECONDS)
