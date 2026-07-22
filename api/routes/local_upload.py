"""Local file upload routes — simple multipart, no TUS.

Copies uploaded files directly into the workspace and indexes them.
/v1/upload/archive 额外支持压缩包(zip/tar[.gz]):服务端解压后按包名
建文件夹逐个入库,复用与普通上传相同的落盘/索引路径。
"""

import asyncio
import hashlib
import uuid
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Request, UploadFile, File, Form

from config import settings
from deps import get_user_id
from domain.file_types import EXTRACTION_TYPES, IMAGE_TYPES, SIMPLE_TEXT_TYPES
from domain.watcher import mark_written
from services.archive_extract import (
    ArchiveError, archive_stem, extract_entries, is_supported_archive,
)

router = APIRouter(tags=["upload"])

# 压缩包内允许入库的类型 = 前端 accept 列表的服务端等价物
_ARCHIVE_ALLOWED_TYPES = SIMPLE_TEXT_TYPES | EXTRACTION_TYPES | IMAGE_TYPES


def _workspace_root() -> Path:
    return Path(settings.WORKSPACE_PATH).resolve()


def _safe_resolve(relative: str) -> Path:
    ws = _workspace_root()
    resolved = (ws / relative).resolve()
    if not resolved.is_relative_to(ws):
        raise HTTPException(status_code=400, detail="Path escapes workspace")
    return resolved


async def _ingest_bytes(db, relative: str, content_bytes: bytes) -> dict:
    """把一份文件字节落盘到工作区并建立索引(单文件上传与解压共用)。"""
    dest = _safe_resolve(relative)
    dest.parent.mkdir(parents=True, exist_ok=True)
    mark_written(str(dest))
    dest.write_bytes(content_bytes)

    filename = relative.rsplit("/", 1)[-1]
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    stem = filename.rsplit(".", 1)[0] if "." in filename else filename
    title = stem.replace("-", " ").replace("_", " ").strip().title()

    dir_path = "/" + "/".join(relative.split("/")[:-1]) + "/" if "/" in relative else "/"
    source_kind = "wiki" if relative.startswith("wiki/") else "source"
    content_hash = hashlib.sha256(content_bytes).hexdigest()

    # HTML is excluded from inline text — it goes through webmd in process_document.
    text_content = None
    needs_processing = ext in EXTRACTION_TYPES
    if ext in SIMPLE_TEXT_TYPES:
        try:
            text_content = content_bytes.decode("utf-8", errors="replace")
        except Exception:
            pass

    from infra.db.sqlite import SQLiteDocumentRepository, SQLiteChunkRepository, serialized_write
    doc_repo = SQLiteDocumentRepository(db)
    chunk_repo = SQLiteChunkRepository(db)

    doc_id = str(uuid.uuid4())

    async with serialized_write():
        try:
            cursor = await db.execute("SELECT COALESCE(MAX(document_number), 0) + 1 FROM documents")
            row = await cursor.fetchone()
            doc_number = row[0]
            await db.execute(
                "INSERT INTO documents (id, user_id, filename, title, path, relative_path, "
                "source_kind, file_type, file_size, status, content, tags, version, "
                "content_hash, mtime_ns, last_indexed_at, document_number) "
                "VALUES (?, (SELECT user_id FROM workspace LIMIT 1), ?, ?, ?, ?, ?, ?, ?, ?, ?, '[]', 0, ?, ?, datetime('now'), ?)",
                (doc_id, filename, title, dir_path, relative, source_kind,
                 ext or "bin", len(content_bytes),
                 "ready" if text_content is not None else "pending",
                 text_content, content_hash,
                 int(dest.stat().st_mtime_ns), doc_number),
            )
            await db.commit()
        except Exception:
            await db.rollback()
            raise

    # Chunk text content or kick off processing for non-text files
    if text_content:
        from services.chunker import chunk_text
        chunks = chunk_text(text_content)
        # SQLite 实现忽略 user/kb 参数(仅为与 hosted 接口对齐)
        await chunk_repo.store(doc_id, "", "", chunks)
    elif needs_processing:
        from domain.local_processor import process_document_isolated
        from infra.tasks import spawn_logged
        spawn_logged(process_document_isolated(_workspace_root(), doc_id),
                     f"upload-process:{doc_id[:8]}")

    return await doc_repo.get(doc_id)


@router.post("/v1/upload", status_code=201)
async def upload_file(
    file: UploadFile = File(...),
    path: str = Form(default="/"),
    user_id: str = Depends(get_user_id),
    request: Request = None,
):
    """Upload a file directly into the workspace and index it."""
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename")

    relative = (path.rstrip("/") + "/" + file.filename).lstrip("/")
    content_bytes = await file.read()
    return await _ingest_bytes(request.app.state.sqlite_db, relative, content_bytes)


@router.post("/v1/upload/archive", status_code=201)
async def upload_archive(
    file: UploadFile = File(...),
    path: str = Form(default="/"),
    user_id: str = Depends(get_user_id),
    request: Request = None,
):
    """上传压缩包并在服务端解压入库。

    以包名(去扩展名)为目标文件夹,保留包内目录结构;与前端上传去重
    同口径:同相对路径或同内容哈希的条目自动跳过,仅在响应中计数。
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename")
    if not is_supported_archive(file.filename):
        raise HTTPException(status_code=400, detail="仅支持 zip / tar / tar.gz / tgz 压缩包")

    data = await file.read()
    try:
        entries = await asyncio.to_thread(extract_entries, file.filename, data)
    except ArchiveError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e

    db = request.app.state.sqlite_db
    base = (path.strip("/") + "/" if path.strip("/") else "") + archive_stem(file.filename)

    cursor = await db.execute(
        "SELECT relative_path, content_hash FROM documents WHERE status != 'failed'"
    )
    rows = await cursor.fetchall()
    existing_paths = {r[0] for r in rows}
    existing_hashes = {r[1] for r in rows if r[1]}

    documents: list[dict] = []
    skipped_duplicate = 0
    skipped_unsupported = 0
    for entry_relative, content in entries:
        ext = entry_relative.rsplit(".", 1)[-1].lower() if "." in entry_relative else ""
        if ext not in _ARCHIVE_ALLOWED_TYPES:
            skipped_unsupported += 1
            continue
        relative = f"{base}/{entry_relative}"
        digest = hashlib.sha256(content).hexdigest()
        if relative in existing_paths or digest in existing_hashes:
            skipped_duplicate += 1
            continue
        existing_paths.add(relative)
        existing_hashes.add(digest)
        documents.append(await _ingest_bytes(db, relative, content))

    return {
        "archive": file.filename,
        "target": "/" + base + "/",
        "created": len(documents),
        "skipped_duplicate": skipped_duplicate,
        "skipped_unsupported": skipped_unsupported,
        "documents": documents,
    }
