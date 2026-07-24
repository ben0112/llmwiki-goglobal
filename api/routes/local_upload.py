"""Local file upload routes — simple multipart, no TUS.

Copies uploaded files directly into the workspace and indexes them.
/v1/upload/archive 额外支持压缩包(zip/tar[.gz]):服务端解压后不另建
目录,按包内结构落在所选文件夹下,复用与普通上传相同的落盘/索引路径。
"""

import asyncio
import hashlib
import json
import os
import re
import uuid
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Request, UploadFile, File, Form
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from config import settings
from deps import get_user_id
from domain.file_types import EXTRACTION_TYPES, IMAGE_TYPES, SIMPLE_TEXT_TYPES
from domain.watcher import mark_written
from services.archive_extract import (
    ArchiveError, extract_entries, is_supported_archive,
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


MAX_UPLOAD_BYTES = 1_073_741_824  # 1 GiB
# 内联正文/建索引的文本上限:超大文本文件只登记不内联,避免整读进内存
_MAX_INLINE_TEXT_BYTES = 50 * 1024 * 1024


async def _ingest_bytes(db, relative: str, content_bytes: bytes) -> dict:
    """把一份文件字节落盘到工作区并建立索引(小文件上传与解压共用)。"""
    dest = _safe_resolve(relative)
    dest.parent.mkdir(parents=True, exist_ok=True)
    mark_written(str(dest))
    dest.write_bytes(content_bytes)
    return await _index_file_on_disk(db, relative, dest,
                                     hashlib.sha256(content_bytes).hexdigest())


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


async def _store_chunks_for_upload(db, doc_id: str, chunks: list, document_version: int) -> None:
    """Write upload chunks on the caller's active SQLite transaction."""
    from domain.local_processor import _store_chunks

    await _store_chunks(db, doc_id, chunks, document_version)


async def _index_file_on_disk(db, relative: str, dest: Path, content_hash: str) -> dict:
    """为已落盘的文件建立索引(哈希由调用方提供,大文件流式计算不进内存)。"""
    filename = relative.rsplit("/", 1)[-1]
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    stem = filename.rsplit(".", 1)[0] if "." in filename else filename
    title = stem.replace("-", " ").replace("_", " ").strip().title()

    dir_path = "/" + "/".join(relative.split("/")[:-1]) + "/" if "/" in relative else "/"
    source_kind = "wiki" if relative.startswith("wiki/") else "source"
    size = dest.stat().st_size

    # HTML is excluded from inline text — it goes through webmd in process_document.
    text_content = None
    needs_processing = ext in EXTRACTION_TYPES
    if ext in SIMPLE_TEXT_TYPES and size <= _MAX_INLINE_TEXT_BYTES:
        try:
            text_content = dest.read_text(encoding="utf-8", errors="replace")
        except Exception:
            pass

    from infra.db.sqlite import SQLiteDocumentRepository, serialized_write
    doc_repo = SQLiteDocumentRepository(db)

    doc_id = str(uuid.uuid4())
    is_simple_text = ext in SIMPLE_TEXT_TYPES
    chunks = []
    if is_simple_text and text_content:
        from services.chunker import chunk_text

        chunks = chunk_text(text_content)

    async with serialized_write(db):
        cursor = await db.execute("SELECT COALESCE(MAX(document_number), 0) + 1 FROM documents")
        row = await cursor.fetchone()
        doc_number = row[0]
        await db.execute(
            "INSERT INTO documents (id, user_id, filename, title, path, relative_path, "
            "source_kind, file_type, file_size, status, content, tags, version, "
            "content_hash, mtime_ns, last_indexed_at, document_number) "
            "VALUES (?, (SELECT user_id FROM workspace LIMIT 1), ?, ?, ?, ?, ?, ?, ?, ?, ?, "
            "'[]', 0, ?, ?, datetime('now'), ?)",
            (
                doc_id,
                filename,
                title,
                dir_path,
                relative,
                source_kind,
                ext or "bin",
                size,
                "processing" if is_simple_text else "pending",
                text_content,
                content_hash,
                int(dest.stat().st_mtime_ns),
                doc_number,
            ),
        )
        if is_simple_text:
            from domain.local_processor import _replace_text_content_on_connection

            await _replace_text_content_on_connection(
                db,
                doc_id,
                text_content,
                chunks,
                store_chunks=_store_chunks_for_upload,
            )
        await db.commit()

    if needs_processing:
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
    if len(content_bytes) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="单个文件超过 1GB 上限")
    return await _ingest_bytes(request.app.state.sqlite_db, relative, content_bytes)


# ---------------------------------------------------------------------------
# 断点续传(本地模式):init → [查询 offset →] 分块 PATCH → complete。
# 分块落在 .llmwiki/tmp/uploads/,完成时原子改名进工作区,哈希流式计算,
# 大文件全程不整份进内存。中断后凭 upload_id 查询 offset 续传。
# ---------------------------------------------------------------------------

_STALE_PART_SECONDS = 24 * 3600
_UPLOAD_ID_RE = re.compile(r"^[0-9a-f]{32}$")


class ResumableInit(BaseModel):
    filename: str
    path: str = "/"
    size: int


def _parts_dir() -> Path:
    d = _workspace_root() / ".llmwiki" / "tmp" / "uploads"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _part_paths(upload_id: str) -> tuple[Path, Path]:
    if not _UPLOAD_ID_RE.match(upload_id):
        raise HTTPException(status_code=400, detail="非法 upload_id")
    d = _parts_dir()
    return d / f"{upload_id}.part", d / f"{upload_id}.json"


def _purge_stale_parts() -> None:
    import time
    now = time.time()
    for p in _parts_dir().glob("*.part"):
        try:
            if now - p.stat().st_mtime > _STALE_PART_SECONDS:
                p.unlink(missing_ok=True)
                p.with_suffix(".json").unlink(missing_ok=True)
        except OSError:
            pass


@router.post("/v1/upload/resumable", status_code=201)
async def resumable_init(body: ResumableInit, user_id: str = Depends(get_user_id)):
    if not body.filename:
        raise HTTPException(status_code=400, detail="No filename")
    if body.size <= 0 or body.size > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="单个文件超过 1GB 上限")
    _purge_stale_parts()
    upload_id = uuid.uuid4().hex
    part, meta = _part_paths(upload_id)
    meta.write_text(json.dumps(
        {"filename": body.filename, "path": body.path, "size": body.size},
        ensure_ascii=False), encoding="utf-8")
    part.touch()
    return {"upload_id": upload_id, "offset": 0}


@router.get("/v1/upload/resumable/{upload_id}")
async def resumable_offset(upload_id: str, user_id: str = Depends(get_user_id)):
    part, meta = _part_paths(upload_id)
    if not meta.is_file():
        raise HTTPException(status_code=404, detail="上传会话不存在或已过期")
    return {"offset": part.stat().st_size if part.is_file() else 0}


@router.patch("/v1/upload/resumable/{upload_id}")
async def resumable_patch(
    upload_id: str,
    request: Request,
    offset: int,
    user_id: str = Depends(get_user_id),
):
    part, meta_path = _part_paths(upload_id)
    if not meta_path.is_file():
        raise HTTPException(status_code=404, detail="上传会话不存在或已过期")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    current = part.stat().st_size if part.is_file() else 0
    if offset != current:
        # 客户端与服务端进度不一致:排空请求体后回报真实 offset
        # (不排空则连接被复位,客户端读不到 409 的 JSON)
        async for _ in request.stream():
            pass
        return JSONResponse(status_code=409, content={"offset": current})

    written = current
    with open(part, "ab") as fh:
        async for chunk in request.stream():
            written += len(chunk)
            if written > meta["size"] or written > MAX_UPLOAD_BYTES:
                fh.close()
                part.unlink(missing_ok=True)
                meta_path.unlink(missing_ok=True)
                raise HTTPException(status_code=413, detail="数据超出声明大小")
            fh.write(chunk)
    return {"offset": written}


@router.post("/v1/upload/resumable/{upload_id}/complete", status_code=201)
async def resumable_complete(
    upload_id: str,
    user_id: str = Depends(get_user_id),
    request: Request = None,
):
    part, meta_path = _part_paths(upload_id)
    if not meta_path.is_file() or not part.is_file():
        raise HTTPException(status_code=404, detail="上传会话不存在或已过期")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    if part.stat().st_size != meta["size"]:
        return JSONResponse(status_code=409, content={"offset": part.stat().st_size})

    relative = (meta["path"].rstrip("/") + "/" + meta["filename"]).lstrip("/")
    dest = _safe_resolve(relative)
    dest.parent.mkdir(parents=True, exist_ok=True)
    content_hash = await asyncio.to_thread(_hash_file, part)
    mark_written(str(dest))
    os.replace(part, dest)
    meta_path.unlink(missing_ok=True)
    return await _index_file_on_disk(request.app.state.sqlite_db, relative, dest, content_hash)


@router.post("/v1/upload/archive", status_code=201)
async def upload_archive(
    file: UploadFile = File(...),
    path: str = Form(default="/"),
    user_id: str = Depends(get_user_id),
    request: Request = None,
):
    """上传压缩包并在服务端解压入库。

    与文件夹上传同口径的剥层:不以包名另建目录,保留包内目录结构,
    直接落在所选文件夹(缺省视图根)下;去重亦与前端上传同口径:
    同相对路径或同内容哈希的条目自动跳过,仅在响应中计数。
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename")
    if not is_supported_archive(file.filename):
        raise HTTPException(status_code=400, detail="仅支持 zip / tar / tar.gz / tgz 压缩包")

    data = await file.read()
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="压缩包超过 1GB 上限")
    try:
        entries = await asyncio.to_thread(extract_entries, file.filename, data)
    except ArchiveError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e

    db = request.app.state.sqlite_db
    base = path.strip("/") + "/" if path.strip("/") else ""

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
        relative = f"{base}{entry_relative}"
        digest = hashlib.sha256(content).hexdigest()
        if relative in existing_paths or digest in existing_hashes:
            skipped_duplicate += 1
            continue
        existing_paths.add(relative)
        existing_hashes.add(digest)
        documents.append(await _ingest_bytes(db, relative, content))

    return {
        "archive": file.filename,
        "target": "/" + base,
        "created": len(documents),
        "skipped_duplicate": skipped_duplicate,
        "skipped_unsupported": skipped_unsupported,
        "documents": documents,
    }
