from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, Request

from auth import get_current_user
from deps import get_document_service
from infra.rate_limit import limiter
from services.base import DocumentService
from services.types import (
    BulkDelete, CreateFromUrl, CreateNote,     ReplaceHighlights, UpdateContent, UpdateMetadata, UpsertHighlight,
)
from services.url_ingest import UrlIngestService

router = APIRouter(tags=["documents"])


@router.get("/v1/knowledge-bases/{kb_id}/documents")
async def list_documents(
    kb_id: UUID,
    service: Annotated[DocumentService, Depends(get_document_service)],
    path: str | None = Query(None),
):
    return await service.list(str(kb_id), path)


@router.get("/v1/documents/regen-status")
async def regen_status(request: Request):
    """删除源文件后维基页面后台重生成的进度(本地模式;前端轮询)。

    必须先于 /v1/documents/{doc_id} 声明,否则会被 UUID 路径参数吞掉。"""
    if getattr(request.app.state, "mode", "") != "local":
        return {"running": False, "total": 0, "done": 0, "failed": 0, "pages": [], "mode": "", "finished_at": None}
    from services.wiki_regen import regen_status as _regen_status
    return _regen_status()


@router.get("/v1/documents/{doc_id}")
async def get_document(doc_id: UUID, service: Annotated[DocumentService, Depends(get_document_service)]):
    row = await service.get(str(doc_id))
    if not row:
        raise HTTPException(status_code=404, detail="Document not found")
    return row


@router.post("/v1/documents/{doc_id}/retry-extraction", status_code=202)
async def retry_extraction(
    doc_id: UUID, request: Request,
    user: Annotated[dict, Depends(get_current_user)],
):
    """手动重试文档提取:清零失败隔离计数并重新排队(本地模式)。

    失败满 FAILED_RETRY_LIMIT 次的文档不再随启动自动重试,这里是普通
    源文档(没有语料条目,走不了「重新识别」)唯一的显式解除通道。
    """
    state = request.app.state
    if getattr(state, "mode", "") != "local":
        raise HTTPException(status_code=400, detail="仅本地模式支持")
    db = state.sqlite_db
    cursor = await db.execute(
        "SELECT status FROM documents WHERE id = ?", (str(doc_id),))
    if await cursor.fetchone() is None:
        raise HTTPException(status_code=404, detail="Document not found")
    await db.execute(
        "UPDATE documents SET status = 'pending', error_message = NULL, "
        "extraction_attempts = 0, updated_at = datetime('now') WHERE id = ?",
        (str(doc_id),))
    await db.commit()

    from pathlib import Path

    from domain.local_processor import process_document_isolated
    from infra.tasks import spawn_logged
    spawn_logged(process_document_isolated(Path(state.workspace_path), str(doc_id)),
                 f"retry:{str(doc_id)[:8]}")
    return {"status": "queued"}


@router.get("/v1/documents/{doc_id}/url")
async def get_document_url(doc_id: UUID, service: Annotated[DocumentService, Depends(get_document_service)]):
    result = await service.get_url(str(doc_id))
    if not result:
        raise HTTPException(status_code=404, detail="Document not found")
    return result


@router.get("/v1/documents/{doc_id}/content")
async def get_document_content(doc_id: UUID, service: Annotated[DocumentService, Depends(get_document_service)]):
    row = await service.get_content(str(doc_id))
    if not row:
        raise HTTPException(status_code=404, detail="Document not found")
    return row


@router.post("/v1/knowledge-bases/{kb_id}/documents/note", status_code=201)
async def create_note(
    kb_id: UUID,
    body: CreateNote,
    service: Annotated[DocumentService, Depends(get_document_service)],
):
    return await service.create_note(str(kb_id), body.filename, body.path, body.content)


@router.post("/v1/documents/from-url", status_code=201)
@limiter.limit("10/minute")
async def create_document_from_url(request: Request, body: CreateFromUrl):
    user_id = await get_current_user(request)
    state = request.app.state
    if not state.s3_service or not state.ocr_service:
        raise HTTPException(status_code=501, detail="URL ingestion is only available in hosted mode")
    service = UrlIngestService(state.pool, state.s3_service, state.ocr_service)
    return await service.ingest_pdf(user_id, str(body.knowledge_base_id), body.url, body.path)


@router.get("/v1/documents/{doc_id}/highlights")
async def get_document_highlights(
    doc_id: UUID,
    service: Annotated[DocumentService, Depends(get_document_service)],
):
    row = await service.get_highlights(str(doc_id))
    if not row:
        raise HTTPException(status_code=404, detail="Document not found")
    return row


@router.patch("/v1/documents/{doc_id}/highlights")
async def replace_document_highlights(
    doc_id: UUID,
    body: ReplaceHighlights,
    service: Annotated[DocumentService, Depends(get_document_service)],
):
    highlights = [h.model_dump() for h in body.highlights]
    row = await service.replace_highlights(
        str(doc_id), highlights, body.expectedVersion,
    )
    if not row:
        raise HTTPException(status_code=404, detail="Document not found")
    if row.get("conflict"):
        raise HTTPException(
            status_code=409,
            detail="Version mismatch — refetch and retry",
        )
    return row


@router.post("/v1/documents/{doc_id}/highlights", status_code=200)
async def upsert_document_highlight(
    doc_id: UUID,
    body: UpsertHighlight,
    service: Annotated[DocumentService, Depends(get_document_service)],
):
    """Idempotent single-highlight upsert. Re-posting the same {id, payload}
    is safe; the wire-level retry behavior on dropped connections matters more
    than strict deduplication."""
    row = await service.upsert_highlight(
        str(doc_id), body.highlight.model_dump(), body.expectedVersion,
    )
    if not row:
        raise HTTPException(status_code=404, detail="Document not found")
    if row.get("conflict"):
        raise HTTPException(
            status_code=409,
            detail="Version mismatch — refetch and retry",
        )
    return row


@router.delete("/v1/documents/{doc_id}/highlights/{highlight_id}", status_code=200)
async def delete_document_highlight(
    doc_id: UUID,
    highlight_id: str,
    service: Annotated[DocumentService, Depends(get_document_service)],
    expectedVersion: int | None = Query(None),
):
    """Idempotent single-highlight delete. Removing an absent id returns the
    current state without bumping the version (200 either way).
    `expectedVersion` is a query param (DELETE bodies are awkward in some
    proxies/clients)."""
    row = await service.delete_highlight(str(doc_id), highlight_id, expectedVersion)
    if not row:
        raise HTTPException(status_code=404, detail="Document not found")
    if row.get("conflict"):
        raise HTTPException(
            status_code=409,
            detail="Version mismatch — refetch and retry",
        )
    return row


@router.put("/v1/documents/{doc_id}/content")
async def update_document_content(
    doc_id: UUID,
    body: UpdateContent,
    service: Annotated[DocumentService, Depends(get_document_service)],
):
    row = await service.update_content(str(doc_id), body.content)
    if not row:
        raise HTTPException(status_code=404, detail="Document not found")
    return row


@router.patch("/v1/documents/{doc_id}")
async def update_document_metadata(
    doc_id: UUID,
    body: UpdateMetadata,
    service: Annotated[DocumentService, Depends(get_document_service)],
):
    fields = {k: v for k, v in body.model_dump().items() if v is not None}
    if not fields:
        raise HTTPException(status_code=400, detail="No fields to update")
    row = await service.update_metadata(str(doc_id), fields)
    if not row:
        raise HTTPException(status_code=404, detail="Document not found")
    return row


_BULK_DELETE_MAX_IDS = 100


@router.post("/v1/documents/delete-impact")
async def delete_impact(
    body: BulkDelete,
    service: Annotated[DocumentService, Depends(get_document_service)],
):
    """删除前预估影响:哪些维基页面引用了这些文档(删除后将自动重生成)。"""
    if not body.ids or len(body.ids) > _BULK_DELETE_MAX_IDS:
        return {"pages": [], "count": 0}
    return await service.delete_impact(body.ids)


@router.post("/v1/documents/bulk-delete", status_code=204)
async def bulk_delete_documents(
    body: BulkDelete,
    service: Annotated[DocumentService, Depends(get_document_service)],
):
    if not body.ids:
        return
    if len(body.ids) > _BULK_DELETE_MAX_IDS:
        raise HTTPException(
            status_code=400,
            detail=f"Too many ids: max {_BULK_DELETE_MAX_IDS} per request",
        )
    await service.bulk_delete(body.ids)


@router.delete("/v1/documents/{doc_id}", status_code=204)
async def delete_document(doc_id: UUID, service: Annotated[DocumentService, Depends(get_document_service)]):
    if not await service.delete(str(doc_id)):
        raise HTTPException(status_code=404, detail="Document not found")
