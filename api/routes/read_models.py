"""Bounded, revision-aware read-model HTTP routes."""

from typing import Annotated
from uuid import UUID

from deps import get_read_service
from fastapi import APIRouter, Body, Depends, Header, HTTPException, Query, Request, Response
from pydantic import BaseModel, Field, model_validator
from services.read_models import (
    StaleReadCursor,
    UploadPreflightRequest,
    etag_for_revision,
    etag_matches,
)

from llmwiki_core.read_cursor import CursorError

router = APIRouter(tags=["read-models"])


class DocumentStatusRequest(BaseModel):
    ids: list[str] | None = Field(default=None, min_length=1, max_length=200)
    document_numbers: list[int] | None = Field(default=None, min_length=1, max_length=200)

    @model_validator(mode="after")
    def exactly_one_key_type(self):
        if (self.ids is None) == (self.document_numbers is None):
            raise ValueError("provide ids or document_numbers")
        return self


async def _revision_or_404(service, kb_id: str) -> int:
    try:
        return await service.revision(kb_id)
    except LookupError:
        raise HTTPException(status_code=404, detail="Knowledge base not found") from None


def _translate_read_error(error: Exception) -> HTTPException:
    if isinstance(error, StaleReadCursor):
        return HTTPException(
            status_code=409,
            detail={"code": "stale_cursor", "revision": error.revision},
        )
    if isinstance(error, CursorError):
        return HTTPException(status_code=422, detail={"code": "invalid_cursor"})
    return HTTPException(status_code=422, detail={"code": "invalid_request"})


@router.get("/v1/knowledge-bases/{kb_id}/documents/browse")
async def browse_documents(
    kb_id: UUID,
    response: Response,
    service: Annotated[object, Depends(get_read_service)],
    path: str | None = Query(default=None),
    query: str | None = Query(default=None),
    sort: str = Query(default="name"),
    direction: str = Query(default="asc"),
    limit: int = Query(default=100, ge=1, le=200),
    cursor: str | None = Query(default=None, max_length=2048),
    if_none_match: str | None = Header(default=None),
):
    kb = str(kb_id)
    revision = await _revision_or_404(service, kb)
    etag = etag_for_revision(revision)
    if etag_matches(if_none_match, revision):
        return Response(status_code=304, headers={"ETag": etag})
    try:
        result = await service.browse(
            kb,
            path=path,
            query=query,
            sort=sort,
            direction=direction,
            limit=limit,
            cursor=cursor,
        )
    except (CursorError, StaleReadCursor, ValueError) as error:
        raise _translate_read_error(error) from None
    response.headers["ETag"] = etag
    return result


@router.get("/v1/knowledge-bases/{kb_id}/wiki/pages")
async def wiki_pages(
    kb_id: UUID,
    response: Response,
    service: Annotated[object, Depends(get_read_service)],
    limit: int = Query(default=100, ge=1, le=200),
    cursor: str | None = Query(default=None, max_length=2048),
    if_none_match: str | None = Header(default=None),
):
    kb = str(kb_id)
    revision = await _revision_or_404(service, kb)
    etag = etag_for_revision(revision)
    if etag_matches(if_none_match, revision):
        return Response(status_code=304, headers={"ETag": etag})
    try:
        result = await service.wiki_pages(kb, limit=limit, cursor=cursor)
    except (CursorError, StaleReadCursor, ValueError) as error:
        raise _translate_read_error(error) from None
    response.headers["ETag"] = etag
    return result


@router.get("/v1/knowledge-bases/{kb_id}/documents/resolve")
async def resolve_document(
    kb_id: UUID,
    service: Annotated[object, Depends(get_read_service)],
    document_number: int | None = Query(default=None, ge=1),
    logical_reference: str | None = Query(default=None, min_length=1, max_length=4096),
):
    try:
        result = await service.resolve(
            str(kb_id),
            document_number=document_number,
            logical_reference=logical_reference,
        )
    except LookupError:
        raise HTTPException(status_code=404, detail="Knowledge base not found") from None
    except (CursorError, ValueError) as error:
        raise _translate_read_error(error) from None
    if result is None:
        raise HTTPException(status_code=404, detail="Document not found")
    return result


@router.post("/v1/knowledge-bases/{kb_id}/documents/status")
async def document_statuses(
    kb_id: UUID,
    body: DocumentStatusRequest,
    service: Annotated[object, Depends(get_read_service)],
):
    try:
        return await service.statuses(str(kb_id), ids=body.ids, document_numbers=body.document_numbers)
    except LookupError:
        raise HTTPException(status_code=404, detail="Knowledge base not found") from None
    except ValueError as error:
        raise _translate_read_error(error) from None


@router.post("/v1/knowledge-bases/{kb_id}/documents/upload-preflight")
async def upload_preflight(
    kb_id: UUID,
    service: Annotated[object, Depends(get_read_service)],
    body: UploadPreflightRequest = Body(...),
):
    try:
        return await service.upload_preflight(str(kb_id), [item.model_dump() for item in body.items])
    except LookupError:
        raise HTTPException(status_code=404, detail="Knowledge base not found") from None
    except ValueError as error:
        raise _translate_read_error(error) from None


_CORPUS_FILTERS = {
    "stage",
    "layer",
    "domain",
    "genre",
    "rule",
    "evidence",
    "origin",
    "dept",
    "country",
    "region",
    "geo",
    "industry",
    "mode",
    "timeliness",
    "state",
    "business",
    "entry_id",
}


def _corpus_filters(request: Request) -> dict[str, str]:
    return {key: value for key, value in request.query_params.items() if key in _CORPUS_FILTERS and value.strip()}


@router.get("/v1/knowledge-bases/{kb_id}/corpus/entries")
async def corpus_entries(
    kb_id: UUID,
    request: Request,
    response: Response,
    service: Annotated[object, Depends(get_read_service)],
    query: str | None = Query(default=None, max_length=1024),
    sort: str = Query(default="name"),
    direction: str = Query(default="asc"),
    limit: int = Query(default=100, ge=1, le=200),
    cursor: str | None = Query(default=None, max_length=2048),
    if_none_match: str | None = Header(default=None),
):
    kb = str(kb_id)
    revision = await _revision_or_404(service, kb)
    etag = etag_for_revision(revision)
    if etag_matches(if_none_match, revision):
        return Response(status_code=304, headers={"ETag": etag})
    try:
        result = await service.corpus_entries(
            kb,
            _corpus_filters(request),
            query=query,
            sort=sort,
            direction=direction,
            limit=limit,
            cursor=cursor,
        )
    except (CursorError, StaleReadCursor, ValueError) as error:
        raise _translate_read_error(error) from None
    response.headers["ETag"] = etag
    return result


@router.get("/v1/knowledge-bases/{kb_id}/corpus/summary")
async def corpus_summary(
    kb_id: UUID,
    request: Request,
    response: Response,
    service: Annotated[object, Depends(get_read_service)],
    query: str | None = Query(default=None, max_length=1024),
    if_none_match: str | None = Header(default=None),
):
    kb = str(kb_id)
    revision = await _revision_or_404(service, kb)
    etag = etag_for_revision(revision)
    if etag_matches(if_none_match, revision):
        return Response(status_code=304, headers={"ETag": etag})
    try:
        result = await service.corpus_summary(kb, _corpus_filters(request), query=query)
    except ValueError as error:
        raise _translate_read_error(error) from None
    response.headers["ETag"] = etag
    return result


@router.get("/v1/knowledge-bases/{kb_id}/graph/summary")
async def graph_summary(
    kb_id: UUID,
    response: Response,
    service: Annotated[object, Depends(get_read_service)],
    if_none_match: str | None = Header(default=None),
):
    kb = str(kb_id)
    revision = await _revision_or_404(service, kb)
    etag = etag_for_revision(revision)
    if etag_matches(if_none_match, revision):
        return Response(status_code=304, headers={"ETag": etag})
    result = await service.graph_summary(kb)
    response.headers["ETag"] = etag
    return result


__all__ = ["router"]
