"""Unauthenticated read-only routes for public wikis."""

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Request, Response
from starlette.responses import StreamingResponse

from infra.rate_limit import limiter
from services.base import PublicWikiService

router = APIRouter(prefix="/v1/public", tags=["public"])

_NO_CACHE_VALUE = "no-store, must-revalidate"
_NO_CACHE_HEADERS = {"Cache-Control": _NO_CACHE_VALUE}
_SLUG_MAX = 80


async def get_public_wiki_service(request: Request) -> PublicWikiService:
    return request.app.state.factory.public_wiki_service()


def _normalize_slug(slug: str) -> str | None:
    slug = slug.strip().lower()
    if not slug or len(slug) > _SLUG_MAX:
        return None
    return slug


@router.get("/wiki/{slug}")
@limiter.limit("60/minute")
async def get_public_wiki(
    request: Request,
    slug: str,
    response: Response,
    service: Annotated[PublicWikiService, Depends(get_public_wiki_service)],
):
    normalized = _normalize_slug(slug)
    if not normalized:
        raise HTTPException(status_code=404, detail="Wiki not found")
    wiki = await service.get_by_slug(normalized)
    if not wiki:
        raise HTTPException(status_code=404, detail="Wiki not found")
    response.headers["Cache-Control"] = _NO_CACHE_VALUE
    return wiki


@router.get("/wiki/{slug}/assets/{document_number}")
@limiter.limit("120/minute")
async def get_public_wiki_asset(
    request: Request,
    slug: str,
    document_number: int,
    service: Annotated[PublicWikiService, Depends(get_public_wiki_service)],
):
    normalized = _normalize_slug(slug)
    if not normalized or document_number < 0:
        raise HTTPException(status_code=404, detail="Asset not found")

    key = await service.get_asset_key(normalized, document_number)
    if not key:
        raise HTTPException(status_code=404, detail="Asset not found")

    s3 = request.app.state.s3_service
    if not s3:
        raise HTTPException(status_code=503, detail="Storage not configured")
    try:
        body = await s3.download_bytes(key)
    except Exception:
        raise HTTPException(status_code=404, detail="Asset not found")

    media_type = _media_type_for_key(key)

    async def _gen():
        yield body

    return StreamingResponse(_gen(), media_type=media_type, headers=_NO_CACHE_HEADERS)


def _media_type_for_key(key: str) -> str:
    ext = key.rsplit(".", 1)[-1].lower() if "." in key else ""
    return {
        "pdf": "application/pdf",
        "png": "image/png",
        "jpg": "image/jpeg",
        "jpeg": "image/jpeg",
        "gif": "image/gif",
        "webp": "image/webp",
        "svg": "image/svg+xml",
        "html": "text/html; charset=utf-8",
        "htm": "text/html; charset=utf-8",
        "txt": "text/plain; charset=utf-8",
        "md": "text/markdown; charset=utf-8",
        "csv": "text/csv; charset=utf-8",
        "json": "application/json",
    }.get(ext, "application/octet-stream")
