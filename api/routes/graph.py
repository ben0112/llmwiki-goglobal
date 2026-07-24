"""Hosted graph routes — reads are scoped and rebuilds are durable jobs."""

from typing import Annotated
from uuid import UUID

from deps import get_job_service, get_scoped_db, get_user_id
from fastapi import APIRouter, Depends, HTTPException, status
from jobs.service import JobResourceNotFound, JobService
from pydantic import BaseModel
from scoped_db import ScopedDB
from services.graph import get_graph_hosted

router = APIRouter(tags=["graph"])


class GraphRebuildAccepted(BaseModel):
    job_id: UUID


async def _get_authenticated_user_id(
    user_id: Annotated[str, Depends(get_user_id)],
) -> UUID:
    try:
        return UUID(user_id)
    except (TypeError, ValueError):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authenticated subject",
        ) from None


@router.get("/v1/knowledge-bases/{kb_id}/graph")
async def get_kb_graph(
    kb_id: UUID,
    db: ScopedDB = Depends(get_scoped_db),
):
    return await get_graph_hosted(db.conn, kb_id, db.user_id)


@router.post(
    "/v1/knowledge-bases/{kb_id}/graph/rebuild",
    response_model=GraphRebuildAccepted,
    status_code=status.HTTP_202_ACCEPTED,
)
async def rebuild_references(
    kb_id: UUID,
    authenticated_user_id: Annotated[UUID, Depends(_get_authenticated_user_id)],
    service: Annotated[JobService, Depends(get_job_service)],
):
    try:
        job = await service.ensure_graph_rebuild(
            knowledge_base_id=kb_id,
            authenticated_user_id=authenticated_user_id,
        )
    except JobResourceNotFound:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Knowledge base not found",
        ) from None
    return GraphRebuildAccepted(job_id=job.id)
