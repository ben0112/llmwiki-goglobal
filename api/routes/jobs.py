"""Hosted HTTP surface for tenant-scoped durable jobs."""

from datetime import datetime
from typing import Annotated
from uuid import UUID

from deps import get_job_service, get_user_id
from fastapi import APIRouter, Depends, HTTPException, status
from jobs.models import JobState, JobType, serialize_public_job
from jobs.service import JobService
from pydantic import BaseModel, JsonValue

router = APIRouter(prefix="/v1/jobs", tags=["jobs"])


class PublicJobError(BaseModel):
    code: str
    message: str


class PublicJobResponse(BaseModel):
    id: UUID
    type: JobType
    state: JobState
    progress: dict[str, JsonValue] | None
    result: dict[str, JsonValue] | None
    attempt_count: int
    max_attempts: int
    cancel_requested: bool
    error: PublicJobError | None
    created_at: datetime
    updated_at: datetime


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


def _not_found() -> HTTPException:
    return HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Job not found")


@router.get("/{job_id}", response_model=PublicJobResponse)
async def get_job(
    job_id: UUID,
    authenticated_user_id: Annotated[UUID, Depends(_get_authenticated_user_id)],
    service: Annotated[JobService, Depends(get_job_service)],
):
    record = await service.get(job_id, authenticated_user_id=authenticated_user_id)
    if record is None:
        raise _not_found()
    return serialize_public_job(record)


@router.post("/{job_id}/cancel", response_model=PublicJobResponse)
async def cancel_job(
    job_id: UUID,
    authenticated_user_id: Annotated[UUID, Depends(_get_authenticated_user_id)],
    service: Annotated[JobService, Depends(get_job_service)],
):
    record = await service.cancel(job_id, authenticated_user_id=authenticated_user_id)
    if record is None:
        raise _not_found()
    return serialize_public_job(record)
