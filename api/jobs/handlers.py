"""Transport-neutral worker handler contracts and initial placeholder registry."""

from __future__ import annotations

import re
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from uuid import UUID

import asyncpg

from jobs.lease import JobLease
from jobs.models import ERROR_MESSAGE_MAX_CHARS, JobRecord, JobType, JSONValue

_ERROR_CODE_PATTERN = re.compile(r"[a-z][a-z0-9_]{0,63}\Z")
_GENERIC_ERROR_MESSAGE = "The job could not be completed."
_VETTED_ERROR_MESSAGES = MappingProxyType(
    {
        "converter_timeout": "Converter timed out.",
        "document_not_found": "The requested document was not found.",
        "extraction_transient": "Document extraction will be retried.",
        "invalid_document": "Document is invalid.",
        "invalid_document_job": "The document extraction job is invalid.",
        "invalid_job_result": "The job produced an invalid result.",
        "quota_exceeded": "The account quota was exceeded.",
        "unsupported_document_type": "This document type is not supported.",
        "unsupported_job_type": "This job type is not supported.",
    }
)


@dataclass(frozen=True, slots=True)
class WorkerContext:
    pool: asyncpg.Pool
    s3: object | None
    converter_url: str
    converter_secret: str


Handler = Callable[
    [JobRecord, JobLease, WorkerContext],
    Awaitable[Mapping[str, JSONValue]],
]


def _sanitize_message(message: str) -> str:
    if not isinstance(message, str):
        raise TypeError("error message must be a string")
    sanitized = " ".join(message.split())[:ERROR_MESSAGE_MAX_CHARS]
    if not sanitized:
        raise ValueError("error message must not be empty")
    return _GENERIC_ERROR_MESSAGE


class JobHandlerError(RuntimeError):
    """A bounded, persistence-safe failure raised by a business handler."""

    def __init__(self, error_code: str, error_message: str) -> None:
        if not isinstance(error_code, str) or not _ERROR_CODE_PATTERN.fullmatch(error_code):
            raise ValueError("error_code must be a stable lowercase identifier")
        self.error_code = error_code
        fallback_message = _sanitize_message(error_message)
        self.error_message = _VETTED_ERROR_MESSAGES.get(error_code, fallback_message)
        super().__init__(self.error_message)


class RetryableJobError(JobHandlerError):
    """A sanitized business failure eligible for durable PostgreSQL retry."""


class TerminalJobError(JobHandlerError):
    """A sanitized business failure that must not be retried."""


class UnsupportedJobHandler(TerminalJobError):
    """Raised until a persisted job type has a concrete business handler."""

    def __init__(self) -> None:
        super().__init__("unsupported_job_type", "This job type is not supported.")


async def handle_document_extract(
    job: JobRecord,
    lease: JobLease,
    context: WorkerContext,
) -> Mapping[str, JSONValue]:
    from services.ocr import (
        OCR_TYPES,
        OCRService,
        RetryableExtractionError,
        TerminalExtractionError,
    )

    await lease.checkpoint()
    _validate_document_job_shape(job)
    terminal_validation_error = await _prepare_document_extraction(job, lease, context, OCR_TYPES)
    if terminal_validation_error is not None:
        raise terminal_validation_error

    if context.s3 is None:
        await _set_extraction_failure_status(job, lease, context, retryable=True)
        raise RetryableJobError("extraction_transient", "Document extraction will be retried.")

    async def final_checkpoint(conn: asyncpg.Connection) -> None:
        await lease.checkpoint(conn)

    async def artifact_checkpoint() -> None:
        await lease.checkpoint()

    try:
        version = await OCRService(context.s3, context.pool).extract_document(
            str(job.document_id),
            str(job.user_id),
            before_write=final_checkpoint,
            before_artifact_write=artifact_checkpoint,
            artifact_namespace=f"{job.id}/attempt-{job.attempt_count}",
        )
    except TerminalExtractionError as exc:
        await _set_extraction_failure_status(job, lease, context, retryable=False)
        raise TerminalJobError(exc.error_code, exc.error_message) from None
    except RetryableExtractionError as exc:
        await _set_extraction_failure_status(job, lease, context, retryable=True)
        raise RetryableJobError(exc.error_code, exc.error_message) from None
    except (ValueError, UnicodeError):
        await _set_extraction_failure_status(job, lease, context, retryable=False)
        raise TerminalJobError("invalid_document", "Document is invalid.") from None
    except Exception as exc:  # noqa: BLE001 - unknown I/O failures are safely retried.
        from jobs.models import JobCancelled, LeaseLost

        if isinstance(exc, (JobCancelled, LeaseLost)):
            raise
        await _set_extraction_failure_status(job, lease, context, retryable=True)
        raise RetryableJobError("extraction_transient", "Document extraction will be retried.") from None

    return {"document_id": str(job.document_id), "derived_version": version}


def _validate_document_job_shape(job: JobRecord) -> None:
    if job.job_type is not JobType.DOCUMENT_EXTRACT:
        raise TerminalJobError("invalid_document_job", "The document extraction job is invalid.")
    if job.document_id is None or job.knowledge_base_id is None:
        raise TerminalJobError("invalid_document_job", "The document extraction job is invalid.")

    payload_document_id = job.payload.get("document_id")
    try:
        parsed_payload_document_id = UUID(payload_document_id) if isinstance(payload_document_id, str) else None
    except ValueError:
        parsed_payload_document_id = None
    if parsed_payload_document_id != job.document_id or set(job.payload) != {"document_id"}:
        raise TerminalJobError("invalid_document_job", "The document extraction job is invalid.")


async def _prepare_document_extraction(
    job: JobRecord,
    lease: JobLease,
    context: WorkerContext,
    ocr_types: set[str],
) -> TerminalJobError | None:
    async with context.pool.acquire() as conn, conn.transaction():
        await lease.checkpoint(conn)
        document = await conn.fetchrow(
            "SELECT filename, file_type FROM documents "
            "WHERE id = $1 AND user_id = $2 AND knowledge_base_id = $3 "
            "AND NOT archived AND source_kind = 'source' FOR UPDATE",
            job.document_id,
            job.user_id,
            job.knowledge_base_id,
        )
        if document is None:
            raise TerminalJobError("document_not_found", "The requested document was not found.")

        filename = document["filename"]
        extension = filename.rsplit(".", 1)[-1].lower() if "." in filename else document["file_type"]
        supported = ocr_types | {"html", "htm", "xlsx", "xls", "csv"}
        if extension not in supported:
            await conn.execute(
                "UPDATE documents SET "
                "status = CASE WHEN status = 'ready' AND version > 0 THEN status ELSE 'failed' END, "
                "error_message = CASE WHEN status = 'ready' AND version > 0 THEN NULL ELSE $2 END, "
                "updated_at = now() "
                "WHERE id = $1 AND user_id = $3 AND knowledge_base_id = $4",
                job.document_id,
                "This document type is not supported.",
                job.user_id,
                job.knowledge_base_id,
            )
            terminal_validation_error = TerminalJobError(
                "unsupported_document_type",
                "This document type is not supported.",
            )
        else:
            await conn.execute(
                "UPDATE documents SET status = CASE WHEN status = 'ready' AND version > 0 "
                "THEN status ELSE 'processing' END, error_message = NULL, updated_at = now() "
                "WHERE id = $1 AND user_id = $2 AND knowledge_base_id = $3",
                job.document_id,
                job.user_id,
                job.knowledge_base_id,
            )
            terminal_validation_error = None

    return terminal_validation_error


async def _set_extraction_failure_status(
    job: JobRecord,
    lease: JobLease,
    context: WorkerContext,
    *,
    retryable: bool,
) -> None:
    status = "processing" if retryable and job.attempt_count < job.max_attempts else "failed"
    error_message = None if status == "processing" else "Document extraction failed."
    async with context.pool.acquire() as conn, conn.transaction():
        await lease.checkpoint(conn)
        updated_document_id = await conn.fetchval(
            "UPDATE documents SET "
            "status = CASE WHEN status = 'ready' AND version > 0 THEN status ELSE $2 END, "
            "error_message = CASE WHEN status = 'ready' AND version > 0 THEN NULL ELSE $3 END, "
            "updated_at = now() "
            "WHERE id = $1 AND user_id = $4 AND knowledge_base_id = $5 "
            "AND NOT archived AND source_kind = 'source' RETURNING id",
            job.document_id,
            status,
            error_message,
            job.user_id,
            job.knowledge_base_id,
        )
        if updated_document_id is None:
            raise TerminalJobError("document_not_found", "The requested document was not found.")


async def handle_graph_rebuild(
    job: JobRecord,
    lease: JobLease,
    context: WorkerContext,
) -> Mapping[str, JSONValue]:
    del job, lease, context
    raise UnsupportedJobHandler


async def handle_upload_cleanup(
    job: JobRecord,
    lease: JobLease,
    context: WorkerContext,
) -> Mapping[str, JSONValue]:
    del job, lease, context
    raise UnsupportedJobHandler


HANDLERS: Mapping[JobType, Handler] = MappingProxyType(
    {
        JobType.DOCUMENT_EXTRACT: handle_document_extract,
        JobType.GRAPH_REBUILD: handle_graph_rebuild,
        JobType.UPLOAD_CLEANUP: handle_upload_cleanup,
    }
)
