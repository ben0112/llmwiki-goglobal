"""Transport-neutral worker handler contracts and initial placeholder registry."""

from __future__ import annotations

import re
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from types import MappingProxyType

import asyncpg

from jobs.lease import JobLease
from jobs.models import ERROR_MESSAGE_MAX_CHARS, JobRecord, JobType, JSONValue

_ERROR_CODE_PATTERN = re.compile(r"[a-z][a-z0-9_]{0,63}\Z")
_GENERIC_ERROR_MESSAGE = "The job could not be completed."
_VETTED_ERROR_MESSAGES = MappingProxyType(
    {
        "converter_timeout": "Converter timed out.",
        "invalid_document": "Document is invalid.",
        "invalid_job_result": "The job produced an invalid result.",
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
    del job, lease, context
    raise UnsupportedJobHandler


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
