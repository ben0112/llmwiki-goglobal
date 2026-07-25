"""Transport-neutral worker handler contracts and initial placeholder registry."""

from __future__ import annotations

import logging
import re
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from math import isfinite
from numbers import Real
from types import MappingProxyType
from uuid import UUID

import asyncpg

from jobs.lease import JobLease
from jobs.models import (
    ERROR_MESSAGE_MAX_CHARS,
    JobCancelled,
    JobRecord,
    JobType,
    JSONValue,
    LeaseLost,
)
from llmwiki_core.models import (
    EmbeddingInputError,
    EmbeddingProfile,
    EmbeddingUnavailable,
    InvalidEmbeddingResponse,
)
from llmwiki_core.search import RetrieverUnavailable

_ERROR_CODE_PATTERN = re.compile(r"[a-z][a-z0-9_]{0,63}\Z")
_GENERIC_ERROR_MESSAGE = "The job could not be completed."
logger = logging.getLogger(__name__)
_VETTED_ERROR_MESSAGES = MappingProxyType(
    {
        "converter_timeout": "Converter timed out.",
        "document_not_found": "The requested document was not found.",
        "extraction_transient": "Document extraction will be retried.",
        "embedding_profile_changed": "The embedding profile changed before execution.",
        "embedding_storage_transient": "Document embeddings will be retried.",
        "embedding_unavailable": "The embedding provider is temporarily unavailable.",
        "graph_transient": "Graph rebuild will be retried.",
        "invalid_document": "Document is invalid.",
        "invalid_document_job": "The document extraction job is invalid.",
        "invalid_embedding_job": "The document embedding job is invalid.",
        "invalid_embedding_response": "The embedding provider returned an invalid response.",
        "invalid_job_result": "The job produced an invalid result.",
        "invalid_graph_job": "The graph rebuild job is invalid.",
        "invalid_upload_job": "The upload cleanup job is invalid.",
        "knowledge_base_not_found": "The requested knowledge base was not found.",
        "quota_exceeded": "The account quota was exceeded.",
        "upload_cleanup_transient": "Upload cleanup will be retried.",
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
    tus_cleanup: object | None = None


Handler = Callable[
    [JobRecord, JobLease, WorkerContext],
    Awaitable[Mapping[str, JSONValue]],
]
EmbeddingSource = tuple[tuple[int, str], ...]
IndexedEmbeddings = tuple[tuple[int, tuple[float, ...]], ...]


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


def _validate_embedding_job_shape(job: JobRecord) -> tuple[int, EmbeddingProfile]:
    invalid = job.job_type is not JobType.DOCUMENT_EMBED
    invalid = invalid or job.document_id is None or job.knowledge_base_id is None
    expected_keys = {"document_id", "document_version", "provider", "model", "dimensions"}
    invalid = invalid or set(job.payload) != expected_keys
    raw_document_id = job.payload.get("document_id")
    try:
        payload_document_id = UUID(raw_document_id) if isinstance(raw_document_id, str) else None
    except ValueError:
        payload_document_id = None
    invalid = invalid or payload_document_id != job.document_id or str(payload_document_id) != raw_document_id
    version = job.payload.get("document_version")
    invalid = invalid or type(version) is not int or not 1 <= version <= 2_147_483_647
    provider = job.payload.get("provider")
    model = job.payload.get("model")
    dimensions = job.payload.get("dimensions")
    invalid = invalid or not isinstance(provider, str) or not provider or provider.strip() != provider or len(provider) > 100
    invalid = invalid or not isinstance(model, str) or not model or model.strip() != model or len(model) > 200
    invalid = invalid or type(dimensions) is not int or not 1 <= dimensions <= 4096
    if invalid:
        raise TerminalJobError("invalid_embedding_job", "The document embedding job is invalid.")
    try:
        profile = EmbeddingProfile(provider=provider, model=model, dimensions=dimensions)
    except (TypeError, ValueError):
        raise TerminalJobError("invalid_embedding_job", "The document embedding job is invalid.") from None
    return version, profile


def _validate_embedding_vectors(
    vectors: object,
    *,
    expected_count: int,
    dimensions: int,
) -> IndexedEmbeddings:
    if isinstance(vectors, (str, bytes)) or not isinstance(vectors, (list, tuple)) or len(vectors) != expected_count:
        raise TerminalJobError(
            "invalid_embedding_response",
            "The embedding provider returned an invalid response.",
        )
    normalized = []
    for index, vector in enumerate(vectors):
        if isinstance(vector, (str, bytes)) or not isinstance(vector, (list, tuple)) or len(vector) != dimensions:
            raise TerminalJobError(
                "invalid_embedding_response",
                "The embedding provider returned an invalid response.",
            )
        coordinates = []
        for coordinate in vector:
            if isinstance(coordinate, bool) or not isinstance(coordinate, Real):
                raise TerminalJobError(
                    "invalid_embedding_response",
                    "The embedding provider returned an invalid response.",
                )
            try:
                value = float(coordinate)
            except (OverflowError, TypeError, ValueError):
                raise TerminalJobError(
                    "invalid_embedding_response",
                    "The embedding provider returned an invalid response.",
                ) from None
            if not isfinite(value):
                raise TerminalJobError(
                    "invalid_embedding_response",
                    "The embedding provider returned an invalid response.",
                )
            coordinates.append(value)
        if not any(coordinates):
            raise TerminalJobError(
                "invalid_embedding_response",
                "The embedding provider returned an invalid response.",
            )
        normalized.append((index, tuple(coordinates)))
    return tuple(normalized)


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
        if isinstance(exc, (JobCancelled, LeaseLost)):
            raise
        await _set_extraction_failure_status(job, lease, context, retryable=True)
        raise RetryableJobError("extraction_transient", "Document extraction will be retried.") from None

    await _enqueue_embedding_after_extraction(job, version, context)
    return {"document_id": str(job.document_id), "derived_version": version}


def _configured_embedding_profile() -> EmbeddingProfile | None:
    from config import settings

    return settings.embedding_profile


async def _enqueue_embedding_after_extraction(
    job: JobRecord,
    document_version: int,
    context: WorkerContext,
) -> None:
    profile = _configured_embedding_profile()
    if profile is None:
        return
    from jobs.service import JobService

    try:
        await JobService(context.pool).ensure_document_embedding(
            document_id=job.document_id,
            document_version=document_version,
            user_id=job.user_id,
            knowledge_base_id=job.knowledge_base_id,
            profile=profile,
        )
    except Exception as exc:  # noqa: BLE001 - reconciliation repairs this post-commit gap.
        logger.warning(
            "embedding_enqueue_failed document_id=%s document_version=%s error_type=%s",
            job.document_id,
            document_version,
            type(exc).__name__,
        )


async def _embed_texts(
    profile: EmbeddingProfile,
    texts: tuple[str, ...],
) -> tuple[tuple[float, ...], ...]:
    from config import settings
    from services.embeddings import OpenAIEmbeddingClient

    async with OpenAIEmbeddingClient(
        profile=profile,
        base_url=settings.EMBEDDING_BASE_URL,
        api_key=settings.EMBEDDING_API_KEY.get_secret_value(),
        batch_size=settings.EMBEDDING_BATCH_SIZE,
        timeout_seconds=settings.EMBEDDING_TIMEOUT_SECONDS,
    ) as client:
        return await client.embed(texts)


def _profile_matches(
    configured: EmbeddingProfile | None,
    submitted: EmbeddingProfile,
) -> bool:
    return configured is not None and configured.identity == submitted.identity


async def _read_embedding_source(
    job: JobRecord,
    document_version: int,
    context: WorkerContext,
) -> EmbeddingSource | None:
    document = await context.pool.fetchrow(
        "SELECT version,status::text FROM documents "
        "WHERE id=$1 AND user_id=$2 AND knowledge_base_id=$3 "
        "AND NOT archived AND source_kind='source'",
        job.document_id,
        job.user_id,
        job.knowledge_base_id,
    )
    if document is None:
        raise TerminalJobError("document_not_found", "The requested document was not found.")
    if document["version"] != document_version:
        return None
    if document["status"] != "ready":
        raise RetryableJobError("embedding_storage_transient", "Document embeddings will be retried.")
    rows = await context.pool.fetch(
        "SELECT chunk_index,content FROM document_chunks "
        "WHERE document_id=$1 AND document_version=$2 AND user_id=$3 AND knowledge_base_id=$4 "
        "ORDER BY chunk_index",
        job.document_id,
        document_version,
        job.user_id,
        job.knowledge_base_id,
    )
    return tuple((row["chunk_index"], row["content"]) for row in rows)


async def handle_document_embed(
    job: JobRecord,
    lease: JobLease,
    context: WorkerContext,
) -> Mapping[str, JSONValue]:
    await lease.checkpoint()
    document_version, profile = _validate_embedding_job_shape(job)
    if not _profile_matches(_configured_embedding_profile(), profile):
        raise TerminalJobError(
            "embedding_profile_changed",
            "The embedding profile changed before execution.",
        )
    try:
        source = await _read_embedding_source(job, document_version, context)
    except (TerminalJobError, RetryableJobError):
        raise
    except asyncpg.PostgresError:
        raise RetryableJobError(
            "embedding_storage_transient",
            "Document embeddings will be retried.",
        ) from None
    if source is None:
        return {"document_id": str(job.document_id), "stale": True}

    await lease.checkpoint()
    indexed_vectors = await _request_embedding_vectors(profile, source)
    count = await _commit_embedding_vectors(
        job,
        lease,
        context,
        document_version=document_version,
        profile=profile,
        source=source,
        indexed_vectors=indexed_vectors,
    )
    if count is None:
        return {"document_id": str(job.document_id), "stale": True}
    return {"document_id": str(job.document_id), "embedded_chunks": count}


async def _request_embedding_vectors(
    profile: EmbeddingProfile,
    source: EmbeddingSource,
) -> IndexedEmbeddings:
    texts = tuple(content for _index, content in source)
    try:
        raw_vectors = await _embed_texts(profile, texts) if texts else ()
    except EmbeddingUnavailable:
        raise RetryableJobError(
            "embedding_unavailable",
            "The embedding provider is temporarily unavailable.",
        ) from None
    except (EmbeddingInputError, InvalidEmbeddingResponse):
        raise TerminalJobError(
            "invalid_embedding_response",
            "The embedding provider returned an invalid response.",
        ) from None
    except (JobCancelled, LeaseLost):
        raise
    except Exception:  # noqa: BLE001 - provider adapters share retry semantics.
        raise RetryableJobError(
            "embedding_unavailable",
            "The embedding provider is temporarily unavailable.",
        ) from None
    vectors = _validate_embedding_vectors(
        raw_vectors,
        expected_count=len(source),
        dimensions=profile.dimensions,
    )
    return tuple((source[index][0], vector) for index, (_ignored, vector) in enumerate(vectors))


async def _commit_embedding_vectors(
    job: JobRecord,
    lease: JobLease,
    context: WorkerContext,
    *,
    document_version: int,
    profile: EmbeddingProfile,
    source: EmbeddingSource,
    indexed_vectors: IndexedEmbeddings,
) -> int | None:
    from services.vector_store import PostgresVectorStore

    try:
        async with context.pool.acquire() as conn, conn.transaction():
            await lease.checkpoint(conn)
            if not _profile_matches(_configured_embedding_profile(), profile):
                raise TerminalJobError(
                    "embedding_profile_changed",
                    "The embedding profile changed before execution.",
                )
            document = await conn.fetchrow(
                "SELECT version,status::text FROM documents "
                "WHERE id=$1 AND user_id=$2 AND knowledge_base_id=$3 "
                "AND NOT archived AND source_kind='source' FOR UPDATE",
                job.document_id,
                job.user_id,
                job.knowledge_base_id,
            )
            if document is None:
                raise TerminalJobError("document_not_found", "The requested document was not found.")
            if document["version"] != document_version:
                return None
            if document["status"] != "ready":
                raise RetryableJobError(
                    "embedding_storage_transient",
                    "Document embeddings will be retried.",
                )
            current_rows = await conn.fetch(
                "SELECT chunk_index,content FROM document_chunks "
                "WHERE document_id=$1 AND document_version=$2 AND user_id=$3 AND knowledge_base_id=$4 "
                "ORDER BY chunk_index FOR SHARE",
                job.document_id,
                document_version,
                job.user_id,
                job.knowledge_base_id,
            )
            current_source = tuple((row["chunk_index"], row["content"]) for row in current_rows)
            if current_source != source:
                return None
            if not _profile_matches(_configured_embedding_profile(), profile):
                raise TerminalJobError(
                    "embedding_profile_changed",
                    "The embedding profile changed before execution.",
                )
            return await PostgresVectorStore(
                context.pool,
                profile=profile,
            ).replace_document_embeddings_in_transaction(
                conn,
                user_id=job.user_id,
                knowledge_base_id=job.knowledge_base_id,
                document_id=job.document_id,
                document_version=document_version,
                embeddings=indexed_vectors,
            )
    except (JobCancelled, LeaseLost, TerminalJobError, RetryableJobError):
        raise
    except ValueError:
        raise TerminalJobError(
            "invalid_embedding_response",
            "The embedding provider returned an invalid response.",
        ) from None
    except (asyncpg.PostgresError, RetrieverUnavailable):
        raise RetryableJobError(
            "embedding_storage_transient",
            "Document embeddings will be retried.",
        ) from None


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
    from services.graph import rebuild_hosted

    try:
        await lease.checkpoint()
        _validate_graph_job_shape(job)
        async with context.pool.acquire() as conn, conn.transaction():
            await lease.checkpoint(conn)
            exists = await conn.fetchval(
                "SELECT EXISTS(SELECT 1 FROM knowledge_bases WHERE id = $1 AND user_id = $2)",
                job.knowledge_base_id,
                job.user_id,
            )
            if not exists:
                raise TerminalJobError(
                    "knowledge_base_not_found",
                    "The requested knowledge base was not found.",
                )

        async def final_checkpoint(conn: asyncpg.Connection) -> None:
            await lease.checkpoint(conn)

        async with context.pool.acquire() as conn:
            raw_result = await rebuild_hosted(
                conn,
                job.knowledge_base_id,
                str(job.user_id),
                before_write=final_checkpoint,
            )
    except (JobCancelled, LeaseLost, TerminalJobError):
        raise
    except (asyncpg.PostgresError, OSError, TimeoutError):
        raise RetryableJobError("graph_transient", "Graph rebuild will be retried.") from None

    expected_keys = {"citations", "links", "facet_rollups"}
    if (
        not isinstance(raw_result, Mapping)
        or set(raw_result) != expected_keys
        or any(
            isinstance(raw_result[key], bool) or not isinstance(raw_result[key], int) or raw_result[key] < 0
            for key in expected_keys
        )
    ):
        raise TerminalJobError("invalid_job_result", "The job produced an invalid result.")
    return {key: raw_result[key] for key in ("citations", "links", "facet_rollups")}


def _validate_graph_job_shape(job: JobRecord) -> None:
    if job.job_type is not JobType.GRAPH_REBUILD or job.knowledge_base_id is None or job.document_id is not None:
        raise TerminalJobError("invalid_graph_job", "The graph rebuild job is invalid.")
    payload_kb_id = job.payload.get("knowledge_base_id")
    try:
        parsed_payload_kb_id = UUID(payload_kb_id) if isinstance(payload_kb_id, str) else None
    except ValueError:
        parsed_payload_kb_id = None
    if parsed_payload_kb_id != job.knowledge_base_id or set(job.payload) != {"knowledge_base_id"}:
        raise TerminalJobError("invalid_graph_job", "The graph rebuild job is invalid.")


async def handle_upload_cleanup(
    job: JobRecord,
    lease: JobLease,
    context: WorkerContext,
) -> Mapping[str, JSONValue]:
    await lease.checkpoint()
    if (
        job.job_type is not JobType.UPLOAD_CLEANUP
        or job.document_id is not None
        or job.knowledge_base_id is not None
        or set(job.payload) != {"upload_id"}
    ):
        raise TerminalJobError("invalid_upload_job", "The upload cleanup job is invalid.")
    raw_upload_id = job.payload.get("upload_id")
    try:
        upload_id = UUID(raw_upload_id) if isinstance(raw_upload_id, str) else None
    except ValueError:
        upload_id = None
    if upload_id is None or str(upload_id) != raw_upload_id:
        raise TerminalJobError("invalid_upload_job", "The upload cleanup job is invalid.")
    if context.tus_cleanup is None:
        raise RetryableJobError("upload_cleanup_transient", "Upload cleanup will be retried.")
    try:
        result = await context.tus_cleanup.cleanup(upload_id, job.user_id)
    except Exception:  # noqa: BLE001 -- cleanup adapters use retryable job semantics
        raise RetryableJobError("upload_cleanup_transient", "Upload cleanup will be retried.") from None
    status = result.get("status") if isinstance(result, Mapping) else None
    if status in {"contended", "retry", "lock_lost"}:
        raise RetryableJobError("upload_cleanup_transient", "Upload cleanup will be retried.")
    if status == "owner_mismatch":
        raise TerminalJobError("invalid_upload_job", "The upload cleanup job is invalid.")
    if status not in {"cleaned", "already_clean", "committed", "active"}:
        raise RetryableJobError("upload_cleanup_transient", "Upload cleanup will be retried.")
    return {"upload_id": str(upload_id), "status": status}


HANDLERS: Mapping[JobType, Handler] = MappingProxyType(
    {
        JobType.DOCUMENT_EXTRACT: handle_document_extract,
        JobType.DOCUMENT_EMBED: handle_document_embed,
        JobType.GRAPH_REBUILD: handle_graph_rebuild,
        JobType.UPLOAD_CLEANUP: handle_upload_cleanup,
    }
)
