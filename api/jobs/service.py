"""Application-scoped background job operations."""

from __future__ import annotations

from uuid import UUID

import asyncpg

from jobs import repository
from jobs.models import JobCreate, JobRecord, JobState, JobType


class JobResourceNotFound(LookupError):
    """A referenced job resource is missing, not owned, or incompatible."""


class JobService:
    """Owns short database transactions for authenticated job operations."""

    def __init__(self, pool: asyncpg.Pool):
        self._pool = pool

    async def create(self, command: JobCreate, *, authenticated_user_id: UUID) -> JobRecord:
        async with self._pool.acquire() as conn, conn.transaction():
            return await self.create_in_transaction(
                conn,
                command,
                authenticated_user_id=authenticated_user_id,
            )

    async def create_in_transaction(
        self,
        conn: asyncpg.Connection,
        command: JobCreate,
        *,
        authenticated_user_id: UUID,
    ) -> JobRecord:
        """Create a job inside a caller-owned business transaction."""
        if not conn.is_in_transaction():
            raise RuntimeError("job creation connection must be in an explicit transaction")
        if command.user_id != authenticated_user_id:
            raise ValueError("command user does not match the authenticated user")
        await self._validate_resources(conn, command, authenticated_user_id)
        return await repository.create(conn, command)

    async def ensure_graph_rebuild(
        self,
        *,
        knowledge_base_id: UUID,
        authenticated_user_id: UUID,
    ) -> JobRecord:
        """Create a graph rebuild or return the database-selected active winner."""
        command = JobCreate(
            job_type=JobType.GRAPH_REBUILD,
            user_id=authenticated_user_id,
            knowledge_base_id=knowledge_base_id,
            payload={"knowledge_base_id": str(knowledge_base_id)},
        )
        async with self._pool.acquire() as conn, conn.transaction():
            await self._validate_resources(conn, command, authenticated_user_id)
            # A conflicting row can become terminal immediately after INSERT
            # observes it. Each new statement gets a fresh READ COMMITTED
            # snapshot, so retry only when no active winner remains visible.
            for _ in range(3):
                created = await repository.create_active_graph(conn, command)
                if created is not None:
                    return created
                active = await repository.get_active_graph(
                    conn,
                    authenticated_user_id,
                    knowledge_base_id,
                )
                if active is not None:
                    return active
        raise RuntimeError("active graph job changed too frequently")

    async def ensure_document_extraction_in_transaction(
        self,
        conn: asyncpg.Connection,
        *,
        document_id: UUID,
        user_id: UUID,
        knowledge_base_id: UUID,
        restart_terminal: bool,
    ) -> tuple[JobRecord, bool]:
        """Return the authoritative extraction job or append one immutable successor."""
        if not conn.is_in_transaction():
            raise RuntimeError("extraction job ensure requires an explicit transaction")
        latest = await conn.fetchrow(
            "SELECT id, state::text FROM background_jobs "
            "WHERE user_id = $1 AND job_type = 'document.extract' AND document_id = $2 "
            "ORDER BY created_at DESC, id DESC LIMIT 1 FOR UPDATE",
            user_id,
            document_id,
        )
        document = await conn.fetchrow(
            "SELECT status::text, version FROM documents "
            "WHERE id = $1 AND user_id = $2 AND knowledge_base_id = $3 "
            "AND NOT archived AND source_kind = 'source' FOR UPDATE",
            document_id,
            user_id,
            knowledge_base_id,
        )
        if document is None:
            raise JobResourceNotFound("referenced extraction document was not found")

        # The initial locking statement may have waited with an older READ COMMITTED
        # snapshot.  Re-read after the document lock without taking a job lock, so
        # newly committed successors are authoritative without reversing job -> doc.
        latest = await conn.fetchrow(
            "SELECT id, state::text FROM background_jobs "
            "WHERE user_id = $1 AND job_type = 'document.extract' AND document_id = $2 "
            "ORDER BY created_at DESC, id DESC LIMIT 1",
            user_id,
            document_id,
        )
        if latest is not None:
            state = JobState(latest["state"])
            if not state.is_terminal or (state is JobState.SUCCEEDED and document["status"] == "ready"):
                record = await repository.get_for_user(conn, latest["id"], user_id)
                if record is None:
                    raise RuntimeError("authoritative extraction job disappeared")
                return record, False
            if document["status"] == "failed" and not restart_terminal:
                record = await repository.get_for_user(conn, latest["id"], user_id)
                if record is None:
                    raise RuntimeError("authoritative extraction job disappeared")
                return record, False

        idempotency_key = f"document.extract:{document_id}"
        if latest is not None:
            idempotency_key = f"{idempotency_key}:after:{latest['id']}"
            if restart_terminal:
                await conn.execute(
                    "UPDATE documents SET "
                    "status = CASE WHEN status = 'ready' AND version > 0 THEN status ELSE 'pending' END, "
                    "error_message = NULL, updated_at = now() "
                    "WHERE id = $1 AND user_id = $2 AND knowledge_base_id = $3",
                    document_id,
                    user_id,
                    knowledge_base_id,
                )
        record = await self.create_in_transaction(
            conn,
            JobCreate(
                job_type=JobType.DOCUMENT_EXTRACT,
                user_id=user_id,
                knowledge_base_id=knowledge_base_id,
                document_id=document_id,
                payload={"document_id": str(document_id)},
                idempotency_key=idempotency_key,
            ),
            authenticated_user_id=user_id,
        )
        return record, True

    async def get(self, job_id: UUID, *, authenticated_user_id: UUID) -> JobRecord | None:
        async with self._pool.acquire() as conn, conn.transaction():
            return await repository.get_for_user(conn, job_id, authenticated_user_id)

    async def cancel(self, job_id: UUID, *, authenticated_user_id: UUID) -> JobRecord | None:
        async with self._pool.acquire() as conn, conn.transaction():
            cancelled = await repository.request_cancel(conn, job_id, authenticated_user_id)

        if (
            cancelled is not None
            and cancelled.job_type.value == "document.extract"
            and cancelled.document_id is not None
        ):
            # Deliberately use a second transaction so cancellation never holds
            # the job and document locks across the status propagation boundary.
            async with self._pool.acquire() as conn, conn.transaction():
                await conn.execute(
                    "UPDATE documents SET "
                    "status = CASE WHEN status = 'ready' AND version > 0 THEN status ELSE 'failed' END, "
                    "error_message = CASE WHEN status = 'ready' AND version > 0 "
                    "THEN NULL ELSE 'Document extraction was cancelled.' END, updated_at = now() "
                    "WHERE id = $1 AND user_id = $2 AND knowledge_base_id = $3 "
                    "AND NOT archived AND source_kind = 'source'",
                    cancelled.document_id,
                    authenticated_user_id,
                    cancelled.knowledge_base_id,
                )
        return cancelled

    @staticmethod
    async def _validate_resources(
        conn: asyncpg.Connection,
        command: JobCreate,
        authenticated_user_id: UUID,
    ) -> None:
        if command.knowledge_base_id is not None:
            knowledge_base_exists = await conn.fetchval(
                "SELECT EXISTS(SELECT 1 FROM knowledge_bases WHERE id = $1 AND user_id = $2)",
                command.knowledge_base_id,
                authenticated_user_id,
            )
            if not knowledge_base_exists:
                raise JobResourceNotFound("referenced job resource was not found")

        if command.document_id is None:
            return

        if command.knowledge_base_id is None:
            document_exists = await conn.fetchval(
                "SELECT EXISTS("
                "SELECT 1 FROM documents "
                "JOIN knowledge_bases ON knowledge_bases.id = documents.knowledge_base_id "
                "WHERE documents.id = $1 "
                "AND documents.user_id = $2 "
                "AND knowledge_bases.user_id = $2"
                ")",
                command.document_id,
                authenticated_user_id,
            )
        else:
            document_exists = await conn.fetchval(
                "SELECT EXISTS(SELECT 1 FROM documents WHERE id = $1 AND user_id = $2 AND knowledge_base_id = $3)",
                command.document_id,
                authenticated_user_id,
                command.knowledge_base_id,
            )
        if not document_exists:
            raise JobResourceNotFound("referenced job resource was not found")
