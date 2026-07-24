"""Application-scoped background job operations."""

from __future__ import annotations

from uuid import UUID

import asyncpg

from jobs import repository
from jobs.models import JobCreate, JobRecord


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
            # Deliberately use a second transaction: final publication locks the
            # document before the job row, while cancellation locks the job first.
            # Releasing the job lock here prevents a job/document lock inversion.
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
