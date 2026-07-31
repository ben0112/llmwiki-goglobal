"""Lease-fenced atomic publication for durable RAG wiki pages."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date
from uuid import UUID

import asyncpg
from jobs import repository as jobs_repository

from llmwiki_adapters.postgres.wiki import WikiWriteResult, write_wiki_bundle_in_transaction
from llmwiki_core.chunking import chunk_text
from llmwiki_core.facets import apply_rollup, rollup_from_metas
from llmwiki_core.rag import RagDomainError, RagUsage
from llmwiki_core.wiki import WikiWriteBundle

from . import repository
from .ports import AtomicPageCommit
from .records import RagPageRecord, RagRunRecord


class PersistedRagLintError(RagDomainError):
    """The transaction-local wiki projection does not agree with its persisted facts."""

    def __init__(self) -> None:
        super().__init__("rag_persisted_lint_failed", "The persisted wiki page failed validation.")


class InjectedCrash(RuntimeError):
    """Test-only crash-window signal."""


@dataclass(frozen=True, slots=True)
class PostgresPersistedRagLinter:
    """Validate the just-written document, chunks, references, and citation facets."""

    async def lint_persisted_in_transaction(  # noqa: C901 - cross-checks coupled persisted projections.
        self,
        conn: asyncpg.Connection,
        *,
        run: RagRunRecord,
        page: RagPageRecord,
        bundle: WikiWriteBundle,
        written: WikiWriteResult,
    ) -> Mapping[str, object]:
        if not conn.is_in_transaction():
            raise RuntimeError("persisted RAG lint requires an explicit transaction")
        if (
            type(run) is not RagRunRecord
            or type(page) is not RagPageRecord
            or type(bundle) is not WikiWriteBundle
            or type(written) is not WikiWriteResult
        ):
            raise PersistedRagLintError()
        if page.run_id != run.id or page.user_id != run.user_id or page.knowledge_base_id != run.knowledge_base_id:
            raise PersistedRagLintError()
        document = await conn.fetchrow(
            "SELECT id,version,path,filename,file_type,content,title,tags,date,metadata,source_kind,status,archived "
            "FROM documents WHERE id=$1 AND user_id=$2 AND knowledge_base_id=$3 FOR UPDATE",
            written.document_id,
            run.user_id,
            run.knowledge_base_id,
        )
        if document is None:
            raise PersistedRagLintError()
        raw_date = document["date"]
        if raw_date is None or type(raw_date) is str:
            persisted_date = raw_date
        elif type(raw_date) is date:
            persisted_date = raw_date.isoformat()
        else:
            raise PersistedRagLintError()
        if (
            document["id"] != written.document_id
            or str(document["id"]) != bundle.document_id
            or document["version"] != written.version
            or written.version != (1 if bundle.expected_version is None else bundle.expected_version + 1)
            or document["path"] != written.path
            or document["path"] != bundle.path
            or document["filename"] != written.filename
            or document["filename"] != bundle.filename
            or f"{document['path']}{document['filename']}" != page.path
            or document["file_type"] != bundle.file_type
            or document["content"] != bundle.content
            or document["title"] != bundle.title
            or tuple(document["tags"] or ()) != bundle.tags
            or persisted_date != bundle.date
            or document["source_kind"] != "wiki"
            or document["status"] not in {"pending", "processing", "ready"}
            or document["archived"] is not False
        ):
            raise PersistedRagLintError()

        expected_chunks = chunk_text(bundle.content)
        rows = await conn.fetch(
            "SELECT document_version,chunk_index,content,page,start_char,token_count,header_breadcrumb "
            "FROM document_chunks WHERE document_id=$1 AND user_id=$2 AND knowledge_base_id=$3 "
            "ORDER BY chunk_index",
            written.document_id,
            run.user_id,
            run.knowledge_base_id,
        )
        if len(rows) != len(expected_chunks):
            raise PersistedRagLintError()
        for row, expected in zip(rows, expected_chunks, strict=True):
            if (
                row["document_version"] != written.version
                or row["chunk_index"] != expected.index
                or row["content"] != expected.content
                or row["page"] != expected.page
                or row["start_char"] != expected.start_char
                or row["token_count"] != expected.token_count
                or row["header_breadcrumb"] != expected.header_breadcrumb
            ):
                raise PersistedRagLintError()

        references = await conn.fetch(
            "SELECT r.target_document_id,r.reference_type,r.page,target.path,target.metadata "
            "FROM document_references r "
            "JOIN documents source ON source.id=r.source_document_id "
            "JOIN documents target ON target.id=r.target_document_id "
            "WHERE r.source_document_id=$1 AND r.knowledge_base_id=$2 "
            "AND r.reference_type=ANY($3::text[]) "
            "AND source.user_id=$4 AND source.knowledge_base_id=$2 "
            "AND target.user_id=$4 AND target.knowledge_base_id=$2 "
            "ORDER BY r.reference_type,r.target_document_id,r.page NULLS FIRST",
            written.document_id,
            run.knowledge_base_id,
            ["cites", "links_to"],
            run.user_id,
        )
        expected_references = sorted(
            (
                UUID(edge.target_id),
                edge.reference_type,
                edge.page,
            )
            for edge in bundle.edges
        )
        persisted_references = sorted(
            (
                UUID(str(reference["target_document_id"])),
                reference["reference_type"],
                reference["page"],
            )
            for reference in references
        )
        if persisted_references != expected_references:
            raise PersistedRagLintError()
        citation_metas: list[dict[str, object]] = []
        citation_count = 0
        for reference in references:
            if reference["reference_type"] == "cites":
                citation_count += 1
                if reference["path"].startswith("/corpus/"):
                    raw = reference["metadata"]
                    try:
                        parsed = json.loads(raw) if isinstance(raw, str) else (raw or {})
                    except (TypeError, ValueError):
                        raise PersistedRagLintError() from None
                    if type(parsed) is not dict:
                        raise PersistedRagLintError()
                    citation_metas.append(parsed)

        raw_metadata = document["metadata"]
        try:
            metadata = json.loads(raw_metadata) if isinstance(raw_metadata, str) else dict(raw_metadata or {})
        except (TypeError, ValueError):
            raise PersistedRagLintError() from None
        expected_metadata = dict(bundle.metadata)
        apply_rollup(expected_metadata, rollup_from_metas(citation_metas, date.today().isoformat()))
        if metadata != expected_metadata:
            raise PersistedRagLintError()

        return {
            "citation_count": citation_count,
            "chunk_count": len(rows),
            "document_version": written.version,
            "facets_verified": True,
            "reference_count": len(references),
        }


class PostgresRagWikiWriter:
    """Publish a page, lint persisted facts, and advance its boundary atomically."""

    def __init__(
        self,
        pool,
        *,
        linter: PostgresPersistedRagLinter | None = None,
        failpoint: str | None = None,
    ) -> None:
        self._pool = pool
        self._linter = linter or PostgresPersistedRagLinter()
        self.failpoint = failpoint

    async def commit(
        self,
        *,
        job_id: UUID,
        lease_owner: str,
        run: RagRunRecord,
        page: RagPageRecord,
        bundle: WikiWriteBundle,
        usage: RagUsage,
    ) -> AtomicPageCommit:
        result: AtomicPageCommit
        async with self._pool.acquire() as conn, conn.transaction():
            job = await jobs_repository.assert_active(conn, job_id, lease_owner)
            await repository.assert_page_job_binding(
                conn,
                job=job,
                run=run,
                page=page,
            )
            written = await write_wiki_bundle_in_transaction(
                conn,
                user_id=run.user_id,
                knowledge_base_id=run.knowledge_base_id,
                bundle=bundle,
            )
            # asyncpg exposes its own UUID scalar subclass; normalize the
            # adapter result before crossing Task 9's exact-type boundary.
            written = WikiWriteResult(
                document_id=UUID(str(written.document_id)),
                filename=written.filename,
                path=written.path,
                version=written.version,
            )
            if self.failpoint == "after_shared_writer_before_boundary":
                raise InjectedCrash("injected crash before RAG boundary")
            lint_summary = await self._linter.lint_persisted_in_transaction(
                conn,
                run=run,
                page=page,
                bundle=bundle,
                written=written,
            )
            updated_run, updated_page = await repository.mark_boundary(
                conn,
                run=run,
                page=page,
                document_id=written.document_id,
                committed_version=written.version,
                usage=usage,
                lint_summary=lint_summary,
            )
            result = AtomicPageCommit(updated_run, updated_page, written, lint_summary)
        if self.failpoint == "after_transaction_return":
            raise InjectedCrash("injected crash after atomic RAG transaction")
        return result


__all__ = [
    "AtomicPageCommit",
    "InjectedCrash",
    "PersistedRagLintError",
    "PostgresPersistedRagLinter",
    "PostgresRagWikiWriter",
]
