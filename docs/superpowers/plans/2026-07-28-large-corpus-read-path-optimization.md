# Large Corpus Read-Path Optimization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace Web full-workspace document loading with revision-aware, indexed, keyset-paginated read models that remain bounded for 75,000-document Local and Hosted knowledge bases.

**Architecture:** Keep the legacy document-array endpoint compatible while adding a shared cursor contract, database-backed knowledge-base revisions, Local SQLite and Hosted Postgres read adapters, and view-specific routes. The Web migrates files, Wiki, corpus, upload support, sidebar search, and document resolution to bounded pages and lookups; no main-screen path retains the legacy full-array hook.

**Tech Stack:** Python 3.11, FastAPI, Pydantic, aiosqlite/SQLite, asyncpg/Postgres, React 19, Next.js 16, TypeScript 5.9, Vitest, pytest

---

## File structure

- `llmwiki_core/read_cursor.py`: dependency-free, versioned keyset cursor value object and Base64URL codec.
- `api/services/read_models.py`: API read-model enums, Pydantic response/request values, ETag helpers, and service protocol.
- `api/services/read_local.py`: SQLite browse, Wiki, corpus, summary, lookup, status, preflight, and revision implementation.
- `api/services/read_hosted.py`: Postgres implementation with explicit user/KB scope.
- `api/routes/read_models.py`: additive HTTP routes and uniform 304/409/422 behavior.
- `shared/sqlite_read_models.sql`: Local indexes and revision triggers, applied after old-schema column backfill.
- `supabase/migrations/016_read_models.sql`: Hosted revision column, triggers, RLS-compatible indexes, and grants.
- `web/src/lib/read-models.ts`: transport types, query encoding, cursor/page cache reducer, and manifest chunking.
- `web/src/hooks/useReadPage.ts`: abortable ETag-aware page loader with stale-cursor recovery.
- `web/src/hooks/useDocumentBrowse.ts`, `useWikiPages.ts`, `useCorpusEntries.ts`: view-specific hooks.
- `web/src/components/kb/DocumentSearchDialog.tsx`: remote sidebar search with bounded results.
- `web/src/components/kb/KBDetail.tsx`, `FilesGrid.tsx`, `WikiOnlyDetail.tsx`, `KBSidenav.tsx`: remove full document-array dependencies.
- `web/src/components/wiki/WikiContent.tsx`: resolve citations and assets on demand.
- `web/src/components/corpus/CorpusView.tsx`: server-side summary and page window.
- `scripts/benchmark_read_models.py`: reproducible synthetic and copied-live-database acceptance probe.

## Task 1: Shared cursor and transport contracts

**Files:**
- Create: `llmwiki_core/read_cursor.py`
- Create: `tests/unit/core/test_read_cursor.py`
- Create: `api/services/read_models.py`
- Modify: `tests/unit/core/test_import_boundaries.py`

- [ ] **Step 1: Write failing cursor tests**

```python
from dataclasses import replace

import pytest

from llmwiki_core.read_cursor import CursorError, ReadCursor, decode_cursor, encode_cursor


def test_cursor_round_trip_preserves_bound_scope_and_key():
    cursor = ReadCursor(
        scope="documents.browse",
        revision=9,
        sort="name",
        direction="asc",
        key=("report.pdf", "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"),
    )
    assert decode_cursor(encode_cursor(cursor), expected_scope="documents.browse") == cursor


@pytest.mark.parametrize("encoded", ["", "%%%", "e30", "W10"])
def test_cursor_rejects_malformed_payload(encoded):
    with pytest.raises(CursorError):
        decode_cursor(encoded, expected_scope="documents.browse")


def test_cursor_rejects_cross_endpoint_reuse():
    encoded = encode_cursor(ReadCursor("wiki.pages", 1, "path", "asc", ("a", "id")))
    with pytest.raises(CursorError, match="scope"):
        decode_cursor(encoded, expected_scope="documents.browse")


def test_cursor_rejects_unsupported_version():
    current = ReadCursor("documents.browse", 1, "name", "asc", ("a", "id"))
    with pytest.raises(CursorError, match="version"):
        encode_cursor(replace(current, version=2))
```

- [ ] **Step 2: Run the cursor tests and verify RED**

Run: `pytest tests/unit/core/test_read_cursor.py -q`

Expected: collection fails with `ModuleNotFoundError: No module named 'llmwiki_core.read_cursor'`.

- [ ] **Step 3: Implement the dependency-free cursor codec**

Implement an immutable `ReadCursor` with fields `scope`, `revision`, `sort`, `direction`, `key`, and `version=1`. Encode canonical compact JSON with `urlsafe_b64encode` and no padding. Decode with a 2048-character input limit, strict JSON object keys, positive integer revision, `asc|desc`, non-empty string tuple members, expected-scope equality, and version equality. Convert every decoding or validation failure to `CursorError` without exposing decoder internals.

```python
@dataclass(frozen=True, slots=True)
class ReadCursor:
    scope: str
    revision: int
    sort: str
    direction: str
    key: tuple[str, ...]
    version: int = 1


def encode_cursor(cursor: ReadCursor) -> str:
    payload = {"v": cursor.version, "s": cursor.scope, "r": cursor.revision,
               "o": cursor.sort, "d": cursor.direction, "k": list(cursor.key)}
    raw = json.dumps(payload, ensure_ascii=True, separators=(",", ":"), sort_keys=True).encode()
    return base64.urlsafe_b64encode(raw).decode().rstrip("=")
```

In `api/services/read_models.py`, define closed `ReadSort`, `SortDirection`, and response models `ReadPage`, `BrowsePage`, `FolderItem`, `CorpusSummary`, `GraphSummary`, `DocumentStatusPage`, `ResolvedDocument`, `UploadPreflightItem`, `UploadPreflightRequest`, and `UploadPreflightResponse`. Every item list uses `Field(max_length=200)`. Add `etag_for_revision(revision)` returning `"kb-read-<revision>"` including quotes and `etag_matches(if_none_match, revision)` using exact token matching.

- [ ] **Step 4: Verify GREEN and core import boundaries**

Run: `pytest tests/unit/core/test_read_cursor.py tests/unit/core/test_import_boundaries.py -q`

Expected: all tests pass; `llmwiki_core` still imports no FastAPI, Pydantic, database, or storage package.

- [ ] **Step 5: Commit**

```bash
git add llmwiki_core/read_cursor.py api/services/read_models.py tests/unit/core/test_read_cursor.py tests/unit/core/test_import_boundaries.py
git commit -m "feat: define bounded read model contracts"
```

## Task 2: Database read revisions and indexes

**Files:**
- Create: `shared/sqlite_read_models.sql`
- Create: `supabase/migrations/016_read_models.sql`
- Create: `tests/unit/test_read_revision.py`
- Create: `tests/integration/test_read_model_schema.py`
- Modify: `shared/sqlite_schema.sql`
- Modify: `api/infra/db/sqlite.py`
- Modify: `mcp/vaultfs/sqlite.py`
- Modify: `tests/helpers/schema.sql`
- Modify: `tests/unit/core/test_schema_invariants.py`

- [ ] **Step 1: Write failing Local revision tests**

Create a file-backed SQLite database through `api.infra.db.sqlite.create_pool`, seed one workspace, and assert:

```python
async def _revision(db):
    row = await (await db.execute("SELECT read_revision FROM workspace WHERE id='ws1'")).fetchone()
    return row[0]


@pytest.mark.asyncio
async def test_document_insert_update_delete_each_bumps_read_revision(tmp_path):
    db = await create_pool(str(tmp_path / "index.db"))
    await db.execute("INSERT INTO workspace (id,name,user_id) VALUES ('ws1','test','u1')")
    await db.commit()
    initial = await _revision(db)
    await db.execute("INSERT INTO documents (id,user_id,filename,path,relative_path,source_kind,file_type) VALUES ('d1','u1','a.md','/','a.md','source','md')")
    await db.commit()
    assert await _revision(db) == initial + 1
    await db.execute("UPDATE documents SET title='A' WHERE id='d1'")
    await db.commit()
    assert await _revision(db) == initial + 2
    await db.execute("DELETE FROM documents WHERE id='d1'")
    await db.commit()
    assert await _revision(db) == initial + 3
```

Add an old-schema fixture with no `read_revision`, open it through both API and MCP initializers, and assert idempotent backfill plus trigger creation.

- [ ] **Step 2: Run Local revision tests and verify RED**

Run: `pytest tests/unit/test_read_revision.py -q`

Expected: failure because `workspace.read_revision` and revision triggers do not exist.

- [ ] **Step 3: Implement Local schema migration and indexes**

Add `read_revision INTEGER NOT NULL DEFAULT 1 CHECK (read_revision > 0)` to new `workspace` tables. In both SQLite initializers, inspect `PRAGMA table_info(workspace)`, add the column for old databases, then execute `shared/sqlite_read_models.sql`.

The shared SQL creates three `AFTER` triggers that run `UPDATE workspace SET read_revision = read_revision + 1`, plus these bounded-query indexes:

```sql
CREATE INDEX IF NOT EXISTS idx_documents_browse_name
  ON documents(path, filename COLLATE NOCASE, id);
CREATE INDEX IF NOT EXISTS idx_documents_browse_date
  ON documents(path, updated_at, id);
CREATE INDEX IF NOT EXISTS idx_documents_wiki_path
  ON documents(source_kind, path, filename, id);
CREATE INDEX IF NOT EXISTS idx_documents_number
  ON documents(document_number, id);
CREATE INDEX IF NOT EXISTS idx_documents_content_hash
  ON documents(content_hash) WHERE content_hash IS NOT NULL AND status != 'failed';
```

- [ ] **Step 4: Write and verify failing Hosted migration tests**

Test that migration 016 adds `knowledge_bases.read_revision`, bumps only the affected knowledge base on document insert/update/delete, bumps both old and new knowledge bases when ownership moves, and creates tenant-first partial indexes. Run `pytest tests/integration/test_read_model_schema.py -q`; expect missing-column failures.

- [ ] **Step 5: Implement Hosted migration**

Create a `SECURITY DEFINER` trigger function with `SET search_path = public, pg_temp`. For `UPDATE`, bump `OLD.knowledge_base_id`, and also bump `NEW.knowledge_base_id` when it differs. For insert/delete, bump the corresponding row. Revoke direct execution from public roles. Add partial indexes beginning with `(knowledge_base_id, user_id, ...)` and `WHERE NOT archived` for browse name/date, Wiki scope, corpus scope, and document number. Hosted has no `content_hash` column, so hash indexing and authoritative hash deduplication remain Local-only. Mirror the resulting schema into `tests/helpers/schema.sql`.

- [ ] **Step 6: Verify schema GREEN**

Run: `pytest tests/unit/test_read_revision.py tests/unit/core/test_schema_invariants.py tests/integration/test_read_model_schema.py -q`

Expected: all tests pass, including migration idempotence.

- [ ] **Step 7: Commit**

```bash
git add shared/sqlite_schema.sql shared/sqlite_read_models.sql api/infra/db/sqlite.py mcp/vaultfs/sqlite.py supabase/migrations/016_read_models.sql tests/helpers/schema.sql tests/unit/test_read_revision.py tests/unit/core/test_schema_invariants.py tests/integration/test_read_model_schema.py
git commit -m "feat: add durable read revisions and indexes"
```

## Task 3: Browse, Wiki, lookup, status, and upload-preflight backend

**Files:**
- Create: `api/services/read_local.py`
- Create: `api/services/read_hosted.py`
- Create: `api/routes/read_models.py`
- Create: `tests/unit/test_local_read_models.py`
- Create: `tests/unit/test_read_model_routes.py`
- Create: `tests/integration/test_hosted_read_models.py`
- Create: `tests/integration/isolation/test_read_model_api_isolation.py`
- Modify: `api/services/base.py`
- Modify: `api/services/local.py`
- Modify: `api/services/hosted.py`
- Modify: `api/routes/local_upload.py`
- Modify: `api/deps.py`
- Modify: `api/main.py`
- Modify: `tests/unit/test_resumable_upload.py`

- [ ] **Step 1: Write failing SQLite adapter tests**

Seed documents with duplicate filenames, mixed case, equal timestamps, Wiki paths, failed rows, content hashes, and two folders. Assert stable keyset pages of 2, folder derivation, global search refusal for an empty query, Wiki-only output, document-number resolution, logical-reference resolution, status order preservation, and preflight decisions.

```python
page1 = await service.browse("ws1", path="/", query=None, sort="name", direction="asc", limit=2, cursor=None)
page2 = await service.browse("ws1", path="/", query=None, sort="name", direction="asc", limit=2, cursor=page1.next_cursor)
assert [item["id"] for item in page1.items + page2.items] == ["d1", "d2", "d3", "d4"]
assert len({item["id"] for item in page1.items + page2.items}) == 4
assert all("content" not in item for item in page1.items)
```

- [ ] **Step 2: Run adapter tests and verify RED**

Run: `pytest tests/unit/test_local_read_models.py -q`

Expected: import failure for `services.read_local`.

- [ ] **Step 3: Implement the service interface and SQLite adapter**

Add `read_service(user_id)` to `ServiceFactory`. The Local implementation validates the singleton workspace id and user id, reads revision before decoding any cursor, compares cursor revision, and uses static SQL selected from closed sort enums. Fetch `limit + 1`, return at most `limit`, and create the next cursor from the last returned row. Select only list fields from `_DOC_COLUMNS`, excluding hashes except for direct preflight queries.

`upload_preflight` performs two bounded indexed lookups: target `(path, lower(filename))` pairs and supplied non-null hashes. It returns decisions in manifest order. It never trusts the preflight as write authorization.

Keep the Local final write authoritative: inside the existing serialized write
span, reject an existing `relative_path` or an existing non-failed
`content_hash` before replacing the destination file or inserting the document.
Return `409` with a stable duplicate code and leave the existing file and row
unchanged. Add regression coverage for direct, resumable, and preflight/write-race
paths. Hosted keeps its existing database uniqueness and tenant checks; optional
client hashes are advisory because Hosted documents do not persist `content_hash`.

- [ ] **Step 4: Write failing route and isolation tests**

Route tests assert `304` short-circuit before adapter `browse`, `409` with `{"detail":{"code":"stale_cursor","revision":N}}`, `422` for malformed cursor or 201 descriptors, and quoted ETag headers. Hosted isolation tests authenticate Alice against Bob's KB for every new endpoint and assert 404 or empty indistinguishable results without Bob identifiers. The Hosted adapter contract test seeds equal sort values and more than one page, then applies the same stable-order, no-duplicate, limit, stale-cursor, resolver, status, and preflight assertions used by the Local adapter tests.

Run: `pytest tests/unit/test_read_model_routes.py tests/integration/test_hosted_read_models.py tests/integration/isolation/test_read_model_api_isolation.py -q`

Expected: 404 for missing routes.

- [ ] **Step 5: Implement Hosted adapter and HTTP routes**

The Hosted adapter uses tenant-first SQL on the authenticated connection and repeats both `knowledge_base_id` and `user_id` conditions even where RLS also applies. Register the additive router for both modes. Routes read revision, return `304` before target data methods, translate `CursorError` to `422 invalid_cursor`, and translate adapter stale-revision errors to `409 stale_cursor`.

- [ ] **Step 6: Verify backend GREEN and legacy compatibility**

Run: `pytest tests/unit/test_local_read_models.py tests/unit/test_read_model_routes.py tests/integration/test_hosted_read_models.py tests/integration/isolation/test_read_model_api_isolation.py tests/integration/test_note_lifecycle.py tests/integration/isolation/test_api_isolation.py -q`

Expected: all pass; legacy `/documents` tests remain unchanged.

- [ ] **Step 7: Commit**

```bash
git add api/services/read_local.py api/services/read_hosted.py api/services/read_models.py api/routes/read_models.py api/services/base.py api/services/local.py api/services/hosted.py api/routes/local_upload.py api/deps.py api/main.py tests/unit/test_local_read_models.py tests/unit/test_read_model_routes.py tests/unit/test_resumable_upload.py tests/integration/test_hosted_read_models.py tests/integration/isolation/test_read_model_api_isolation.py
git commit -m "feat: add bounded document read APIs"
```

## Task 4: Corpus entries, corpus summary, and graph summary backend

**Files:**
- Create: `tests/unit/test_local_corpus_read_models.py`
- Create: `tests/integration/isolation/test_corpus_read_model_isolation.py`
- Modify: `api/services/read_local.py`
- Modify: `api/services/read_hosted.py`
- Modify: `api/services/read_models.py`
- Modify: `api/routes/read_models.py`

- [ ] **Step 1: Write failing corpus and graph tests**

Seed valid, malformed, pending-review, expired, multi-facet, cited, and uncited entries. Assert entries are paginated, facet counts apply all filters except their own facet, coverage and business summaries match existing `web/src/lib/corpus` semantics, malformed metadata is skipped without 500, and graph summary excludes cross-KB references.

```python
summary = await service.corpus_summary("ws1", {"stage": "S2", "state": "待复核"})
assert summary.filtered_count == 2
assert summary.facets["stage"]["S1"] == 1
assert summary.facets["stage"]["S2"] == 2
assert summary.kpis["pending_review"] == 2
```

- [ ] **Step 2: Run tests and verify RED**

Run: `pytest tests/unit/test_local_corpus_read_models.py tests/integration/isolation/test_corpus_read_model_isolation.py -q`

Expected: missing service methods or 404 routes.

- [ ] **Step 3: Implement bounded corpus entries and summaries**

Reuse `llmwiki_core.facets.validate_facets` and the current corpus metadata parser. Keep SQL parameterized and static. Cache only `CorpusSummary` values in a 256-entry `OrderedDict` keyed by `(user_id, kb_id, normalized_filters, revision)`; check database revision before every hit. The Postgres version operates on the RLS-scoped connection. The Local version tolerates invalid JSON using `json_valid` guards.

- [ ] **Step 4: Implement graph summary and routes**

Return fixed scalar counts and cited target ids required by corpus KPI calculations. Do not call `get_graph_local` or `get_graph_hosted`; query `document_references` directly under KB/user scope. Add ETag handling identical to other read routes.

- [ ] **Step 5: Verify GREEN and cache invalidation**

Run: `pytest tests/unit/test_local_corpus_read_models.py tests/integration/isolation/test_corpus_read_model_isolation.py tests/unit/test_graph_local.py -q`

Expected: all pass; a document mutation invalidates prior-revision cache entries.

- [ ] **Step 6: Commit**

```bash
git add api/services/read_local.py api/services/read_hosted.py api/services/read_models.py api/routes/read_models.py tests/unit/test_local_corpus_read_models.py tests/integration/isolation/test_corpus_read_model_isolation.py
git commit -m "feat: add bounded corpus and graph summaries"
```

## Task 5: Frontend page cache, ETag loader, and manifest batching

**Files:**
- Create: `web/vitest.config.ts`
- Create: `web/src/lib/read-models.ts`
- Create: `web/src/lib/__tests__/read-models.test.ts`
- Create: `web/src/hooks/useReadPage.ts`
- Create: `web/src/hooks/__tests__/useReadPage.test.tsx`
- Modify: `web/package.json`
- Modify: `web/package-lock.json`
- Modify: `web/src/lib/types.ts`

- [ ] **Step 1: Add the test runner and write failing pure-state tests**

Install `vitest`, `jsdom`, and `@testing-library/react` as dev dependencies and add scripts `test` and `test:watch`. Test a reducer that holds current/previous/next pages only, handles 304 without identity churn, resets on 409 while retaining visible items, and chunks a 40,000-item manifest into 200 arrays of at most 200.

```typescript
it('chunks an unlimited folder manifest into bounded requests', () => {
  const files = Array.from({ length: 40_000 }, (_, i) => ({ key: String(i) }))
  const chunks = chunkManifest(files, 200)
  expect(chunks).toHaveLength(200)
  expect(Math.max(...chunks.map((chunk) => chunk.length))).toBe(200)
  expect(chunks.flat()).toHaveLength(40_000)
})
```

- [ ] **Step 2: Run frontend tests and verify RED**

Run: `cd web && npm test -- --run src/lib/__tests__/read-models.test.ts`

Expected: import failure for `@/lib/read-models`.

- [ ] **Step 3: Implement transport values, three-page reducer, and manifest helper**

Define types matching Pydantic field names. `chunkManifest<T>` rejects non-positive bounds and returns fresh arrays. The reducer stores at most three page payloads and a cursor-history array. A filter key change aborts the old state and starts at page one.

- [ ] **Step 4: Write failing hook tests**

Use a real `AbortController` and a fetch stub returning 200, 304, and 409 responses. Assert the old request is aborted when query parameters change, 304 keeps the same items reference, 409 keeps visible data until replacement succeeds, and hidden-document polling does not call fetch.

- [ ] **Step 5: Implement `useReadPage`**

The hook accepts URL, token, query key, poll interval, and initial limit. It sends the last ETag, treats 304 as success, parses the 409 revision detail, restarts at page one without clearing visible data, and cancels every superseded request. It exposes `items`, `loading`, `refreshing`, `error`, `hasNext`, `loadNext`, `reload`, and `revision`.

- [ ] **Step 6: Verify GREEN and production type safety**

Run: `cd web && npm test -- --run && npx tsc --noEmit`

Expected: all tests and type checks pass.

- [ ] **Step 7: Commit**

```bash
git add web/package.json web/package-lock.json web/vitest.config.ts web/src/lib/read-models.ts web/src/lib/types.ts web/src/lib/__tests__/read-models.test.ts web/src/hooks/useReadPage.ts web/src/hooks/__tests__/useReadPage.test.tsx
git commit -m "feat: add bounded web read state"
```

## Task 6: Migrate file browsing, sidebar search, and folder upload

**Files:**
- Create: `web/src/hooks/useDocumentBrowse.ts`
- Create: `web/src/components/kb/DocumentSearchDialog.tsx`
- Create: `web/src/lib/upload-preflight.ts`
- Create: `web/src/lib/__tests__/upload-preflight.test.ts`
- Modify: `web/src/components/kb/KBDetail.tsx`
- Modify: `web/src/components/kb/FilesGrid.tsx`
- Modify: `web/src/components/kb/KBSidenav.tsx`
- Modify: `web/src/stores/useUploadStore.ts`

- [ ] **Step 1: Write failing folder-upload pipeline tests**

Test 201 and 40,000 descriptors, exactly two concurrent preflight calls, cross-chunk filename/hash duplicates, one-file-at-a-time buffer release per hash worker, cancellation before unsent chunks, and status reconciliation in batches of 200.

```typescript
expect(Math.max(...concurrencySamples)).toBe(2)
expect(preflightCalls).toHaveLength(200)
expect(preflightCalls.every((call) => call.items.length <= 200)).toBe(true)
```

- [ ] **Step 2: Run upload tests and verify RED**

Run: `cd web && npm test -- --run src/lib/__tests__/upload-preflight.test.ts`

Expected: missing `upload-preflight` module.

- [ ] **Step 3: Implement bounded upload preflight and status reconciliation**

Use two async workers over manifest chunks and two hash workers over eligible files. Maintain manifest-wide `Set<string>` values for destination/name and digest. Start accepted uploads as each chunk finishes. Change `useUploadStore` reconciliation input from all documents to bounded status results keyed by active upload identity.

- [ ] **Step 4: Migrate file browsing and sidebar search**

`KBDetail` owns `useDocumentBrowse` for the active path and passes `items`, `folders`, paging state, and refresh callbacks to `FilesGrid`. Remove client-side descendant folder discovery and full-array sorting. `FilesGrid` renders only the current page and a load-more control.

Replace `KBSidenav.sourceDocs` with scalar source/failed/corpus counts plus `DocumentSearchDialog`. The dialog queries only after trimmed input is non-empty, debounces 250 ms, and renders at most 100 source hits. Wiki results still come from the small Wiki-only set.

- [ ] **Step 5: Remove upload and navigation full-array lookups**

Use upload preflight for duplicates, status batches for progress, and `documents/resolve` for document-number open requests. Selected-row names for delete confirmation come from the active page. No code in `KBDetail`, `FilesGrid`, or `KBSidenav` may import `useKBDocuments`.

- [ ] **Step 6: Verify frontend GREEN**

Run: `cd web && npm test -- --run && npx tsc --noEmit && npm run build`

Expected: all pass and `rg -n 'useKBDocuments' src/components/kb/KBDetail.tsx src/components/kb/FilesGrid.tsx src/components/kb/KBSidenav.tsx` returns no matches.

- [ ] **Step 7: Commit**

```bash
git add web/src/hooks/useDocumentBrowse.ts web/src/components/kb/DocumentSearchDialog.tsx web/src/lib/upload-preflight.ts web/src/lib/__tests__/upload-preflight.test.ts web/src/components/kb/KBDetail.tsx web/src/components/kb/FilesGrid.tsx web/src/components/kb/KBSidenav.tsx web/src/stores/useUploadStore.ts
git commit -m "feat: bound file browsing and folder uploads"
```

## Task 7: Migrate Wiki navigation, direct links, citations, and assets

**Files:**
- Create: `web/src/hooks/useWikiPages.ts`
- Create: `web/src/hooks/useDocumentResolver.ts`
- Create: `web/src/hooks/__tests__/useDocumentResolver.test.tsx`
- Modify: `web/src/components/kb/WikiOnlyDetail.tsx`
- Modify: `web/src/components/kb/KBDetail.tsx`
- Modify: `web/src/components/wiki/WikiContent.tsx`
- Modify: `web/src/app/(dashboard)/wikis/[slug]/[[...path]]/page.tsx`

- [ ] **Step 1: Write failing resolver tests**

Assert deduplicated concurrent resolution, cache keys scoped by KB and revision, abort on Wiki navigation, and negative-result caching only within one revision.

- [ ] **Step 2: Run resolver tests and verify RED**

Run: `cd web && npm test -- --run src/hooks/__tests__/useDocumentResolver.test.tsx`

Expected: missing resolver hook.

- [ ] **Step 3: Implement Wiki-only loading and resolver cache**

`useWikiPages` wraps `useReadPage` with Wiki scope and follows pages only until the small navigation set is complete, retaining no source docs. `useDocumentResolver` keys requests by `kbId`, revision, and either document number or logical reference and coalesces identical in-flight requests.

- [ ] **Step 4: Replace every Wiki full-array dependency**

Build trees, course progress, stale state, and facet rollups from Wiki items. Resolve legacy `?page=`, source citations, relative images, graph-node source opens, and document-number deep links through the resolver endpoint. Pass the active Wiki document directly to `WikiContent`; remove its `documents` prop.

- [ ] **Step 5: Verify Wiki GREEN and removal gate**

Run: `cd web && npm test -- --run && npx tsc --noEmit && npm run build`

Expected: all pass and `rg -n 'useKBDocuments' src/components/kb/WikiOnlyDetail.tsx src/components/wiki/WikiContent.tsx 'src/app/(dashboard)/wikis/[slug]/[[...path]]/page.tsx'` returns no matches.

- [ ] **Step 6: Commit**

```bash
git add web/src/hooks/useWikiPages.ts web/src/hooks/useDocumentResolver.ts web/src/hooks/__tests__/useDocumentResolver.test.tsx web/src/components/kb/WikiOnlyDetail.tsx web/src/components/kb/KBDetail.tsx web/src/components/wiki/WikiContent.tsx 'web/src/app/(dashboard)/wikis/[slug]/[[...path]]/page.tsx'
git commit -m "feat: bound wiki document resolution"
```

## Task 8: Migrate corpus browsing and bounded batch review

**Files:**
- Create: `web/src/hooks/useCorpusEntries.ts`
- Create: `web/src/hooks/__tests__/useCorpusEntries.test.tsx`
- Modify: `web/src/components/corpus/CorpusView.tsx`
- Modify: `web/src/lib/corpus.ts`

- [ ] **Step 1: Write failing corpus hook tests**

Assert summary and first page run in parallel, filter changes abort both old requests, stale cursors restart without blanking, only three pages remain cached, and batch-review target generation rejects 201 ids.

- [ ] **Step 2: Run tests and verify RED**

Run: `cd web && npm test -- --run src/hooks/__tests__/useCorpusEntries.test.tsx`

Expected: missing corpus hook.

- [ ] **Step 3: Implement corpus hook and server-response adapters**

Move client-only aggregate result types into transport types. Keep presentation label helpers in `web/src/lib/corpus.ts`, but remove full-array `filterEntries`, `collectFacetOptions`, `buildCoverageGrid`, `buildBusinessView`, and `computeKpis` calls from `CorpusView`.

- [ ] **Step 4: Render bounded pages and summary**

`CorpusView` renders summary facets/KPIs/coverage/business data and at most the current 200 entry rows. Filtering, text query, sort, and page navigation are server parameters. Batch approval is limited to selected/current-page ids and its label states the exact bounded count.

- [ ] **Step 5: Verify corpus GREEN and full-array removal**

Run: `cd web && npm test -- --run && npx tsc --noEmit && npm run build`

Expected: all pass; `rg -n 'useKBDocuments|/graph' web/src/components/corpus/CorpusView.tsx` returns no matches.

- [ ] **Step 6: Commit**

```bash
git add web/src/hooks/useCorpusEntries.ts web/src/hooks/__tests__/useCorpusEntries.test.tsx web/src/components/corpus/CorpusView.tsx web/src/lib/corpus.ts
git commit -m "feat: paginate corpus read models"
```

## Task 9: Performance regression gates, documentation, live acceptance, and publication

**Files:**
- Create: `scripts/benchmark_read_models.py`
- Create: `tests/unit/test_read_model_performance.py`
- Modify: `docs/architecture/data-and-retrieval.md`
- Modify: `docs/architecture/system-overview.md`
- Modify: `docs/self-hosting.md`
- Modify: `README.md`

- [ ] **Step 1: Write the failing structural performance test**

Seed 75,000 narrow document rows in one transaction. Run `EXPLAIN QUERY PLAN` for the first and second browse pages and assert the bounded composite index is used and `USE TEMP B-TREE FOR ORDER BY` is absent. Assert 201 requested items are rejected and 200 returned items serialize below 1 MiB.

- [ ] **Step 2: Run and verify RED**

Run: `pytest tests/unit/test_read_model_performance.py -q`

Expected: failure until the final query/index names and serializers are wired.

- [ ] **Step 3: Implement the benchmark script**

Support `--database`, `--api-base`, `--kb-id`, and optional `--token-env`. Print document count, revision latency, first-page latency, second-page latency, ETag 304 latency, item count, response bytes, and query plan. Never print tokens, filenames, paths, metadata, or response bodies. Exit nonzero when warm page latency exceeds 200 ms, 304 exceeds 50 ms, item count exceeds 200, response exceeds 1 MiB, or a temporary sort appears.

- [ ] **Step 4: Update architecture and operations documentation**

Document the legacy compatibility endpoint, new read APIs, revision/ETag behavior, cursor reset semantics, Local/Hosted indexes, three-page Web cache, 200-item request bounds, unlimited folder manifests through 200-item chunks, and the benchmark command. Remove statements that Web loads or polls all documents.

- [ ] **Step 5: Run focused and full verification**

Run:

```bash
pytest tests/unit/core/test_read_cursor.py tests/unit/test_read_revision.py tests/unit/test_local_read_models.py tests/unit/test_local_corpus_read_models.py tests/unit/test_read_model_routes.py tests/unit/test_read_model_performance.py -q
pytest tests/integration/test_read_model_schema.py tests/integration/isolation/test_read_model_api_isolation.py tests/integration/isolation/test_corpus_read_model_isolation.py -q
pytest tests/unit -q
cd web && npm test -- --run && npx tsc --noEmit && npm run build
```

Expected: every command exits 0 with no unexpected warnings.

- [ ] **Step 6: Run copied-live-database acceptance**

Create an APFS copy-on-write clone of the complete live workspace so startup
reconciliation sees the same files without modifying the live database. Start the
new API against the clone on a non-conflicting port, wait for readiness, and run:

```bash
READ_MODEL_TMP_ROOT="$(mktemp -d /tmp/llmwiki-read-model.XXXXXX)"
cp -cR /Users/benjamin/Downloads/llmwiki-goglobal/workspace "$READ_MODEL_TMP_ROOT/workspace"
READ_MODEL_DB_COPY="$READ_MODEL_TMP_ROOT/workspace/.llmwiki/index.db"
MODE=local STAGE=test WORKSPACE_PATH="$READ_MODEL_TMP_ROOT/workspace" \
  API_URL=http://127.0.0.1:19000 APP_URL=http://127.0.0.1:3000 \
  .venv/bin/uvicorn main:app --app-dir api --host 127.0.0.1 --port 19000 \
  >"$READ_MODEL_TMP_ROOT/api.log" 2>&1 &
READ_MODEL_API_PID=$!
for attempt in $(seq 1 60); do
  curl -fsS http://127.0.0.1:19000/ready >/dev/null && break
  sleep 1
done
python scripts/benchmark_read_models.py \
  --database "$READ_MODEL_DB_COPY" \
  --api-base http://127.0.0.1:19000 \
  --kb-id 23736668-ca62-45b9-89fa-a9b8ccb30d0b
kill "$READ_MODEL_API_PID"
wait "$READ_MODEL_API_PID" 2>/dev/null || true
case "$READ_MODEL_TMP_ROOT" in
  /tmp/llmwiki-read-model.*) rm -rf -- "$READ_MODEL_TMP_ROOT" ;;
  *) echo "refusing to remove unexpected path" >&2; exit 1 ;;
esac
```

Expected: at most 200 items, under 1 MiB, warm pages under 200 ms, 304 under 50 ms, and no temporary sort. Delete only the validated temporary copy after recording results; do not modify the user's live database or running container.

- [ ] **Step 7: Commit documentation and benchmark gates**

```bash
git add scripts/benchmark_read_models.py tests/unit/test_read_model_performance.py docs/architecture/data-and-retrieval.md docs/architecture/system-overview.md docs/self-hosting.md README.md
git commit -m "test: gate large corpus read performance"
```

- [ ] **Step 8: Final branch verification and publication**

Run `git status --short`, confirm only intended changes are committed, compare `git diff --stat origin/feat/platform-architecture-evolution...HEAD`, then use the `github:yeet` workflow to push `feat/platform-architecture-evolution`. Verify the resulting GitHub Actions run before claiming completion.
