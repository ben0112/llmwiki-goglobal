# Large Corpus Read-Path Optimization Design

**Date:** 2026-07-28
**Status:** Approved for implementation planning
**Scope:** Local SQLite and Hosted Postgres read paths, Web document browsing,
Wiki navigation, corpus browsing, and graph-derived corpus metrics

## 1. Problem statement

The current Web application obtains one unbounded document array and reuses it
for file browsing, Wiki navigation, corpus filtering, corpus metrics, and active
document resolution. Local mode polls that array periodically. This design works
for small workspaces but causes database scans, large JSON responses, repeated
client reconciliation, and unbounded React rendering at large corpus sizes.

The live local workspace measured on 2026-07-28 contains:

- 75,403 documents;
- 213,256 document chunks;
- 62,035 document pages;
- a 4.2 GiB SQLite database and a 276 MiB WAL;
- a 1.78 GiB `document_chunks` table, 1.38 GiB FTS data, and an
  866.5 MiB `documents` table.

The existing document-list query performs `SCAN documents` plus a temporary
B-tree sort. It took 4.13 seconds when executed directly. The HTTP endpoint
returned 96.1 MB in 9.83 seconds cold and 3.92 seconds warm. Local polling repeats
that response every 15 seconds. The corpus screen also downloads a 21.7 MB graph
response to derive one KPI and renders every matching corpus entry.

SQLite remains fast for bounded indexed work: document and chunk counts completed
in 16 ms and 154 ms respectively, and indexed FTS lookup was effectively
instantaneous. The problem is therefore the unbounded read shape, not a row-count
limit in SQLite.

## 2. Goals

1. Bound every interactive list request to at most 200 records.
2. Remove the legacy unbounded document endpoint from Web main-screen data paths
   without changing its public response contract.
3. Give Local and Hosted the same read-model API and response schemas.
4. Replace whole-workspace polling with database-backed revision validation and
   conditional current-page refreshes.
5. Move corpus filtering, counts, coverage, business distribution, and citation
   metrics to bounded server-side read services.
6. Prevent the corpus screen from mounting more than 200 entry rows at once.
7. Preserve tenant isolation, current write invariants, and existing document,
   page, chunk, and MCP behaviors.
8. Remove hidden full-document-array dependencies from upload deduplication,
   upload-status reconciliation, sidebar search, direct document-number lookup,
   and Wiki reference or asset resolution.

## 3. Non-goals

- Removing the legacy `GET /v1/knowledge-bases/{kb_id}/documents` endpoint.
- Moving Local mode from SQLite to Postgres.
- Replacing FTS5 or implementing a new two-character CJK search index.
- Building an all-corpus asynchronous batch-review job. This design bounds review
  actions to explicitly selected entries or the current page.
- Replacing the full graph endpoint used by the graph viewer.
- Copying large document content, page content, or chunk content into a new store.

## 4. Chosen architecture

### 4.1 Compatibility boundary

The existing document-list endpoint continues returning a bare JSON array. It
remains available to external callers and existing tests, but the Web main screens
stop calling it.

New read APIs are additive:

- `GET /v1/knowledge-bases/{kb_id}/documents/browse`
- `GET /v1/knowledge-bases/{kb_id}/wiki/pages`
- `GET /v1/knowledge-bases/{kb_id}/corpus/entries`
- `GET /v1/knowledge-bases/{kb_id}/corpus/summary`
- `GET /v1/knowledge-bases/{kb_id}/graph/summary`
- `GET /v1/knowledge-bases/{kb_id}/documents/status`
- `GET /v1/knowledge-bases/{kb_id}/documents/resolve`
- `POST /v1/knowledge-bases/{kb_id}/documents/upload-preflight`

The API service layer gains a read-only interface implemented by SQLite and
Postgres adapters. Transport models, cursor handling, and invariants are shared;
SQL remains adapter-specific.

### 4.2 Database-backed read revision

Each knowledge base has a persisted, monotonically increasing read revision.
Local SQLite stores it for the singleton workspace. Hosted Postgres stores it per
knowledge base. Database triggers increment the revision for every document
`INSERT`, `UPDATE`, and `DELETE`, including changes made outside the API process.

This revision is authoritative across API replicas. It is not an in-process
counter and does not depend on WebSocket delivery. Existing databases are
backfilled with an initial positive revision by idempotent migrations.

Every new read response includes the current revision and an `ETag`. A request
with a matching `If-None-Match` returns `304 Not Modified` before executing the
list or aggregate query. A changed revision causes only the active query to run;
the client never refreshes an unbounded workspace snapshot.

### 4.3 Stable keyset pagination

List endpoints use keyset pagination, not offset pagination. An opaque,
versioned Base64URL cursor contains:

- the endpoint/scope identifier;
- the knowledge-base read revision;
- the normalized sort identifier and direction;
- the last record's stable sort tuple, ending with document `id`.

The cursor is decoded into a strictly validated model. It cannot select SQL
fragments. Each adapter maps a closed sort enum to static SQL. The maximum limit
is 200 and the default is 100.

If the current database revision differs from the cursor revision, the endpoint
returns `409 stale_cursor` with the new revision. The Web client retains the
currently visible page while restarting from page one, preventing visual blanking
and preventing duplicate or skipped rows across a changing snapshot.

### 4.4 Indexed bounded queries

No large text is duplicated. List projections continue reading from `documents`
but select only fields needed by the target view. New composite indexes match
scope, tenant, path/filter, sort tuple, and `id`. A representative Local files
index is ordered by source scope, exact path, normalized filename, and id; Hosted
indexes include knowledge-base and user scope first.

Tests assert that the 75,000-row SQLite fixture uses an index-driven bounded plan
and does not reintroduce an unbounded temporary sort. Postgres queries remain
tenant-scoped in both application SQL and RLS.

## 5. Read APIs

### 5.1 Document browse

`documents/browse` accepts exact `path`, a closed sort enum, direction, optional
name query, cursor, and limit. It returns:

- current-directory document items;
- the next cursor;
- the read revision;
- exact current-filter count;
- immediate child folders with bounded document counts.

Folder discovery reads distinct indexed paths and derives only immediate children.
It does not fetch document rows from descendant folders. The first page defaults
to 100 documents; subsequent pages contain documents only and may reuse the
folder data already held by the client.

When `path` is omitted and a non-empty text query is present, the same endpoint
performs paginated knowledge-base-wide source-document search for the sidebar
command palette. An empty query cannot request an unbounded global source list.

### 5.2 Bounded lookup and upload support

`documents/resolve` returns at most one lightweight document. It accepts exactly
one of a document number or normalized logical reference. Logical-reference
resolution implements the existing filename, title, relative-path, and Wiki
asset lookup rules on the server. Direct links, citation popovers, and relative
Wiki images use this endpoint instead of searching a client-side workspace array.

`documents/status` accepts at most 200 document ids or upload-result document
numbers and returns only status, error, document number, and revision fields. The
upload progress store polls this endpoint for active uploads and removes finished
items from subsequent requests.

`documents/upload-preflight` accepts at most 200 file descriptors per request.
Each descriptor contains its destination path, filename, byte size, and optional
SHA-256 digest. It returns a per-descriptor decision for existing target-path
names, existing content hashes, unsupported files, and oversize files. This is a
user-experience optimization, not the authority for uniqueness: the final upload
write repeats the database checks so concurrent uploads cannot bypass invariants.

A selected folder may contain any number of files. The browser enumerates the
folder once, preserves every relative path, divides the manifest into chunks of
at most 200 descriptors, and runs no more than two preflight requests at once.
Passing files may enter the existing bounded upload queue without waiting for the
entire manifest. Client-side filename and hash sets identify duplicates across
manifest chunks. Hashing reads at most two file bodies concurrently and retains
no file-body buffer after its digest is produced. Cancellation stops unsent
manifest chunks and active hashing without invalidating already completed uploads.

### 5.3 Wiki pages

`wiki/pages` returns only non-archived Wiki Markdown documents and the fields
required to construct navigation, resolve document numbers, display stale state,
and open a page. It remains paginated even though the measured workspace contains
only 115 Wiki pages.

### 5.4 Corpus entries

`corpus/entries` accepts the existing supported facet selections, review state,
business scene, text query, sort, cursor, and limit. It returns lightweight entry
items and paging metadata. Filtering and sorting occur in the database adapter.
No client operation requires a complete corpus entry array.

### 5.5 Corpus summary

`corpus/summary` returns fixed-size aggregate data for:

- total and filtered counts;
- facet value counts with all other active filters applied;
- stage-by-layer coverage;
- business-class and business-scene distribution;
- completeness, timeliness, pending-review, shelf coverage, citation, and Wiki
  coverage KPIs.

The service may use an in-process cache keyed by knowledge base, normalized
filters, and database read revision. Correctness never depends on cache sharing:
each replica reads the authoritative database revision before accepting a cache
entry. Each replica retains at most 256 summary entries and evicts least-recently
used entries; a revision change makes every older entry ineligible immediately.

### 5.6 Graph summary

`graph/summary` returns only the counts and cited target document identifiers
needed by corpus metrics. The corpus screen stops calling the full graph endpoint.
The graph viewer continues using the existing full graph contract.

## 6. Web data flow

### 6.1 Files

The files screen requests the current path and first 100 records. Scrolling or a
user-visible load-more action requests the next cursor. Search, sort, or path
changes cancel the previous request chain and discard its cursors.

The client keeps only the current page and its immediate previous and next pages
for the active folder, for a maximum of 600 records. It retains cursor history,
not evicted records, so backward navigation can refetch an older page. New, moved,
renamed, and deleted documents update the active page optimistically, then
revalidate against the database revision.

Sidebar search sends a debounced global query to `documents/browse` and displays
only returned pages. Counts shown in navigation come from knowledge-base or corpus
summary responses, not from counting a source-document array in the browser.

Folder upload uses the chunked preflight pipeline in section 5.2. The upload
progress store requests bounded status batches for processing items. Neither path
loads existing workspace documents.

### 6.2 Wiki

Wiki navigation uses the Wiki-only endpoint. Source documents are never loaded for
Wiki tree construction. Direct `document_number` resolution operates on Wiki
items for Wiki routes and uses a bounded server lookup for source-document routes.
Citation sources and relative Wiki assets are resolved on demand through the same
bounded lookup contract.

### 6.3 Corpus

The corpus screen fetches `corpus/summary` and the first `corpus/entries` page in
parallel. Filter changes cancel both prior requests and start a new summary plus
first-page pair. It uses the same three-page, 600-record cache as file browsing,
while the list window mounts at most 200 entry rows.

Batch approval is limited to explicitly selected entries or the current page,
with an API-enforced maximum of 200 document ids. Whole-corpus review requires a
future durable job and is outside this change.

### 6.4 Conditional refresh

Local mode retains a foreground-only periodic refresh, but sends the current ETag.
An unchanged database returns an empty `304`; a change refreshes only the active
view/page. Hosted mode may keep WebSocket invalidation, with the same ETag path as
a recovery mechanism. A hidden browser tab does not poll.

## 7. Error semantics

- Invalid limits and filters return `422`.
- Upload-preflight and status requests containing more than 200 descriptors or
  identities return `422`; this per-request limit does not limit folder size.
- Malformed, cross-endpoint, cross-sort, or otherwise invalid cursors return
  `422 invalid_cursor`.
- A valid cursor tied to an older revision returns `409 stale_cursor` and the
  current revision.
- A matching ETag returns `304` with no body and without running the target data
  query.
- A missing or unauthorized knowledge base returns the existing indistinguishable
  not-found response and does not reveal tenant data.
- Client request cancellation is silent. Other failures preserve the last good
  page and expose a retry action.

## 8. Testing strategy

Implementation follows test-driven development.

1. Shared cursor unit tests cover encoding, decoding, endpoint/sort binding,
   malformed data, and stale revisions.
2. SQLite and Postgres adapter contract tests cover stable paging, exact limits,
   filters, deletion, movement, renaming, and no duplicates across an unchanged
   revision.
3. Migration tests prove idempotent revision initialization and revision changes
   on document insert, update, and delete.
4. Isolation tests prove that paging, aggregates, ETags, and caches remain scoped
   to the authenticated tenant and knowledge base.
5. A 75,000-row SQLite regression fixture checks bounded query plans and response
   cardinality. Timing assertions use generous ceilings and accompany structural
   plan assertions to avoid flaky CI failures.
6. Frontend unit tests cover page merge, 304 retention, 409 reset, stale request
   cancellation, filter changes, and the 200-row mount bound. A lightweight
   TypeScript test runner is added because the Web package currently has no unit
   test command.
7. Upload tests cover manifests with 201 and 40,000 descriptors, two-request
   preflight concurrency, cross-chunk duplicates, cancellation, status batching,
   and authoritative duplicate detection at final write.
8. Resolver and sidebar-search tests prove that Wiki assets, citations, legacy
   links, and command-palette source results no longer need a full document array.
9. Existing Python suites, Web type checks, and the production Web build remain
   required.
10. Final read-only acceptance runs against the measured 75,403-document local
   workspace before publishing the implementation.

## 9. Acceptance criteria

- The legacy array endpoint and existing callers remain compatible.
- No new list endpoint returns more than 200 items.
- A list response against the measured workspace is below 1 MB.
- A warm bounded list query against that workspace completes in under 200 ms.
- A matching revision check returns `304` in under 50 ms with an empty body.
- The corpus first screen mounts at most 200 entry rows.
- The corpus first screen does not request the 96.1 MB legacy document response or
  the 21.7 MB full graph response.
- Uploading a folder with more than 200 files completes through bounded manifest
  and status requests without loading the legacy document response.
- Sidebar search, direct links, Wiki citations, and Wiki assets resolve through
  bounded endpoints.
- Local and Hosted expose the same response models and cursor/error semantics.
- Existing tenant-isolation and shared-kernel invariants remain green.

## 10. Delivery sequence

1. Add shared read models, cursor validation, database revision migrations, and
   adapter contract tests.
2. Add indexed document browse and Wiki endpoints, then migrate the file and Wiki
   Web views.
3. Add corpus entries, aggregate summary, and graph summary endpoints, then
   migrate the corpus Web view.
4. Add conditional refresh, request cancellation, and bounded batch selection.
5. Run structural, integration, frontend, build, and live-workspace acceptance
   gates before pushing the implementation branch.
