# Bounded read models for large workspaces

## Contract

The Web main screens do not load the complete knowledge-base document array.
`GET /v1/knowledge-bases/{kb_id}/documents` remains available as a legacy
compatibility endpoint, but new file, Wiki, corpus, upload, sidebar-search, and
document-resolution flows use bounded read models:

| Endpoint | Purpose and bound |
|---|---|
| `GET .../documents/browse` | Exact-folder browse or non-empty global source search; at most 200 documents. |
| `GET .../wiki/pages` | Wiki-only metadata pages; at most 200 per response. |
| `GET .../documents/resolve` | One document by `document_number` or logical reference. |
| `POST .../documents/status` | Status for 1–200 ids or document numbers. |
| `POST .../documents/upload-preflight` | Duplicate/type/size decisions for at most 200 descriptors. |
| `GET .../corpus/entries` | Server-filtered corpus page; at most 200 entries. |
| `GET .../corpus/summary` | Fixed-size facets, coverage, business counts, and KPIs. |
| `GET .../graph/summary` | Fixed-size graph counts and bounded cited-document ids. |

Every list cursor binds its endpoint, read revision, sort, direction, and final
stable `id`. Malformed or cross-endpoint cursors return `422 invalid_cursor`.
If a document mutation advances the knowledge-base revision, an older cursor
returns `409 stale_cursor` with the current revision. The client keeps the last
visible page while restarting at page one, so a changing workspace does not
briefly render an empty list.

List and summary responses carry an ETag derived from the durable read
revision. `If-None-Match` is checked before the data query; an unchanged view
returns an empty `304`. Local foreground polling uses that conditional request
and stops while the browser tab is hidden. Hosted invalidation may use
LISTEN/NOTIFY and WebSockets, with ETag refresh as the recovery path.

## Storage and indexes

Local mode stores `workspace.read_revision` in SQLite. Insert, update, and
delete triggers advance it. `shared/sqlite_read_models.sql` installs keyset
indexes for `(path, filename COLLATE NOCASE, id)`, date browse, Wiki paths,
document numbers, and non-failed content hashes. Workspace files remain the
content source of truth; SQLite remains a rebuildable index.

The first Local start after upgrading an older large database creates these
indexes before readiness. This can take several minutes and may use one CPU
core heavily; do not kill the process while it still reports application
startup. Later starts reuse the indexes. On the measured 75,403-document copy,
the post-migration warm browse completed in 1.113 ms and its ETag `304` in
0.575 ms; these measurements describe that machine and are not universal
capacity promises.

Hosted migration `016_read_models.sql` adds `knowledge_bases.read_revision`, a
document trigger, and tenant-first partial Postgres indexes. Their leading keys
are `knowledge_base_id` and `user_id`, followed by the target path/scope and
stable sort tuple. RLS and explicit SQL predicates both retain tenant scope.

## Web memory and upload behavior

The generic read hook retains at most the current, previous, and next pages:
three pages or 600 narrow records at the maximum page size. Wiki navigation
accumulates only Wiki metadata. Citations, relative assets, legacy links, and
file deep links resolve one document on demand through a revision-keyed cache.
Corpus facets and KPIs come from `corpus/summary`; the corpus table mounts at
most the current 200 rows, and batch review cannot target more than 200 ids.

Folder size is not capped at 200 files. The browser walks and hashes the local
manifest with two workers, sends at most two concurrent 200-descriptor
preflight requests, and releases each file buffer after hashing. It preserves
cross-chunk duplicate detection and can cancel before unsent chunks. Upload
status reconciliation is likewise split into requests of at most 200 ids.

## Regression benchmark

The structural test seeds 75,000 SQLite document rows and asserts that first
and second browse pages use `idx_documents_browse_name`, avoid a temporary
ORDER BY B-tree, return at most 200 items, and serialize below 1 MiB:

```bash
.venv/bin/pytest tests/unit/test_read_model_performance.py -q
```

For a copied Local workspace, start an isolated API and run:

```bash
.venv/bin/python scripts/benchmark_read_models.py \
  --database /copied/workspace/.llmwiki/index.db \
  --api-base http://127.0.0.1:19000 \
  --kb-id 00000000-0000-0000-0000-000000000001
```

The command prints only counts, latencies, byte size, and query-plan operators;
it never prints tokens, filenames, paths, metadata, or response bodies. It
fails when a warm page exceeds 200 ms, a `304` exceeds 50 ms, a page exceeds
200 items or 1 MiB, or SQLite introduces a temporary ORDER BY sort. For Hosted
mode, load a REST token through an environment variable and pass only its name
with `--token-env`; never place the token on the command line.
