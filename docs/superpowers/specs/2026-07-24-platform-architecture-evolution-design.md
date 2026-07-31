# Platform Architecture Evolution

## Status

Implemented on `feat/platform-architecture-evolution`. The platform uses one
shared domain kernel, Postgres-backed durable Hosted work, measured optional
hybrid retrieval, and optional server-side RAG. The latter two capabilities
remain independently gated and default off.

The current system map and operational contracts are canonical in the
[architecture overview](../../architecture/overview.md) and its four linked
domain documents. This specification records the design decisions and
boundaries behind that implementation.

## Goals and non-goals

The architecture provides:

- identical domain identities and write invariants across Local and Hosted
  adapters;
- durable, observable work that survives API or worker loss;
- stateless Hosted API replicas and independently scalable workers;
- a deterministic lexical retrieval baseline before model-dependent ranking;
- opt-in hybrid recall promoted only by measured quality and latency; and
- a bounded server-owned RAG workflow built on the same jobs, retrieval, and
  wiki-write contracts.

It does not provide a general workflow engine, arbitrary agent tools, vector-
only retrieval, automatic hybrid promotion, client-supplied model endpoints,
or Local server-side orchestration. MCP remains the interactive Local and
Hosted agent surface.

## Dependency direction

`llmwiki_core` contains framework-free values, invariants, and pure algorithms.
Storage and remote integrations live in adapters. API and MCP translate their
transport contracts into the same core values, while Hosted workers execute
persisted jobs through those adapters.

The allowed direction is:

```text
API / MCP / workers -> application services and adapters -> llmwiki_core
```

The core never imports FastAPI, MCP, database clients, S3 clients, runtime
configuration, or process lifecycle code. Local and Hosted implementations may
use different transaction mechanisms but expose the same domain result.

## Shared kernel and data invariants

Documents are tenant scoped by `(user_id, knowledge_base_id, document_id)`.
Logical paths are normalized before persistence, search hits retain document
version and chunk identity, and citations retain their source identity rather
than only rendered text.

A `ready` document publishes one coherent version of content, pages, chunks,
and derived state. Hosted adapters make that publication atomic in Postgres.
Local adapters make each file replacement atomic and reconcile SQLite from the
filesystem after an interrupted cross-store write. Wiki writes use one
`WikiWriteBundle` contract and compare-and-swap versions to protect concurrent
edits.

The canonical invariant list is
[shared-kernel.md](../../architecture/shared-kernel.md).

## Durable Hosted work and scaling

Postgres is the job ledger and lease authority. Redis/ARQ carries job UUIDs but
does not own business state or results. Workers claim due rows with database
time, heartbeat under a lease generation, and publish final state only when the
same live lease is validated in the transaction.

Document extraction, graph rebuild, current-version embeddings, upload cleanup,
and server-side RAG use this substrate. A reaper fences expired owners and
redelivers recoverable work. Cancellation is durable and observed at handler
checkpoints. API replicas therefore keep no authoritative accepted work and can
scale independently from workers.

Multipart uploads keep bytes in S3/MinIO, offsets and short-lived coordination
in Redis, and completed document/job state in Postgres. Uncertain commits are
reconciled before destructive cleanup. The complete contract is
[durable-jobs.md](../../architecture/durable-jobs.md).

## Retrieval evaluation and hybrid recall

Lexical retrieval is the default in both runtime modes. Hosted lexical queries
use the production PGroonga compiler and apply tenant, path, tag, kind, area,
facet, annotation, status, and archive filters before candidate limits.

Hybrid retrieval is Hosted-only and opt-in per request. Durable embedding jobs
write version- and profile-fenced pgvector rows. The service retrieves bounded
lexical and vector candidates, fuses unique identities with reciprocal-rank
fusion, and may apply bounded injected reranking or graph expansion. Only a
typed vector-availability failure can return a lexical fallback; configuration,
authorization, lexical, and unexpected failures remain visible.

Promotion uses one representative snapshot and requires hybrid Recall@10 to be
at least 110% of lexical while hybrid p95 latency is at most 2.0 times lexical.
Passing makes one exact profile eligible for operator rollout and never changes
the default automatically. See [retrieval.md](../../architecture/retrieval.md).

## Server-side RAG orchestration

Server-side RAG is a purpose-built `build_wiki` job, not an autonomous agent or
general DAG engine. An authenticated caller supplies a knowledge base, bounded
goal, target wiki prefix, server-owned model profile, optional retrieval
profile, lower budget overrides, and optional dry-run.

The run, immutable page worklist, and bounded step trace are durable. A worker
plans once and executes pages through shared retrieval and wiki-write ports. A
wiki revision and its RAG page boundary commit in one Postgres transaction.
After a crash, a different worker resumes from the last committed ordinal;
compare-and-swap conflicts cause bounded reread and redraft rather than
overwriting a concurrent edit.

Model endpoints and credentials are configured only on the server. Prompts,
raw model responses, evidence bodies, goals, paths, DSNs, and provider secrets
are excluded from public results and telemetry. The complete operational
contract is [server-rag.md](../../architecture/server-rag.md).

## Failure, isolation, and security properties

- Tenant filters and ownership checks occur in repositories and again before
  worker side effects; opaque IDs are not authorization.
- Postgres transactions publish coherent business state; Redis or object-store
  failures cannot manufacture a successful ledger transition.
- Stale worker generations cannot heartbeat, update progress, or commit.
- Hybrid fallback is typed and narrow, so failures do not become false lexical
  successes.
- RAG budgets cap worklist size, steps, tokens, context, output, retries, and
  call duration.
- Structured telemetry uses stable identifiers, counts, durations, and error
  codes while excluding content, credentials, and raw exceptions.

## Rollout boundaries

The migration order is additive: shared invariants first, durable work and
multi-replica deployment second, measured hybrid retrieval third, and
server-side RAG last. API producers and workers run the same release during
rollout and rollback.

Hybrid retrieval rolls back independently with `HYBRID_SEARCH_ENABLED=false`;
lexical remains available and embedding rows stay in place. Server-side RAG
rolls back with `SERVER_RAG_ENABLED=false` on API and worker; committed wiki
versions and audit rows remain. Durable Hosted jobs and multipart uploads are
not disabled as part of either feature rollback.

## Verification and canonical documentation

The pre-documentation-rewrite publication baseline is
`0173f560c6fec03b87ce4f6803f663d2d6983ead`. All six jobs passed in
[GitHub Actions run 30251808426](https://github.com/ben0112/llmwiki-goglobal/actions/runs/30251808426),
including isolated Postgres RAG/retrieval coverage and the required two-API,
two-worker recovery smoke.

Current facts and detailed evidence live in:

- [Platform overview](../../architecture/overview.md)
- [Shared kernel](../../architecture/shared-kernel.md)
- [Durable jobs](../../architecture/durable-jobs.md)
- [Retrieval](../../architecture/retrieval.md)
- [Server-side RAG](../../architecture/server-rag.md)
- [Self-hosting](../../self-hosting.md)
- [Agent integration](../../agent-integration.md)
