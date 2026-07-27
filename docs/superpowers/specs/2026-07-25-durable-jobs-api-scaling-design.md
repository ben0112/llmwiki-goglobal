# Durable Jobs and Hosted API Scaling

## Status

Implemented on `feat/platform-architecture-evolution`. Hosted accepted work is
durable, API replicas are stateless, workers recover through Postgres leases,
and multipart uploads can move between API replicas.

The operational source of truth is
[durable-jobs.md](../../architecture/durable-jobs.md). This specification keeps
the design rationale and failure boundaries.

## Goals and non-goals

The design ensures that accepted Hosted work survives API or worker loss,
duplicate delivery cannot duplicate side effects, cancellation and progress
are observable, and API/worker replica counts scale independently. It also
keeps multipart offsets recoverable across API restarts and publishes document
state only after validated object completion.

It does not replace Local SQLite/filesystem execution, provide exactly-once
message transport, make Redis a business database, coordinate old and new
worker protocols during a mixed-release rollout, or support arbitrary job
payloads and handlers.

## Selected architecture

Postgres owns the job ledger, lease generation, progress, result summary, safe
error, attempts, cancellation, and idempotency identity. Redis/ARQ delivers
opaque job UUIDs and runs maintenance scheduling. S3/MinIO stores upload bytes;
Redis stores bounded, expiring multipart coordination.

This separation makes message delivery at-least-once while making business
publication compare-and-swap guarded. Losing an ARQ message is repairable from
Postgres, and receiving it twice is harmless because only one live database
claim wins.

## Ledger, dispatch, and leases

Jobs move through `queued`, `running`, `retry_wait`, and terminal `succeeded`,
`failed`, or `cancelled` states. A claim uses database time, increments the
attempt, and records a unique owner plus heartbeat and expiry. Heartbeat,
progress, retry, and completion require the same running owner and an unexpired
lease.

The API commits a ledger row before dispatching its UUID. Dispatch scans
redeliver committed due rows when Redis delivery was lost. The reaper locks
expired rows with `SKIP LOCKED`, clears ownership, and chooses retry, failure,
or cancellation from the persisted attempt and request state.

ARQ deduplication reduces duplicate work but is not a correctness boundary. A
duplicate message that loses the Postgres claim exits without invoking the
handler.

## Handler and transaction boundaries

Handlers receive a persisted tenant/resource identity rather than trusting a
message payload. They revalidate ownership and bounded payload shape before
side effects.

Document extraction replaces content, pages, chunks, derived assets, version,
and job completion in a transaction that validates the live lease. Graph
rebuild similarly replaces its derived edge set with the matching generation.
Embedding handlers publish one complete current-version/profile vector set.
Server-side RAG commits each wiki write with its page boundary. A stale worker
cannot publish any of those results.

Failures map to bounded retryable or terminal codes. Retryable failures use a
deterministic backoff while attempts remain; a final retry becomes
`attempts_exhausted`. Process shutdown is retryable, cancellation uses its own
terminal state, and control signals are not flattened into ordinary errors.

## Resumable uploads

A TUS creation reserves quota, creates an S3 multipart upload, then creates a
Redis reservation marker and closed-schema session under a token-fenced lock.
A PATCH writes an S3 part before advancing the Redis offset with its ETag.
Non-final parts respect the S3 minimum, request size is bounded, and no session
can exceed the declared length or 10,000 parts.

The final PATCH completes and validates the object before one Postgres
transaction creates the document and extraction job. A confirmed database
rollback deletes the orphan. An uncertain commit preserves the object and
session for reconciliation rather than risking deletion of committed data.
Repeated final PATCH returns the persisted document/job identity.

Redis AOF preserves active offsets across ordinary restarts. If session state
is lost, committed documents and jobs remain safe in Postgres; stale multipart
uploads are reclaimed through idempotent cleanup jobs.

## Multi-replica topology

The gateway is the only published API listener and distributes requests across
stateless API replicas. Any replica can create or observe a job and continue a
Redis-backed upload. Workers share the ledger, and exactly one live lease owner
per job generation may publish.

Graceful worker shutdown stops new claims and makes unfinished work available
to another owner. SIGKILL recovery waits for lease expiry and reaping. API and
worker counts are independent, but both roles deploy the same release and
configuration schema.

Hosted startup fails closed unless `DURABLE_JOBS_ENABLED=true` and
`TUS_MULTIPART_ENABLED=true`. Local startup imports neither Redis nor Hosted
job dependencies.

## Security and failure handling

- Job reads and cancellation are tenant filtered.
- Worker resource lookup revalidates the persisted tenant before side effects.
- Redis keys use opaque IDs but possession of an ID grants no authorization.
- Job payloads, upload metadata, progress, result summaries, and errors have
  closed schemas and size limits.
- JWTs, API keys, object URLs, multipart IDs, source content, DSNs, and raw
  exception text are excluded from persisted public state and telemetry.
- Uncertain commits favor reconciliation over destructive compensation.
- Redis loss cannot erase a committed job; Postgres/S3 backup remains the
  durable business recovery boundary.

## Alternatives rejected

### API-owned background tasks

In-process tasks bind accepted work to one replica and cannot distinguish an
API crash from an unfinished side effect. They are not used in Hosted mode.

### Redis-owned job truth

Treating ARQ results as the ledger would separate job completion from document
publication and make Redis recovery a business-data recovery event. Redis is
therefore limited to delivery and expiring upload coordination.

### Long database transactions around remote calls

Holding locks while invoking converter or object storage harms throughput and
still cannot create a distributed transaction. Remote work happens outside the
publication transaction; the live lease is checked again at commit.

### Automatic mixed-release compatibility

Protocol negotiation between arbitrary API and worker releases adds an
unbounded compatibility surface. Deployments instead roll API and workers as a
coordinated release and reject malformed/unsupported job types.

## Verification and operations

The closing candidate `08892110d85c09b8d64027cd8ab468c107ee5490`
passed all six jobs in
[GitHub Actions run 30152538145](https://github.com/ben0112/llmwiki-goglobal/actions/runs/30152538145).
The required scaled path covered two APIs, two workers, Redis AOF restart, API
loss, worker TERM/SIGKILL recovery, multipart uploads, extraction, graph
rebuild, one-owner claims, and stale-owner rejection.

The later platform publication baseline
`0173f560c6fec03b87ce4f6803f663d2d6983ead` also passed all six jobs in
[run 30251808426](https://github.com/ben0112/llmwiki-goglobal/actions/runs/30251808426).

Current operating instructions and exact rollback boundaries are in:

- [Durable jobs](../../architecture/durable-jobs.md)
- [Platform overview](../../architecture/overview.md)
- [Self-hosting guide](../../self-hosting.md)
