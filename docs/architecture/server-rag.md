# Server-side RAG operations

## Operational contract

Server-side RAG is an optional Hosted-mode workflow that plans and writes a
bounded wiki through the existing durable job ledger. `background_jobs` is the
only scheduler and lease authority. `rag_runs`, `rag_run_pages`, and
`rag_steps` persist the recoverable worklist and audit trace; a wiki write and
its page boundary commit in one Postgres transaction. API replicas therefore
remain stateless, and any worker can resume from the last committed page.

The feature is off unless both API and worker receive
`SERVER_RAG_ENABLED=true`. Local mode remains unchanged. Clients select a
server-owned profile name and never submit a provider URL, model id, or key.

The [platform overview](overview.md) owns the end-to-end topology. RAG reuses
the [shared write invariants](shared-kernel.md), executes through the
[durable-job ledger](durable-jobs.md), and consumes the independently gated
[retrieval contract](retrieval.md).

## Configuration and secrets

Keep profile metadata and provider credentials in separate secret-manager
entries. Both mappings must have the same profile names:

```dotenv
SERVER_RAG_ENABLED=false
RAG_MODEL_PROFILES_JSON={"primary":{"base_url":"https://models.internal.example/v1","model":"wiki-writer","timeout_seconds":60,"version":"wiki-writer-2026-07"}}
RAG_MODEL_API_KEYS_JSON={"primary":"load-from-secret-manager"}
```

`RAG_MODEL_PROFILES_JSON` is treated as sensitive configuration even though it
does not contain the key. `RAG_MODEL_API_KEYS_JSON` must never be committed,
passed by a client, placed in a CLI argument, or shared with the web/MCP
process. The LLMWiki bearer token used by `scripts.rag` authenticates the REST
caller and is not a model-provider credential. Profile resolution fails closed
if either mapping is malformed, too large, mismatched, duplicated, or contains
an unapproved field or URL.

Only the profile name and non-secret `version` are durable. Provider URLs,
model ids, credentials, prompts, raw responses, evidence bodies, DSNs, and
private exception text are excluded from public responses, persisted step
summaries, and structured logs.

## Migration and deployment order

Apply `supabase/migrations/001` through `015` in order before enabling RAG.
Migration `015_server_rag.sql` is additive: it extends the durable job type and
adds RAG runs, pages, steps, constraints, indexes, and RLS. Keep it installed
during rollback.

Deploy API and worker from the same release with the flag false. Configure and
validate the named profiles, run the deterministic fake-model smoke, then
enable one internal profile for a bounded tenant cohort. Never enable RAG when
durable jobs are disabled, and do not mix API producers with workers from
different releases.

## REST and CLI

The authenticated REST surface is:

- `POST /v1/rag/build-wiki` with a required `Idempotency-Key`;
- `GET /v1/rag/runs/{run_id}`;
- `GET /v1/rag/runs/{run_id}/steps?after=0&limit=50`;
- `POST /v1/rag/runs/{run_id}/resume` with a new idempotency key; and
- `POST /v1/jobs/{job_id}/cancel` for cancellation.

Create accepts a knowledge base, goal, `/wiki/.../` target prefix, model
profile, optional retrieval profile, optional lower budget, and `dry_run`.
Unknown fields are rejected. Reads, cancel, and resume are tenant scoped.

`PYTHONPATH=api python -m scripts.rag` provides `build-wiki`, `status`,
`steps`, and `resume`. It only calls the REST API. Exit codes are `0` success,
`2` local argument/configuration failure, `3` REST/transport failure, and `4`
output failure. Load `LLMWIKI_ACCESS_TOKEN` from a secret store; CLI errors do
not echo it, the goal, response body, provider configuration, or exception
chain.

## Budgets and retrieval

Requests may reduce defaults but cannot exceed hard caps:

| Limit | Default | Hard cap |
|---|---:|---:|
| Pages | 8 | 32 |
| Persisted steps | 96 | 512 |
| Model tokens | 64,000 | 250,000 |
| Context characters per page | 120,000 | 240,000 |
| Page characters | 40,000 | 120,000 |
| Model-call timeout | 60 seconds | 180 seconds |
| Attempts per page | 2 | 3 |
| Conflict retries per page | 1 | 3 |

Lexical retrieval is the default and requires PGroonga. `hybrid` is available
only when the separately gated hybrid configuration and current-version
embedding coverage are valid. A typed vector availability failure may fall
back to lexical under the shared retrieval contract; lexical, authorization,
configuration, and unexpected failures do not become false successes. The
full hybrid promotion contract remains in
[`retrieval.md`](retrieval.md).

## Recovery, cancellation, and partial commits

A lease checkpoint fences every model or database boundary. If a worker dies
before the atomic page commit, the reaper marks the interrupted step and a
different worker repeats that page. If it dies after commit, the next worker
observes the durable ordinal and does not rewrite the page. Compare-and-swap
refreshes reread and redraft after a version conflict up to the configured
limit; they never overwrite a concurrent revision.

Cancellation uses the generic job endpoint and is observed at checkpoints. It
does not roll back pages already committed. Budget exhaustion and other
terminal failures likewise preserve the committed prefix. `resume` creates a
new run/job with root and parent lineage, copies the immutable worklist, skips
committed ordinals, and accepts only budget increases within server caps.
Cancelled and successful runs are not resumable.

`dry_run` still persists a run and bounded validated trace, but commits no
wiki, chunk, reference, or facet rows. Each page stores only a capped preview,
its full-character count, digest, and truncation flag. A dry run completes with
boundary `-1` and `dry_run_complete` pages.

## Observability and privacy

Use the structured `rag_run_started`, `rag_step_finished`,
`rag_page_committed`, `rag_run_finished`, and `rag_run_failed` events. A
conflict is represented by the relevant finished step and its stable error
code. Safe dimensions include run/job ids, tenant-scoped knowledge
base identity, ordinal, step type, stable error code, non-secret profile name
and version, token/count totals, and bounded latency. Do not add goals, paths,
page bodies, prompts, evidence, provider values, DSNs, raw errors, or bearer
tokens to logs or metrics.

Alert on queue age, retry/terminal rates, validation and lint errors, token and
latency budgets, conflicts, cancellations, and time to first/final committed
page. Health remains process liveness. Startup validates configured profiles,
and readiness checks the ordinary Hosted dependencies; neither contacts an
external model.

## Fake-model and scaled smoke

CI owns a deterministic OpenAI-compatible server in
`tests/fixtures/fake_rag_model`. It accepts only
`POST /v1/chat/completions`, caps requests at 1 MiB, returns exact plan/draft
JSON and token usage, has no host-published port, and is enabled only by the
Compose `ci` profile. It must never be replaced by an external provider in CI.

The `integration-rag` matrix runs each RAG integration file in a fresh pytest
process against the pinned pgvector Postgres service. The required scaled
Compose job verifies two API and two worker replicas, cross-replica create and
observe, owner death before page commit, lease recovery by another worker,
cancellation, explicit resume, and unique page versions and step sequences.

## Rollout and rollback

Roll out in this order:

1. apply migration `015` and deploy with the flag false;
2. configure allowlisted profiles and separate credentials;
3. pass fake-model, integration, and two-API/two-worker smoke gates;
4. enable one internal profile for a small tenant cohort;
5. observe errors, tokens, latency, conflicts, cancellation, and partial
   commits; and
6. expand the cohort without changing default budgets or retrieval defaults.

The operational rollback is flag-only: set `SERVER_RAG_ENABLED=false` on API
and worker and roll/restart both roles. New create/resume requests are rejected;
running work stops at the next page boundary, while committed wiki versions and
audit rows remain intact. Do not drop migration `015`, delete partial history,
disable durable jobs, or change the independent hybrid retrieval flag as part
of this rollback.

## Verification evidence

The substantive candidate is
`f4c22afa39c249e000ac5becf10d8bfb212be75c`. GitHub Actions
[run 30251148719](https://github.com/ben0112/llmwiki-goglobal/actions/runs/30251148719)
matched that SHA and passed all six jobs, including the isolated Postgres RAG
matrix and the required two-API/two-worker live Compose smoke.

Local gates for the candidate recorded these exact results:

| Gate | Result |
|---|---:|
| Core RAG | 74 passed |
| RAG and durable-job unit tests | 926 passed |
| MCP plus wiki-write invariants | 180 passed |
| Combined RAG integration files | 205 passed |
| RAG API isolation | 7 passed |
| Fresh-process `integration-rag` partition | 212 passed |
| Fresh-process `integration-api` partition | 402 passed, 36 skipped |
| Fresh-process `integration-retrieval` partition | 391 passed |
| Scaled Compose and CI static contracts | 19 passed, 1 live opt-in skipped |
| Live two-API/two-worker scaled recovery smoke | 1 passed in 361.66 seconds |

Focused and changed-file Ruff reported zero errors. A repository-wide Ruff
scan still reports the pre-existing baseline of 258 errors, 149 automatically
fixable; this milestone does not expand that unrelated cleanup scope.

The deterministic dataset contains authoritative launch evidence plus an
adversarial prompt-injection string. The successful path produced the exact
seven-step trace and 42 model tokens (18 plan and 24 draft), cited the source,
and excluded the injection and provider secrets. The seven live E2E scenarios
cover initial generation/injection resistance, refresh, dry-run, budget
exhaustion plus explicit resume, timeout persistence, real version conflict,
and private configuration exclusion.

Independent specification and quality reviews examined the exact candidate
SHA and each concluded Critical 0, Important 0, Minor 0, `Ready: Yes`.

The verified publication state remains disabled by default. Deploy migration
and matching API/worker code with `SERVER_RAG_ENABLED=false`; enable only one
internal allowlisted profile for a bounded tenant cohort after these gates.
Rollback remains the flag-only procedure above.
