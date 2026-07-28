# Self-Hosting Guide (multi-user hosted mode)

This guide deploys the full multi-user stack on your own infrastructure — the
same code that runs llmwiki.app, with the SaaS pieces swapped for self-hosted
equivalents:

| Managed piece | Self-hosted replacement |
|---|---|
| Supabase cloud (Postgres + Auth) | **Self-hosted Supabase** (official docker compose) |
| AWS S3 | **MinIO** (or any S3-compatible store) |
| Railway (api / worker / mcp / converter) | **docker compose** (`deploy/docker-compose.selfhost.yml`) |
| Netlify (web) | Docker image built from `web/Dockerfile` |

Nothing needs to be deleted from the codebase: local mode simply never runs,
and the Railway/Netlify config files are inert.

> **Scope note.** The environment variable names, ports, and service contracts
> below are taken directly from the code (`api/config.py`, `mcp/config.py`,
> `converter/main.py`, the Dockerfiles). The compose file is a starting
> skeleton — review resource limits, secrets handling, and networking for your
> environment before production use.

For system boundaries and storage ownership, start with the
[platform architecture overview](architecture/overview.md). The canonical
runtime contracts are [durable jobs](architecture/durable-jobs.md),
[retrieval](architecture/retrieval.md), and
[server-side RAG](architecture/server-rag.md), and
[bounded read models](architecture/read-models.md); this guide keeps only deployable
operator steps.

---

## Architecture

```
                       ┌──────────────── reverse proxy (TLS) ────────────────┐
  browsers ──────────► │ app.example.com      → web:3000        (Next.js)    │
  Agent (MCP) ───────► │ mcp.example.com/mcp  → mcp:8080/mcp    (FastMCP)    │
  browsers ──────────► │ api.example.com      → gateway:8000    (Nginx)      │
  browsers ──────────► │ s3.example.com       → minio:9000      (presigned)  │
                       │ supabase.example.com → kong:8000       (/auth/v1/*) │
                       └─────────────────────────────────────────────────────┘
                                    │
        gateway ──► api:8000 (two or more independent FastAPI replicas)
        api/worker ──► redis:6379 (internal only; AOF enabled)
        worker ──► converter:8000 (internal only; bearer-authenticated)
        api/worker/mcp ──► Supabase Postgres (RLS + LISTEN/NOTIFY + PGroonga + pgvector)
        api/mcp ──► MinIO (S3 API)
```

Five public hostnames (subpaths behind one hostname also work if you adjust
the URLs consistently). The converter must **not** be exposed publicly.
The shown two-API/two-worker shape is the required recovery-smoke topology, not
a production replica mandate; size each role independently and keep the gateway
as the only published API listener.

---

## 1. Prerequisites

- Docker + docker compose on the host(s)
- A domain with DNS for the five hostnames, TLS certificates (Let's Encrypt)
- ~4 GB RAM minimum for the app services; the converter (LibreOffice + JVM)
  spikes during Office/PDF extraction — give it 2 GB of its own

## 2. Supabase (Postgres + Auth)

The app authenticates with Supabase-issued JWTs (verified against the stack's
JWKS by `api/auth.py` and `mcp/auth.py`) and the web app logs in through
`@supabase/supabase-js`. The lowest-friction path is Supabase's official
self-hosting compose — you do **not** need most of its services.

1. Follow https://supabase.com/docs/guides/self-hosting/docker. Generate fresh
   `JWT_SECRET`, `ANON_KEY`, `SERVICE_ROLE_KEY` per their instructions.
2. Required services: **db** (Postgres — the image ships PGroonga), **auth**
   (GoTrue), **kong** (gateway serving `/auth/v1/*`), and **studio** if you
   want the admin UI. `rest`, `realtime`, `storage`, `imgproxy`, and
   `functions` are unused by this app and can be disabled.
   The database image must provide pgvector 0.8.x (the CI/self-host profile
   pins `pgvector/pgvector:0.8.0-pg16`); enable the `vector` extension before
   hybrid retrieval is used. PGroonga remains required for lexical search.
3. In GoTrue, configure your signup policy (email/password works out of the
   box; disable open signups if this is an internal platform and invite users
   from Studio instead). Email/password is the only login method — no OAuth
   provider configuration is needed (see [Notes on auth](#notes-on-auth)).
4. Record: the Kong URL (this is `SUPABASE_URL`), the `ANON_KEY`, and the
   Postgres connection string.

### Apply the migrations

Run `supabase/migrations/001…016` in order against the stack's database:

```bash
for f in supabase/migrations/*.sql; do
  psql "$DATABASE_URL" -f "$f"
done
```

They create the schema, RLS policies, PGroonga full-text indexes, versioned
pgvector chunk storage, durable embedding jobs, server-side RAG runs, bounded
read revisions/indexes, the
`document_changes` NOTIFY trigger, and the `auth.users` trigger that
provisions a `public.users` row (with page/storage quotas) on signup — which
is why this must run on the Supabase database, not a bare Postgres.

## 3. MinIO

Included in the compose file, with an init job that creates the bucket and an
app-scoped access key. Compose pins the server to
`RELEASE.2025-04-22T22-12-26Z` and its compatible `mc` client to
`RELEASE.2025-04-16T18-13-26Z` so deployments do not silently change when a
new image is published. Two things matter:

- **Browsers fetch presigned URLs directly**, so MinIO's S3 port must be
  publicly reachable — put `minio:9000` behind your proxy as
  `https://s3.example.com` and set `S3_ENDPOINT_URL` to that public URL.
  (The backend containers will use the same URL; ensure it resolves from
  inside the compose network, or use split-horizon DNS.)
- **Path-style addressing** is on (`S3_FORCE_PATH_STYLE=true` in the compose
  file) — MinIO needs it unless you configure wildcard DNS.

Any other S3-compatible store (Ceph RGW, cloud object storage with an S3 API)
works the same way via `S3_ENDPOINT_URL`.

## 4. Application services

```bash
cp deploy/.env.selfhost.example deploy/.env.selfhost
# fill in every CHANGE-ME and URL
docker compose -f deploy/docker-compose.selfhost.yml --env-file deploy/.env.selfhost up -d --build
```

What each service needs (full matrix in the compose file):

| Service | Key env | Notes |
|---|---|---|
| **gateway** :8000 | no secrets | The only host-published API port. It resolves the current Compose `api` replicas through Docker DNS and forwards HTTP, TUS bodies, and WebSocket upgrades. |
| **api** :8000 (internal) | `MODE=hosted`, `DATABASE_URL`, `DIRECT_DATABASE_URL`, `SUPABASE_URL`, `APP_URL`, S3 vars, `REDIS_URL` | `APP_URL` is the CORS allowlist. `DIRECT_DATABASE_URL` must be non-pooled: each replica owns one LISTEN/NOTIFY socket and one local WebSocket manager. Each durable replica also owns exactly one Redis client. |
| **worker** (internal) | same DB/Redis/S3/converter vars as api | Runs durable extraction, graph, and cleanup jobs. Redis transports only job IDs; Postgres remains the job ledger and lease authority. |
| **mcp** :8080 | same DB/S3/Supabase vars + `MCP_URL` | Serves streamable HTTP at `/mcp`; health at `/health`. |
| **converter** :8000 (internal) | `CONVERTER_SECRET`, `S3_BUCKET`, `S3_ENDPOINT` | Refuses to boot without the secret. Its URL allowlist locks presigned downloads to your endpoint + bucket. |
| **web** :3000 | build args: `NEXT_PUBLIC_MODE=hosted`, `NEXT_PUBLIC_API_URL`, `NEXT_PUBLIC_MCP_URL`, `NEXT_PUBLIC_SUPABASE_URL`, `NEXT_PUBLIC_SUPABASE_ANON_KEY` | These are **baked in at build time** — changing any of them means rebuilding the image. |

Redis-backed multipart sessions and the Postgres job ledger make API and
worker replica counts independent. Start with two of each:

```bash
docker compose -f deploy/docker-compose.selfhost.yml --env-file deploy/.env.selfhost \
  up -d --build --scale api=2 --scale worker=2
docker compose -f deploy/docker-compose.selfhost.yml --env-file deploy/.env.selfhost \
  restart gateway
```

### Optional server-side RAG

Server-side RAG is Hosted-only and defaults off. Apply migration `015`, then
inject the same non-secret profile mapping and separate provider-key mapping
into API and worker. Start with `SERVER_RAG_ENABLED=false`; enable it only after
the deterministic fake-model and scaled smoke gates pass. The LLMWiki REST
token used by operators is not a provider key.

The Compose example forwards `SERVER_RAG_ENABLED`,
`RAG_MODEL_PROFILES_JSON`, and `RAG_MODEL_API_KEYS_JSON` to API and worker. The
example env file intentionally leaves the mappings empty. Load real values
from your deployment secret manager rather than committing them. Migration,
profile schema, budgets, REST/CLI usage, recovery, privacy, cohort rollout, and
the exact flag-only rollback are documented in
[`docs/architecture/server-rag.md`](architecture/server-rag.md).

Restarting `gateway` after every replica-count change drops cached upstream
connections and makes the new Docker DNS task set effective immediately.
Each worker accepts up to 10 concurrent ARQ jobs; prefer adding worker
replicas before increasing per-process concurrency because PDF and Office
extraction can consume substantial memory. Its Postgres pool has 12
connections, reserving two beyond handler concurrency for lease heartbeats
and dispatcher/reaper cron scans.

## 5. Reverse proxy

nginx example (repeat the `server` block per hostname; certbot/caddy as you
prefer):

```nginx
# api.example.com; the Compose gateway is the upstream load balancer
server {
    listen 443 ssl http2;
    server_name api.example.com;
    client_max_body_size 65m;             # bounded TUS PATCH size

    location /v1/ws/ {                    # WebSocket live updates
        proxy_pass http://127.0.0.1:8000;  # gateway, never a specific api replica
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_read_timeout 300s;          # server pings every 30s
    }
    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Forwarded-Proto https;
    }
}

# app.example.com  → 127.0.0.1:3000
# mcp.example.com  → 127.0.0.1:8080   (proxy_read_timeout 300s; MCP streams responses)
# s3.example.com   → 127.0.0.1:9000   (client_max_body_size 0; presigned PUT/GET)
# supabase.example.com → your Supabase kong port
```

## 6. Verification checklist

```bash
curl -fsS https://api.example.com/health            # {"status":"ok"} — 进程存活
curl -fsS https://api.example.com/ready             # Postgres + Redis + S3 ready
curl -fsS https://mcp.example.com/health            # ok
curl -fsS https://supabase.example.com/auth/v1/.well-known/jwks.json | head -c 200
```

Then end-to-end: sign up in the web app → create a wiki → upload a PDF
(exercises TUS → MinIO → converter → chunking) → watch it turn "ready"
without a page reload (exercises LISTEN/NOTIFY → WebSocket) → search for a
term from the PDF (exercises PGroonga) → connect an MCP agent and run the
`guide` tool.

For large Local workspaces, never benchmark the live database in place. Make a
copy-on-write clone, start an isolated Local API on another port, and run
`scripts/benchmark_read_models.py` against that copy. The acceptance thresholds
are 200 items, 1 MiB, 200 ms for warm pages, 50 ms for ETag `304`, and no
temporary SQLite ORDER BY sort; the full safe procedure is in
[`architecture/read-models.md`](architecture/read-models.md).

### Optional hybrid retrieval

Lexical search remains the default and needs no embedding service. To make the
hosted hybrid profile available, provide the same values to API, worker, and
MCP, then restart/roll all three roles:

```dotenv
HYBRID_SEARCH_ENABLED=true
EMBEDDING_PROVIDER=openai_compatible
EMBEDDING_BASE_URL=https://embeddings.example.com/v1
EMBEDDING_API_KEY=replace-with-secret
EMBEDDING_MODEL=text-embedding-model
EMBEDDING_DIMENSIONS=1536
EMBEDDING_BATCH_SIZE=32
EMBEDDING_TIMEOUT_SECONDS=30
HYBRID_LEXICAL_CANDIDATES=50
HYBRID_VECTOR_CANDIDATES=50
HYBRID_RRF_K=60
```

The Compose skeleton does not forward these optional values yet. Add explicit
`${VARIABLE}` mappings for every value above to the `environment` section of
the `api`, `worker`, and `mcp` services (or inject the same values through your
orchestrator). Values present only in `deploy/.env.selfhost` are available for
Compose substitution but are not automatically copied into containers.

Startup validation rejects hybrid in local mode, hybrid without durable jobs,
blank endpoint/model values, invalid dimensions, and out-of-range numeric
settings. Migrations `013` and `014` must already be applied, and the database
must provide pgvector 0.8.x in addition to PGroonga.

Newly extracted document versions enqueue durable embedding work. Backfill an
existing corpus, or re-embed after changing model/dimensions, while workers are
running:

```bash
PYTHONPATH=api MODE=hosted .venv/bin/python -m scripts.enqueue_embeddings \
  --missing --page-size 100
```

Repeat after the queue settles until it returns
`{"enqueued":0,"scanned":0}`. Vectors are fenced by document version and the
exact provider/model/dimensions tuple, so a model change cannot mix profiles.
Run the supported private Postgres comparison from a deployment shell that has
the same embedding and `HYBRID_*` values. Load `DATABASE_URL` and API keys from
your secret store; do not put either one on the command line:

```bash
: "${DATABASE_URL:?load DATABASE_URL from the deployment secret store}"
export HYBRID_SEARCH_ENABLED=true
export RETRIEVAL_EVAL_USER_ID=00000000-0000-0000-0000-000000000001
export RETRIEVAL_EVAL_KNOWLEDGE_BASE_ID=00000000-0000-0000-0000-000000000002
PYTHONPATH=api MODE=hosted .venv/bin/python -m scripts.retrieval_eval \
  --dataset /controlled/private-retrieval-cases.jsonl \
  --compare --hosted --require-promotion-gate \
  --output-json /controlled/private-retrieval-report.json
```

Before changing model or dimensions, stop new hybrid opt-ins and leave serving
entry points on lexical. Roll API producers and workers together onto the new
profile, reconcile until that exact profile has complete current-version
coverage, and run the command above from a separately configured evaluator.
Only after exit `0` and operator review should serving processes receive the
new profile and selected canary callers resume hybrid. Exit `2` or `3`, or
incomplete coverage, leaves serving hybrid disabled; restoring the previously
tested profile is safe because old profile rows remain isolated and intact.

A passing gate never changes the lexical default automatically. Never commit a
production evaluation corpus or report containing user data. The full
operational and promotion contract is in
[`docs/architecture/retrieval.md`](architecture/retrieval.md).

## 7. Operations

- **Backups**: `pg_dump` the Supabase database, mirror the MinIO bucket
  (`mc mirror`), and preserve the `redis-data` volume. Postgres is the job and
  business source of truth; S3 holds source/derived objects; Redis AOF holds
  dispatch hints and active multipart coordination state.
- **Redis AOF**: Compose pins Redis 7.4.2 with `appendonly yes` and
  `appendfsync everysec`.
  Before a backup, run `docker compose -f deploy/docker-compose.selfhost.yml
  --env-file deploy/.env.selfhost exec redis redis-cli BGREWRITEAOF`, wait for
  `aof_rewrite_in_progress:0`, then snapshot the `redis-data` volume together
  with Postgres and S3. To restore, stop API/workers/Redis, restore the
  volume's `/data/appendonlydir`, start Redis, verify `redis-cli ping`, then
  start workers and APIs. At most the last second of Redis hints may be
  absent; Postgres dispatch/reaper scans reconstruct durable work.
- **Role readiness**: `/health` is process liveness and deliberately ignores
  temporary dependency failures. Hosted API `/ready` checks `SELECT 1`, Redis
  `PING`, S3 `head_bucket`, and the current Postgres LISTEN subscription under
  a bounded timeout without returning raw exception text. Local `/ready`
  checks SQLite only. Workers have no HTTP port: startup requires a non-empty
  `CONVERTER_SECRET`, then performs the same three dependency checks plus the
  converter's anonymous `/health` under one bounded startup timeout and accepts
  no jobs until all pass. The preflight does not send `CONVERTER_SECRET` and
  cannot detect a converter-secret mismatch because the
  converter exposes no separate auth-safe validation endpoint; authenticated
  job calls still carry the secret and fail through the normal classifier. A
  later dependency loss likewise never acknowledges the job as a success.
- **Recovery behaviors**: every API replica has a supervised Postgres LISTEN
  task that reconnects with backoff. Workers reap expired leases so another
  replica safely retries interrupted work.
- **Quotas**: per-user page/storage limits are columns on `public.users`
  (defaults 500 pages / 1 GiB, set by the signup trigger); adjust in SQL.
  `GLOBAL_MAX_USERS` caps registrations at KB-creation time.
- **Timeliness**: `M1` corpus entries want frequent review — schedule a
  nightly agent routine against the MCP server and use
  `search(mode="references", query="due")` as its worklist.

### Zero-downtime rollout and rollback

Apply migrations through `016` before deploying application code; these
migrations are additive for the rollout. Then update Redis/MinIO/converter,
roll workers, roll API replicas one at a time, and update/restart the gateway
last. Confirm `/ready`, one real upload, and one graph rebuild before removing
old containers.

`DURABLE_JOBS_ENABLED=true` and `TUS_MULTIPART_ENABLED=true` are required in
Hosted mode. Current binaries fail fast if either is false; the process-local
Hosted rollback path has been removed. Roll back API and worker together to a
known-good release. Database migrations are additive, so leave the schema,
Redis AOF, and objects intact. Never mix API producers and workers from
different releases.

Hybrid retrieval has an independent, data-preserving rollback: set
`HYBRID_SEARCH_ENABLED=false` for API, worker, and MCP and roll/restart those
roles. All callers return to lexical behavior; leave migrations `013`/`014`,
embedding rows, and job history in place. Passing the promotion gate never
flips this flag or changes the default profile automatically.

Server-side RAG has a separate flag-only rollback: set
`SERVER_RAG_ENABLED=false` for API and worker and roll/restart both roles. New
create/resume work is rejected and running work stops at the next page
boundary. Leave migration `015`, committed wiki versions, partial-run rows, and
step traces intact; do not disable durable jobs as part of this rollback.

Copyable incident rollback (set `ROLLBACK_REF` to a tested tag or commit):

```bash
(
  set -e
  : "${ROLLBACK_REF:?set ROLLBACK_REF to a tested release}"
  rollback_dir="$(mktemp -d)"
  cleanup() { rm -rf "$rollback_dir"; }
  trap cleanup EXIT
  git archive "$ROLLBACK_REF" | tar -x -C "$rollback_dir"
  docker compose -f "$rollback_dir/deploy/docker-compose.selfhost.yml" \
    --env-file "$(pwd)/deploy/.env.selfhost" up -d --build --force-recreate \
    --scale api=1 --scale worker=1 api worker gateway
)
```

### Security notes for an internal deployment

- Keep the converter and Postgres off the public network; only the five
  proxy-fronted hostnames should be reachable.
- The API's rate limiter keys on an **unverified** JWT `sub` claim (a known
  upstream weakness — one IP can mint fresh buckets). Add per-IP limits at
  the proxy (`limit_req`) rather than trusting it.
- Set `SENTRY_DSN` only if you run your own Sentry —
  the default sends nothing anywhere.
- The web bundle contains the Supabase `ANON_KEY`; that is by design (it is
  RLS-scoped), but make sure you generated fresh keys and never expose the
  `SERVICE_ROLE_KEY`.

## Notes on auth

- **Email/password** login works with stock GoTrue and is the only login
  method — Google OAuth has been removed from the web app, so
  no external identity provider is involved.
- **MCP and API access use API keys** — no OAuth-capable auth server is
  required. Each user creates a key in **Settings → Connect AI Assistant (MCP)**;
  the generated config carries it as a static `Authorization: Bearer sv_…`
  header, which both the MCP server and the REST API verify against its
  stored SHA-256 hash (revocable in Settings, `last_used_at` tracked).
  Supabase JWTs are also accepted everywhere, so the web app is unaffected.
- **Connecting agents** (desktop MCP clients, Codex CLI, Hermes, OpenClaw,
  or any other MCP client) to this deployment — including per-client config
  and headless key creation — is covered in
  [`docs/agent-integration.md`](agent-integration.md).

## Importing the corpus (八维标注)

The corpus importer writes directly into the hosted database — entries land
as documents + search chunks in the target user's knowledge base, so facet
search, the web corpus browser, lint, and relations all work immediately:

```bash
python3 -m corpus.import_annotations \
    --csv 标注结果/标注明细_业务视图.csv \
    --database-url "$DATABASE_URL" \
    --user-email corpus-admin@example.com \
    --kb goglobal-corpus \
    --raw 审核结果_deepseek/收录
```

The account must exist (sign in once first); the knowledge base is created on
first import. Re-imports are idempotent.

### Automated classification (pipeline CLI)

Instead of importing pre-annotated CSVs, the pipeline CLI can audit +
classify raw text documents already sitting in a hosted knowledge base
(uploaded via the app or MCP), using any OpenAI-compatible LLM endpoint:

```bash
python3 -m corpus.pipeline \
    --database-url "$DATABASE_URL" \
    --user-email corpus-admin@example.com \
    --kb goglobal-corpus \
    --base-url https://api.deepseek.com/v1 --model deepseek-chat --api-key "$KEY"
```

State lives in the `corpus_pipeline` table (migration 010): re-runs are
idempotent, failures retry up to 3 times, entries land with search chunks
in one transaction. Schedule it with cron for continuous ingestion;
`--mock` runs a rule-stub end-to-end test without an LLM.
 S3 is not involved: corpus entries
are markdown and hosted mode stores text content in Postgres — S3 only holds
binary sources, which the annotation pipeline does not produce.

## Scaling verification

With `STAGE=test` and the opt-in `SCALED_TEST_*` variables in
`deploy/.env.selfhost`, run from the repository root. The pytest fixture
safely reads `deploy/.env.selfhost` without sourcing or printing its secrets:

```bash
(
  set -e

  compose() {
    docker compose -f deploy/docker-compose.selfhost.yml --env-file deploy/.env.selfhost "$@"
  }
  cleanup() {
    compose down -v
  }
  trap cleanup EXIT

  compose up -d --build --scale api=2 --scale worker=2
  compose restart gateway
  SCALED_COMPOSE_TEST=1 PYTHONPATH=api .venv/bin/pytest \
    tests/integration/test_scaled_compose.py -q
)
```

The live test requires a reachable self-hosted Supabase database and a valid
JWT/user ID; without those explicit inputs it is skipped rather than reporting
a false pass. Always run `down -v` after success or failure of the disposable
smoke stack.
