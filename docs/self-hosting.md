# Self-Hosting Guide (multi-user hosted mode)

This guide deploys the full multi-user stack on your own infrastructure — the
same code that runs llmwiki.app, with the SaaS pieces swapped for self-hosted
equivalents:

| Managed piece | Self-hosted replacement |
|---|---|
| Supabase cloud (Postgres + Auth) | **Self-hosted Supabase** (official docker compose) |
| AWS S3 | **MinIO** (or any S3-compatible store) |
| Railway (api / mcp / converter) | **docker compose** (`deploy/docker-compose.selfhost.yml`) |
| Netlify (web) | Docker image built from `web/Dockerfile` |

Nothing needs to be deleted from the codebase: local mode simply never runs,
and the Railway/Netlify config files are inert.

> **Scope note.** The environment variable names, ports, and service contracts
> below are taken directly from the code (`api/config.py`, `mcp/config.py`,
> `converter/main.py`, the Dockerfiles). The compose file is a starting
> skeleton — review resource limits, secrets handling, and networking for your
> environment before production use.

---

## Architecture

```
                       ┌──────────────── reverse proxy (TLS) ────────────────┐
  browsers ──────────► │ app.example.com      → web:3000        (Next.js)    │
  Claude (MCP) ──────► │ mcp.example.com/mcp  → mcp:8080/mcp    (FastMCP)    │
  browsers + ext ────► │ api.example.com      → api:8000        (FastAPI)    │
  browsers ──────────► │ s3.example.com       → minio:9000      (presigned)  │
                       │ supabase.example.com → kong:8000       (/auth/v1/*) │
                       └─────────────────────────────────────────────────────┘
                                    │
        api ──► converter:8000 (internal only; bearer-authenticated)
        api/mcp ──► Supabase Postgres (RLS + LISTEN/NOTIFY + PGroonga)
        api/mcp ──► MinIO (S3 API)
```

Five public hostnames (subpaths behind one hostname also work if you adjust
the URLs consistently). The converter must **not** be exposed publicly.

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
3. In GoTrue, configure your signup policy (email/password works out of the
   box; disable open signups if this is an internal platform and invite users
   from Studio instead). Email/password is the only login method — no OAuth
   provider configuration is needed (see [Notes on auth](#notes-on-auth)).
4. Record: the Kong URL (this is `SUPABASE_URL`), the `ANON_KEY`, and the
   Postgres connection string.

### Apply the migrations

Run `supabase/migrations/001…009` in order against the stack's database:

```bash
for f in supabase/migrations/*.sql; do
  psql "$DATABASE_URL" -f "$f"
done
```

They create the schema, RLS policies, PGroonga full-text indexes, the
`document_changes` NOTIFY trigger, and the `auth.users` trigger that
provisions a `public.users` row (with page/storage quotas) on signup — which
is why this must run on the Supabase database, not a bare Postgres.

## 3. MinIO

Included in the compose file, with an init job that creates the bucket and an
app-scoped access key. Two things matter:

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
| **api** :8000 | `MODE=hosted`, `DATABASE_URL`, `DIRECT_DATABASE_URL`, `SUPABASE_URL`, `APP_URL`, S3 vars, `CONVERTER_URL`, `CONVERTER_SECRET` | `APP_URL` is the CORS allowlist — it must exactly match the web origin. `DIRECT_DATABASE_URL` must be a non-pooled connection: poolers silently kill the LISTEN/NOTIFY socket that powers live updates. |
| **mcp** :8080 | same DB/S3/Supabase vars + `MCP_URL` | Serves streamable HTTP at `/mcp`; health at `/health`. |
| **converter** :8000 (internal) | `CONVERTER_SECRET`, `S3_BUCKET`, `S3_ENDPOINT` | Refuses to boot without the secret. Its URL allowlist locks presigned downloads to your endpoint + bucket. |
| **web** :3000 | build args: `NEXT_PUBLIC_MODE=hosted`, `NEXT_PUBLIC_API_URL`, `NEXT_PUBLIC_MCP_URL`, `NEXT_PUBLIC_SUPABASE_URL`, `NEXT_PUBLIC_SUPABASE_ANON_KEY` | These are **baked in at build time** — changing any of them means rebuilding the image. |

**Run exactly one `api` replica.** Resumable-upload (TUS) state, the WebSocket
connection manager, and graph rebuild locks are process-local; horizontal
scaling requires externalizing them first. One replica of each of the other
services is also the sane default.

## 5. Reverse proxy

nginx example (repeat the `server` block per hostname; certbot/caddy as you
prefer):

```nginx
# api.example.com
server {
    listen 443 ssl http2;
    server_name api.example.com;
    client_max_body_size 110m;            # TUS upload chunks (100 MB cap + headroom)

    location /v1/ws/ {                    # WebSocket live updates
        proxy_pass http://127.0.0.1:8000;
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

## 6. Chrome extension (optional)

The published extension points at llmwiki.app. For your deployment, rebuild it
with your endpoints (`extension/.env`):

```
VITE_API_BASE_URL=https://api.example.com
VITE_SUPABASE_URL=https://supabase.example.com
VITE_SUPABASE_ANON_KEY=eyJ...
```

`cd extension && npm ci && npm run zip`, then distribute the zip via your
organization's Chrome policy (or load unpacked for testing).

## 7. Verification checklist

```bash
curl -fsS https://api.example.com/health            # {"status":"ok"}
curl -fsS https://mcp.example.com/health            # ok
curl -fsS https://supabase.example.com/auth/v1/.well-known/jwks.json | head -c 200
```

Then end-to-end: sign up in the web app → create a wiki → upload a PDF
(exercises TUS → MinIO → converter → chunking) → watch it turn "ready"
without a page reload (exercises LISTEN/NOTIFY → WebSocket) → search for a
term from the PDF (exercises PGroonga) → connect Claude via MCP and run the
`guide` tool.

## 8. Operations

- **Backups**: `pg_dump` the Supabase database + mirror the MinIO bucket
  (`mc mirror`). The database is the source of truth in hosted mode; S3 holds
  original files, page images, and assets.
- **Recovery behaviors already built in**: documents stuck in `processing`
  are re-fired at api boot; the LISTEN loop reconnects with backoff.
- **Quotas**: per-user page/storage limits are columns on `public.users`
  (defaults 500 pages / 1 GiB, set by the signup trigger); adjust in SQL.
  `GLOBAL_MAX_USERS` caps registrations at KB-creation time.
- **Timeliness**: `M1` corpus entries want frequent review — schedule a
  nightly Claude routine against the MCP server and use
  `search(mode="references", query="due")` as its worklist.

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
  method — Google OAuth has been removed from the web app and extension, so
  no external identity provider is involved.
- **MCP and API access use API keys** — no OAuth-capable auth server is
  required. Each user creates a key in **Settings → Connect Claude (MCP)**;
  the generated config carries it as a static `Authorization: Bearer sv_…`
  header, which both the MCP server and the REST API verify against its
  stored SHA-256 hash (revocable in Settings, `last_used_at` tracked).
  Supabase JWTs are also accepted everywhere, so the web app is unaffected.
- **Connecting agents** (Claude Desktop/Code, Codex CLI, Hermes, OpenClaw,
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
first import. Re-imports are idempotent. S3 is not involved: corpus entries
are markdown and hosted mode stores text content in Postgres — S3 only holds
binary sources, which the annotation pipeline does not produce.

## Known limitations

- **Single-instance api** (see above).
