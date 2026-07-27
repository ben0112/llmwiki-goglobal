# Current-state Documentation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace historical, process-oriented documentation with a verified current-state architecture set, retain concise design rationale, and remove every completed implementation plan.

**Architecture:** `README.md` becomes the navigation entry point, `docs/architecture/overview.md` explains the whole platform, and four domain architecture documents become canonical contracts. Deployment and agent guides remain task-oriented, retained specifications describe the as-built design, and completed plans are removed after durable evidence and links are migrated.

**Tech Stack:** Markdown, Git, ripgrep, Python 3 link validation, pytest static contracts, GitHub Actions

---

## File responsibility map

- `README.md`: product entry point, supported modes, capability status, and links to canonical documentation.
- `docs/architecture/overview.md`: end-to-end layers, storage ownership, data flow, deployment topology, and milestone status.
- `docs/architecture/shared-kernel.md`: cross-runtime types and data invariants.
- `docs/architecture/durable-jobs.md`: Hosted durable work, leases, recovery, and scaling contract.
- `docs/architecture/retrieval.md`: lexical baseline, evaluation, hybrid retrieval, fallback, and promotion contract.
- `docs/architecture/server-rag.md`: model profiles, bounded orchestration, recovery, rollout, and privacy contract.
- `docs/self-hosting.md`: executable Hosted deployment and operations guide.
- `docs/agent-integration.md`: MCP, REST, durable-job, retrieval, and RAG client boundaries.
- `docs/superpowers/specs/2026-07-24-platform-architecture-evolution-design.md`: current platform design and milestone decisions.
- `docs/superpowers/specs/2026-07-25-durable-jobs-api-scaling-design.md`: current durable-job and multi-replica design rationale.
- `docs/superpowers/specs/2026-07-26-server-rag-orchestration-design.md`: current server-RAG design rationale.
- `docs/superpowers/specs/2026-07-27-current-state-documentation-design.md`: approved documentation architecture.
- `docs/superpowers/plans/*.md`: temporary execution records; all are removed when this work is complete.

## Task 1: Add the platform overview and README navigation

**Files:**
- Create: `docs/architecture/overview.md`
- Modify: `README.md`

- [ ] **Step 1: Capture the failing navigation contract**

Run:

```bash
test -f docs/architecture/overview.md
rg -n "architecture/overview\.md" README.md
```

Expected: the file check fails because `overview.md` does not exist, and the
README search returns no match.

- [ ] **Step 2: Write the current-state architecture overview**

Create `docs/architecture/overview.md` with these exact top-level sections and
content boundaries:

```markdown
# Platform architecture overview

## Capability status
```

State that all four milestones are implemented on
`feat/platform-architecture-evolution`, while hybrid retrieval and server-side
RAG remain independently gated and default off.

```markdown
## Layers and dependency direction
```

Describe `llmwiki_core` as the dependency-free contract, adapters as storage
and remote integrations, API/MCP as transports, and workers as durable Hosted
execution. Include a compact Mermaid flow from client to API/MCP, shared core,
adapters, and Local/Hosted storage.

```markdown
## Runtime modes
```

Contrast Local SQLite/filesystem execution with Hosted Postgres, Redis, MinIO,
stateless API replicas, and durable workers. State that server-side RAG is
Hosted-only and the MCP workflow remains the offline path.

```markdown
## Storage ownership
```

Map SQLite, Postgres, Redis, MinIO, PGroonga, and pgvector to their current
responsibilities. Explicitly state that Postgres is the authority for Hosted
jobs, RAG runs, wiki versions, and derived document state.

```markdown
## Request and data flows
```

Document synchronous reads, durable document processing, embedding lifecycle,
and server-RAG page commits. Link each flow to its canonical domain document.

```markdown
## Scaling and failure boundaries
```

Describe stateless APIs, lease-owned workers, idempotency, page-boundary RAG
recovery, and lexical fallback. Do not claim that a process-local cache is a
source of truth.

```markdown
## Documentation map
## Verification baseline
```

Link the four domain contracts, self-hosting guide, agent guide, and retained
specifications. Record the last verified pre-rewrite publication SHA
`0173f560c6fec03b87ce4f6803f663d2d6983ead` and Actions run `30251808426` as a
historical baseline; state that the rewrite's final exact-SHA run is the
publication gate reported at handoff.

- [ ] **Step 3: Add README architecture navigation**

Add a concise section after “与上游的差异”:

```markdown
## 架构与运维文档

- [总体架构](docs/architecture/overview.md)：分层、数据流、存储职责与部署拓扑。
- [共享内核与数据不变量](docs/architecture/shared-kernel.md)
- [持久任务与多副本 Hosted 部署](docs/architecture/durable-jobs.md)
- [检索评测与混合召回](docs/architecture/retrieval.md)
- [服务端 RAG](docs/architecture/server-rag.md)
- [自托管指南](docs/self-hosting.md) · [智能体接入](docs/agent-integration.md)
```

Keep detailed CLI and setup sections in place; remove only duplicated status or
architecture prose that the new overview owns.

- [ ] **Step 4: Verify navigation and terminology**

Run:

```bash
test -f docs/architecture/overview.md
rg -n "architecture/overview\.md" README.md
test "$(rg -c '^## (Capability status|Layers and dependency direction|Runtime modes|Storage ownership|Request and data flows|Scaling and failure boundaries|Documentation map|Verification baseline)$' docs/architecture/overview.md)" -eq 8
git diff --check
```

Expected: the file and README link exist, all eight overview sections match,
and `git diff --check` is silent.

- [ ] **Step 5: Commit the overview**

```bash
git add README.md docs/architecture/overview.md
git commit -m "docs: add platform architecture overview"
```

## Task 2: Make the four domain documents canonical

**Files:**
- Modify: `docs/architecture/shared-kernel.md`
- Modify: `docs/architecture/durable-jobs.md`
- Modify: `docs/architecture/retrieval.md`
- Modify: `docs/architecture/server-rag.md`

- [ ] **Step 1: Record missing current-state sections**

Run:

```bash
rg -L '^## (Configuration|Verification evidence|Rollout and rollback)' docs/architecture/shared-kernel.md docs/architecture/durable-jobs.md
rg -n "will implement|planned|single[- ]replica|001…014|001\.\.014" docs/architecture/*.md || true
```

Expected: at least `shared-kernel.md` and `durable-jobs.md` are reported as
missing one or more canonical sections. Any second-command match becomes an
explicit rewrite target.

- [ ] **Step 2: Rewrite the shared-kernel contract**

Keep its existing invariant detail and add or normalize these sections:

```markdown
## Scope and dependency rule
## Identities, paths, hashes, and citations
## Lifecycle and version invariants
## Local and Hosted transaction boundaries
## Compatibility facades
## Verification evidence
```

Use present tense. State that shared core imports no transport or database
implementation, `ready` implies current derived state, writes preserve version
and citation identities, Local repairs happen through the shared contract, and
Hosted multi-row writes commit atomically. Move durable-job, retrieval, and RAG
details to links.

- [ ] **Step 3: Rewrite the durable-job contract**

Normalize it around:

```markdown
## Operational contract
## Configuration and topology
## Job states, leases, and compare-and-swap rules
## Dispatch, cancellation, and recovery
## Upload and object-storage coordination
## Security and observability
## Rollout and rollback
## Verification evidence
```

State that Postgres is authoritative, Redis/ARQ dispatch is not the job ledger,
API replicas are stateless, workers are interchangeable after lease expiry,
and document extraction, graph rebuild, embeddings, and server RAG reuse the
same durable substrate. Preserve exact recovery and tenant-isolation rules.

- [ ] **Step 4: Normalize retrieval and server-RAG cross-links**

In `retrieval.md`, keep lexical as the default, clarify that hybrid requires
Hosted durable embedding coverage and explicit promotion, and link to
`overview.md`, `durable-jobs.md`, and `server-rag.md` without duplicating their
operational sections.

In `server-rag.md`, preserve budgets, endpoint contracts, atomic page commits,
resume rules, secret separation, and flag-only rollback. Add links back to the
overview, shared kernel, durable jobs, and retrieval contract. Keep
`f4c22afa39c249e000ac5becf10d8bfb212be75c` and Actions run `30251148719` as
the substantive server-RAG evidence, not as the final documentation SHA.

- [ ] **Step 5: Verify canonical section ownership**

Run:

```bash
rg -n '^## Verification evidence$' docs/architecture/shared-kernel.md docs/architecture/durable-jobs.md docs/architecture/retrieval.md docs/architecture/server-rag.md
rg -n "overview\.md" docs/architecture/shared-kernel.md docs/architecture/durable-jobs.md docs/architecture/retrieval.md docs/architecture/server-rag.md
rg -n "single[- ]replica|001…014|001\.\.014|rag_page_conflict" docs/architecture/*.md && exit 1 || true
git diff --check
```

Expected: each domain document has verification evidence and an overview link;
the stale-claim scan produces no matches; diff check is silent.

- [ ] **Step 6: Commit the canonical domain documents**

```bash
git add docs/architecture/shared-kernel.md docs/architecture/durable-jobs.md \
  docs/architecture/retrieval.md docs/architecture/server-rag.md
git commit -m "docs: align architecture domain contracts"
```

## Task 3: Align deployment and client integration guides

**Files:**
- Modify: `docs/self-hosting.md`
- Modify: `docs/agent-integration.md`

- [ ] **Step 1: Capture the integration-guide gap**

Run:

```bash
rg -n "durable job|/v1/jobs|/v1/rag|server-side RAG|服务端 RAG" docs/agent-integration.md
```

Expected: the current guide does not cover the full REST job and RAG boundary,
so one or more required terms are absent.

- [ ] **Step 2: Refocus the self-hosting guide**

Keep executable deployment instructions and normalize the document around:

- migrations `001` through `015` in order;
- Supabase/Postgres, Redis, MinIO, converter, API, MCP, worker, and gateway
  roles;
- two API and two worker replicas as the required scaling smoke topology, not a
  production replica mandate;
- independent default-off hybrid and server-RAG flags;
- health/readiness and deterministic verification commands; and
- separate hybrid, server-RAG, and release rollback boundaries.

Replace duplicated architecture explanations with links to
`architecture/overview.md` and the relevant domain document.

- [ ] **Step 3: Add the complete client capability matrix**

Update `docs/agent-integration.md` so its opening matrix distinguishes:

| Surface | Local | Hosted | Purpose |
|---|---|---|---|
| MCP tools | yes | yes | interactive search, read, write, and governance |
| REST document/job APIs | no | yes | uploads, durable status, and cancellation |
| REST server-RAG APIs | no | gated | bounded server-owned wiki builds |
| `scripts.rag` CLI | no | gated | operator client over REST |

Add short sections for Hosted durable job observation/cancellation and
server-RAG create/status/steps/resume. Link to the canonical API and operations
documents; do not expose provider keys or imply that MCP clients can submit
model endpoints.

- [ ] **Step 4: Verify operator and client claims**

Run:

```bash
rg -n "001.*015|SERVER_RAG_ENABLED|HYBRID_RETRIEVAL_ENABLED|architecture/overview\.md" docs/self-hosting.md
rg -n "/v1/jobs|/v1/rag|scripts\.rag|architecture/overview\.md" docs/agent-integration.md
rg -n "001…014|001\.\.014" docs/self-hosting.md docs/agent-integration.md && exit 1 || true
rg -n "provider (key|credential).*not|不接受.*provider|never.*provider (key|credential)" docs/self-hosting.md docs/agent-integration.md
git diff --check
```

Expected: current migrations, flags, navigation, job/RAG surfaces, and CLI are
present; stale or unsafe claims are absent; diff check is silent.

- [ ] **Step 5: Commit deployment and integration guidance**

```bash
git add docs/self-hosting.md docs/agent-integration.md
git commit -m "docs: align deployment and agent integration"
```

## Task 4: Rewrite retained specifications as as-built designs

**Files:**
- Modify: `docs/superpowers/specs/2026-07-24-platform-architecture-evolution-design.md`
- Modify: `docs/superpowers/specs/2026-07-25-durable-jobs-api-scaling-design.md`
- Modify: `docs/superpowers/specs/2026-07-26-server-rag-orchestration-design.md`

- [ ] **Step 1: Capture process-oriented content that must disappear**

Run:

```bash
rg -n "Implementation plan|Acceptance criteria|will (add|implement|create|move)|Task [0-9]|RED|GREEN|git commit|pytest" docs/superpowers/specs/2026-07-24-platform-architecture-evolution-design.md docs/superpowers/specs/2026-07-25-durable-jobs-api-scaling-design.md docs/superpowers/specs/2026-07-26-server-rag-orchestration-design.md
```

Expected: one or more future-tense, plan-link, acceptance, or process matches
identify content to rewrite.

- [ ] **Step 2: Rewrite the platform specification**

Use these top-level sections:

```markdown
# Platform Architecture Evolution
## Status
## Goals and non-goals
## Dependency direction
## Shared kernel and data invariants
## Durable Hosted work and scaling
## Retrieval evaluation and hybrid recall
## Server-side RAG orchestration
## Failure, isolation, and security properties
## Rollout boundaries
## Verification and canonical documentation
```

Describe the implemented system in present tense. Keep design reasons and
rejected shortcuts, but replace milestone acceptance checklists and repeated
test commands with links to the overview and four canonical domain documents.

- [ ] **Step 3: Rewrite the durable-jobs specification**

Use these top-level sections:

```markdown
# Durable Jobs and Hosted API Scaling
## Status
## Goals and non-goals
## Selected architecture
## Ledger, dispatch, and leases
## Handler and transaction boundaries
## Resumable uploads
## Multi-replica topology
## Security and failure handling
## Alternatives rejected
## Verification and operations
```

Remove its link to the deleted implementation plan and any supersession
language that forces readers to compare historical drafts. Keep rationale that
is not repeated in `durable-jobs.md` and link operational facts there.

- [ ] **Step 4: Rewrite the server-RAG specification**

Preserve the purpose-built workflow decision, persistent model, budgets,
planning, per-page execution, atomicity, model safety, errors, and rejected DAG
alternative. Consolidate test subsections into:

```markdown
## Verification and operations
```

Link to `docs/architecture/server-rag.md` for current counts, rollout, and
rollback. Remove acceptance checklists, future tense, and process commands.

- [ ] **Step 5: Verify specifications are current-state documents**

Run:

```bash
rg -n "\.\./plans/|Implementation plan|^## Acceptance criteria$|git commit|pytest" docs/superpowers/specs/2026-07-24-platform-architecture-evolution-design.md docs/superpowers/specs/2026-07-25-durable-jobs-api-scaling-design.md docs/superpowers/specs/2026-07-26-server-rag-orchestration-design.md && exit 1 || true
rg -n "architecture/(overview|shared-kernel|durable-jobs|retrieval|server-rag)\.md" docs/superpowers/specs/2026-07-24-platform-architecture-evolution-design.md docs/superpowers/specs/2026-07-25-durable-jobs-api-scaling-design.md docs/superpowers/specs/2026-07-26-server-rag-orchestration-design.md
git diff --check
```

Expected: no plan links, acceptance headings, commit commands, or pytest
commands remain; each specification links to at least one canonical
architecture document; diff check is silent.

- [ ] **Step 6: Commit the rewritten specifications**

```bash
git add docs/superpowers/specs/2026-07-24-platform-architecture-evolution-design.md \
  docs/superpowers/specs/2026-07-25-durable-jobs-api-scaling-design.md \
  docs/superpowers/specs/2026-07-26-server-rag-orchestration-design.md
git commit -m "docs: rewrite architecture specifications as built"
```

## Task 5: Remove the six historical implementation plans

**Files:**
- Delete: `docs/superpowers/plans/2026-07-24-shared-kernel-data-invariants.md`
- Delete: `docs/superpowers/plans/2026-07-25-durable-jobs-api-scaling.md`
- Delete: `docs/superpowers/plans/2026-07-25-durable-jobs-quality-followup.md`
- Delete: `docs/superpowers/plans/2026-07-25-retrieval-evaluation-hybrid.md`
- Delete: `docs/superpowers/plans/2026-07-25-task14-spec-review.md`
- Delete: `docs/superpowers/plans/2026-07-26-server-rag-orchestration.md`

- [ ] **Step 1: Prove every durable fact has a destination**

For each plan, compare its final evidence, rollout, rollback, and design-decision
sections against the corresponding architecture document and retained
specification:

```bash
for f in docs/superpowers/plans/2026-07-24-shared-kernel-data-invariants.md docs/superpowers/plans/2026-07-25-durable-jobs-api-scaling.md docs/superpowers/plans/2026-07-25-durable-jobs-quality-followup.md docs/superpowers/plans/2026-07-25-retrieval-evaluation-hybrid.md docs/superpowers/plans/2026-07-25-task14-spec-review.md docs/superpowers/plans/2026-07-26-server-rag-orchestration.md; do
  rg -n "SHA|Actions|rollback|Rollout|evidence|passed|Ready" "$f" | tail -30
done
```

Expected: all still-current facts are already present in the rewritten
architecture/specification files. If a durable fact is missing, add it to its
canonical destination and rerun the comparison before deletion.

- [ ] **Step 2: Delete the six confirmed plans with `apply_patch`**

Use one patch containing these exact directives:

```text
*** Delete File: docs/superpowers/plans/2026-07-24-shared-kernel-data-invariants.md
*** Delete File: docs/superpowers/plans/2026-07-25-durable-jobs-api-scaling.md
*** Delete File: docs/superpowers/plans/2026-07-25-durable-jobs-quality-followup.md
*** Delete File: docs/superpowers/plans/2026-07-25-retrieval-evaluation-hybrid.md
*** Delete File: docs/superpowers/plans/2026-07-25-task14-spec-review.md
*** Delete File: docs/superpowers/plans/2026-07-26-server-rag-orchestration.md
```

- [ ] **Step 3: Verify no active link targets a deleted plan**

Run:

```bash
rg -n "\]\([^)]*superpowers/plans/|\]\(\.\./plans/" README.md docs --glob '!docs/superpowers/plans/2026-07-27-current-state-documentation.md' && exit 1 || true
find docs/superpowers/plans -maxdepth 1 -type f -print
git diff --check
```

Expected: the link scan is empty, only
`docs/superpowers/plans/2026-07-27-current-state-documentation.md` remains, and
diff check is silent.

- [ ] **Step 4: Commit historical plan removal**

```bash
git add \
  docs/superpowers/plans/2026-07-24-shared-kernel-data-invariants.md \
  docs/superpowers/plans/2026-07-25-durable-jobs-api-scaling.md \
  docs/superpowers/plans/2026-07-25-durable-jobs-quality-followup.md \
  docs/superpowers/plans/2026-07-25-retrieval-evaluation-hybrid.md \
  docs/superpowers/plans/2026-07-25-task14-spec-review.md \
  docs/superpowers/plans/2026-07-26-server-rag-orchestration.md
git commit -m "docs: remove completed implementation plans"
```

## Task 6: Run final documentation gates and publish

**Files:**
- Delete: `docs/superpowers/plans/2026-07-27-current-state-documentation.md`

- [ ] **Step 1: Validate every relative Markdown file link**

Run this read-only validator from the repository root:

```bash
python3 - <<'PY'
from pathlib import Path
import re

roots = [Path("README.md"), *Path("docs").rglob("*.md")]
pattern = re.compile(r"\[[^\]]+\]\((?!https?://|mailto:|#)([^)#]+)(?:#[^)]+)?\)")
missing = []
for source in roots:
    text = source.read_text(encoding="utf-8")
    for target in pattern.findall(text):
        resolved = (source.parent / target).resolve()
        if not resolved.exists():
            missing.append(f"{source}: {target}")
if missing:
    raise SystemExit("\n".join(missing))
print(f"validated {len(roots)} Markdown files")
PY
```

Expected: prints the number of validated Markdown files and exits zero.

- [ ] **Step 2: Scan for stale process and architecture claims**

Run:

```bash
rg -n "TBD|TODO|FIXME|\- \[ \]|001…014|001\.\.014|single[- ]replica|Implementation plan|\.\./plans/|rag_page_conflict" README.md docs --glob '!docs/superpowers/plans/2026-07-27-current-state-documentation.md' && exit 1 || true
rg -n "SERVER_RAG_ENABLED: bool = False|HYBRID_RETRIEVAL_ENABLED: bool = False" api/config.py
ls supabase/migrations/015_server_rag.sql
```

Expected: the documentation stale scan is empty; both default-off settings and
migration `015` exist in code.

- [ ] **Step 3: Run formatting and static contract gates**

Run:

```bash
git diff --check
.venv/bin/ruff check tests/test_ci_matrix_contract.py
.venv/bin/ruff format --check tests/test_ci_matrix_contract.py
PYTHONPATH=api .venv/bin/pytest tests/test_ci_matrix_contract.py -q
```

Expected: diff check is silent, Ruff passes, format check reports the file is
already formatted, and the CI matrix contract tests pass.

- [ ] **Step 4: Remove this now-completed execution plan**

After Steps 1–3 pass, delete
`docs/superpowers/plans/2026-07-27-current-state-documentation.md` with
`apply_patch`. Then verify the plans directory contains no Markdown files:

```bash
find docs/superpowers/plans -type f -name '*.md' -print
```

Expected: no output. Remove the empty directory only if Git naturally stops
tracking it; do not add a placeholder file.

- [ ] **Step 5: Commit the publication state**

```bash
git add docs/superpowers/plans/2026-07-27-current-state-documentation.md
git diff --cached --check
git commit -m "docs: publish current-state architecture"
```

Expected: the commit contains only documentation changes and deletion of this
completed plan. If Steps 1–3 required a factual correction, stage each corrected
file by its exact path in the same commit; do not use `git add -A`.

- [ ] **Step 6: Push the same branch and wait for the exact SHA**

```bash
git push -u origin feat/platform-architecture-evolution
FINAL_SHA=$(git rev-parse HEAD)
gh run list --repo ben0112/llmwiki-goglobal \
  --branch feat/platform-architecture-evolution --limit 5
```

Wait for the run whose `headSha` equals `FINAL_SHA`. Expected: all six jobs,
including `Required scaled Compose (2 API / 2 worker)`, complete successfully.
Do not create a pull request or merge.

- [ ] **Step 7: Verify clean handoff**

```bash
git fetch origin feat/platform-architecture-evolution
test "$(git rev-parse HEAD)" = "$(git rev-parse origin/feat/platform-architecture-evolution)"
test -z "$(git status --porcelain)"
```

Expected: both commands exit zero; local HEAD equals the remote branch and the
worktree is clean.
