# Current-state Documentation Design

## Status

Approved for implementation on `feat/platform-architecture-evolution`.

## Context

The platform architecture evolution is implemented and verified, but the
documentation still mixes three different purposes:

- reader-facing setup and integration guidance;
- current architecture and operational contracts; and
- historical implementation plans containing future-tense tasks, transient
  commands, and superseded intermediate evidence.

That mixture makes it difficult to identify the current system of record and
creates multiple places where migration numbers, feature flags, topology, and
verification claims can drift.

## Goals

The documentation update will:

1. provide one discoverable overview of the implemented architecture;
2. make the four architecture documents the canonical domain contracts;
3. rewrite the three retained specifications as descriptions of the current
   design and its decisions;
4. remove completed implementation plans after migrating durable evidence;
5. keep deployment and agent-integration guides task-oriented; and
6. verify every drift-prone claim against the current branch before
   publication.

## Non-goals

This update will not change runtime behavior, schemas, migrations, feature
defaults, API contracts, CI behavior, or rollout state. It will not open or
merge a pull request. It will not retain transient RED/GREEN instructions,
step-by-step implementation commands, or evidence superseded by a newer exact
SHA gate.

## Chosen approach

Use a current-state, single-source-of-truth documentation model.

- `README.md` is the product entry point and architecture navigation surface.
- `docs/architecture/overview.md` explains the whole system and links to the
  four domain contracts.
- The domain architecture documents own implementation and operational facts.
- The retained specifications explain current design decisions, alternatives,
  boundaries, and security properties without duplicating operational detail.
- Completed implementation plans are deleted from the active documentation
  tree.

This preserves design rationale without forcing readers through an execution
log or maintaining the same contract in multiple files.

## Target information architecture

```text
README.md
docs/
├── architecture/
│   ├── overview.md
│   ├── shared-kernel.md
│   ├── durable-jobs.md
│   ├── retrieval.md
│   └── server-rag.md
├── self-hosting.md
├── agent-integration.md
└── superpowers/specs/
    ├── 2026-07-24-platform-architecture-evolution-design.md
    ├── 2026-07-25-durable-jobs-api-scaling-design.md
    ├── 2026-07-26-server-rag-orchestration-design.md
    └── 2026-07-27-current-state-documentation-design.md
```

## Source-of-truth boundaries

### README

The README will summarize supported modes and capabilities, link the
architecture overview, and route readers to deployment, agent integration,
retrieval, durable jobs, and server-side RAG guidance. It will not duplicate
long configuration or recovery procedures.

### Architecture overview

The new overview will describe:

- the shared core, adapter, API/MCP, durable-worker, and storage layers;
- Local and Hosted mode boundaries;
- the request-to-job-to-commit data flow;
- Postgres, Redis, MinIO, SQLite, PGroonga, and pgvector responsibilities;
- stateless API and horizontally scaled worker topology;
- lexical, hybrid, and server-side RAG relationships; and
- the completed state of all four architecture milestones.

### Domain architecture documents

Each domain document will own its configuration, invariants, failure model,
deployment boundary, observability contract, verification evidence, and
rollback behavior:

- `shared-kernel.md`: shared types, identities, paths, hashes, citations,
  serialization, and write invariants;
- `durable-jobs.md`: job ledger, leases, idempotency, cancellation, recovery,
  and multi-replica deployment;
- `retrieval.md`: evaluation corpus, lexical baseline, filters, hybrid recall,
  fallbacks, and promotion gates; and
- `server-rag.md`: profiles, budgets, orchestration, atomic page commits,
  resume, privacy, rollout, and flag-only rollback.

Cross-domain documents will link to these contracts rather than restating
them.

### Deployment and integration guides

`docs/self-hosting.md` will remain an executable operator guide: migration
order, configuration, startup, scaling, health checks, promotion, and rollback.
`docs/agent-integration.md` will describe the currently supported MCP, REST,
job, retrieval, and RAG interaction boundaries from a client perspective.

### Retained specifications

The three existing architecture specifications will be rewritten in the
present tense. They will retain:

- goals and non-goals;
- accepted decisions and rejected alternatives;
- component and data-flow boundaries;
- failure, isolation, and security properties; and
- links to canonical operational and verification evidence.

They will remove implementation checklists, provisional wording, obsolete
future work, repeated commands, and superseded test snapshots.

## Plan deletion and evidence migration

Delete these six completed plans:

- `docs/superpowers/plans/2026-07-24-shared-kernel-data-invariants.md`;
- `docs/superpowers/plans/2026-07-25-durable-jobs-api-scaling.md`;
- `docs/superpowers/plans/2026-07-25-durable-jobs-quality-followup.md`;
- `docs/superpowers/plans/2026-07-25-retrieval-evaluation-hybrid.md`;
- `docs/superpowers/plans/2026-07-25-task14-spec-review.md`; and
- `docs/superpowers/plans/2026-07-26-server-rag-orchestration.md`.

Before deletion, migrate only durable information that is not already recorded
by a newer source:

- final substantive and publication SHAs;
- exact-SHA GitHub Actions links and conclusions;
- stable local gate counts where they clarify coverage;
- feature-default and rollout boundaries;
- incident rollback contracts; and
- important design conclusions not present in a retained specification.

Intermediate failed SHAs, temporary environment workarounds, mechanical task
steps, and evidence replaced by later gates will not be migrated.

## Consistency and link migration

All references to deleted plans will be removed or redirected to the relevant
current architecture document or retained specification. Terminology will be
normalized across files, including Hosted/Local mode, durable jobs, knowledge
bases, hybrid retrieval, server-side RAG, API/worker roles, and feature flags.

Drift-prone facts will be checked against the branch rather than copied from an
older document. These include migration range, default feature state, required
dependencies, replica topology, environment variable names, endpoint paths,
budget limits, test ownership, and exact verification SHA.

## Verification

The documentation change is complete only when:

1. no repository link targets a deleted plan;
2. no active document contains stale task checkboxes, unresolved placeholders,
   obsolete migration ranges, or claims that Hosted API requires one replica;
3. every relative Markdown file link in the changed documentation resolves;
4. configuration, migration, Compose, workflow, and endpoint claims match the
   current code;
5. `git diff --check` passes;
6. relevant documentation and CI-contract tests pass; and
7. the final pushed documentation SHA completes all six GitHub Actions jobs
   successfully.

## Publication

Implementation stays on `feat/platform-architecture-evolution`. Stage only the
approved documentation changes and the six confirmed plan deletions, commit
them intentionally, and push the same branch. Do not create a pull request or
merge the branch unless the user separately requests it.
