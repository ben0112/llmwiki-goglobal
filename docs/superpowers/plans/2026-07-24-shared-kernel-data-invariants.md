# Shared Kernel and Data Invariants Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Establish one shared Python kernel for document, chunk, facet, reference, and search contracts, then make local and hosted document readiness transactionally consistent and version-auditable.

**Architecture:** Add a dependency-light `llmwiki_core` package at the repository root and migrate API/MCP pure logic behind compatibility modules. Add explicit document kind and derived-version columns to both schemas, then route hosted extraction and MCP wiki writes through atomic adapter operations. Keep the local filesystem as truth and use reconciliation to repair the unavoidable filesystem/SQLite boundary.

**Tech Stack:** Python 3.11, setuptools, FastAPI, FastMCP, asyncpg/Postgres, aiosqlite/SQLite, pytest, Ruff, Docker.

---

## File map

New shared files:

- `pyproject.toml`: installable `llmwiki-core` package metadata.
- `llmwiki_core/__init__.py`: stable public exports.
- `llmwiki_core/documents.py`: kinds, statuses, transitions, and normalized logical paths.
- `llmwiki_core/chunking.py`: backend-neutral `Chunk` and chunking functions.
- `llmwiki_core/facets.py`: facet validation plus rollup functions.
- `llmwiki_core/references.py`: pure citation/link parsing and edge extraction.
- `llmwiki_core/search.py`: backend-neutral search query/result contracts.
- `llmwiki_core/wiki.py`: wiki write bundle and expected-version conflict contract.
- `api/infra/db/derived_documents.py`: atomic hosted page/chunk/readiness writer.
- `supabase/migrations/011_document_invariants.sql`: hosted kind and derived-version migration.
- `tests/unit/core/`: pure shared-kernel tests.
- `tests/integration/test_document_invariants.py`: hosted transaction and version tests.
- `tests/integration/mcp/test_wiki_write_invariants.py`: SQLite/Postgres-compatible wiki write contract tests.

Compatibility files retained but reduced to adapters/re-exports:

- `api/services/chunker.py`
- `mcp/services/chunker.py`
- `api/services/references.py`
- `mcp/tools/references.py`
- `api/services/facet_rollup.py`
- `mcp/vaultfs/facet_rollup.py`
- `mcp/vaultfs/facets.py`

Schema and packaging files modified:

- `shared/sqlite_schema.sql`
- `api/infra/db/sqlite.py`
- `mcp/vaultfs/sqlite.py`
- `tests/helpers/schema.sql`
- `api/Dockerfile`
- `mcp/Dockerfile`
- `Dockerfile.local`
- `deploy/docker-compose.selfhost.yml`
- `.github/workflows/test.yml`
- `README.md`

## Task 1: Package the shared kernel and prove every runtime can import it

**Files:**

- Create: `pyproject.toml`
- Create: `llmwiki_core/__init__.py`
- Create: `tests/unit/core/__init__.py`
- Create: `tests/unit/core/test_import_boundaries.py`
- Modify: `.github/workflows/test.yml`

- [ ] **Step 1: Write the failing import-boundary test**

```python
# tests/unit/core/test_import_boundaries.py
import ast
from pathlib import Path


FORBIDDEN = {"fastapi", "mcp", "asyncpg", "aiosqlite", "aioboto3", "boto3"}


def test_core_imports_without_service_dependencies():
    import llmwiki_core

    assert llmwiki_core.__version__ == "0.1.0"


def test_core_source_does_not_import_infrastructure_packages():
    root = Path(__file__).parents[3] / "llmwiki_core"
    offenders = []
    for path in root.glob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = {alias.name.split(".")[0] for alias in node.names}
            elif isinstance(node, ast.ImportFrom) and node.module:
                names = {node.module.split(".")[0]}
            else:
                continue
            if names & FORBIDDEN:
                offenders.append(f"{path.name}:{node.lineno}")
    assert offenders == []
```

- [ ] **Step 2: Run the test and confirm the missing-package failure**

Run: `pytest tests/unit/core/test_import_boundaries.py -v`

Expected: FAIL with `ModuleNotFoundError: No module named 'llmwiki_core'`.

- [ ] **Step 3: Add minimal package metadata and public version**

```toml
# pyproject.toml
[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[project]
name = "llmwiki-core"
version = "0.1.0"
requires-python = ">=3.11"

[tool.setuptools.packages.find]
include = ["llmwiki_core*"]
```

```python
# llmwiki_core/__init__.py
__version__ = "0.1.0"
```

- [ ] **Step 4: Install editable and rerun the import test**

Run: `python -m pip install -e . --no-deps && pytest tests/unit/core/test_import_boundaries.py -v`

Expected: 2 passed.

- [ ] **Step 5: Teach all CI Python jobs to install the shared package**

Add `pip install -e . --no-deps` immediately after Python setup in the unit,
MCP integration, and Postgres integration jobs. Do not add runtime dependencies
to `pyproject.toml`. Add `"feat/**"` to the workflow's `push.branches` so every
incremental push on this implementation branch runs GitHub Actions.

- [ ] **Step 6: Commit the package boundary**

```bash
git add pyproject.toml llmwiki_core tests/unit/core .github/workflows/test.yml
git commit -m "refactor: establish shared core package"
git push
```

## Task 2: Define document kinds, states, and logical paths

**Files:**

- Create: `llmwiki_core/documents.py`
- Create: `tests/unit/core/test_documents.py`
- Modify: `llmwiki_core/__init__.py`

- [ ] **Step 1: Write failing contract tests**

```python
# tests/unit/core/test_documents.py
import pytest

from llmwiki_core.documents import (
    DocumentIdentity,
    DocumentKind,
    DocumentStatus,
    InvalidStatusTransition,
    assert_status_transition,
    join_logical_path,
    normalize_directory_path,
)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [("", "/"), ("wiki", "/wiki/"), ("/wiki", "/wiki/"), ("/wiki/a/", "/wiki/a/")],
)
def test_normalize_directory_path(raw, expected):
    assert normalize_directory_path(raw) == expected


@pytest.mark.parametrize("raw", ["../x", "/wiki/../x", "a\x00b"])
def test_normalize_directory_path_rejects_unsafe_paths(raw):
    with pytest.raises(ValueError):
        normalize_directory_path(raw)


def test_join_logical_path_uses_one_canonical_form():
    assert join_logical_path("wiki/concepts", "risk.md") == "/wiki/concepts/risk.md"


@pytest.mark.parametrize(
    ("old", "new"),
    [
        (DocumentStatus.PENDING, DocumentStatus.PROCESSING),
        (DocumentStatus.PROCESSING, DocumentStatus.READY),
        (DocumentStatus.PROCESSING, DocumentStatus.FAILED),
        (DocumentStatus.FAILED, DocumentStatus.PENDING),
    ],
)
def test_allowed_status_transitions(old, new):
    assert_status_transition(old, new)


def test_ready_cannot_skip_processing():
    with pytest.raises(InvalidStatusTransition):
        assert_status_transition(DocumentStatus.PENDING, DocumentStatus.READY)


def test_ready_can_only_return_to_pending_for_system_repair():
    with pytest.raises(InvalidStatusTransition):
        assert_status_transition(DocumentStatus.READY, DocumentStatus.PENDING)
    assert_status_transition(
        DocumentStatus.READY,
        DocumentStatus.PENDING,
        for_repair=True,
    )


def test_document_kinds_are_stable_wire_values():
    assert [kind.value for kind in DocumentKind] == ["source", "wiki", "asset"]


def test_document_identity_is_immutable_and_tenant_scoped():
    identity = DocumentIdentity(document_id="doc", knowledge_base_id="kb", user_id="user")
    assert identity.scope == ("user", "kb", "doc")
    with pytest.raises(AttributeError):
        identity.document_id = "other"
```

- [ ] **Step 2: Run the tests and verify the missing module failure**

Run: `pytest tests/unit/core/test_documents.py -v`

Expected: collection FAIL because `llmwiki_core.documents` does not exist.

- [ ] **Step 3: Implement the minimal document contracts**

```python
# llmwiki_core/documents.py
from dataclasses import dataclass
from enum import StrEnum
from pathlib import PurePosixPath


class DocumentKind(StrEnum):
    SOURCE = "source"
    WIKI = "wiki"
    ASSET = "asset"


class DocumentStatus(StrEnum):
    PENDING = "pending"
    PROCESSING = "processing"
    READY = "ready"
    FAILED = "failed"


class InvalidStatusTransition(ValueError):
    pass


@dataclass(frozen=True)
class DocumentIdentity:
    document_id: str
    knowledge_base_id: str
    user_id: str

    @property
    def scope(self) -> tuple[str, str, str]:
        return (self.user_id, self.knowledge_base_id, self.document_id)


_ALLOWED = {
    DocumentStatus.PENDING: {DocumentStatus.PROCESSING, DocumentStatus.FAILED},
    DocumentStatus.PROCESSING: {DocumentStatus.READY, DocumentStatus.FAILED},
    DocumentStatus.READY: set(),
    DocumentStatus.FAILED: {DocumentStatus.PENDING},
}


def assert_status_transition(
    old: DocumentStatus,
    new: DocumentStatus,
    *,
    for_repair: bool = False,
) -> None:
    if for_repair and old is DocumentStatus.READY and new is DocumentStatus.PENDING:
        return
    if new not in _ALLOWED[old]:
        raise InvalidStatusTransition(f"{old.value} -> {new.value}")


def normalize_directory_path(raw: str) -> str:
    if "\x00" in raw:
        raise ValueError("path contains NUL")
    parts = [part for part in raw.replace("\\", "/").split("/") if part and part != "."]
    if ".." in parts:
        raise ValueError("path traversal is not allowed")
    return "/" + "/".join(parts) + ("/" if parts else "")


def join_logical_path(directory: str, filename: str) -> str:
    if not filename or PurePosixPath(filename).name != filename:
        raise ValueError("filename must be a basename")
    return normalize_directory_path(directory) + filename
```

- [ ] **Step 4: Export contracts and rerun tests**

Export the six public document symbols from `llmwiki_core/__init__.py`.

Run: `pytest tests/unit/core/test_documents.py tests/unit/core/test_import_boundaries.py -v`

Expected: all tests pass.

- [ ] **Step 5: Commit document contracts**

```bash
git add llmwiki_core tests/unit/core
git commit -m "refactor: centralize document contracts"
git push
```

## Task 3: Define backend-neutral search contracts

**Files:**

- Create: `llmwiki_core/search.py`
- Create: `tests/unit/core/test_search.py`
- Modify: `mcp/tools/search.py`
- Modify: `api/infra/db/sqlite.py`
- Modify: `llmwiki_core/__init__.py`

- [ ] **Step 1: Write failing normalization and result-contract tests**

```python
# tests/unit/core/test_search.py
import pytest

from llmwiki_core.search import SearchArea, SearchHit, SearchQuery, SearchScope


def test_search_query_normalizes_shared_filters():
    query = SearchQuery.build(
        text="  data localization  ",
        limit=20,
        area="wiki",
        scope="annotations",
        facets={"stage": "S2"},
    )
    assert query.text == "data localization"
    assert query.area is SearchArea.WIKI
    assert query.scope is SearchScope.ANNOTATIONS
    assert query.facets == {"stage": "S2"}


@pytest.mark.parametrize("text,limit", [("", 10), ("query", 0), ("query", 101)])
def test_search_query_rejects_invalid_inputs(text, limit):
    with pytest.raises(ValueError):
        SearchQuery.build(text=text, limit=limit)


def test_search_hit_has_stable_identity_fields():
    hit = SearchHit(
        document_id="doc",
        document_version=3,
        chunk_index=2,
        content="matched text",
        score=0.75,
        path="/wiki/policy.md",
        title="Policy",
    )
    assert hit.identity == ("doc", 3, 2)
```

- [ ] **Step 2: Run and verify the missing-contract failure**

Run: `pytest tests/unit/core/test_search.py -v`

Expected: collection FAIL because `llmwiki_core.search` does not exist.

- [ ] **Step 3: Implement immutable query and hit contracts**

Use `StrEnum` for `SearchArea` (`all`, `wiki`, `sources`) and `SearchScope`
(`all`, `annotations`, `source`). Implement frozen `SearchQuery` and `SearchHit`
dataclasses. `SearchQuery.build` strips text, enforces a limit from 1 through
100, copies facet mappings, and converts wire strings to enums. `SearchHit`
exposes an `identity` property returning `(document_id, document_version,
chunk_index)`.

- [ ] **Step 4: Normalize API and MCP requests with the shared query**

At the start of `SearchHandler.search_chunks`, build a `SearchQuery` after
computing the existing `path_filter`, then use its normalized text, limit,
area, scope, and facets for the existing adapter call. At the start of
`SQLiteChunkRepository.search_fulltext`, build the same contract with the
existing query, limit, and path filter, then use its normalized values. Keep
the public MCP/API compatibility signatures and dictionary response shape
unchanged in this milestone.

- [ ] **Step 5: Run core and search regressions**

Run:

```bash
pytest tests/unit/core/test_search.py tests/unit/test_graph_local.py -v
cd mcp && PYTHONPATH=.. pytest ../tests/integration/mcp/test_tool_handlers.py ../tests/integration/mcp/test_vaultfs_contract.py -q
```

Expected: all selected tests pass.

- [ ] **Step 6: Commit search contracts**

```bash
git add llmwiki_core/search.py llmwiki_core/__init__.py api/infra/db/sqlite.py mcp/tools/search.py tests/unit/core/test_search.py
git commit -m "refactor: centralize search contracts"
git push
```

## Task 4: Make one chunking implementation authoritative

**Files:**

- Create: `llmwiki_core/chunking.py`
- Create: `tests/unit/core/test_chunking.py`
- Modify: `api/services/chunker.py`
- Modify: `mcp/services/chunker.py`
- Modify: `tests/unit/test_chunker.py`

- [ ] **Step 1: Write a cross-runtime parity test before moving code**

```python
# tests/unit/core/test_chunking.py
from llmwiki_core.chunking import MAX_CHUNK_CHARS, chunk_pages, chunk_text


def test_cjk_chunks_preserve_page_breadcrumb_and_overlap():
    content = "# 印尼合规\n\n" + "数据本地化要求。" * 400
    chunks = chunk_text(content, chunk_size=128, overlap=32, page=7)
    assert len(chunks) > 1
    assert all(chunk.page == 7 for chunk in chunks)
    assert all(len(chunk.content) <= MAX_CHUNK_CHARS for chunk in chunks)
    assert chunks[0].header_breadcrumb == "印尼合规"
    assert chunks[1].start_char < chunks[0].start_char + len(chunks[0].content)


def test_chunk_pages_assigns_global_indexes():
    chunks = chunk_pages([(1, "A sentence. " * 200), (2, "B sentence. " * 200)])
    assert [chunk.index for chunk in chunks] == list(range(len(chunks)))
    assert {chunk.page for chunk in chunks} == {1, 2}
```

- [ ] **Step 2: Run and verify the missing-module failure**

Run: `pytest tests/unit/core/test_chunking.py -v`

Expected: collection FAIL because `llmwiki_core.chunking` does not exist.

- [ ] **Step 3: Move only pure chunking code into the core**

Copy `Chunk`, constants, `_estimate_tokens`, `_split_paragraphs`,
`_get_overlap`, `_split_oversized`, `chunk_text`, and `chunk_pages` from
`api/services/chunker.py` into `llmwiki_core/chunking.py`. Remove `asyncpg`,
logging, and persistence functions from the core.

In `api/services/chunker.py`, import and re-export the pure symbols, retaining
only `store_chunks` and `_store_chunks_on_conn` as Postgres adapter functions:

```python
from llmwiki_core.chunking import Chunk, chunk_pages, chunk_text
```

In `mcp/services/chunker.py`, import the same symbols and retain only
`store_chunks_pg` and `store_chunks_sqlite`.

- [ ] **Step 4: Prove API, MCP, and core return identical chunks**

Extend `tests/unit/core/test_chunking.py`:

```python
def test_compatibility_modules_export_core_functions():
    from services.chunker import chunk_text as api_chunk_text

    assert api_chunk_text is chunk_text
```

Run API context:

`PYTHONPATH=api pytest tests/unit/core/test_chunking.py tests/unit/test_chunker.py -v`

Run MCP context:

`cd mcp && PYTHONPATH=.. pytest ../tests/unit/core/test_chunking.py ../tests/integration/mcp/test_vaultfs_contract.py -q`

Expected: all selected tests pass.

- [ ] **Step 5: Remove duplicate algorithms and commit**

Verify: `rg -n "def chunk_text|def chunk_pages" api mcp llmwiki_core`

Expected: definitions exist only in `llmwiki_core/chunking.py`.

```bash
git add llmwiki_core api/services/chunker.py mcp/services/chunker.py tests
git commit -m "refactor: share document chunking"
git push
```

## Task 5: Centralize facet and reference semantics

**Files:**

- Create: `llmwiki_core/facets.py`
- Create: `llmwiki_core/references.py`
- Create: `tests/unit/core/test_facets.py`
- Create: `tests/unit/core/test_references.py`
- Modify: `mcp/vaultfs/facets.py`
- Modify: `api/services/facet_rollup.py`
- Modify: `mcp/vaultfs/facet_rollup.py`
- Modify: `api/services/references.py`
- Modify: `mcp/tools/references.py`

- [ ] **Step 1: Write failing shared-semantics tests**

```python
# tests/unit/core/test_facets.py
import pytest

from llmwiki_core.facets import UnknownFacetError, apply_rollup, rollup_from_metas, validate_facets


def test_validate_facets_rejects_unknown_keys():
    with pytest.raises(UnknownFacetError):
        validate_facets({"planet": "Mars"})


def test_rollup_merges_cited_corpus_dimensions():
    rollup = rollup_from_metas(
        [
            {"stage": "S2", "geo_country": ["IDN"], "timeliness": "M2"},
            {"stage": "S3", "geo_country": ["VNM"], "timeliness": "M1"},
        ],
        "2026-07-24",
    )
    assert rollup["stage"] == ["S2", "S3"]
    assert rollup["country"] == ["IDN", "VNM"]
    assert rollup["timeliness_worst"] == "M1"
    assert apply_rollup({}, rollup)["facet_rollup"] == rollup
```

```python
# tests/unit/core/test_references.py
from llmwiki_core.references import build_lookup_maps, extract_references, parse_citation_filename


def test_reference_extraction_deduplicates_edges():
    docs = [
        {"id": "src", "filename": "law.pdf", "title": "Law", "path": "/"},
        {"id": "page", "filename": "risk.md", "title": "Risk", "path": "/wiki/"},
    ]
    names, bases, paths = build_lookup_maps(docs)
    edges = extract_references(
        "Claim[^1].\n\n[^1]: law.pdf, p.3\n[^2]: law.pdf, p.4",
        "page", "", names, bases, paths,
    )
    assert edges == [{"target_id": "src", "type": "cites", "page": 3}]


def test_chinese_page_suffix_is_parsed():
    assert parse_citation_filename("法规汇编.pdf，第 12 页") == ("法规汇编.pdf", 12)
```

- [ ] **Step 2: Run and confirm both modules are missing**

Run: `pytest tests/unit/core/test_facets.py tests/unit/core/test_references.py -v`

Expected: collection FAIL for missing core modules.

- [ ] **Step 3: Move pure logic and leave persistence in adapters**

Move `FACET_KEYS`, `UnknownFacetError`, and `validate_facets` from
`mcp/vaultfs/facets.py` into `llmwiki_core/facets.py`. Move
`rollup_from_metas` and `apply_rollup` from the duplicated facet-rollup modules
into the same core module. Keep SQLite/Postgres SQL builders in
`mcp/vaultfs/facets.py` and refresh queries in their existing adapter modules.

Move all pure parsing and lookup functions from `api/services/references.py`
into `llmwiki_core/references.py`. Make the API module a compatibility re-export.
Change `mcp/tools/references.py` to call `build_lookup_maps` and
`extract_references`, then persist returned edges through `VaultFS`.

- [ ] **Step 4: Run shared and legacy tests**

Run:

```bash
PYTHONPATH=api pytest tests/unit/core/test_facets.py tests/unit/core/test_references.py tests/unit/test_facet_rollup.py tests/unit/test_references.py tests/unit/test_references_links.py -v
cd mcp && PYTHONPATH=.. pytest ../tests/integration/mcp/test_corpus_facets.py ../tests/integration/mcp/test_corpus_relations.py ../tests/integration/mcp/test_tool_handlers.py -q
```

Expected: all selected tests pass.

- [ ] **Step 5: Verify one authoritative definition per behavior and commit**

Run:

`rg -n "def (validate_facets|rollup_from_metas|apply_rollup|parse_citation_filename|parse_wiki_links|extract_references)" api mcp llmwiki_core`

Expected: pure definitions appear only under `llmwiki_core`; adapter refresh and
SQL compilation functions remain in API/MCP.

```bash
git add llmwiki_core api/services mcp/tools/references.py mcp/vaultfs tests
git commit -m "refactor: share facet and reference semantics"
git push
```

## Task 6: Add explicit kind and derived-version schema invariants

**Files:**

- Create: `supabase/migrations/011_document_invariants.sql`
- Create: `tests/unit/core/test_schema_invariants.py`
- Modify: `shared/sqlite_schema.sql`
- Modify: `tests/helpers/schema.sql`
- Modify: `api/infra/db/sqlite.py`
- Modify: `mcp/vaultfs/sqlite.py`

- [ ] **Step 1: Write failing schema declaration tests**

```python
# tests/unit/core/test_schema_invariants.py
from pathlib import Path


ROOT = Path(__file__).parents[3]


def test_sqlite_schema_tracks_derived_versions():
    schema = (ROOT / "shared/sqlite_schema.sql").read_text(encoding="utf-8")
    assert "document_version INTEGER NOT NULL DEFAULT 0" in schema


def test_postgres_migration_adds_explicit_kind_and_versions():
    sql = (ROOT / "supabase/migrations/011_document_invariants.sql").read_text(encoding="utf-8")
    assert "ADD COLUMN source_kind" in sql
    assert "document_pages" in sql and "document_chunks" in sql
    assert "document_version" in sql
```

- [ ] **Step 2: Run and verify missing migration/version failures**

Run: `pytest tests/unit/core/test_schema_invariants.py -v`

Expected: both tests fail.

- [ ] **Step 3: Add the Postgres migration**

```sql
-- supabase/migrations/011_document_invariants.sql
ALTER TABLE documents
    ADD COLUMN source_kind text;

UPDATE documents
SET source_kind = CASE
    WHEN COALESCE(metadata->>'asset', 'false') = 'true'
         OR metadata->>'kind' = 'pdf_image' THEN 'asset'
    WHEN path LIKE '/wiki/%' THEN 'wiki'
    ELSE 'source'
END
WHERE source_kind IS NULL;

ALTER TABLE documents
    ALTER COLUMN source_kind SET DEFAULT 'source',
    ALTER COLUMN source_kind SET NOT NULL,
    ADD CONSTRAINT documents_source_kind_check
        CHECK (source_kind IN ('source', 'wiki', 'asset'));

ALTER TABLE document_pages ADD COLUMN document_version integer NOT NULL DEFAULT 0;
ALTER TABLE document_chunks ADD COLUMN document_version integer NOT NULL DEFAULT 0;

UPDATE document_pages p SET document_version = d.version
FROM documents d WHERE d.id = p.document_id;
UPDATE document_chunks c SET document_version = d.version
FROM documents d WHERE d.id = c.document_id;

CREATE INDEX idx_pages_document_version ON document_pages(document_id, document_version);
CREATE INDEX idx_chunks_document_version ON document_chunks(document_id, document_version);
```

- [ ] **Step 4: Add SQLite columns and idempotent existing-database migration**

Add `document_version INTEGER NOT NULL DEFAULT 0` to `document_pages` and
`document_chunks` in `shared/sqlite_schema.sql` and `tests/helpers/schema.sql`.

Add this helper to both SQLite initialization modules and invoke it after the
base schema is executed:

```python
async def _ensure_derived_version_columns(db) -> None:
    for table in ("document_pages", "document_chunks"):
        cursor = await db.execute(f"PRAGMA table_info({table})")
        columns = {row[1] for row in await cursor.fetchall()}
        if "document_version" not in columns:
            await db.execute(
                f"ALTER TABLE {table} ADD COLUMN document_version INTEGER NOT NULL DEFAULT 0"
            )
    await db.execute(
        "UPDATE document_pages SET document_version = "
        "COALESCE((SELECT version FROM documents WHERE id = document_pages.document_id), 0)"
    )
    await db.execute(
        "UPDATE document_chunks SET document_version = "
        "COALESCE((SELECT version FROM documents WHERE id = document_chunks.document_id), 0)"
    )
    await db.commit()
```

- [ ] **Step 5: Add migration execution to Postgres integration setup**

Append the migration contents to `tests/helpers/schema.sql` in schema order, so
integration fixtures match production. Extend the Postgres fixture assertion
to check `documents.source_kind` and both derived version columns.

- [ ] **Step 6: Run schema and database contract tests**

Run:

```bash
pytest tests/unit/core/test_schema_invariants.py tests/unit/test_sqlite_document_repo.py -v
PYTHONPATH=api MODE=hosted pytest tests/integration/isolation/test_application_only.py -v
```

Expected: all selected tests pass with Postgres available on port 5434.

- [ ] **Step 7: Commit schema invariants**

```bash
git add supabase/migrations/011_document_invariants.sql shared/sqlite_schema.sql tests/helpers/schema.sql api/infra/db/sqlite.py mcp/vaultfs/sqlite.py tests
git commit -m "feat: track document kind and derived versions"
git push
```

## Task 7: Make hosted extraction readiness atomic

**Files:**

- Create: `api/infra/db/derived_documents.py`
- Create: `tests/integration/test_document_invariants.py`
- Modify: `api/services/ocr.py`
- Modify: `api/services/chunker.py`
- Modify: `api/infra/tus.py`

- [ ] **Step 1: Write failing hosted transaction tests**

```python
# tests/integration/test_document_invariants.py
import uuid

import pytest


@pytest.fixture
def seed_pending_document(pool):
    async def seed(*, status: str, version: int) -> dict:
        user_id = uuid.uuid4()
        kb_id = uuid.uuid4()
        doc_id = uuid.uuid4()
        await pool.execute(
            "INSERT INTO users (id, email, display_name) VALUES ($1, $2, 'Invariant Test')",
            user_id,
            f"{user_id}@test.invalid",
        )
        await pool.execute(
            "INSERT INTO knowledge_bases (id, user_id, name, slug) VALUES ($1, $2, $3, $4)",
            kb_id,
            user_id,
            f"KB {kb_id}",
            f"kb-{kb_id}",
        )
        await pool.execute(
            "INSERT INTO documents "
            "(id, knowledge_base_id, user_id, filename, path, file_type, status, version, source_kind) "
            "VALUES ($1, $2, $3, 'source.md', '/', 'md', $4, $5, 'source')",
            doc_id,
            kb_id,
            user_id,
            status,
            version,
        )
        return {"id": doc_id, "user_id": user_id, "knowledge_base_id": kb_id}

    return seed


@pytest.mark.asyncio
async def test_replace_derived_content_commits_ready_with_matching_versions(seed_pending_document, pool):
    from infra.db.derived_documents import replace_derived_content
    from llmwiki_core.chunking import chunk_pages

    doc = await seed_pending_document(status="processing", version=3)
    pages = [(1, "Indonesia data localization requirements.")]
    await replace_derived_content(
        pool,
        document_id=doc["id"],
        user_id=doc["user_id"],
        knowledge_base_id=doc["knowledge_base_id"],
        pages=pages,
        chunks=chunk_pages(pages),
        parser="test",
    )
    row = await pool.fetchrow("SELECT status, version FROM documents WHERE id=$1", doc["id"])
    versions = await pool.fetch(
        "SELECT DISTINCT document_version FROM document_chunks WHERE document_id=$1", doc["id"]
    )
    assert row["status"] == "ready"
    assert {r["document_version"] for r in versions} == {row["version"]}


@pytest.mark.asyncio
async def test_replace_derived_content_rolls_back_before_ready(seed_pending_document, pool, monkeypatch):
    from infra.db import derived_documents

    doc = await seed_pending_document(status="processing", version=1)

    async def fail_chunks(*args, **kwargs):
        raise RuntimeError("chunk write failed")

    monkeypatch.setattr(derived_documents, "_insert_chunks", fail_chunks)
    with pytest.raises(RuntimeError, match="chunk write failed"):
        await derived_documents.replace_derived_content(
            pool,
            document_id=doc["id"],
            user_id=doc["user_id"],
            knowledge_base_id=doc["knowledge_base_id"],
            pages=[(1, "content")],
            chunks=[],
            parser="test",
        )
    assert await pool.fetchval("SELECT status FROM documents WHERE id=$1", doc["id"]) == "processing"
    assert await pool.fetchval("SELECT count(*) FROM document_pages WHERE document_id=$1", doc["id"]) == 0
```

- [ ] **Step 2: Run and verify the missing adapter failure**

Run: `PYTHONPATH=api MODE=hosted pytest tests/integration/test_document_invariants.py -v`

Expected: collection FAIL because `infra.db.derived_documents` does not exist.

- [ ] **Step 3: Implement one transactional hosted writer**

`replace_derived_content` must acquire one connection and transaction, lock the
document row with `FOR UPDATE`, verify `user_id`, increment `version`, replace
pages and chunks with that version, patch metadata, and set `ready` last.

The final update must be:

```sql
UPDATE documents
SET status = 'ready', content = $2, page_count = $3, parser = $4,
    version = $5, metadata = COALESCE(metadata, '{}'::jsonb) || $6::jsonb,
    error_message = NULL, updated_at = now()
WHERE id = $1 AND user_id = $7
```

The adapter must raise `LookupError` if the locked document is absent and must
never catch transaction exceptions.

- [ ] **Step 4: Route every hosted extracted-text path through the adapter**

Replace direct page/chunk/status sequences in these `OCRService` methods:

- `_store_extracted_pages`
- `_store_ocr_result`
- `_process_html`
- `_process_spreadsheet`

Image-only documents may continue to set ready without chunks, but must bump
their version in the same update. Add `source_kind='source'` to the TUS
document insert and `source_kind='asset'` to extracted asset inserts.

- [ ] **Step 5: Run transaction, OCR, and isolation tests**

Run:

```bash
PYTHONPATH=api MODE=hosted pytest tests/integration/test_document_invariants.py tests/integration/test_converter_isolation.py tests/integration/isolation/test_api_isolation.py -v
PYTHONPATH=api pytest tests/unit/test_chunker.py tests/unit/test_extraction_fixes.py -v
```

Expected: all selected tests pass.

- [ ] **Step 6: Commit atomic hosted readiness**

```bash
git add api/infra/db/derived_documents.py api/services/ocr.py api/services/chunker.py api/infra/tus.py tests/integration
git commit -m "fix: commit hosted document indexes atomically"
git push
```

## Task 8: Make local text ingestion version-consistent

**Files:**

- Create: `tests/unit/test_local_document_invariants.py`
- Modify: `tests/unit/test_local_reconcile.py`
- Modify: `api/domain/local_processor.py`
- Modify: `api/routes/local_upload.py`
- Modify: `api/infra/db/sqlite.py`
- Modify: `mcp/services/chunker.py`

- [ ] **Step 1: Write failing local invariant tests**

```python
# tests/unit/test_local_document_invariants.py
import hashlib

import pytest


@pytest.mark.asyncio
async def test_text_upload_ready_implies_current_chunks(tmp_path, monkeypatch):
    from config import settings
    from infra.db.sqlite import create_pool
    from routes.local_upload import _index_file_on_disk

    monkeypatch.setattr(settings, "WORKSPACE_PATH", str(tmp_path))
    db = await create_pool(str(tmp_path / "index.db"))
    await db.execute(
        "INSERT INTO workspace (id, name, description, user_id) VALUES ('ws', 'ws', '', 'user')"
    )
    await db.commit()
    content = "Policy sentence. " * 100
    source = tmp_path / "policy.md"
    source.write_text(content, encoding="utf-8")
    doc = await _index_file_on_disk(
        db,
        "policy.md",
        source,
        hashlib.sha256(content.encode()).hexdigest(),
    )
    row = await (
        await db.execute("SELECT status, version FROM documents WHERE id=?", (doc["id"],))
    ).fetchone()
    versions = await (
        await db.execute(
            "SELECT DISTINCT document_version FROM document_chunks WHERE document_id=?",
            (doc["id"],),
        )
    ).fetchall()
    assert row[0] == "ready"
    assert {v[0] for v in versions} == {row[1]}
    await db.close()


@pytest.mark.asyncio
async def test_text_upload_rolls_back_document_when_chunk_write_fails(tmp_path, monkeypatch):
    from config import settings
    from infra.db.sqlite import create_pool
    import routes.local_upload as local_upload

    monkeypatch.setattr(settings, "WORKSPACE_PATH", str(tmp_path))
    db = await create_pool(str(tmp_path / "index.db"))
    await db.execute(
        "INSERT INTO workspace (id, name, description, user_id) VALUES ('ws', 'ws', '', 'user')"
    )
    await db.commit()
    source = tmp_path / "policy.md"
    source.write_text("Policy sentence. " * 100, encoding="utf-8")

    async def fail_chunk_write(*args, **kwargs):
        raise RuntimeError("chunk write failed")

    monkeypatch.setattr(local_upload, "_store_chunks_for_upload", fail_chunk_write)
    with pytest.raises(RuntimeError, match="chunk write failed"):
        await local_upload._index_file_on_disk(db, "policy.md", source, "digest")
    assert await db.execute_fetchall("SELECT id FROM documents") == []
    await db.close()
```

Extend `tests/unit/test_local_reconcile.py` with the existing `_init_db` helper:

```python
async def test_inconsistent_ready_scan_finds_stale_chunk_version(tmp_path):
    from domain.local_processor import _inconsistent_ready_document_ids

    workspace = tmp_path / "research"
    workspace.mkdir()
    db = await _init_db(workspace)
    doc_id = str(uuid.uuid4())
    await db.execute(
        "INSERT INTO documents "
        "(id, user_id, filename, title, path, relative_path, source_kind, file_type, "
        "status, content, tags, parser, version, document_number) "
        "VALUES (?, ?, 'policy.md', 'Policy', '/', 'policy.md', 'source', 'md', "
        "'ready', 'current body', '[]', 'text', 2, 3)",
        (doc_id, USER_ID),
    )
    await db.execute(
        "INSERT INTO document_chunks "
        "(id, document_id, chunk_index, content, source_content, token_count, document_version) "
        "VALUES (?, ?, 0, 'stale body', 'stale body', 2, 1)",
        (str(uuid.uuid4()), doc_id),
    )
    await db.commit()

    assert await _inconsistent_ready_document_ids(db) == [doc_id]
    await db.close()
```

- [ ] **Step 2: Run and confirm version assertions fail**

Run: `PYTHONPATH=api pytest tests/unit/test_local_document_invariants.py -v`

Expected: FAIL because local chunk inserts do not set `document_version` and
the upload marks ready before chunk storage commits.

- [ ] **Step 3: Store local text documents, chunks, and ready status together**

Change local chunk persistence functions to require a `document_version` and
write it on every row. For simple-text upload, hold the existing serialized
SQLite write gate while inserting the document as `processing`, inserting its
chunks, then setting `ready` and committing once.

Make `api.infra.db.sqlite.serialized_write` the one API-local write gate: it
accepts the active connection, rolls back on exceptions, and is used by
`local_processor` instead of the separate `_db_write_gate`. The upload helper
must insert chunks directly on that connection; it must not call a decorated
repository method while already holding the gate.

Add an explicit `SIMPLE_TEXT_TYPES` branch to `process_document`: read the
current file from disk and call the same local replacement operation used by
upload. This makes reconciliation of a stale Markdown/text document rebuild
content and chunks rather than merely flipping it back to ready. For all local
text and extracted-page replacements, calculate
`next_version = current_version + 1`, write pages/chunks with that value, and
update document content/status/version last in the same `_gated_write`
transaction.

- [ ] **Step 4: Add stale-derived reconciliation**

Implement `_inconsistent_ready_document_ids` with this query and, at the start
of local startup reconciliation, set the returned documents to `pending` before
the existing backlog is collected:

```sql
SELECT d.id
FROM documents d
WHERE d.status = 'ready'
  AND d.source_kind != 'asset'
  AND (
    EXISTS (SELECT 1 FROM document_chunks c
            WHERE c.document_id = d.id AND c.document_version != d.version)
    OR (COALESCE(d.content, '') != '' AND NOT EXISTS
        (SELECT 1 FROM document_chunks c WHERE c.document_id = d.id))
  )
```

Set those documents to `pending` before the existing extraction backlog kick.

- [ ] **Step 5: Run local upload, reconciliation, and MCP SQLite tests**

Run:

```bash
PYTHONPATH=api pytest tests/unit/test_local_document_invariants.py tests/unit/test_local_reconcile.py tests/unit/test_resumable_upload.py tests/unit/test_extraction_fixes.py -v
cd mcp && PYTHONPATH=.. pytest ../tests/integration/mcp/test_vaultfs_contract.py ../tests/integration/mcp/test_tool_handlers.py -q
```

Expected: all selected tests pass.

- [ ] **Step 6: Commit local invariants**

```bash
git add api/domain/local_processor.py api/routes/local_upload.py api/infra/db/sqlite.py mcp/services/chunker.py tests/unit
git commit -m "fix: keep local derived indexes version-consistent"
git push
```

## Task 9: Make MCP wiki writes atomic inside each database

**Files:**

- Create: `llmwiki_core/wiki.py`
- Create: `tests/unit/core/test_wiki.py`
- Create: `tests/integration/mcp/test_wiki_write_invariants.py`
- Modify: `mcp/vaultfs/base.py`
- Modify: `mcp/vaultfs/postgres.py`
- Modify: `mcp/vaultfs/sqlite.py`
- Modify: `mcp/tools/write.py`
- Modify: `mcp/tools/references.py`

- [ ] **Step 1: Write the failing core bundle test**

```python
# tests/unit/core/test_wiki.py
from llmwiki_core.references import ReferenceEdge
from llmwiki_core.wiki import WikiWriteBundle


def test_wiki_bundle_deduplicates_derived_edges():
    bundle = WikiWriteBundle.build(
        document_id="page",
        expected_version=4,
        filename="page.md",
        path="/wiki/",
        file_type="md",
        content="body",
        title="Page",
        tags=["risk"],
        date="2026-07-24",
        metadata={"description": "Page"},
        edges=[
            ReferenceEdge("src", "cites", 3),
            ReferenceEdge("src", "cites", 4),
        ],
    )
    assert bundle.edges == (ReferenceEdge("src", "cites", 3),)
```

- [ ] **Step 2: Write failing adapter contract tests**

The SQLite test must create a wiki page, update it with an expected version,
and assert one commit contains the new content version, chunks, derived
references, and facet rollup. A second update with the old expected version
must raise `VersionConflict` without changing any row.

The Postgres variant must use the same assertions and fixtures under
`tests/integration/mcp/`. Name the common test function
`assert_atomic_wiki_write(fs, kb_id)` and call it for each adapter fixture.

- [ ] **Step 3: Run and verify missing bundle/adapter APIs**

Run:

```bash
pytest tests/unit/core/test_wiki.py -v
cd mcp && PYTHONPATH=.. pytest ../tests/integration/mcp/test_wiki_write_invariants.py -v
```

Expected: collection failures for `WikiWriteBundle` and `write_wiki_bundle`.

- [ ] **Step 4: Implement the core bundle and conflict**

```python
# llmwiki_core/wiki.py
from dataclasses import dataclass

from .references import ReferenceEdge


class VersionConflict(RuntimeError):
    pass


@dataclass(frozen=True)
class WikiWriteBundle:
    document_id: str
    expected_version: int | None
    filename: str
    path: str
    file_type: str
    content: str
    title: str | None
    tags: tuple[str, ...]
    date: str | None
    metadata: dict
    edges: tuple[ReferenceEdge, ...]

    @classmethod
    def build(cls, *, edges, tags, **values):
        deduped = {}
        for edge in edges:
            deduped.setdefault((edge.target_id, edge.reference_type), edge)
        return cls(tags=tuple(tags), edges=tuple(deduped.values()), **values)
```

Add a frozen `ReferenceEdge` dataclass to `llmwiki_core/references.py` and
return it from `extract_references` instead of dictionaries. Update compatibility
tests and callers in the same red-green cycle.

- [ ] **Step 5: Add one atomic adapter operation**

Add to `VaultFS`:

```python
@abstractmethod
async def write_wiki_bundle(self, kb_id: str, bundle: WikiWriteBundle) -> dict:
    raise NotImplementedError
```

`expected_version=None` means create; an integer means compare-and-swap update.
Postgres implementation uses one acquired connection and transaction to:

1. insert the caller-generated document id at version 1 when creating, or update
   with `WHERE version = expected_version` when editing;
2. map active-path uniqueness failures to `DuplicateDocumentError`, and raise
   `VersionConflict` when an expected-version update returns no row;
3. replace chunks with the committed document version;
4. replace only `cites`/`links_to` edges;
5. propagate link staleness;
6. compute and update the page facet rollup;
7. commit.

All seven steps must execute through the transaction's own `conn`/`db` object;
do not call public reference or facet helpers that acquire another connection
or commit independently.

SQLite performs the same database sequence under its serialized write lock.
Change SQLite `write_to_disk` to write a temporary file in the destination
directory, flush it, and call `os.replace` immediately before the database
transaction. Postgres keeps its no-op disk adapter. Startup reconciliation
repairs a database rollback after a successful local file replacement.

- [ ] **Step 6: Route MCP edit, append, and overwrite through the bundle**

`WriteHandler` must fetch all documents, use shared reference lookup/extraction,
construct `WikiWriteBundle`, call `write_wiki_bundle`, and remove the
`_sync_references` sequence. For a new wiki page it generates the document UUID
and passes `expected_version=None`; edits, appends, and overwrites pass the
current integer version. Thus document creation/update, chunks, derived edges,
staleness, and facet rollup share one database transaction. Non-wiki assets and
notes retain `create_document`/`update_document`.

- [ ] **Step 7: Run MCP and graph regression tests**

Run:

```bash
pytest tests/unit/core/test_wiki.py tests/unit/test_graph_local.py tests/unit/test_facet_rollup.py -v
cd mcp && PYTHONPATH=.. pytest ../tests/integration/mcp/test_wiki_write_invariants.py ../tests/integration/mcp/test_tool_handlers.py ../tests/integration/mcp/test_corpus_relations.py -v
```

Expected: all selected tests pass.

- [ ] **Step 8: Commit atomic wiki updates**

```bash
git add llmwiki_core mcp tests
git commit -m "feat: commit wiki updates as atomic bundles"
git push
```

## Task 10: Audit and repair hosted derived-version drift at startup

**Files:**

- Create: `tests/integration/test_derived_recovery.py`
- Modify: `api/main.py`
- Modify: `api/infra/db/derived_documents.py`

- [ ] **Step 1: Write failing audit-query tests**

```python
# tests/integration/test_derived_recovery.py
import pytest


@pytest.mark.asyncio
async def test_find_inconsistent_ready_documents(pool, seed_pending_document):
    from infra.db.derived_documents import find_inconsistent_ready_documents

    good = await seed_pending_document(status="ready", version=2)
    bad = await seed_pending_document(status="ready", version=3)
    await pool.execute(
        "INSERT INTO document_chunks "
        "(document_id,user_id,knowledge_base_id,chunk_index,content,source_content,token_count,document_version) "
        "VALUES ($1,$2,$3,0,'good','good',1,2)",
        good["id"], good["user_id"], good["knowledge_base_id"],
    )
    await pool.execute(
        "INSERT INTO document_chunks "
        "(document_id,user_id,knowledge_base_id,chunk_index,content,source_content,token_count,document_version) "
        "VALUES ($1,$2,$3,0,'stale','stale',1,2)",
        bad["id"], bad["user_id"], bad["knowledge_base_id"],
    )
    rows = await find_inconsistent_ready_documents(pool)
    assert {row["id"] for row in rows} == {bad["id"]}
```

- [ ] **Step 2: Run and verify missing audit function**

Run: `PYTHONPATH=api MODE=hosted pytest tests/integration/test_derived_recovery.py -v`

Expected: FAIL importing `find_inconsistent_ready_documents`.

- [ ] **Step 3: Implement the version audit query**

Select non-asset ready documents whose chunks are absent while content is
non-empty, or whose page/chunk versions differ from the document version. The
query must scope each mismatch by document id and return id plus user id.

At hosted startup, set inconsistent documents to `pending` in one transaction,
log the count, then let the existing recovery loop schedule them. Do not mark
image-only `asset` documents inconsistent.

- [ ] **Step 4: Run recovery and startup tests**

Run:

```bash
PYTHONPATH=api MODE=hosted pytest tests/integration/test_derived_recovery.py tests/integration/test_converter_isolation.py -v
PYTHONPATH=api pytest tests/unit/test_auth_provider_invariant.py -v
```

Expected: all selected tests pass.

- [ ] **Step 5: Commit recovery audit**

```bash
git add api/main.py api/infra/db/derived_documents.py tests/integration/test_derived_recovery.py
git commit -m "feat: recover stale hosted document indexes"
git push
```

## Task 11: Package the shared core in every service image

**Files:**

- Modify: `api/Dockerfile`
- Modify: `mcp/Dockerfile`
- Modify: `Dockerfile.local`
- Modify: `deploy/docker-compose.selfhost.yml`
- Modify: `.dockerignore`
- Modify: `README.md`

- [ ] **Step 1: Add a packaging smoke test that initially fails for hosted contexts**

Create `tests/unit/core/test_docker_packaging.py`:

```python
from pathlib import Path


ROOT = Path(__file__).parents[3]


def test_hosted_compose_builds_api_and_mcp_from_repo_root():
    compose = (ROOT / "deploy/docker-compose.selfhost.yml").read_text(encoding="utf-8")
    assert "context: ..\n      dockerfile: api/Dockerfile" in compose
    assert "context: ..\n      dockerfile: mcp/Dockerfile" in compose


def test_all_python_images_install_core_package():
    for dockerfile in ("api/Dockerfile", "mcp/Dockerfile", "Dockerfile.local"):
        text = (ROOT / dockerfile).read_text(encoding="utf-8")
        assert "pip install --no-deps" in text and "llmwiki_core" in text
```

- [ ] **Step 2: Run and verify packaging assertions fail**

Run: `pytest tests/unit/core/test_docker_packaging.py -v`

Expected: both tests fail against the existing Dockerfiles and compose contexts.

- [ ] **Step 3: Change hosted build contexts and install the package**

Use repository-root build contexts:

```yaml
api:
  build:
    context: ..
    dockerfile: api/Dockerfile
mcp:
  build:
    context: ..
    dockerfile: mcp/Dockerfile
```

In each hosted Dockerfile, copy its own hash lock first, install dependencies,
then copy `pyproject.toml` and `llmwiki_core/` and run:

```dockerfile
RUN pip install --no-cache-dir --no-deps .
```

Copy only the relevant service directory into its runtime path. Add
`pyproject.toml` and `llmwiki_core/` to the local image and install the same
package before copying API/MCP code.

- [ ] **Step 4: Document editable installation for source development**

Add `pip install -e . --no-deps` after the virtual environment is activated in
the README local installation commands.

- [ ] **Step 5: Run smoke tests and build all affected images**

Run:

```bash
pytest tests/unit/core/test_docker_packaging.py -v
docker build -f api/Dockerfile -t llmwiki-api:core .
docker build -f mcp/Dockerfile -t llmwiki-mcp:core .
docker build -f Dockerfile.local -t llmwiki-local:core .
```

Expected: tests pass and all three builds exit 0.

- [ ] **Step 6: Commit packaging changes**

```bash
git add api/Dockerfile mcp/Dockerfile Dockerfile.local deploy/docker-compose.selfhost.yml .dockerignore README.md tests/unit/core/test_docker_packaging.py
git commit -m "build: package shared core in all runtimes"
git push
```

## Task 12: Run milestone-wide verification and record the boundary

**Files:**

- Create: `docs/architecture/shared-kernel.md`
- Modify: `docs/superpowers/plans/2026-07-24-shared-kernel-data-invariants.md`

- [ ] **Step 1: Write the architecture boundary document**

Document:

- allowed dependency direction;
- core module responsibilities;
- status transitions and archive semantics;
- `ready` and `document_version` invariants;
- local filesystem repair behavior;
- hosted transaction boundary;
- compatibility facade removal criteria.

Use concrete module names from this plan and include no future milestone APIs.

- [ ] **Step 2: Run Ruff over every changed Python path**

Run:

```bash
ruff check llmwiki_core api mcp tests
ruff format --check llmwiki_core api mcp tests
```

Expected: exit 0 for both commands.

- [ ] **Step 3: Run the complete Python test suite**

Run with Postgres test service available on port 5434:

```bash
pytest tests/unit -v
PYTHONPATH=api MODE=hosted pytest tests/integration/isolation/ tests/integration/test_api_key_auth.py tests/integration/test_converter_isolation.py tests/integration/test_converter_service.py tests/integration/test_corpus_hosted_import.py tests/integration/test_corpus_hosted_pipeline.py tests/integration/test_kb_lifecycle.py tests/integration/test_note_lifecycle.py tests/integration/test_document_invariants.py tests/integration/test_derived_recovery.py -v
cd mcp && PYTHONPATH=.. pytest ../tests/integration/mcp/ -v
```

Expected: all tests pass with zero failures.

- [ ] **Step 4: Run schema and duplicate-definition audits**

Run:

```bash
git diff --check origin/master...HEAD
rg -n "def (chunk_text|chunk_pages|validate_facets|rollup_from_metas|apply_rollup|parse_citation_filename|parse_wiki_links|extract_references)" api mcp llmwiki_core
```

Expected: no whitespace errors; each pure definition appears once under
`llmwiki_core`.

- [ ] **Step 5: Mark completed checkboxes only after fresh evidence**

Update this plan's checkboxes for steps actually completed. If a Docker build
or Postgres suite cannot run, leave its checkbox open and record the exact
command and blocker below that step.

- [ ] **Step 6: Commit and push the milestone boundary**

```bash
git add docs/architecture/shared-kernel.md docs/superpowers/plans/2026-07-24-shared-kernel-data-invariants.md
git commit -m "docs: record shared kernel architecture"
git push
```

## Subsequent plans on the same branch

After this plan passes milestone-wide verification, create and execute these
separate detailed plans without changing branches:

1. `docs/superpowers/plans/2026-07-24-durable-jobs-api-scaling.md`
2. `docs/superpowers/plans/2026-07-24-retrieval-evaluation-hybrid-search.md`
3. `docs/superpowers/plans/2026-07-24-server-rag-orchestration.md`

Each plan begins from the verified commit produced by Task 12, uses the same
red-green-refactor and commit/push discipline, and must not weaken local offline
mode or the MCP compatibility contract.
