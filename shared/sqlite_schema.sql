-- LLM Wiki local index schema (SQLite + FTS5)
-- This is derived state — deletable and rebuildable from the workspace filesystem.

PRAGMA journal_mode=WAL;
PRAGMA foreign_keys=ON;

CREATE TABLE IF NOT EXISTS workspace (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    description TEXT DEFAULT '',
    kind TEXT NOT NULL DEFAULT 'wiki',
    user_id TEXT NOT NULL,
    created_at TEXT DEFAULT (datetime('now')),
    UNIQUE(user_id)
);

CREATE TABLE IF NOT EXISTS documents (
    id TEXT PRIMARY KEY DEFAULT (lower(hex(randomblob(16)))),
    user_id TEXT NOT NULL,
    filename TEXT NOT NULL,
    title TEXT,
    path TEXT DEFAULT '/' NOT NULL,
    relative_path TEXT NOT NULL,
    source_kind TEXT NOT NULL CHECK (source_kind IN ('wiki', 'source', 'asset')),
    file_type TEXT NOT NULL,
    file_size INTEGER DEFAULT 0,
    document_number INTEGER,
    status TEXT DEFAULT 'pending' CHECK (status IN ('pending', 'processing', 'ready', 'failed')),
    page_count INTEGER,
    content TEXT,
    tags TEXT DEFAULT '[]',
    date TEXT,
    metadata TEXT,
    error_message TEXT,
    extraction_attempts INTEGER NOT NULL DEFAULT 0,
    version INTEGER DEFAULT 0,
    parser TEXT,
    content_hash TEXT,
    mtime_ns INTEGER,
    last_indexed_at TEXT,
    stale_since TEXT,
    highlights TEXT DEFAULT '[]',
    created_at TEXT DEFAULT (datetime('now')),
    updated_at TEXT DEFAULT (datetime('now')),
    UNIQUE(relative_path)
);

CREATE TABLE IF NOT EXISTS document_pages (
    id TEXT PRIMARY KEY DEFAULT (lower(hex(randomblob(16)))),
    document_id TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    document_version INTEGER NOT NULL DEFAULT 0,
    page INTEGER NOT NULL,
    content TEXT NOT NULL,
    elements TEXT,
    UNIQUE(document_id, page)
);

CREATE TABLE IF NOT EXISTS document_chunks (
    id TEXT PRIMARY KEY DEFAULT (lower(hex(randomblob(16)))),
    document_id TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    document_version INTEGER NOT NULL DEFAULT 0,
    chunk_index INTEGER NOT NULL,
    -- `content` is the materialized form (source + annotations) used by FTS.
    -- `source_content` is the immutable raw chunk text; `annotations_text`
    -- holds the materialized footnote-body block of highlights/comments
    -- touching this chunk, maintained by the highlight CRUD service methods.
    -- See docs/highlights-in-search-spec.md.
    content TEXT NOT NULL,
    source_content TEXT NOT NULL DEFAULT '',
    annotations_text TEXT,
    has_highlight INTEGER NOT NULL DEFAULT 0,
    page INTEGER,
    start_char INTEGER,
    token_count INTEGER NOT NULL,
    header_breadcrumb TEXT,
    created_at TEXT DEFAULT (datetime('now')),
    UNIQUE(document_id, chunk_index)
);

-- Reference types: 'cites'/'links_to' are derived from page content on every
-- write; the five relation-layer types (关系层, corpus spec v2026.06 §2.5) are
-- curated via the MCP `relate` tool and must survive content-driven rebuilds —
-- rebuild paths only delete the derived citation types.
--   is_a 上下位 · next 前后置 · routes_to 路径衔接 ·
--   governed_by 归口映射 · serves 阶段服务包
CREATE TABLE IF NOT EXISTS document_references (
    id TEXT PRIMARY KEY DEFAULT (lower(hex(randomblob(16)))),
    source_document_id TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    target_document_id TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    reference_type TEXT NOT NULL CHECK (reference_type IN (
        'cites', 'links_to',
        'is_a', 'next', 'routes_to', 'governed_by', 'serves'
    )),
    page INTEGER,
    UNIQUE(source_document_id, target_document_id, reference_type)
);

-- FTS5 full-text search (replaces pgroonga).
-- Tokenizer is trigram (SQLite >= 3.34) so CJK text is searchable — the
-- default `porter unicode61` treats a run of Chinese characters as a single
-- token, making Chinese queries unmatchable. Trigram trades English stemming
-- for substring matching in any script. Queries shorter than 3 characters
-- can't use the trigram index; the search layers fall back to a LIKE scan
-- for those (see mcp/vaultfs/sqlite.py and api/infra/db/sqlite.py, which
-- also migrate pre-existing databases to this tokenizer on startup).
CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
    content,
    content='document_chunks',
    content_rowid='rowid',
    tokenize='trigram'
);

-- Keep FTS in sync with document_chunks
CREATE TRIGGER IF NOT EXISTS chunks_fts_insert AFTER INSERT ON document_chunks BEGIN
    INSERT INTO chunks_fts(rowid, content) VALUES (new.rowid, new.content);
END;

CREATE TRIGGER IF NOT EXISTS chunks_fts_delete AFTER DELETE ON document_chunks BEGIN
    INSERT INTO chunks_fts(chunks_fts, rowid, content) VALUES('delete', old.rowid, old.content);
END;

CREATE TRIGGER IF NOT EXISTS chunks_fts_update AFTER UPDATE ON document_chunks BEGIN
    INSERT INTO chunks_fts(chunks_fts, rowid, content) VALUES('delete', old.rowid, old.content);
    INSERT INTO chunks_fts(rowid, content) VALUES (new.rowid, new.content);
END;

CREATE INDEX IF NOT EXISTS idx_documents_relative_path ON documents(relative_path);
CREATE INDEX IF NOT EXISTS idx_documents_path ON documents(path);
CREATE INDEX IF NOT EXISTS idx_documents_source_kind ON documents(source_kind);
CREATE INDEX IF NOT EXISTS idx_documents_status ON documents(status);
CREATE INDEX IF NOT EXISTS idx_chunks_doc ON document_chunks(document_id);
-- Partial index for the "search but only chunks I've annotated" filter.
CREATE INDEX IF NOT EXISTS idx_chunks_annotated
  ON document_chunks(document_id) WHERE has_highlight = 1;
CREATE INDEX IF NOT EXISTS idx_refs_source ON document_references(source_document_id);
CREATE INDEX IF NOT EXISTS idx_refs_target ON document_references(target_document_id);

-- 语料分类流水线:逐源文档状态(L3-P1)
CREATE TABLE IF NOT EXISTS corpus_pipeline (
    doc_id TEXT PRIMARY KEY REFERENCES documents(id) ON DELETE CASCADE,
    state TEXT NOT NULL CHECK (state IN ('imported', 'excluded', 'failed')),
    attempts INTEGER DEFAULT 1,
    error TEXT DEFAULT '',
    entry_id TEXT DEFAULT '',
    updated_at TEXT DEFAULT (datetime('now'))
);

-- 工作区级键值设置(如 corpus_llm:分类 LLM 端点,前端可配)
CREATE TABLE IF NOT EXISTS settings (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
