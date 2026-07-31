-- Bounded read-model state for the Local singleton workspace.

CREATE TRIGGER IF NOT EXISTS documents_read_revision_insert
AFTER INSERT ON documents
BEGIN
    UPDATE workspace SET read_revision = read_revision + 1;
END;

CREATE TRIGGER IF NOT EXISTS documents_read_revision_update
AFTER UPDATE ON documents
BEGIN
    UPDATE workspace SET read_revision = read_revision + 1;
END;

CREATE TRIGGER IF NOT EXISTS documents_read_revision_delete
AFTER DELETE ON documents
BEGIN
    UPDATE workspace SET read_revision = read_revision + 1;
END;

CREATE INDEX IF NOT EXISTS idx_documents_browse_name
    ON documents(path, filename COLLATE NOCASE, id);
CREATE INDEX IF NOT EXISTS idx_documents_browse_date
    ON documents(path, updated_at, id);
CREATE INDEX IF NOT EXISTS idx_documents_wiki_path
    ON documents(source_kind, path, filename, id);
CREATE INDEX IF NOT EXISTS idx_documents_number
    ON documents(document_number, id);
CREATE INDEX IF NOT EXISTS idx_documents_content_hash
    ON documents(content_hash)
    WHERE content_hash IS NOT NULL AND status != 'failed';
