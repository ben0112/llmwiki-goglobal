-- Make document kind explicit and bind derived rows to their parent version.

ALTER TABLE documents
    ADD COLUMN source_kind TEXT;

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

ALTER TABLE document_pages
    ADD COLUMN document_version INTEGER NOT NULL DEFAULT 0;

ALTER TABLE document_chunks
    ADD COLUMN document_version INTEGER NOT NULL DEFAULT 0;

UPDATE document_pages AS page
SET document_version = document.version
FROM documents AS document
WHERE document.id = page.document_id;

UPDATE document_chunks AS chunk
SET document_version = document.version
FROM documents AS document
WHERE document.id = chunk.document_id;

CREATE INDEX idx_pages_document_version
    ON document_pages(document_id, document_version);
CREATE INDEX idx_chunks_document_version
    ON document_chunks(document_id, document_version);
