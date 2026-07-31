-- Durable read revisions and tenant-first indexes for bounded read models.

ALTER TABLE knowledge_bases
    ADD COLUMN IF NOT EXISTS read_revision BIGINT NOT NULL DEFAULT 1
    CHECK (read_revision > 0);

CREATE OR REPLACE FUNCTION bump_knowledge_base_read_revision()
RETURNS TRIGGER
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = public, pg_temp
AS $$
BEGIN
    IF TG_OP = 'INSERT' THEN
        UPDATE knowledge_bases
        SET read_revision = read_revision + 1
        WHERE id = NEW.knowledge_base_id;
        RETURN NEW;
    END IF;

    IF TG_OP = 'DELETE' THEN
        UPDATE knowledge_bases
        SET read_revision = read_revision + 1
        WHERE id = OLD.knowledge_base_id;
        RETURN OLD;
    END IF;

    UPDATE knowledge_bases
    SET read_revision = read_revision + 1
    WHERE id = OLD.knowledge_base_id;
    IF NEW.knowledge_base_id IS DISTINCT FROM OLD.knowledge_base_id THEN
        UPDATE knowledge_bases
        SET read_revision = read_revision + 1
        WHERE id = NEW.knowledge_base_id;
    END IF;
    RETURN NEW;
END;
$$;

REVOKE ALL ON FUNCTION bump_knowledge_base_read_revision() FROM PUBLIC;

DROP TRIGGER IF EXISTS documents_bump_read_revision ON documents;
CREATE TRIGGER documents_bump_read_revision
    AFTER INSERT OR UPDATE OR DELETE ON documents
    FOR EACH ROW
    EXECUTE FUNCTION bump_knowledge_base_read_revision();

CREATE INDEX IF NOT EXISTS idx_documents_read_browse_name
    ON documents (knowledge_base_id, user_id, path, lower(filename), id)
    WHERE NOT archived;
CREATE INDEX IF NOT EXISTS idx_documents_read_browse_date
    ON documents (knowledge_base_id, user_id, path, updated_at, id)
    WHERE NOT archived;
CREATE INDEX IF NOT EXISTS idx_documents_read_wiki_path
    ON documents (knowledge_base_id, user_id, source_kind, path, filename, id)
    WHERE NOT archived;
CREATE INDEX IF NOT EXISTS idx_documents_read_corpus
    ON documents (knowledge_base_id, user_id, source_kind, updated_at, id)
    WHERE NOT archived;
CREATE INDEX IF NOT EXISTS idx_documents_read_number
    ON documents (knowledge_base_id, user_id, document_number, id)
    WHERE NOT archived;
