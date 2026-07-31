-- Enable durable document embedding work without rewriting Task 7 migration 013.
ALTER TABLE background_jobs
    DROP CONSTRAINT IF EXISTS background_jobs_job_type_check;
ALTER TABLE background_jobs
    ADD CONSTRAINT background_jobs_job_type_check CHECK (job_type IN (
        'document.extract',
        'document.embed',
        'graph.rebuild',
        'upload.cleanup'
    ));

CREATE INDEX IF NOT EXISTS chunk_embeddings_reconciliation_idx
    ON chunk_embeddings (
        user_id, document_id, document_version, provider, model, dimensions, chunk_index
    );
