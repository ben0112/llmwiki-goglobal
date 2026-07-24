-- Durable, tenant-owned ledger for background jobs. Delivery remains external.

CREATE TABLE background_jobs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    job_type TEXT NOT NULL CHECK (job_type IN (
        'document.extract',
        'graph.rebuild',
        'upload.cleanup'
    )),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    knowledge_base_id UUID REFERENCES knowledge_bases(id) ON DELETE CASCADE,
    document_id UUID REFERENCES documents(id) ON DELETE CASCADE,
    state TEXT NOT NULL DEFAULT 'queued' CHECK (state IN (
        'queued',
        'running',
        'retry_wait',
        'succeeded',
        'failed',
        'cancelled'
    )),
    payload JSONB NOT NULL DEFAULT '{}'::jsonb
        CHECK (jsonb_typeof(payload) = 'object')
        CHECK (octet_length(payload::text) <= 16384),
    progress JSONB
        CHECK (progress IS NULL OR jsonb_typeof(progress) = 'object')
        CHECK (octet_length(progress::text) <= 8192),
    result JSONB
        CHECK (result IS NULL OR jsonb_typeof(result) = 'object')
        CHECK (octet_length(result::text) <= 16384),
    idempotency_key TEXT,
    attempt_count INTEGER NOT NULL DEFAULT 0 CHECK (attempt_count >= 0),
    max_attempts INTEGER NOT NULL DEFAULT 3 CHECK (max_attempts BETWEEN 1 AND 20),
    run_after TIMESTAMPTZ NOT NULL DEFAULT now(),
    lease_owner TEXT,
    lease_expires_at TIMESTAMPTZ,
    heartbeat_at TIMESTAMPTZ,
    last_dispatched_at TIMESTAMPTZ,
    dispatch_attempts INTEGER NOT NULL DEFAULT 0 CHECK (dispatch_attempts >= 0),
    error_code TEXT CHECK (char_length(error_code) <= 2000),
    error_message TEXT CHECK (char_length(error_message) <= 2000),
    cancel_requested_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE UNIQUE INDEX background_jobs_idempotency_key_unique
    ON background_jobs(user_id, job_type, idempotency_key)
    WHERE idempotency_key IS NOT NULL;

CREATE INDEX background_jobs_due_dispatch_idx
    ON background_jobs(state, run_after, last_dispatched_at)
    WHERE state IN ('queued', 'retry_wait');

CREATE INDEX background_jobs_lease_expiry_idx
    ON background_jobs(state, lease_expires_at)
    WHERE state = 'running';

CREATE INDEX background_jobs_user_created_idx
    ON background_jobs(user_id, created_at DESC);

ALTER TABLE background_jobs ENABLE ROW LEVEL SECURITY;

CREATE POLICY background_jobs_select ON background_jobs
    FOR SELECT TO authenticated
    USING (user_id = auth.uid());

GRANT SELECT ON background_jobs TO authenticated;

CREATE TRIGGER set_background_jobs_updated_at
    BEFORE UPDATE ON background_jobs
    FOR EACH ROW
    EXECUTE FUNCTION set_updated_at();
