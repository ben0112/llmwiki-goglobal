-- Durable, tenant-owned server-side RAG runs, normalized pages, and step audit rows.

ALTER TABLE background_jobs
    DROP CONSTRAINT IF EXISTS background_jobs_job_type_check;
ALTER TABLE background_jobs
    ADD CONSTRAINT background_jobs_job_type_check CHECK (job_type IN (
        'document.extract',
        'document.embed',
        'graph.rebuild',
        'upload.cleanup',
        'build_wiki'
    ));

ALTER TABLE background_jobs
    ADD COLUMN rag_run_id UUID;

CREATE OR REPLACE FUNCTION set_background_job_rag_run_id()
RETURNS TRIGGER
LANGUAGE plpgsql
AS $$
BEGIN
    IF NEW.job_type = 'build_wiki'
        AND jsonb_typeof(NEW.payload) = 'object'
        AND NEW.payload ? 'run_id'
        AND NEW.payload - 'run_id' = '{}'::jsonb
        AND jsonb_typeof(NEW.payload->'run_id') = 'string'
        AND NEW.payload->>'run_id' ~
            '^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$'
    THEN
        NEW.rag_run_id := (NEW.payload->>'run_id')::uuid;
    ELSE
        NEW.rag_run_id := NULL;
    END IF;
    RETURN NEW;
END;
$$;

CREATE TRIGGER set_background_job_rag_run_id
    BEFORE INSERT OR UPDATE OF job_type, payload, rag_run_id ON background_jobs
    FOR EACH ROW
    EXECUTE FUNCTION set_background_job_rag_run_id();

ALTER TABLE background_jobs
    ADD CONSTRAINT background_jobs_build_wiki_shape_check CHECK (
        job_type <> 'build_wiki' OR (
            knowledge_base_id IS NOT NULL
            AND document_id IS NULL
            AND idempotency_key IS NOT NULL
            AND char_length(idempotency_key) BETWEEN 1 AND 200
            AND idempotency_key !~ '^[[:space:]]|[[:space:]]$'
            AND jsonb_typeof(payload) = 'object'
            AND payload ? 'run_id'
            AND payload - 'run_id' = '{}'::jsonb
            AND jsonb_typeof(payload->'run_id') = 'string'
            AND payload->>'run_id' ~
                '^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$'
            AND rag_run_id IS NOT NULL
        )
    );

ALTER TABLE background_jobs
    ADD CONSTRAINT background_jobs_kb_owner_fk
        FOREIGN KEY (knowledge_base_id, user_id)
        REFERENCES knowledge_bases (id, user_id) ON DELETE CASCADE;

CREATE UNIQUE INDEX background_jobs_rag_owner_ref
    ON background_jobs (id, user_id, knowledge_base_id);

CREATE UNIQUE INDEX background_jobs_rag_run_ref
    ON background_jobs (id, user_id, knowledge_base_id, rag_run_id);

CREATE TABLE rag_runs (
    id UUID PRIMARY KEY,
    job_id UUID NOT NULL UNIQUE,
    root_run_id UUID NOT NULL,
    parent_run_id UUID,
    user_id UUID NOT NULL,
    knowledge_base_id UUID NOT NULL,
    goal TEXT NOT NULL CHECK (char_length(goal) BETWEEN 1 AND 4000),
    goal_digest TEXT NOT NULL CHECK (goal_digest ~ '^[0-9a-f]{64}$'),
    target_path_prefix TEXT NOT NULL CHECK (
        target_path_prefix = '/wiki/' OR target_path_prefix LIKE '/wiki/%/'
    ),
    model_profile TEXT NOT NULL CHECK (char_length(model_profile) BETWEEN 1 AND 100),
    model_profile_version TEXT NOT NULL CHECK (char_length(model_profile_version) BETWEEN 1 AND 128),
    retrieval_profile TEXT NOT NULL CHECK (retrieval_profile IN ('lexical', 'hybrid')),
    dry_run BOOLEAN NOT NULL DEFAULT false,
    budget JSONB NOT NULL CHECK (jsonb_typeof(budget) = 'object' AND octet_length(budget::text) <= 4096),
    usage JSONB NOT NULL DEFAULT '{"steps":0,"model_tokens":0}'::jsonb
        CHECK (jsonb_typeof(usage) = 'object' AND octet_length(usage::text) <= 4096),
    idempotency_key TEXT NOT NULL CHECK (char_length(idempotency_key) BETWEEN 1 AND 200),
    request_digest TEXT NOT NULL CHECK (request_digest ~ '^[0-9a-f]{64}$'),
    completion_reason TEXT CHECK (completion_reason IN (
        'completed', 'no_work', 'dry_run', 'budget_exhausted', 'partial_failure'
    )),
    last_committed_ordinal INTEGER NOT NULL DEFAULT -1 CHECK (last_committed_ordinal >= -1),
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (user_id, idempotency_key),
    FOREIGN KEY (job_id, user_id, knowledge_base_id)
        REFERENCES background_jobs (id, user_id, knowledge_base_id) ON DELETE CASCADE,
    FOREIGN KEY (root_run_id) REFERENCES rag_runs (id) DEFERRABLE INITIALLY DEFERRED,
    FOREIGN KEY (parent_run_id) REFERENCES rag_runs (id),
    FOREIGN KEY (knowledge_base_id, user_id)
        REFERENCES knowledge_bases (id, user_id) ON DELETE CASCADE
);

ALTER TABLE rag_runs
    ADD CONSTRAINT rag_runs_job_run_fk
        FOREIGN KEY (job_id, user_id, knowledge_base_id, id)
        REFERENCES background_jobs (
            id, user_id, knowledge_base_id, rag_run_id
        ) ON DELETE CASCADE;

ALTER TABLE rag_runs ADD CONSTRAINT rag_runs_target_path_normalized_check CHECK (
    octet_length(target_path_prefix) <= 8000
    AND target_path_prefix !~ '^[[:space:]]|[[:space:]]$'
    AND target_path_prefix !~ '[[:cntrl:]]'
    AND position(E'\\' in target_path_prefix) = 0
    AND target_path_prefix !~ '(^|/)[.]{1,2}(/|$)'
    AND target_path_prefix ~ '^/wiki/([^/]+/)*$'
);

ALTER TABLE rag_runs ADD CONSTRAINT rag_runs_budget_shape_check CHECK (
    budget ?& ARRAY[
        'max_pages',
        'max_steps',
        'max_model_tokens',
        'max_context_chars',
        'max_page_chars',
        'per_call_timeout_seconds',
        'max_page_attempts',
        'max_conflict_retries'
    ]
    AND budget - ARRAY[
        'max_pages',
        'max_steps',
        'max_model_tokens',
        'max_context_chars',
        'max_page_chars',
        'per_call_timeout_seconds',
        'max_page_attempts',
        'max_conflict_retries'
    ] = '{}'::jsonb
    AND jsonb_typeof(budget->'max_pages') = 'number'
    AND (budget->>'max_pages') ~ '^[0-9]+$'
    AND (budget->>'max_pages')::numeric BETWEEN 1 AND 32
    AND jsonb_typeof(budget->'max_steps') = 'number'
    AND (budget->>'max_steps') ~ '^[0-9]+$'
    AND (budget->>'max_steps')::numeric BETWEEN 1 AND 512
    AND jsonb_typeof(budget->'max_model_tokens') = 'number'
    AND (budget->>'max_model_tokens') ~ '^[0-9]+$'
    AND (budget->>'max_model_tokens')::numeric BETWEEN 1 AND 250000
    AND jsonb_typeof(budget->'max_context_chars') = 'number'
    AND (budget->>'max_context_chars') ~ '^[0-9]+$'
    AND (budget->>'max_context_chars')::numeric BETWEEN 1 AND 240000
    AND jsonb_typeof(budget->'max_page_chars') = 'number'
    AND (budget->>'max_page_chars') ~ '^[0-9]+$'
    AND (budget->>'max_page_chars')::numeric BETWEEN 1 AND 120000
    AND jsonb_typeof(budget->'per_call_timeout_seconds') = 'number'
    AND (budget->>'per_call_timeout_seconds') ~ '^[0-9]+$'
    AND (budget->>'per_call_timeout_seconds')::numeric BETWEEN 1 AND 180
    AND jsonb_typeof(budget->'max_page_attempts') = 'number'
    AND (budget->>'max_page_attempts') ~ '^[0-9]+$'
    AND (budget->>'max_page_attempts')::numeric BETWEEN 1 AND 3
    AND jsonb_typeof(budget->'max_conflict_retries') = 'number'
    AND (budget->>'max_conflict_retries') ~ '^[0-9]+$'
    AND (budget->>'max_conflict_retries')::numeric BETWEEN 1 AND 3
);

ALTER TABLE rag_runs ADD CONSTRAINT rag_runs_usage_shape_check CHECK (
    usage ?& ARRAY['steps', 'model_tokens']
    AND usage - ARRAY['steps', 'model_tokens'] = '{}'::jsonb
    AND jsonb_typeof(usage->'steps') = 'number'
    AND (usage->>'steps') ~ '^[0-9]+$'
    AND (usage->>'steps')::numeric BETWEEN 0 AND 512
    AND jsonb_typeof(usage->'model_tokens') = 'number'
    AND (usage->>'model_tokens') ~ '^[0-9]+$'
    AND (usage->>'model_tokens')::numeric BETWEEN 0 AND 250000
);

ALTER TABLE rag_runs ADD CONSTRAINT rag_runs_usage_within_budget_check CHECK (
    CASE
        WHEN jsonb_typeof(budget) = 'object'
            AND jsonb_typeof(usage) = 'object'
            AND jsonb_typeof(budget->'max_steps') = 'number'
            AND (budget->>'max_steps') ~ '^[0-9]+$'
            AND jsonb_typeof(budget->'max_model_tokens') = 'number'
            AND (budget->>'max_model_tokens') ~ '^[0-9]+$'
            AND jsonb_typeof(usage->'steps') = 'number'
            AND (usage->>'steps') ~ '^[0-9]+$'
            AND jsonb_typeof(usage->'model_tokens') = 'number'
            AND (usage->>'model_tokens') ~ '^[0-9]+$'
        THEN (usage->>'steps')::numeric <= (budget->>'max_steps')::numeric
            AND (usage->>'model_tokens')::numeric
                <= (budget->>'max_model_tokens')::numeric
        ELSE false
    END
);

CREATE UNIQUE INDEX rag_runs_owner_ref
    ON rag_runs (id, user_id, knowledge_base_id);

ALTER TABLE rag_runs
    ADD CONSTRAINT rag_runs_root_owner_fk
        FOREIGN KEY (root_run_id, user_id, knowledge_base_id)
        REFERENCES rag_runs (id, user_id, knowledge_base_id)
        DEFERRABLE INITIALLY DEFERRED,
    ADD CONSTRAINT rag_runs_parent_owner_fk
        FOREIGN KEY (parent_run_id, user_id, knowledge_base_id)
        REFERENCES rag_runs (id, user_id, knowledge_base_id);

CREATE TABLE rag_run_pages (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    run_id UUID NOT NULL,
    user_id UUID NOT NULL,
    knowledge_base_id UUID NOT NULL,
    ordinal INTEGER NOT NULL CHECK (ordinal >= 0),
    path TEXT NOT NULL CHECK (
        char_length(path) BETWEEN 1 AND 2000
        AND octet_length(path) <= 8000
        AND path = btrim(path)
        AND path ~ '^/wiki/([^/]+/)*[^/]+[.]md$'
        AND path !~ '(^|/)[.]{1,2}(/|$)'
        AND position(E'\\' in path) = 0
    ),
    intent TEXT NOT NULL CHECK (
        char_length(intent) BETWEEN 1 AND 2000 AND intent = btrim(intent)
    ),
    query TEXT NOT NULL CHECK (
        char_length(query) BETWEEN 1 AND 2000 AND query = btrim(query)
    ),
    state TEXT NOT NULL DEFAULT 'planned' CHECK (state IN (
        'planned', 'running', 'committed', 'dry_run_complete', 'failed'
    )),
    document_id UUID,
    version_read INTEGER CHECK (version_read IS NULL OR version_read >= 1),
    version_committed INTEGER CHECK (version_committed IS NULL OR version_committed >= 1),
    attempt_count INTEGER NOT NULL DEFAULT 0 CHECK (attempt_count >= 0),
    conflict_retry_count INTEGER NOT NULL DEFAULT 0 CHECK (conflict_retry_count >= 0),
    last_completed_step_sequence INTEGER NOT NULL DEFAULT 0
        CHECK (last_completed_step_sequence >= 0),
    preview TEXT CHECK (preview IS NULL OR octet_length(preview) <= 16384),
    preview_digest TEXT CHECK (
        preview_digest IS NULL OR preview_digest ~ '^[0-9a-f]{64}$'
    ),
    preview_full_char_count INTEGER CHECK (
        preview_full_char_count IS NULL OR preview_full_char_count >= 0
    ),
    preview_truncated BOOLEAN NOT NULL DEFAULT false,
    lint_summary JSONB CHECK (
        lint_summary IS NULL OR (
            jsonb_typeof(lint_summary) = 'object'
            AND octet_length(lint_summary::text) <= 16384
        )
    ),
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    CHECK (
        (state = 'committed' AND lint_summary IS NOT NULL)
        OR (state <> 'committed' AND lint_summary IS NULL)
    ),
    UNIQUE (run_id, ordinal),
    UNIQUE (run_id, path),
    UNIQUE (id, run_id, user_id, knowledge_base_id),
    FOREIGN KEY (run_id, user_id, knowledge_base_id)
        REFERENCES rag_runs (id, user_id, knowledge_base_id) ON DELETE CASCADE,
    FOREIGN KEY (document_id, user_id, knowledge_base_id)
        REFERENCES documents (id, user_id, knowledge_base_id)
);

CREATE TABLE rag_steps (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    run_id UUID NOT NULL,
    run_page_id UUID,
    user_id UUID NOT NULL,
    knowledge_base_id UUID NOT NULL,
    sequence INTEGER NOT NULL CHECK (sequence >= 1),
    step_type TEXT NOT NULL CHECK (step_type IN (
        'plan', 'retrieve', 'read', 'draft', 'validate', 'write', 'lint', 'conflict'
    )),
    status TEXT NOT NULL CHECK (status IN ('running', 'succeeded', 'failed')),
    input_digest TEXT NOT NULL CHECK (input_digest ~ '^[0-9a-f]{64}$'),
    output_summary JSONB NOT NULL DEFAULT '{}'::jsonb CHECK (
        jsonb_typeof(output_summary) = 'object'
        AND octet_length(output_summary::text) <= 16384
    ),
    citation_identities JSONB NOT NULL DEFAULT '[]'::jsonb CHECK (
        jsonb_typeof(citation_identities) = 'array'
        AND jsonb_array_length(citation_identities) <= 128
        AND octet_length(citation_identities::text) <= 16384
    ),
    prompt_version TEXT CHECK (
        prompt_version IS NULL OR char_length(prompt_version) BETWEEN 1 AND 128
    ),
    prompt_digest TEXT CHECK (prompt_digest IS NULL OR prompt_digest ~ '^[0-9a-f]{64}$'),
    model_profile_version TEXT NOT NULL CHECK (
        char_length(model_profile_version) BETWEEN 1 AND 128
    ),
    input_tokens INTEGER NOT NULL DEFAULT 0 CHECK (input_tokens >= 0),
    output_tokens INTEGER NOT NULL DEFAULT 0 CHECK (output_tokens >= 0),
    total_tokens INTEGER NOT NULL DEFAULT 0 CHECK (
        total_tokens >= 0 AND total_tokens = input_tokens + output_tokens
    ),
    latency_ms DOUBLE PRECISION NOT NULL DEFAULT 0 CHECK (
        latency_ms >= 0
        AND latency_ms < 'Infinity'::double precision
        AND latency_ms <> 'NaN'::double precision
    ),
    error_code TEXT CHECK (
        error_code IS NULL OR error_code ~ '^[a-z0-9_]{1,128}$'
    ),
    error_message TEXT CHECK (
        error_message IS NULL OR (
            char_length(error_message) <= 2000
            AND octet_length(error_message) <= 8000
            AND error_message !~ '[[:cntrl:]]'
        )
    ),
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (run_id, sequence),
    FOREIGN KEY (run_id, user_id, knowledge_base_id)
        REFERENCES rag_runs (id, user_id, knowledge_base_id) ON DELETE CASCADE,
    FOREIGN KEY (run_page_id, run_id, user_id, knowledge_base_id)
        REFERENCES rag_run_pages (id, run_id, user_id, knowledge_base_id) ON DELETE CASCADE
);

CREATE UNIQUE INDEX rag_steps_one_running_per_run
    ON rag_steps (run_id)
    WHERE status = 'running';

CREATE INDEX rag_run_pages_state_idx
    ON rag_run_pages (run_id, state, ordinal);

ALTER TABLE rag_runs ENABLE ROW LEVEL SECURITY;
ALTER TABLE rag_run_pages ENABLE ROW LEVEL SECURITY;
ALTER TABLE rag_steps ENABLE ROW LEVEL SECURITY;

CREATE POLICY rag_runs_select ON rag_runs
    FOR SELECT TO authenticated
    USING (user_id = auth.uid());

CREATE POLICY rag_run_pages_select ON rag_run_pages
    FOR SELECT TO authenticated
    USING (user_id = auth.uid());

CREATE POLICY rag_steps_select ON rag_steps
    FOR SELECT TO authenticated
    USING (user_id = auth.uid());

REVOKE INSERT, UPDATE, DELETE ON rag_runs FROM authenticated;
REVOKE INSERT, UPDATE, DELETE ON rag_run_pages FROM authenticated;
REVOKE INSERT, UPDATE, DELETE ON rag_steps FROM authenticated;

GRANT SELECT ON rag_runs TO authenticated;
GRANT SELECT ON rag_run_pages TO authenticated;
GRANT SELECT ON rag_steps TO authenticated;

CREATE TRIGGER set_rag_runs_updated_at
    BEFORE UPDATE ON rag_runs
    FOR EACH ROW
    EXECUTE FUNCTION set_updated_at();

CREATE TRIGGER set_rag_run_pages_updated_at
    BEFORE UPDATE ON rag_run_pages
    FOR EACH ROW
    EXECUTE FUNCTION set_updated_at();

CREATE TRIGGER set_rag_steps_updated_at
    BEFORE UPDATE ON rag_steps
    FOR EACH ROW
    EXECUTE FUNCTION set_updated_at();
