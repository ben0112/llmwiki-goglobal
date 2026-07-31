-- Test schema: mock Supabase auth functions + main migration (no PGroonga)

DROP SCHEMA IF EXISTS auth CASCADE;
CREATE SCHEMA auth;

CREATE TABLE auth.users (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    email TEXT,
    raw_user_meta_data JSONB DEFAULT '{}'
);

CREATE OR REPLACE FUNCTION auth.uid() RETURNS UUID
LANGUAGE sql STABLE
AS $$
    SELECT COALESCE(
        nullif(current_setting('request.jwt.claims', true), '')::json->>'sub',
        NULL
    )::uuid
$$;

CREATE OR REPLACE FUNCTION auth.jwt() RETURNS JSON
LANGUAGE sql STABLE
AS $$
    SELECT COALESCE(
        nullif(current_setting('request.jwt.claims', true), ''),
        '{}'
    )::json
$$;

DO $$ BEGIN
    CREATE ROLE authenticated NOLOGIN;
EXCEPTION WHEN duplicate_object THEN NULL;
END $$;

CREATE TYPE document_status AS ENUM ('pending', 'processing', 'ready', 'failed', 'archived');

CREATE TABLE users (
    id UUID PRIMARY KEY,
    email TEXT NOT NULL UNIQUE,
    display_name TEXT,
    onboarded BOOLEAN NOT NULL DEFAULT false,
    page_limit INTEGER NOT NULL DEFAULT 500,
    storage_limit_bytes BIGINT NOT NULL DEFAULT 1073741824,
    created_at TIMESTAMPTZ DEFAULT now() NOT NULL,
    updated_at TIMESTAMPTZ DEFAULT now() NOT NULL
);

CREATE TABLE api_keys (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    name TEXT,
    key_hash TEXT NOT NULL UNIQUE,
    key_prefix TEXT NOT NULL,
    created_at TIMESTAMPTZ DEFAULT now() NOT NULL,
    last_used_at TIMESTAMPTZ,
    revoked_at TIMESTAMPTZ
);

CREATE TABLE knowledge_bases (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    name TEXT NOT NULL,
    slug TEXT NOT NULL,
    description TEXT,
    kind TEXT NOT NULL DEFAULT 'wiki' CHECK (kind IN ('wiki', 'course')),
    created_at TIMESTAMPTZ DEFAULT now() NOT NULL,
    updated_at TIMESTAMPTZ DEFAULT now() NOT NULL,
    UNIQUE(user_id, slug),
    UNIQUE(user_id, name)
);

CREATE TABLE documents (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    knowledge_base_id UUID NOT NULL REFERENCES knowledge_bases(id) ON DELETE CASCADE,
    user_id UUID NOT NULL REFERENCES users(id),
    filename TEXT NOT NULL,
    title TEXT,
    path TEXT DEFAULT '/' NOT NULL,
    source_kind TEXT DEFAULT 'source' NOT NULL CHECK (source_kind IN ('source', 'wiki', 'asset')),
    file_type TEXT NOT NULL,
    file_size BIGINT DEFAULT 0 NOT NULL,
    document_number INTEGER,
    status document_status DEFAULT 'pending' NOT NULL,
    page_count INTEGER CHECK (page_count IS NULL OR page_count <= 300),
    content TEXT,
    tags TEXT[] DEFAULT '{}' NOT NULL,
    url TEXT,
    date TEXT,
    metadata JSONB,
    error_message TEXT,
    version INTEGER DEFAULT 0 NOT NULL,
    sort_order INTEGER DEFAULT 0,
    parser TEXT,
    archived BOOLEAN DEFAULT false NOT NULL,
    created_at TIMESTAMPTZ DEFAULT now() NOT NULL,
    updated_at TIMESTAMPTZ DEFAULT now() NOT NULL
);

CREATE TABLE document_pages (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    document_id UUID NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    document_version INTEGER NOT NULL DEFAULT 0,
    page INTEGER NOT NULL,
    content TEXT NOT NULL CHECK (length(content) <= 500000),
    elements JSONB,
    UNIQUE(document_id, page)
);

CREATE TABLE document_chunks (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    document_id UUID NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    document_version INTEGER NOT NULL DEFAULT 0,
    user_id UUID NOT NULL REFERENCES users(id),
    knowledge_base_id UUID NOT NULL REFERENCES knowledge_bases(id) ON DELETE CASCADE,
    chunk_index INTEGER NOT NULL,
    -- `content` is the materialized form (source + annotations) used by FTS.
    content TEXT NOT NULL,
    source_content TEXT NOT NULL DEFAULT '' CHECK (length(source_content) <= 10000),
    annotations_text TEXT,
    has_highlight BOOLEAN NOT NULL DEFAULT false,
    page INTEGER,
    start_char INTEGER,
    token_count INTEGER NOT NULL,
    header_breadcrumb TEXT,
    created_at TIMESTAMPTZ DEFAULT now() NOT NULL,
    UNIQUE(document_id, chunk_index)
);
CREATE INDEX IF NOT EXISTS idx_chunks_annotated
    ON document_chunks(knowledge_base_id) WHERE has_highlight = true;

ALTER TABLE documents ADD COLUMN IF NOT EXISTS stale_since TIMESTAMPTZ;
ALTER TABLE documents ADD COLUMN IF NOT EXISTS highlights JSONB NOT NULL DEFAULT '[]'::jsonb;
CREATE INDEX IF NOT EXISTS idx_documents_source_url
    ON documents (user_id, (metadata->>'source_url'))
    WHERE metadata ? 'source_url' AND NOT archived;

CREATE TABLE document_references (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    source_document_id UUID NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    target_document_id UUID NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    knowledge_base_id UUID NOT NULL REFERENCES knowledge_bases(id) ON DELETE CASCADE,
    reference_type TEXT NOT NULL CHECK (reference_type IN (
        'cites', 'links_to',
        'is_a', 'next', 'routes_to', 'governed_by', 'serves'
    )),  -- mirrors supabase/migrations/009_relation_types.sql
    page INTEGER,
    created_at TIMESTAMPTZ DEFAULT now() NOT NULL,
    UNIQUE(source_document_id, target_document_id, reference_type)
);

CREATE INDEX idx_refs_source ON document_references(source_document_id);
CREATE INDEX idx_refs_target ON document_references(target_document_id);

CREATE POLICY document_references_select ON document_references
    FOR SELECT TO authenticated
    USING (knowledge_base_id IN (
        SELECT id FROM knowledge_bases WHERE user_id = auth.uid()
    ));

CREATE POLICY document_references_write ON document_references
    FOR ALL TO authenticated
    USING (knowledge_base_id IN (
        SELECT id FROM knowledge_bases WHERE user_id = auth.uid()
    ))
    WITH CHECK (knowledge_base_id IN (
        SELECT id FROM knowledge_bases WHERE user_id = auth.uid()
    ));

ALTER TABLE document_references ENABLE ROW LEVEL SECURITY;

CREATE INDEX idx_documents_knowledge_base_id ON documents(knowledge_base_id);
CREATE INDEX idx_documents_user_id ON documents(user_id);
CREATE INDEX idx_documents_tags ON documents USING GIN(tags);
CREATE INDEX idx_documents_kb_path ON documents(knowledge_base_id, path);
CREATE INDEX idx_documents_kb_status ON documents(knowledge_base_id, status) WHERE NOT archived;
CREATE INDEX idx_documents_date ON documents(date) WHERE date IS NOT NULL;
CREATE INDEX idx_api_keys_user_id ON api_keys(user_id);
CREATE INDEX idx_chunks_kb ON document_chunks(knowledge_base_id);
CREATE INDEX idx_chunks_doc ON document_chunks(document_id);
CREATE INDEX idx_pages_document_version ON document_pages(document_id, document_version);
CREATE INDEX idx_chunks_document_version ON document_chunks(document_id, document_version);

-- PGroonga indexes intentionally omitted (requires C extension)

ALTER TABLE users ENABLE ROW LEVEL SECURITY;
ALTER TABLE api_keys ENABLE ROW LEVEL SECURITY;
ALTER TABLE knowledge_bases ENABLE ROW LEVEL SECURITY;
ALTER TABLE documents ENABLE ROW LEVEL SECURITY;
ALTER TABLE document_chunks ENABLE ROW LEVEL SECURITY;
ALTER TABLE document_pages ENABLE ROW LEVEL SECURITY;

CREATE POLICY users_select ON users
    FOR SELECT USING (id = auth.uid());

CREATE POLICY users_update ON users
    FOR UPDATE USING (id = auth.uid());

CREATE POLICY api_keys_select ON api_keys
    FOR SELECT USING (user_id = auth.uid());

CREATE POLICY knowledge_bases_select ON knowledge_bases
    FOR SELECT USING (user_id = auth.uid());

CREATE POLICY documents_select ON documents
    FOR SELECT USING (user_id = auth.uid());

CREATE POLICY document_pages_select ON document_pages
    FOR SELECT USING (
        EXISTS (
            SELECT 1 FROM documents
            WHERE documents.id = document_pages.document_id
              AND documents.user_id = auth.uid()
        )
    );

CREATE POLICY document_chunks_select ON document_chunks
    FOR SELECT USING (user_id = auth.uid());

CREATE OR REPLACE FUNCTION generate_slug(name TEXT, p_user_id UUID)
RETURNS TEXT
LANGUAGE plpgsql
AS $$
DECLARE
    base_slug TEXT;
    candidate TEXT;
    counter INTEGER := 0;
BEGIN
    base_slug := lower(regexp_replace(trim(name), '[^a-zA-Z0-9]+', '-', 'g'));
    base_slug := trim(both '-' from base_slug);
    IF base_slug = '' THEN
        base_slug := 'untitled';
    END IF;
    candidate := base_slug;
    LOOP
        IF NOT EXISTS (
            SELECT 1 FROM knowledge_bases
            WHERE slug = candidate AND user_id = p_user_id
        ) THEN
            RETURN candidate;
        END IF;
        counter := counter + 1;
        candidate := base_slug || '-' || counter;
    END LOOP;
END;
$$;

CREATE OR REPLACE FUNCTION handle_new_user()
RETURNS TRIGGER
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = public
AS $$
BEGIN
    INSERT INTO public.users (id, email, display_name)
    VALUES (
        NEW.id,
        NEW.email,
        COALESCE(NEW.raw_user_meta_data ->> 'display_name', NEW.raw_user_meta_data ->> 'full_name')
    );
    RETURN NEW;
END;
$$;

CREATE OR REPLACE FUNCTION set_updated_at()
RETURNS TRIGGER
LANGUAGE plpgsql
AS $$
BEGIN
    NEW.updated_at := now();
    RETURN NEW;
END;
$$;

CREATE OR REPLACE FUNCTION set_knowledge_base_slug()
RETURNS TRIGGER
LANGUAGE plpgsql
AS $$
BEGIN
    IF NEW.slug IS NULL OR NEW.slug = '' THEN
        NEW.slug := generate_slug(NEW.name, NEW.user_id);
    END IF;
    RETURN NEW;
END;
$$;

CREATE TRIGGER on_auth_user_created
    AFTER INSERT ON auth.users
    FOR EACH ROW
    EXECUTE FUNCTION handle_new_user();

CREATE TRIGGER set_knowledge_base_slug
    BEFORE INSERT ON knowledge_bases
    FOR EACH ROW
    EXECUTE FUNCTION set_knowledge_base_slug();

CREATE TRIGGER set_users_updated_at
    BEFORE UPDATE ON users
    FOR EACH ROW
    EXECUTE FUNCTION set_updated_at();

CREATE TRIGGER set_knowledge_bases_updated_at
    BEFORE UPDATE ON knowledge_bases
    FOR EACH ROW
    EXECUTE FUNCTION set_updated_at();

CREATE TRIGGER set_documents_updated_at
    BEFORE UPDATE ON documents
    FOR EACH ROW
    EXECUTE FUNCTION set_updated_at();

CREATE OR REPLACE FUNCTION set_document_number()
RETURNS TRIGGER
LANGUAGE plpgsql
AS $$
BEGIN
    PERFORM pg_advisory_xact_lock(hashtext(NEW.knowledge_base_id::text));
    NEW.document_number := COALESCE(
        (SELECT MAX(document_number) FROM documents WHERE knowledge_base_id = NEW.knowledge_base_id),
        0
    ) + 1;
    RETURN NEW;
END;
$$;

CREATE TRIGGER set_document_number
    BEFORE INSERT ON documents
    FOR EACH ROW
    EXECUTE FUNCTION set_document_number();

CREATE UNIQUE INDEX idx_documents_kb_number ON documents(knowledge_base_id, document_number);

GRANT USAGE ON SCHEMA public TO authenticated;
GRANT SELECT ON ALL TABLES IN SCHEMA public TO authenticated;
-- Supabase grants full CRUD to authenticated by default; mirror those grants
-- so RLS write tests and scoped_execute work correctly.
GRANT INSERT, UPDATE, DELETE ON document_references TO authenticated;
GRANT INSERT, UPDATE, DELETE ON documents TO authenticated;
GRANT INSERT, UPDATE, DELETE ON knowledge_bases TO authenticated;
GRANT UPDATE ON users TO authenticated;

-- Document change notification trigger (mirrors 003_document_notify.sql)
CREATE OR REPLACE FUNCTION notify_document_change() RETURNS trigger AS $$
BEGIN
  IF TG_OP = 'DELETE' THEN
    PERFORM pg_notify('document_changes', json_build_object(
      'event', TG_OP,
      'id', OLD.id::text,
      'knowledge_base_id', OLD.knowledge_base_id::text,
      'user_id', OLD.user_id::text
    )::text);
    RETURN OLD;
  ELSE
    PERFORM pg_notify('document_changes', json_build_object(
      'event', TG_OP,
      'id', NEW.id::text,
      'knowledge_base_id', NEW.knowledge_base_id::text,
      'user_id', NEW.user_id::text
    )::text);
    RETURN NEW;
  END IF;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER document_change_trigger
  AFTER INSERT OR UPDATE OR DELETE ON documents
  FOR EACH ROW EXECUTE FUNCTION notify_document_change();

-- Mirror of 006_kb_sharing.sql so the test DB reflects the live schema.
CREATE TYPE kb_visibility AS ENUM ('private', 'shared', 'public');

ALTER TABLE knowledge_bases
    ADD COLUMN visibility kb_visibility NOT NULL DEFAULT 'private',
    ADD COLUMN public_slug TEXT,
    ADD COLUMN share_token TEXT NOT NULL DEFAULT replace(gen_random_uuid()::text, '-', ''),
    ADD COLUMN visibility_updated_at TIMESTAMPTZ,
    ADD COLUMN published_at TIMESTAMPTZ,
    ADD CONSTRAINT knowledge_bases_public_slug_format
        CHECK (public_slug IS NULL OR public_slug ~ '^[a-z0-9][a-z0-9-]{0,78}[a-z0-9]$'),
    ADD CONSTRAINT knowledge_bases_public_requires_slug
        CHECK (visibility <> 'public' OR public_slug IS NOT NULL);

CREATE UNIQUE INDEX idx_knowledge_bases_public_slug
    ON knowledge_bases (public_slug)
    WHERE public_slug IS NOT NULL;

CREATE UNIQUE INDEX idx_knowledge_bases_share_token
    ON knowledge_bases (share_token);

CREATE INDEX idx_knowledge_bases_public_lookup
    ON knowledge_bases (public_slug, updated_at)
    WHERE visibility = 'public';

-- 010: corpus pipeline state (hosted parity)
CREATE TABLE IF NOT EXISTS corpus_pipeline (
    doc_id UUID PRIMARY KEY REFERENCES documents(id) ON DELETE CASCADE,
    state TEXT NOT NULL CHECK (state IN ('imported', 'excluded', 'failed')),
    attempts INTEGER DEFAULT 1,
    error TEXT DEFAULT '',
    entry_id TEXT DEFAULT '',
    updated_at TIMESTAMPTZ DEFAULT now()
);

-- Mirror of 012_background_jobs.sql. Keep this definition aligned with the
-- migration because integration tests build a schema directly from this file.
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

CREATE UNIQUE INDEX background_jobs_one_active_graph
    ON background_jobs(user_id, knowledge_base_id, job_type)
    WHERE job_type = 'graph.rebuild'
      AND state IN ('queued', 'running', 'retry_wait');

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

-- Version-fenced pgvector storage. Mixed embedding dimensions intentionally use
-- exact cosine scans; a global ANN index cannot safely span those dimensions.

CREATE EXTENSION IF NOT EXISTS vector;

CREATE UNIQUE INDEX chunk_embeddings_kb_owner_ref
    ON knowledge_bases (id, user_id);
CREATE UNIQUE INDEX chunk_embeddings_document_owner_ref
    ON documents (id, user_id, knowledge_base_id);
CREATE UNIQUE INDEX chunk_embeddings_chunk_owner_version_ref
    ON document_chunks (
        document_id, document_version, chunk_index, user_id, knowledge_base_id
    );

CREATE TABLE chunk_embeddings (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL,
    knowledge_base_id UUID NOT NULL,
    document_id UUID NOT NULL,
    document_version INTEGER NOT NULL CHECK (document_version >= 0),
    chunk_index INTEGER NOT NULL CHECK (chunk_index >= 0),
    provider TEXT NOT NULL CHECK (provider <> '' AND length(provider) <= 100),
    model TEXT NOT NULL CHECK (model <> '' AND length(model) <= 200),
    dimensions INTEGER NOT NULL CHECK (dimensions BETWEEN 1 AND 4096),
    embedding vector NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    CONSTRAINT chunk_embeddings_vector_dimensions
        CHECK (vector_dims(embedding) = dimensions),
    CONSTRAINT chunk_embeddings_user_fk
        FOREIGN KEY (user_id) REFERENCES users (id) ON DELETE CASCADE,
    CONSTRAINT chunk_embeddings_kb_owner_fk
        FOREIGN KEY (knowledge_base_id, user_id)
        REFERENCES knowledge_bases (id, user_id) ON DELETE CASCADE,
    CONSTRAINT chunk_embeddings_document_owner_fk
        FOREIGN KEY (document_id, user_id, knowledge_base_id)
        REFERENCES documents (id, user_id, knowledge_base_id)
        ON DELETE CASCADE,
    CONSTRAINT chunk_embeddings_chunk_owner_version_fk
        FOREIGN KEY (
            document_id, document_version, chunk_index, user_id, knowledge_base_id
        ) REFERENCES document_chunks (
            document_id, document_version, chunk_index, user_id, knowledge_base_id
        ) ON DELETE CASCADE,
    UNIQUE (document_id, document_version, chunk_index, provider, model, dimensions)
);

CREATE INDEX chunk_embeddings_scope_idx
    ON chunk_embeddings (
        user_id, knowledge_base_id, provider, model, dimensions, document_version
    );
CREATE INDEX chunk_embeddings_document_idx
    ON chunk_embeddings (document_id, document_version);

ALTER TABLE chunk_embeddings ENABLE ROW LEVEL SECURITY;

CREATE POLICY chunk_embeddings_select ON chunk_embeddings
    FOR SELECT TO authenticated
    USING (user_id = auth.uid());

GRANT SELECT ON chunk_embeddings TO authenticated;

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
    reserved_tokens INTEGER NOT NULL DEFAULT 0 CHECK (
        reserved_tokens BETWEEN 0 AND 250000
        AND ((step_type = 'draft' AND reserved_tokens > 0)
            OR (step_type <> 'draft' AND reserved_tokens = 0))
    ),
    input_tokens INTEGER NOT NULL DEFAULT 0 CHECK (input_tokens >= 0),
    output_tokens INTEGER NOT NULL DEFAULT 0 CHECK (output_tokens >= 0),
    total_tokens INTEGER NOT NULL DEFAULT 0 CHECK (
        total_tokens >= 0 AND total_tokens = input_tokens + output_tokens
        AND (step_type <> 'draft' OR total_tokens <= reserved_tokens)
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
