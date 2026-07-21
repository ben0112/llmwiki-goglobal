-- 010: corpus classification pipeline state (L3-P4 hosted parity).
-- Tracks per-source-document decisions made by corpus/pipeline.py
-- (audit LLM → eight-dimension classify → import). Service-role access
-- only — written by the pipeline CLI / backend, never via PostgREST.

CREATE TABLE IF NOT EXISTS corpus_pipeline (
    doc_id UUID PRIMARY KEY REFERENCES documents(id) ON DELETE CASCADE,
    state TEXT NOT NULL CHECK (state IN ('imported', 'excluded', 'failed')),
    attempts INTEGER DEFAULT 1,
    error TEXT DEFAULT '',
    entry_id TEXT DEFAULT '',
    updated_at TIMESTAMPTZ DEFAULT now()
);

ALTER TABLE corpus_pipeline ENABLE ROW LEVEL SECURITY;
-- No policies on purpose: only the service role (bypasses RLS) touches it.
