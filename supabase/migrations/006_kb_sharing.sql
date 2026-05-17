-- Knowledge base sharing: three visibility levels.
--   private  — default; owner-only via RLS, existing behavior.
--   shared   — anyone with the share_token URL can read; not indexed.
--   public   — anyone can read at /w/<public_slug>; indexed, OG previews.
--
-- The public read path does NOT go through RLS / the anon Supabase key.
-- Instead, FastAPI exposes /v1/public/wiki/{slug} which queries Postgres
-- with the service-role pool and a hardcoded `visibility = 'public'`
-- filter. That keeps the public surface to one auditable endpoint.
--
-- share_token is generated for every KB up front so flipping a KB to
-- "shared" later doesn't require a backfill. public_slug is global
-- (cross-user) and only set when the owner opts in to "public".

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

-- Partial unique index: public_slug is globally unique among KBs that have one.
CREATE UNIQUE INDEX idx_knowledge_bases_public_slug
    ON knowledge_bases (public_slug)
    WHERE public_slug IS NOT NULL;

-- share_token is unique across all KBs (the URL is the security model
-- for the "shared" tier — a non-unique token would let two owners share
-- collision-prone links).
CREATE UNIQUE INDEX idx_knowledge_bases_share_token
    ON knowledge_bases (share_token);

-- Lookup index for the public read path. Partial so it only indexes
-- KBs that are actually published; cheap to maintain.
CREATE INDEX idx_knowledge_bases_public_lookup
    ON knowledge_bases (public_slug, updated_at)
    WHERE visibility = 'public';
