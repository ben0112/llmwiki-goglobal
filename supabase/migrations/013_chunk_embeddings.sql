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
