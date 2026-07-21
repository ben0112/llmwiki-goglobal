-- 009: relation-layer edge types (corpus spec v2026.06 §2.5 关系层).
--
-- 'cites' and 'links_to' remain content-derived (rebuilt from wiki page text
-- on every write). The five new types are curated via the MCP `relate` tool:
--   is_a 上下位 · next 前后置 · routes_to 路径衔接 ·
--   governed_by 归口映射 · serves 阶段服务包
-- Rebuild paths delete only the two derived types, so curated relation edges
-- survive content rebuilds.

ALTER TABLE document_references
    DROP CONSTRAINT IF EXISTS document_references_reference_type_check;

ALTER TABLE document_references
    ADD CONSTRAINT document_references_reference_type_check CHECK (
        reference_type IN (
            'cites', 'links_to',
            'is_a', 'next', 'routes_to', 'governed_by', 'serves'
        )
    );
