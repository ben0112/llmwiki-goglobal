from pathlib import Path

ROOT = Path(__file__).parents[3]


def test_sqlite_schema_tracks_derived_versions():
    schema = (ROOT / "shared/sqlite_schema.sql").read_text(encoding="utf-8")
    assert schema.count("document_version INTEGER NOT NULL DEFAULT 0") == 2


def test_sqlite_read_model_schema_is_revisioned_and_indexed():
    schema = (ROOT / "shared/sqlite_schema.sql").read_text(encoding="utf-8")
    read_schema = (ROOT / "shared/sqlite_read_models.sql").read_text(encoding="utf-8")

    assert "read_revision INTEGER NOT NULL DEFAULT 1" in schema
    assert read_schema.count("read_revision = read_revision + 1") == 3
    for name in (
        "idx_documents_browse_name",
        "idx_documents_browse_date",
        "idx_documents_wiki_path",
        "idx_documents_number",
        "idx_documents_content_hash",
    ):
        assert name in read_schema


def test_postgres_migration_adds_explicit_kind_and_versions():
    sql = (ROOT / "supabase/migrations/011_document_invariants.sql").read_text(encoding="utf-8")
    assert "ADD COLUMN source_kind" in sql
    assert "document_pages" in sql and "document_chunks" in sql
    assert sql.count("document_version") >= 4
