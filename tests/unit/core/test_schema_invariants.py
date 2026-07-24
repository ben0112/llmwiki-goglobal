from pathlib import Path

ROOT = Path(__file__).parents[3]


def test_sqlite_schema_tracks_derived_versions():
    schema = (ROOT / "shared/sqlite_schema.sql").read_text(encoding="utf-8")
    assert schema.count("document_version INTEGER NOT NULL DEFAULT 0") == 2


def test_postgres_migration_adds_explicit_kind_and_versions():
    sql = (ROOT / "supabase/migrations/011_document_invariants.sql").read_text(encoding="utf-8")
    assert "ADD COLUMN source_kind" in sql
    assert "document_pages" in sql and "document_chunks" in sql
    assert sql.count("document_version") >= 4
