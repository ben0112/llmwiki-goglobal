"""Facet filters over documents.metadata — the 八维 classification dimensions.

Corpus entries imported by corpus/import_annotations.py carry a structured
eight-dimension record in documents.metadata. Each facet below maps to a SQL
condition on that JSON; multi-value dimensions match the primary label or any
secondary label. Documents without corpus metadata simply never match, so a
facet filter implicitly narrows the search to classified corpus entries.

The same facet keys work on both backends; only the SQL dialect differs.
"""

from llmwiki_core.facets import (
    FACET_KEYS,
    UnknownFacetError,
    postgres_facet_conditions,
    sqlite_facet_conditions,
    validate_facets,
)

__all__ = [
    "FACET_KEYS",
    "UnknownFacetError",
    "postgres_facet_conditions",
    "sqlite_facet_conditions",
    "validate_facets",
]

# facet key -> ("scalar", json path) | ("array", json path) | special-cased
