"""Atomic wiki-write values shared by storage adapters and tools."""

from dataclasses import dataclass
from typing import Any, Iterable

from .references import ReferenceEdge


class VersionConflict(RuntimeError):
    """Raised when a compare-and-swap wiki update observes a stale version."""


@dataclass(frozen=True)
class WikiWriteBundle:
    """Everything derived from one wiki revision and committed together."""

    document_id: str
    expected_version: int | None
    filename: str
    path: str
    file_type: str
    content: str
    title: str | None
    tags: tuple[str, ...]
    date: str | None
    metadata: dict[str, Any]
    edges: tuple[ReferenceEdge, ...]

    @classmethod
    def build(
        cls,
        *,
        edges: Iterable[ReferenceEdge],
        tags: Iterable[str],
        **values: Any,
    ) -> "WikiWriteBundle":
        deduped: dict[tuple[str, str], ReferenceEdge] = {}
        for edge in edges:
            deduped.setdefault((edge.target_id, edge.reference_type), edge)
        return cls(tags=tuple(tags), edges=tuple(deduped.values()), **values)
