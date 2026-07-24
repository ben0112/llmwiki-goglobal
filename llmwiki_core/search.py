"""Backend-neutral search request and result contracts."""

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from types import MappingProxyType
from typing import Any


class SearchArea(StrEnum):
    ALL = "all"
    WIKI = "wiki"
    SOURCES = "sources"


class SearchScope(StrEnum):
    ALL = "all"
    ANNOTATIONS = "annotations"
    SOURCE = "source"


@dataclass(frozen=True)
class SearchQuery:
    text: str
    limit: int = 20
    area: SearchArea = SearchArea.ALL
    scope: SearchScope = SearchScope.ALL
    facets: Mapping[str, Any] = field(default_factory=lambda: MappingProxyType({}))

    @classmethod
    def build(
        cls,
        *,
        text: str,
        limit: int = 20,
        area: str | SearchArea | None = None,
        scope: str | SearchScope = SearchScope.ALL,
        facets: Mapping[str, Any] | None = None,
    ) -> "SearchQuery":
        normalized_text = text.strip()
        if not normalized_text:
            raise ValueError("search text must not be empty")
        if not 1 <= limit <= 100:
            raise ValueError("search limit must be between 1 and 100")
        return cls(
            text=normalized_text,
            limit=limit,
            area=SearchArea(area or SearchArea.ALL),
            scope=SearchScope(scope),
            facets=MappingProxyType(dict(facets or {})),
        )


@dataclass(frozen=True)
class SearchHit:
    document_id: str
    document_version: int
    chunk_index: int
    content: str
    score: float
    path: str
    title: str | None = None

    @property
    def identity(self) -> tuple[str, int, int]:
        return (self.document_id, self.document_version, self.chunk_index)
