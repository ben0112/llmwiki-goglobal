"""Transport-only contracts for bounded knowledge-base read APIs."""

from __future__ import annotations

from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, Field

MAX_READ_ITEMS = 200


class ReadSort(StrEnum):
    """Sort identifiers that adapters may map to static query fragments."""

    NAME = "name"
    DATE = "date"
    TYPE = "type"
    PATH = "path"


class SortDirection(StrEnum):
    ASC = "asc"
    DESC = "desc"


class StaleReadCursor(ValueError):
    """A cursor was issued for an older authoritative read revision."""

    def __init__(self, revision: int):
        super().__init__("stale read cursor")
        self.revision = revision


class ResolvedDocument(BaseModel):
    """Lightweight document projection shared by browse and resolver reads."""

    id: str = Field(min_length=1, max_length=64)
    filename: str = Field(min_length=1, max_length=1024)
    path: str = Field(min_length=1, max_length=4096)
    file_type: str = Field(min_length=1, max_length=64)
    status: str = Field(min_length=1, max_length=64)
    knowledge_base_id: str | None = Field(default=None, max_length=64)
    title: str | None = Field(default=None, max_length=2048)
    file_size: int | None = Field(default=None, ge=0)
    page_count: int | None = Field(default=None, ge=0)
    tags: list[str] = Field(default_factory=list, max_length=MAX_READ_ITEMS)
    date: str | None = Field(default=None, max_length=64)
    metadata: dict[str, Any] | None = None
    error_message: str | None = None
    version: int | None = Field(default=None, ge=0)
    document_number: int | None = Field(default=None, ge=1)
    sort_order: int | None = None
    archived: bool = False
    stale_since: str | None = Field(default=None, max_length=64)
    created_at: str | None = Field(default=None, max_length=64)
    updated_at: str | None = Field(default=None, max_length=64)


class ReadPage(BaseModel):
    revision: int = Field(strict=True, ge=1)
    items: list[dict[str, Any]] = Field(default_factory=list, max_length=MAX_READ_ITEMS)
    next_cursor: str | None = Field(default=None, max_length=2048)
    total_count: int = Field(default=0, ge=0)


class FolderItem(BaseModel):
    name: str = Field(min_length=1, max_length=1024)
    path: str = Field(min_length=1, max_length=4096)
    document_count: int = Field(ge=0)


class BrowsePage(ReadPage):
    items: list[ResolvedDocument] = Field(default_factory=list, max_length=MAX_READ_ITEMS)
    folders: list[FolderItem] = Field(default_factory=list, max_length=MAX_READ_ITEMS)
    source_count: int = Field(default=0, ge=0)
    failed_count: int = Field(default=0, ge=0)
    corpus_count: int = Field(default=0, ge=0)


class CorpusSummary(BaseModel):
    """Fixed-size corpus aggregates; never contains corpus entry rows."""

    revision: int = Field(strict=True, ge=1)
    total_count: int = Field(ge=0)
    filtered_count: int = Field(ge=0)
    facets: dict[str, dict[str, int]] = Field(default_factory=dict)
    coverage: dict[str, Any] = Field(default_factory=dict)
    business_classes: dict[str, int] = Field(default_factory=dict)
    business_scenes: dict[str, int] = Field(default_factory=dict)
    kpis: dict[str, int | float | None] = Field(default_factory=dict)


class GraphSummary(BaseModel):
    revision: int = Field(strict=True, ge=1)
    node_count: int = Field(default=0, ge=0)
    edge_count: int = Field(default=0, ge=0)
    cited_document_ids: list[str] = Field(default_factory=list, max_length=MAX_READ_ITEMS)


class DocumentStatus(BaseModel):
    id: str | None = Field(default=None, max_length=64)
    document_number: int | None = Field(default=None, ge=1)
    status: str = Field(min_length=1, max_length=64)
    error_message: str | None = None
    version: int | None = Field(default=None, ge=0)


class DocumentStatusPage(BaseModel):
    revision: int = Field(strict=True, ge=1)
    items: list[DocumentStatus] = Field(default_factory=list, max_length=MAX_READ_ITEMS)


class UploadPreflightItem(BaseModel):
    """One upload descriptor and its optional preflight decision."""

    path: str = Field(min_length=1, max_length=4096)
    filename: str = Field(min_length=1, max_length=1024)
    size: int = Field(ge=0)
    sha256: str | None = Field(default=None, min_length=64, max_length=64)
    accepted: bool | None = None
    code: (
        Literal[
            "accepted",
            "duplicate_name",
            "duplicate_content",
            "unsupported",
            "too_large",
        ]
        | None
    ) = None
    existing_document_id: str | None = Field(default=None, max_length=64)


class UploadPreflightRequest(BaseModel):
    items: list[UploadPreflightItem] = Field(min_length=1, max_length=MAX_READ_ITEMS)


class UploadPreflightResponse(BaseModel):
    revision: int = Field(strict=True, ge=1)
    items: list[UploadPreflightItem] = Field(default_factory=list, max_length=MAX_READ_ITEMS)


def etag_for_revision(revision: int) -> str:
    return f'"kb-read-{revision}"'


def etag_matches(if_none_match: str | None, revision: int) -> bool:
    """Return whether a complete If-None-Match list token is current."""

    if not if_none_match:
        return False
    expected = etag_for_revision(revision)
    for raw_token in if_none_match.split(","):
        token = raw_token.strip()
        if token == "*":
            return True
        if token.startswith("W/"):
            token = token[2:]
        if token == expected:
            return True
    return False


__all__ = [
    "BrowsePage",
    "CorpusSummary",
    "DocumentStatus",
    "DocumentStatusPage",
    "FolderItem",
    "GraphSummary",
    "MAX_READ_ITEMS",
    "ReadPage",
    "ReadSort",
    "ResolvedDocument",
    "SortDirection",
    "StaleReadCursor",
    "UploadPreflightItem",
    "UploadPreflightRequest",
    "UploadPreflightResponse",
    "etag_for_revision",
    "etag_matches",
]
