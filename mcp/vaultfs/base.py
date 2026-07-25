from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from llmwiki_core.models import EmbeddingProfile
from llmwiki_core.search import RetrieverUnavailable, SearchHit, SearchQuery, SearchResult
from llmwiki_core.wiki import WikiWriteBundle

# Content-derived edge types, rebuilt from wiki page text on every write.
CITATION_TYPES = ("cites", "links_to")

# Curated relation-layer edges (corpus spec v2026.06 §2.5) — created via the
# `relate` tool and preserved across content-driven rebuilds.
RELATION_TYPES = {
    "is_a": "上下位",
    "next": "前后置",
    "routes_to": "路径衔接",
    "governed_by": "归口映射",
    "serves": "阶段服务包",
}


class DuplicateDocumentError(Exception):
    """Raised when create_document hits a uniqueness constraint on (kb, path, filename)."""

    def __init__(self, dir_path: str, filename: str):
        self.dir_path = dir_path
        self.filename = filename
        super().__init__(f"document already exists at {dir_path}{filename}")


class VaultFS(ABC):
    """Abstract virtual filesystem for the knowledge vault."""

    user_id: str

    @abstractmethod
    async def resolve_kb(self, slug: str) -> dict | None: ...

    @abstractmethod
    async def list_knowledge_bases(self) -> list[dict]: ...

    @abstractmethod
    async def create_knowledge_base(self, name: str, description: str | None = None, kind: str = "wiki") -> dict: ...

    @abstractmethod
    async def update_knowledge_base(self, kb_id: str, name: str | None = None, description: str | None = None, kind: str | None = None) -> dict | None: ...

    @abstractmethod
    async def get_document(self, kb_id: str, filename: str, dir_path: str) -> dict | None: ...

    @abstractmethod
    async def find_document_by_name(self, kb_id: str, name: str) -> dict | None: ...

    @abstractmethod
    async def create_document(self, kb_id: str, filename: str, title: str, dir_path: str, file_type: str, content: str, tags: list[str], date: str | None = None, metadata: dict | None = None) -> dict: ...

    @abstractmethod
    async def update_document(self, doc_id: str, content: str, tags: list[str] | None = None, title: str | None = None, date: str | None = None, metadata: dict | None = None) -> dict | None: ...

    @abstractmethod
    async def write_wiki_bundle(self, kb_id: str, bundle: WikiWriteBundle) -> dict:
        """Atomically commit a wiki revision and all of its derived state."""
        raise NotImplementedError

    @abstractmethod
    async def archive_documents(self, doc_ids: list[str]) -> int: ...

    @abstractmethod
    async def list_documents(self, kb_id: str, facets: dict | None = None) -> list[dict]: ...

    @abstractmethod
    async def list_documents_with_content(self, kb_id: str) -> list[dict]: ...

    @abstractmethod
    async def get_pages(self, doc_id: str, page_nums: list[int]) -> list[dict]: ...

    @abstractmethod
    async def get_all_pages(self, doc_id: str) -> list[dict]: ...

    async def retrieve(self, kb_id: str, query: SearchQuery) -> SearchResult:
        """Typed retrieval must be implemented natively by each adapter."""
        raise NotImplementedError("typed contract requires native retrieve")

    async def retrieve_vector(
        self,
        kb_id: str,
        query: SearchQuery,
        *,
        embedding: tuple[float, ...],
        profile: EmbeddingProfile,
    ) -> SearchResult:
        """Hosted adapters may implement tenant-scoped current-version vectors."""
        raise RetrieverUnavailable("vector retrieval is unavailable")

    async def expand_references(
        self,
        kb_id: str,
        query: SearchQuery,
        hits: tuple[SearchHit, ...],
        *,
        limit: int,
    ) -> tuple[SearchHit, ...]:
        """Return bounded one-hop context; local adapters deliberately do nothing."""
        return ()

    async def search_chunks(
        self, kb_id: str, query: str, limit: int,
        path_filter: str | None = None,
        annotated_only: bool = False,
        scope: str = "all",
        facets: dict | None = None,
    ) -> list[dict]:
        """Compatibility facade over the typed lexical retrieval contract."""
        request = SearchQuery.build(
            text=query,
            limit=limit,
            candidate_limit=limit,
            area=path_filter,
            scope=scope,
            facets=facets,
            annotated_only=annotated_only,
        )
        result = await self.retrieve(kb_id, request)
        return [search_hit_to_legacy_dict(hit) for hit in result.hits]

    async def corpus_search_context(self, kb_id: str, relpaths: list[str]) -> dict:
        """搜索结果的语料折叠/可信度标记上下文(默认无流水线,普通库零开销)。

        返回 {"has_pipeline": bool,
              "entries": {源文件相对路径: 语料条目相对路径},
              "states":  {源文件相对路径: {"state": ..., "reason": ...}}}
        """
        return {"has_pipeline": False, "entries": {}, "states": {}}

    @abstractmethod
    async def load_source_bytes(self, doc: dict) -> bytes | None: ...

    @abstractmethod
    async def load_image_bytes(self, doc_id: str, image_id: str) -> bytes | None: ...

    @abstractmethod
    async def load_asset_bytes(self, asset_doc_id: str) -> bytes | None: ...

    @abstractmethod
    def write_to_disk(self, dir_path: str, filename: str, content: str) -> bool: ...

    @abstractmethod
    def delete_from_disk(self, docs: list[dict]) -> None: ...

    @abstractmethod
    async def delete_references(self, source_doc_id: str, ref_types: tuple | None = None) -> None:
        """Delete outgoing references; `ref_types` scopes deletion (None = all).

        Content-driven rebuilds pass CITATION_TYPES so curated relation-layer
        edges survive page edits.
        """

    @abstractmethod
    async def delete_reference(self, source_id: str, target_id: str, ref_type: str) -> bool:
        """Delete one edge; returns True when a row was removed."""

    @abstractmethod
    async def upsert_reference(self, source_id: str, target_id: str, kb_id: str, ref_type: str, page: int | None) -> None: ...

    @abstractmethod
    async def propagate_staleness(self, doc_id: str) -> None: ...

    async def mark_cites_stale(self, target_doc_ids: list[str]) -> int:
        """把通过 cites 引用这些文档的维基页面标记为待复查(stale)。

        语料条目复核变更/复审到期时的联动入口;返回新标记的页面数。"""
        return 0

    async def refresh_facet_rollup(self, doc_id: str) -> None:
        """重算单个维基页的 facet_rollup(从其引用的语料条目聚合八维)。"""
        return None

    @abstractmethod
    async def get_backlinks(self, doc_id: str) -> list[dict]: ...

    @abstractmethod
    async def get_forward_references(self, doc_id: str) -> list[dict]: ...

    @abstractmethod
    async def find_uncited_sources(self, kb_id: str) -> list[dict]: ...

    @abstractmethod
    async def find_stale_pages(self, kb_id: str) -> list[dict]: ...


def logical_glob_to_sql_like(path_glob: str) -> str:
    """Translate a normalized logical glob into an escaped SQL LIKE value.

    SQL's own `%`, `_`, and escape character stay literal. `?` is also
    literal; only `*` and `**` are wildcards. Both star forms retain the
    historical `fnmatch` behavior where a wildcard may span `/`. A trailing
    slash (or a bare extensionless path) denotes a directory prefix.
    """
    directory = path_glob.endswith("/") or (
        "*" not in path_glob
        and "." not in path_glob.rsplit("/", 1)[-1]
        and path_glob != "/"
    )
    escaped: list[str] = []
    index = 0
    while index < len(path_glob):
        char = path_glob[index]
        if char == "*":
            if index + 1 < len(path_glob) and path_glob[index + 1] == "*":
                index += 1
            escaped.append("%")
        elif char in {"%", "_", "\\"}:
            escaped.append("\\" + char)
        else:
            escaped.append(char)
        index += 1
    pattern = "".join(escaped)
    if path_glob == "/":
        return "/%"
    if directory:
        return pattern.rstrip("/") + "/%"
    return pattern


def is_wiki_directory(path: str) -> bool:
    """Return whether a logical directory is exactly wiki or below wiki/."""
    normalized = path.replace("\\", "/").strip("/")
    return normalized == "wiki" or normalized.startswith("wiki/")


@dataclass(frozen=True, slots=True)
class _LegacyFields:
    filename: str
    directory: str
    file_type: str
    source_content: str
    annotations_text: str | None
    has_highlight: bool
    raw_tags: tuple[str, ...] | None
    source_hit: bool
    annotation_hit: bool


@dataclass(frozen=True, kw_only=True)
class _VaultSearchHit(SearchHit):
    """Internal transport envelope for the legacy dictionary facade."""

    legacy_fields: _LegacyFields = field(compare=False, repr=False)


def _vault_search_hit(
    *,
    document_id: str,
    document_version: int,
    chunk_index: int,
    content: str,
    score: float,
    path: str,
    title: str | None,
    page: int | None,
    header_breadcrumb: str | None,
    tags: list[str] | tuple[str, ...] | None,
    document_kind: object,
    metadata: dict,
    filename: str,
    directory: str,
    file_type: str,
    source_content: str,
    annotations_text: str | None,
    has_highlight: bool,
    source_hit: bool,
    annotation_hit: bool,
) -> SearchHit:
    """Build a typed hit without storing compatibility fields in metadata."""
    raw_tags = None if tags is None else tuple(tags)
    return _VaultSearchHit(
        document_id=document_id,
        document_version=document_version,
        chunk_index=chunk_index,
        content=content,
        score=score,
        path=path,
        title=title,
        page=page,
        header_breadcrumb=header_breadcrumb,
        tags=() if raw_tags is None else raw_tags,
        document_kind=document_kind,
        metadata=metadata,
        legacy_fields=_LegacyFields(
            filename=filename,
            directory=directory,
            file_type=file_type,
            source_content=source_content,
            annotations_text=annotations_text,
            has_highlight=has_highlight,
            raw_tags=raw_tags,
            source_hit=source_hit,
            annotation_hit=annotation_hit,
        ),
    )


def _typed_path_parts(path: str) -> tuple[str, str]:
    stripped = path.rstrip("/")
    if not stripped:
        return "", "/"
    directory, separator, filename = stripped.rpartition("/")
    if not separator:
        return filename, "/"
    return filename, f"{directory or ''}/"


def search_hit_to_legacy_dict(hit: SearchHit) -> dict:
    """Convert a typed hit back to the dictionary schema used by MCP tools."""
    metadata = dict(hit.metadata)
    if isinstance(hit, _VaultSearchHit):
        fields = hit.legacy_fields
        filename = fields.filename
        directory = fields.directory
        file_type = fields.file_type
        source_content = fields.source_content
        annotations_text = fields.annotations_text
        has_highlight = fields.has_highlight
        legacy_tags = None if fields.raw_tags is None else list(fields.raw_tags)
        source_hit = fields.source_hit
        annotation_hit = fields.annotation_hit
    else:
        filename, directory = _typed_path_parts(hit.path)
        file_type = filename.rsplit(".", 1)[-1] if "." in filename else ""
        source_content = ""
        annotations_text = None
        has_highlight = False
        legacy_tags = list(hit.tags)
        source_hit = False
        annotation_hit = False
    return {
        "document_id": hit.document_id,
        "document_version": hit.document_version,
        "content": hit.content,
        "source_content": source_content,
        "annotations_text": annotations_text,
        "has_highlight": has_highlight,
        "source_hit": source_hit,
        "annotation_hit": annotation_hit,
        "page": hit.page,
        "header_breadcrumb": hit.header_breadcrumb,
        "chunk_index": hit.chunk_index,
        "filename": filename,
        "title": hit.title,
        "path": directory,
        "file_type": file_type,
        "tags": legacy_tags,
        "source_kind": hit.document_kind.value if hit.document_kind is not None else None,
        "metadata": metadata,
        "score": hit.score,
    }
