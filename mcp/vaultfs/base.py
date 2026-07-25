import json
from abc import ABC, abstractmethod
from time import perf_counter

from llmwiki_core.documents import DocumentKind
from llmwiki_core.search import SearchHit, SearchQuery, SearchResult
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
        """Bridge legacy-only adapters into the typed retrieval contract."""
        legacy_search = type(self).search_chunks
        if legacy_search is VaultFS.search_chunks:
            raise NotImplementedError(
                "VaultFS subclass must implement retrieve or search_chunks"
            )
        started_at = perf_counter()
        path_filter = None if query.area.value == "all" else query.area.value
        rows = await legacy_search(
            self,
            kb_id,
            query.text,
            query.candidate_limit,
            path_filter,
            query.annotated_only,
            query.scope.value,
            dict(query.facets),
        )
        if not isinstance(rows, list):
            raise TypeError("legacy search_chunks must return a list of dictionaries")
        hits = tuple(
            _legacy_search_row_to_hit(row, index)
            for index, row in enumerate(rows[: query.candidate_limit])
        )
        return SearchResult(
            hits=hits,
            candidate_count=len(rows),
            latency_ms=(perf_counter() - started_at) * 1000,
            profile="lexical",
        )

    async def search_chunks(
        self, kb_id: str, query: str, limit: int,
        path_filter: str | None = None,
        annotated_only: bool = False,
        scope: str = "all",
        facets: dict | None = None,
    ) -> list[dict]:
        """Compatibility facade over the typed lexical retrieval contract."""
        if type(self).retrieve is VaultFS.retrieve:
            raise NotImplementedError(
                "VaultFS subclass must implement retrieve or search_chunks"
            )
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


def _legacy_search_row_to_hit(row: object, index: int) -> SearchHit:
    if not isinstance(row, dict):
        raise TypeError("legacy search_chunks rows must be dictionaries")
    filename = row.get("filename")
    directory = row.get("path", "/")
    if not isinstance(filename, str) or not filename.strip():
        raise ValueError("legacy search_chunks row filename must be nonblank")
    if not isinstance(directory, str) or not directory.startswith("/"):
        raise ValueError("legacy search_chunks row path must be absolute")
    raw_metadata = row.get("metadata")
    if isinstance(raw_metadata, str):
        try:
            raw_metadata = json.loads(raw_metadata)
        except (json.JSONDecodeError, TypeError) as exc:
            raise ValueError("legacy search_chunks row metadata must be valid JSON") from exc
    if raw_metadata is not None and not isinstance(raw_metadata, dict):
        raise TypeError("legacy search_chunks row metadata must be a dictionary")
    metadata = dict(raw_metadata or {})
    raw_tags = row.get("tags")
    metadata.update(
        {
            "_filename": filename,
            "_directory": directory,
            "_file_type": row.get("file_type", ""),
            "_source_content": row.get("source_content") or "",
            "_annotations_text": row.get("annotations_text"),
            "_has_highlight": bool(row.get("has_highlight", False)),
            "_legacy_tags": raw_tags,
            "source_hit": bool(row.get("source_hit", False)),
            "annotation_hit": bool(row.get("annotation_hit", False)),
        }
    )
    source_kind = row.get("source_kind")
    if source_kind is None:
        source_kind = "wiki" if is_wiki_directory(directory) else "source"
    return SearchHit(
        document_id=str(
            row.get("document_id")
            or row.get("id")
            or f"legacy:{directory}{filename}"
        ),
        document_version=row.get("document_version", 0),
        chunk_index=row.get("chunk_index", index),
        content=row.get("content"),
        score=row.get("score", 0.0),
        path=f"{directory}{filename}",
        title=row.get("title"),
        page=row.get("page"),
        header_breadcrumb=row.get("header_breadcrumb"),
        tags=() if raw_tags is None else raw_tags,
        document_kind=DocumentKind(source_kind),
        metadata=metadata,
    )


def search_hit_to_legacy_dict(hit: SearchHit) -> dict:
    """Convert a typed hit back to the dictionary schema used by MCP tools."""
    metadata = dict(hit.metadata)
    filename = metadata.pop("_filename", hit.path.rsplit("/", 1)[-1])
    directory = metadata.pop("_directory", hit.path.removesuffix(filename))
    file_type = metadata.pop("_file_type", filename.rsplit(".", 1)[-1] if "." in filename else "")
    source_content = metadata.pop("_source_content", "")
    annotations_text = metadata.pop("_annotations_text", None)
    has_highlight = bool(metadata.pop("_has_highlight", False))
    raw_tags = metadata.pop("_legacy_tags", _NO_LEGACY_TAGS)
    source_hit = bool(metadata.pop("source_hit", False))
    annotation_hit = bool(metadata.pop("annotation_hit", False))
    if raw_tags is _NO_LEGACY_TAGS:
        legacy_tags = list(hit.tags)
    elif raw_tags is None:
        legacy_tags = None
    else:
        legacy_tags = list(raw_tags)
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


_NO_LEGACY_TAGS = object()
