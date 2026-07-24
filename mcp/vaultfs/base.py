from abc import ABC, abstractmethod

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

    @abstractmethod
    async def search_chunks(
        self, kb_id: str, query: str, limit: int,
        path_filter: str | None = None,
        annotated_only: bool = False,
        scope: str = "all",
        facets: dict | None = None,
    ) -> list[dict]: ...

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
