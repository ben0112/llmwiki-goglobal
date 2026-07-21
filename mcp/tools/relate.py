"""Relate tool — curate the relation layer (关系层) between documents.

The corpus spec (v2026.06 §2.5) models five relation types beyond the
content-derived citation graph. These edges are created deliberately (by
Claude or an operator), stored in document_references, and preserved across
content-driven rebuilds:

  is_a        上下位     concept hierarchy / controlled-vocabulary broader-narrower
  next        前后置     process order — source comes first, target follows
  routes_to   路径衔接   path handoff, e.g. 调解 → 仲裁 → 诉讼
  governed_by 归口映射   rule/entry → its responsible department's page
  serves      阶段服务包 stage/readiness page → recommended service or corpus set
"""

import logging
from typing import Literal

from mcp.server.fastmcp import Context, FastMCP
from vaultfs import VaultFS
from vaultfs.base import RELATION_TYPES

from .helpers import resolve_path

logger = logging.getLogger(__name__)

Relation = Literal["is_a", "next", "routes_to", "governed_by", "serves"]


class RelateHandler:
    """Adds and removes typed relation edges between two documents."""

    def __init__(self, fs: VaultFS, kb: dict):
        self.fs = fs
        self.kb = kb
        self.kb_id = str(kb["id"])

    async def _resolve(self, path: str) -> dict | None:
        dir_path, filename = resolve_path(path)
        doc = await self.fs.get_document(self.kb_id, filename, dir_path)
        if not doc:
            doc = await self.fs.find_document_by_name(self.kb_id, filename)
        return doc

    async def run(self, source: str, target: str, relation: str, action: str) -> str:
        if relation not in RELATION_TYPES:
            valid = ", ".join(f"{k} ({v})" for k, v in RELATION_TYPES.items())
            return f"Unknown relation '{relation}'. Valid relations: {valid}"

        src = await self._resolve(source)
        if not src:
            return f"Source document '{source}' not found."
        tgt = await self._resolve(target)
        if not tgt:
            return f"Target document '{target}' not found."
        if str(src["id"]) == str(tgt["id"]):
            return "Source and target are the same document — a relation must connect two documents."

        label = RELATION_TYPES[relation]
        src_path = f"{src['path']}{src['filename']}"
        tgt_path = f"{tgt['path']}{tgt['filename']}"

        if action == "remove":
            removed = await self.fs.delete_reference(str(src["id"]), str(tgt["id"]), relation)
            if not removed:
                return f"No {relation} ({label}) edge exists from {src_path} to {tgt_path}."
            return f"Removed: {src_path} —{relation} ({label})→ {tgt_path}"

        await self.fs.upsert_reference(str(src["id"]), str(tgt["id"]), self.kb_id, relation, None)
        return (
            f"Related: {src_path} —{relation} ({label})→ {tgt_path}\n"
            f"The edge is curated and survives content rebuilds; "
            f"see it via search(mode=\"references\", path=\"{src_path}\")."
        )


def register(mcp: FastMCP, get_user_id, fs_factory) -> None:
    @mcp.tool(
        name="relate",
        description=(
            "Create or remove a curated relation edge between two documents (关系层).\n\n"
            "Relations (direction: source → target):\n"
            "- is_a 上下位: source is a narrower concept of target\n"
            "- next 前后置: source comes first in a process, target follows "
            "(e.g. ODI 备案 → 落地设立)\n"
            "- routes_to 路径衔接: source hands off to target "
            "(e.g. 调解 → 仲裁 → 诉讼)\n"
            "- governed_by 归口映射: source is governed by the department described in target\n"
            "- serves 阶段服务包: source (stage/readiness page) recommends target "
            "(service or corpus set)\n\n"
            "Unlike footnote citations and wiki links — which are re-derived from page "
            "content on every write — these edges persist until removed with "
            "action=\"remove\". They appear in search(mode=\"references\") and the graph view."
        ),
    )
    async def relate(
        ctx: Context,
        knowledge_base: str,
        source: str,
        target: str,
        relation: Relation,
        action: Literal["add", "remove"] = "add",
    ) -> str:
        user_id = get_user_id(ctx)
        fs = fs_factory(user_id)

        kb = await fs.resolve_kb(knowledge_base)
        if not kb:
            return f"Knowledge base '{knowledge_base}' not found."

        return await RelateHandler(fs, kb).run(source, target, relation, action)
