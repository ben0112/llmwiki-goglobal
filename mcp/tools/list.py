"""Knowledge base tools — create and enumerate knowledge bases."""

from mcp.server.fastmcp import FastMCP, Context

from config import settings


def register(mcp: FastMCP, get_user_id, fs_factory) -> None:

    @mcp.tool(
        name="create_knowledge_base",
        description=(
            "Create a new knowledge base/wiki and scaffold starter overview/log pages.\n\n"
            "In hosted mode this creates a separate knowledge base with a unique slug. "
            "In local MCP mode there is one workspace per server, so this returns the "
            "existing workspace if it has already been initialized."
        ),
    )
    async def create_knowledge_base(
        ctx: Context,
        name: str,
        description: str = "",
    ) -> str:
        name = name.strip()
        description = description.strip()
        if not name:
            return "Error: name is required when creating a knowledge base."
        if len(name) > 120:
            return "Error: knowledge base name must be 120 characters or fewer."

        user_id = get_user_id(ctx)
        fs = fs_factory(user_id)
        kb = await fs.create_knowledge_base(name, description or None)

        if kb.get("already_exists"):
            if kb.get("local_singleton"):
                return (
                    "Local MCP mode uses one workspace per server. "
                    f"Existing knowledge base: **{kb['name']}** (`{kb['slug']}`). "
                    "Use that slug with the other tools."
                )
            return f"Knowledge base already exists: **{kb['name']}** (`{kb['slug']}`)."

        return (
            f"Created knowledge base **{kb['name']}** (`{kb['slug']}`). "
            "Starter pages were added at `/wiki/overview.md` and `/wiki/log.md`. "
            f"Use `knowledge_base=\"{kb['slug']}\"` with the other tools."
        )

    @mcp.tool(
        name="list_knowledge_bases",
        description=(
            "List the user's knowledge bases with their names and slugs.\n\n"
            "Every other tool takes a `knowledge_base` slug — call this first to "
            "discover the valid slugs, or whenever you need to confirm which "
            "knowledge bases exist."
        ),
    )
    async def list_knowledge_bases(ctx: Context) -> str:
        user_id = get_user_id(ctx)
        fs = fs_factory(user_id)
        kbs = await fs.list_knowledge_bases()
        if not kbs:
            return (
                "No knowledge bases yet. Use `create_knowledge_base`, "
                f"or create one at {settings.APP_URL}/wikis."
            )

        lines = [f"- **{kb['name']}** (`{kb['slug']}`)" for kb in kbs]
        return "\n".join(lines)
