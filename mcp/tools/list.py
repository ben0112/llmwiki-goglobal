"""List tool — enumerate the user's knowledge bases."""

from mcp.server.fastmcp import FastMCP, Context

from config import settings


def register(mcp: FastMCP, get_user_id, fs_factory) -> None:

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
            return f"No knowledge bases yet. Create one at {settings.APP_URL}/wikis"

        lines = [f"- **{kb['name']}** (`{kb['slug']}`)" for kb in kbs]
        return "\n".join(lines)
