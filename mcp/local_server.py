"""Local MCP server for stdio (any MCP-capable client).

One workspace = one MCP server. Filesystem is truth. SQLite is the index.

Usage:
    python -m local_server --workspace ~/research
    python -m local_server ~/research
"""

import argparse
import asyncio
import logging
import os
import sys
import uuid
from datetime import date
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("llmwiki.local")

_LOCAL_USER_ID = os.environ.get("LLMWIKI_USER_ID", str(uuid.uuid5(uuid.NAMESPACE_DNS, "local")))
os.environ["SUPAVAULT_USER_ID"] = _LOCAL_USER_ID


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LLM Wiki local MCP server")
    parser.add_argument("workspace", nargs="?", default=".", help="Path to workspace folder")
    parser.add_argument("--workspace", dest="workspace_flag", default=None, help="Path to workspace folder")
    parser.add_argument("--http", action="store_true",
                        help="Serve streamable HTTP on --port instead of stdio (Docker deployments)")
    parser.add_argument("--port", type=int, default=8080, help="Port for --http mode")
    return parser.parse_args()


async def _init_workspace(workspace_path: str) -> None:
    """Initialize workspace: create dirs, SQLite, default workspace row, scaffold wiki files."""
    ws = Path(workspace_path).resolve()

    (ws / "wiki").mkdir(parents=True, exist_ok=True)
    (ws / ".llmwiki").mkdir(parents=True, exist_ok=True)
    (ws / ".llmwiki" / "cache").mkdir(parents=True, exist_ok=True)

    from vaultfs import SqliteVaultFS
    await SqliteVaultFS.init(str(ws))

    fs = SqliteVaultFS(_LOCAL_USER_ID)
    existing = await fs.get_workspace()
    if not existing:
        ws_name = ws.name
        ws_id = await fs.ensure_workspace(ws_name)
        today = date.today().isoformat()
        overview_content = (
            "---\n"
            "title: Overview\n"
            f"description: Research hub for {ws_name}.\n"
            f"date: {today}\n"
            "tags: [overview, wiki]\n"
            "---\n\n"
            f"This wiki tracks research on {ws_name}.\n\n"
            "## Key Findings\n\n"
            "No sources ingested yet.\n\n"
            "## Recent Updates\n\n"
            "No activity yet."
        )
        log_content = "Chronological record of ingests, queries, and maintenance passes."

        await fs.create_document(
            ws_id, "overview.md", "Overview", "/wiki/", "md",
            overview_content,
            ["overview", "wiki"],
            date=today,
            metadata={"description": f"Research hub for {ws_name}."},
        )
        await fs.create_document(
            ws_id, "log.md", "Log", "/wiki/", "md",
            log_content,
            ["log"],
        )

        overview_path = ws / "wiki" / "overview.md"
        if not overview_path.exists():
            overview_path.write_text(overview_content + "\n", encoding="utf-8")
        log_path = ws / "wiki" / "log.md"
        if not log_path.exists():
            log_path.write_text(log_content + "\n", encoding="utf-8")

        logger.info("Initialized workspace: %s", ws)
    else:
        logger.info("Workspace ready: %s", ws)


def main():
    args = _parse_args()
    workspace = args.workspace_flag or args.workspace
    workspace = str(Path(workspace).resolve())

    sys.modules["local_server"] = sys.modules[__name__]

    loop = asyncio.new_event_loop()
    loop.run_until_complete(_init_workspace(workspace))

    from mcp.server.fastmcp import FastMCP
    from tools import register
    from vaultfs import SqliteVaultFS

    mcp = FastMCP(
        name="LLM Wiki",
        instructions=(
            "You are connected to an LLM Wiki workspace. The user has uploaded files, notes, "
            "and documents that you can read, search, edit, and organize. "
            "Call the `guide` tool first to see available knowledge bases and learn the full workflow."
        ),
        # HTTP mode is stateless like the hosted server: every request is
        # self-contained, so curl checks and client reconnects just work.
        stateless_http=args.http,
    )

    def _get_user_id(ctx):
        return _LOCAL_USER_ID

    register(mcp, _get_user_id, lambda user_id: SqliteVaultFS(user_id))

    @mcp.tool(name="ping", description="Test connectivity")
    async def ping() -> str:
        return "pong"

    if args.http:
        # No auth: local mode is single-user and the API on the neighbouring
        # port is equally open — bind/publish on loopback unless you mean to
        # share it (the compose file defaults to 127.0.0.1).
        import uvicorn
        from starlette.requests import Request
        from starlette.responses import PlainTextResponse
        from starlette.routing import Route

        app = mcp.streamable_http_app()

        async def health(request: Request) -> PlainTextResponse:
            return PlainTextResponse("ok")

        app.router.routes.insert(0, Route("/health", health))
        logger.info("Local MCP server (HTTP) ready — workspace: %s, port: %d", workspace, args.port)
        uvicorn.run(app, host="0.0.0.0", port=args.port)
    else:
        logger.info("Local MCP server ready — workspace: %s", workspace)
        asyncio.run(mcp.run_stdio_async())


if __name__ == "__main__":
    main()
