"""WebSocket endpoint for real-time document change notifications.

Replaces Supabase Realtime. The API listens to Postgres NOTIFY on the
'document_changes' channel and pushes events to connected clients.
"""

import asyncio
import json
import logging

import asyncpg
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from auth import verify_token

logger = logging.getLogger(__name__)

router = APIRouter()


class DocumentWSManager:
    """Tracks WebSocket connections and broadcasts document change events."""

    def __init__(self):
        self._connections: dict[tuple[str, str], set[WebSocket]] = {}

    async def connect(self, user_id: str, kb_id: str, ws: WebSocket):
        key = (user_id, kb_id)
        if key not in self._connections:
            self._connections[key] = set()
        self._connections[key].add(ws)
        logger.debug("WS connected: user=%s kb=%s (%d total)", user_id[:8], kb_id[:8], self._count())

    def disconnect(self, user_id: str, kb_id: str, ws: WebSocket):
        key = (user_id, kb_id)
        if key in self._connections:
            self._connections[key].discard(ws)
            if not self._connections[key]:
                del self._connections[key]

    async def broadcast(self, user_id: str, kb_id: str, event: dict):
        key = (user_id, kb_id)
        conns = self._connections.get(key)
        if not conns:
            return
        snapshot = list(conns)
        dead = []
        for ws in snapshot:
            try:
                await ws.send_json(event)
            except Exception:
                dead.append(ws)
        for ws in dead:
            conns.discard(ws)

    def _count(self) -> int:
        return sum(len(s) for s in self._connections.values())


manager = DocumentWSManager()

# Ping the LISTEN connection on this interval. Short enough that the pooler
# never idle-kills the socket, and so a dead socket is noticed within seconds
# instead of silently swallowing every NOTIFY until the next reconnect.
KEEPALIVE_SECONDS = 30
RECONNECT_DELAY_SECONDS = 5


async def setup_listener(database_url: str) -> asyncio.Task:
    """Start a supervised Postgres LISTEN loop that reconnects on failure."""
    return asyncio.create_task(_supervise_listener(database_url))


async def _supervise_listener(database_url: str) -> None:
    """Reconnect forever around a single LISTEN connection's lifetime."""
    while True:
        try:
            await _listen_until_closed(database_url)
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.warning("LISTEN connection lost (%s), reconnecting in %ds", e, RECONNECT_DELAY_SECONDS)
            await asyncio.sleep(RECONNECT_DELAY_SECONDS)


async def _listen_until_closed(database_url: str) -> None:
    """Hold one LISTEN connection open, pinging it so it stays alive and a drop surfaces."""
    conn = await asyncpg.connect(database_url)
    try:
        await conn.add_listener("document_changes", _on_notify)
        logger.info("Postgres LISTEN on 'document_changes' active")
        while True:
            await asyncio.sleep(KEEPALIVE_SECONDS)
            await conn.execute("SELECT 1")
    finally:
        if not conn.is_closed():
            await conn.close()


def _on_notify(conn: asyncpg.Connection, pid: int, channel: str, payload: str) -> None:
    asyncio.get_running_loop().create_task(_handle_notify(payload))


async def _handle_notify(payload: str) -> None:
    try:
        data = json.loads(payload)
    except json.JSONDecodeError:
        logger.warning("Bad NOTIFY payload: %s", payload[:100])
        return
    user_id = data.get("user_id")
    kb_id = data.get("knowledge_base_id")
    if user_id and kb_id:
        await manager.broadcast(user_id, kb_id, {
            "event": data.get("event"),
            "id": data.get("id"),
        })


@router.websocket("/v1/ws/documents/{kb_id}")
async def document_ws(websocket: WebSocket, kb_id: str):
    await websocket.accept()

    # First-message auth: client sends the token, we verify before registering.
    # Keeps the JWT out of URLs and logs.
    try:
        token = await asyncio.wait_for(websocket.receive_text(), timeout=5)
    except (asyncio.TimeoutError, WebSocketDisconnect):
        await websocket.close(code=4001, reason="Auth timeout")
        return

    try:
        user_id = await verify_token(token)
    except ValueError:
        await websocket.close(code=4001, reason="Invalid token")
        return

    await manager.connect(user_id, kb_id, websocket)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        pass
    finally:
        manager.disconnect(user_id, kb_id, websocket)
