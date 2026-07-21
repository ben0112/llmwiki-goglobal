"""API-key bearer auth: sv_ keys work anywhere a Supabase JWT does.

Completes the previously-inert /v1/api-keys feature so self-hosted
deployments need no OAuth-capable auth server for MCP/API access.
"""

import hashlib
import uuid

import pytest

from tests.helpers.jwt import auth_headers

USER_ID = str(uuid.uuid4())
OTHER_ID = str(uuid.uuid4())


@pytest.fixture
async def seeded_users(pool):
    for uid, email in ((USER_ID, "keyowner@example.com"), (OTHER_ID, "other@example.com")):
        await pool.execute(
            "INSERT INTO users (id, email, display_name) VALUES ($1, $2, 'U') "
            "ON CONFLICT (id) DO NOTHING",
            uid, email,
        )
    yield
    await pool.execute(
        "DELETE FROM api_keys WHERE user_id = ANY($1::uuid[])", [USER_ID, OTHER_ID],
    )
    await pool.execute(
        "DELETE FROM knowledge_bases WHERE user_id = ANY($1::uuid[])", [USER_ID, OTHER_ID],
    )
    await pool.execute(
        "DELETE FROM users WHERE id = ANY($1::uuid[])", [USER_ID, OTHER_ID],
    )


async def _create_key(client) -> tuple[str, str]:
    resp = await client.post("/v1/api-keys", json={"name": "ops"}, headers=auth_headers(USER_ID))
    assert resp.status_code == 201, resp.text
    body = resp.json()
    assert body["key"].startswith("sv_")
    return body["key"], body["id"]


async def test_api_key_authenticates_requests(client, pool, seeded_users):
    raw_key, _ = await _create_key(client)

    headers = {"Authorization": f"Bearer {raw_key}"}
    resp = await client.post("/v1/knowledge-bases", json={"name": "Key KB"}, headers=headers)
    assert resp.status_code in (200, 201), resp.text
    kb_id = resp.json()["id"]

    resp = await client.get("/v1/knowledge-bases", headers=headers)
    assert resp.status_code == 200
    assert any(kb["id"] == kb_id for kb in resp.json())

    # The key is scoped to its owner: the KB belongs to USER_ID.
    owner = await pool.fetchval("SELECT user_id FROM knowledge_bases WHERE id = $1", uuid.UUID(kb_id))
    assert str(owner) == USER_ID

    # last_used_at is stamped on use.
    last_used = await pool.fetchval(
        "SELECT last_used_at FROM api_keys WHERE key_hash = $1",
        hashlib.sha256(raw_key.encode()).hexdigest(),
    )
    assert last_used is not None


async def test_revoked_key_rejected(client, seeded_users):
    raw_key, key_id = await _create_key(client)
    resp = await client.delete(f"/v1/api-keys/{key_id}", headers=auth_headers(USER_ID))
    assert resp.status_code == 204

    resp = await client.get("/v1/knowledge-bases", headers={"Authorization": f"Bearer {raw_key}"})
    assert resp.status_code == 401


async def test_unknown_or_malformed_key_rejected(client, seeded_users):
    for bad in ("sv_" + "a" * 43, "sv_", "sv_short"):
        resp = await client.get("/v1/knowledge-bases", headers={"Authorization": f"Bearer {bad}"})
        assert resp.status_code == 401, bad


async def test_mcp_verifier_accepts_api_key(client, pool, seeded_users, monkeypatch):
    """The MCP-side verifier resolves the same key to the owning user."""
    raw_key, _ = await _create_key(client)

    import importlib.util
    import sys
    from pathlib import Path

    repo_root = Path(__file__).resolve().parents[2]
    # Provide the `db` and `config` modules mcp/auth.py expects, backed by the
    # test pool, without dragging in the whole mcp package.
    import types
    db_stub = types.ModuleType("db")

    async def service_queryrow(sql, *args):
        row = await pool.fetchrow(sql, *args)
        return dict(row) if row else None

    db_stub.service_queryrow = service_queryrow
    config_stub = types.ModuleType("config")
    config_stub.settings = types.SimpleNamespace(SUPABASE_URL="https://example.supabase.co")
    monkeypatch.setitem(sys.modules, "db", db_stub)
    monkeypatch.setitem(sys.modules, "config", config_stub)

    spec = importlib.util.spec_from_file_location("mcp_auth_mod", repo_root / "mcp" / "auth.py")
    mcp_auth = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mcp_auth)

    token = await mcp_auth.SupabaseTokenVerifier().verify_token(raw_key)
    assert token is not None and token.client_id == USER_ID

    assert await mcp_auth.SupabaseTokenVerifier().verify_token("sv_bogus") is None
