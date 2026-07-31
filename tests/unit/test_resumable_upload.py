"""本地断点续传上传(routes/local_upload 断点续传部分)的单元测试。

覆盖:upload_id 合法性校验(防路径拼接)、过期分块清理、流式哈希与
整读一致、init/offset 两个无 DB 依赖的端点行为、1GiB 上限常量。
PATCH/complete 的全链路(409 排空、原子落盘、索引)由浏览器 e2e 与
协议级联调覆盖,这里不重复。
"""

import hashlib
import os
import time
from types import SimpleNamespace

import httpx
import pytest
import routes.local_upload as lu
from config import settings
from fastapi import FastAPI, HTTPException, Response
from infra.db.sqlite import create_pool


@pytest.fixture
def ws(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "WORKSPACE_PATH", str(tmp_path))
    return tmp_path


def test_upload_limits_are_1gib():
    assert lu.MAX_UPLOAD_BYTES == 1_073_741_824
    from infra import tus

    assert tus.MAX_SIZE == 1_073_741_824


@pytest.mark.parametrize(
    "bad",
    [
        "../../etc/passwd",
        "..",
        "abc",
        "A" * 32,
        "0" * 31,
        "0" * 33,
        "g" * 32,
        "0" * 16 + "/" + "0" * 15,
    ],
)
def test_part_paths_rejects_malformed_ids(ws, bad):
    with pytest.raises(HTTPException) as exc:
        lu._part_paths(bad)
    assert exc.value.status_code == 400


def test_part_paths_accepts_hex_id(ws):
    part, meta = lu._part_paths("0123456789abcdef" * 2)
    assert part.parent == ws / ".llmwiki" / "tmp" / "uploads"
    assert part.suffix == ".part" and meta.suffix == ".json"


def test_purge_stale_parts_keeps_fresh(ws):
    d = lu._parts_dir()
    stale = d / ("a" * 32 + ".part")
    stale.write_bytes(b"x")
    (d / ("a" * 32 + ".json")).write_text("{}")
    old = time.time() - lu._STALE_PART_SECONDS - 60
    os.utime(stale, (old, old))
    fresh = d / ("b" * 32 + ".part")
    fresh.write_bytes(b"y")
    (d / ("b" * 32 + ".json")).write_text("{}")

    lu._purge_stale_parts()

    assert not stale.exists() and not stale.with_suffix(".json").exists()
    assert fresh.exists() and fresh.with_suffix(".json").exists()


def test_hash_file_matches_whole_read(ws):
    payload = bytes(range(256)) * 8192  # 2MB,跨多个 1MB 分块
    p = ws / "blob.bin"
    p.write_bytes(payload)
    assert lu._hash_file(p) == hashlib.sha256(payload).hexdigest()


async def test_init_rejects_bad_sizes(ws):
    for size in (0, -1, lu.MAX_UPLOAD_BYTES + 1):
        with pytest.raises(HTTPException) as exc:
            await lu.resumable_init(lu.ResumableInit(filename="a.bin", size=size), user_id="u")
        assert exc.value.status_code == 413


async def test_init_then_offset_roundtrip(ws):
    created = await lu.resumable_init(lu.ResumableInit(filename="大文件.bin", path="/子目录/", size=123), user_id="u")
    uid = created["upload_id"]
    assert created["offset"] == 0 and lu._UPLOAD_ID_RE.match(uid)

    part, meta = lu._part_paths(uid)
    assert part.is_file() and meta.is_file()

    # 模拟已写入 5 字节后查询进度
    part.write_bytes(b"12345")
    got = await lu.resumable_offset(uid, user_id="u")
    assert got == {"offset": 5}


async def test_offset_unknown_id_404(ws):
    with pytest.raises(HTTPException) as exc:
        await lu.resumable_offset("c" * 32, user_id="u")
    assert exc.value.status_code == 404


async def test_direct_duplicate_rejects_without_overwriting_existing_file(ws):
    db = await create_pool(str(ws / "index.db"))
    await db.execute("INSERT INTO workspace(id,name,user_id) VALUES('ws','test','u1')")
    target = ws / "same.md"
    target.write_bytes(b"original")
    await db.execute(
        "INSERT INTO documents"
        "(id,user_id,filename,path,relative_path,source_kind,file_type,status,content_hash) "
        "VALUES('d1','u1','same.md','/','same.md','source','md','ready',?)",
        (hashlib.sha256(b"original").hexdigest(),),
    )
    await db.commit()

    with pytest.raises(HTTPException) as exc:
        await lu._ingest_bytes(db, "same.md", b"replacement")
    assert exc.value.status_code == 409
    assert exc.value.detail["code"] == "duplicate_path"
    assert target.read_bytes() == b"original"
    assert await db.execute_fetchall("SELECT id FROM documents") == [("d1",)]
    await db.close()


async def test_content_duplicate_rejects_without_creating_destination(ws):
    db = await create_pool(str(ws / "index.db"))
    await db.execute("INSERT INTO workspace(id,name,user_id) VALUES('ws','test','u1')")
    digest = hashlib.sha256(b"same bytes").hexdigest()
    await db.execute(
        "INSERT INTO documents"
        "(id,user_id,filename,path,relative_path,source_kind,file_type,status,content_hash) "
        "VALUES('d1','u1','old.md','/','old.md','source','md','ready',?)",
        (digest,),
    )
    await db.commit()

    with pytest.raises(HTTPException) as exc:
        await lu._ingest_bytes(db, "new.md", b"same bytes")
    assert exc.value.status_code == 409
    assert exc.value.detail["code"] == "duplicate_content"
    assert not (ws / "new.md").exists()
    await db.close()


async def test_resumable_duplicate_keeps_existing_file_and_upload_session(ws):
    db = await create_pool(str(ws / "index.db"))
    await db.execute("INSERT INTO workspace(id,name,user_id) VALUES('ws','test','u1')")
    target = ws / "same.md"
    target.write_bytes(b"original")
    await db.execute(
        "INSERT INTO documents"
        "(id,user_id,filename,path,relative_path,source_kind,file_type,status,content_hash) "
        "VALUES('d1','u1','same.md','/','same.md','source','md','ready',?)",
        (hashlib.sha256(b"original").hexdigest(),),
    )
    await db.commit()
    upload_id = "d" * 32
    part, meta = lu._part_paths(upload_id)
    part.write_bytes(b"replacement")
    meta.write_text('{"filename":"same.md","path":"/","size":11}', encoding="utf-8")

    with pytest.raises(HTTPException) as exc:
        await lu.resumable_complete(
            upload_id, user_id="u1", request=SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(sqlite_db=db)))
        )
    assert exc.value.status_code == 409
    assert target.read_bytes() == b"original"
    assert part.exists() and meta.exists()
    await db.close()


async def test_hosted_tus_flag_delegates_to_application_scoped_service(monkeypatch):
    from infra import tus

    calls = []

    class HostedService:
        async def create(self, request, user_id):
            calls.append(("create", user_id))
            return Response(status_code=201, headers={"Location": "/v1/uploads/shared"})

        async def head(self, upload_id, request, user_id):
            calls.append(("head", upload_id, user_id))
            return Response(status_code=200, headers={"Upload-Offset": "7"})

        async def patch(self, upload_id, request, user_id):
            calls.append(("patch", upload_id, user_id))
            return Response(status_code=204, headers={"Upload-Offset": "9"})

    async def authenticated(_request):
        return "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"

    monkeypatch.setattr(tus, "_get_user_id", authenticated)
    monkeypatch.setattr(tus.settings, "TUS_MULTIPART_ENABLED", True)
    app = FastAPI()
    app.state.tus_service = HostedService()
    app.include_router(tus.router)

    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        created = await client.post("/v1/uploads")
        headed = await client.head("/v1/uploads/shared")
        patched = await client.patch("/v1/uploads/shared")

    assert created.status_code == 201
    assert headed.headers["upload-offset"] == "7"
    assert patched.headers["upload-offset"] == "9"
    assert calls == [
        ("create", "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"),
        ("head", "shared", "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"),
        ("patch", "shared", "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"),
    ]
