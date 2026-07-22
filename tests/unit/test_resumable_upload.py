"""本地断点续传上传(routes/local_upload 断点续传部分)的单元测试。

覆盖:upload_id 合法性校验(防路径拼接)、过期分块清理、流式哈希与
整读一致、init/offset 两个无 DB 依赖的端点行为、1GiB 上限常量。
PATCH/complete 的全链路(409 排空、原子落盘、索引)由浏览器 e2e 与
协议级联调覆盖,这里不重复。
"""

import hashlib
import os
import time

import pytest
from fastapi import HTTPException

import routes.local_upload as lu
from config import settings


@pytest.fixture
def ws(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "WORKSPACE_PATH", str(tmp_path))
    return tmp_path


def test_upload_limits_are_1gib():
    assert lu.MAX_UPLOAD_BYTES == 1_073_741_824
    from infra import tus
    assert tus.MAX_SIZE == 1_073_741_824


@pytest.mark.parametrize("bad", [
    "../../etc/passwd", "..", "abc", "A" * 32, "0" * 31, "0" * 33,
    "g" * 32, "0" * 16 + "/" + "0" * 15,
])
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
            await lu.resumable_init(
                lu.ResumableInit(filename="a.bin", size=size), user_id="u")
        assert exc.value.status_code == 413


async def test_init_then_offset_roundtrip(ws):
    created = await lu.resumable_init(
        lu.ResumableInit(filename="大文件.bin", path="/子目录/", size=123),
        user_id="u")
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
