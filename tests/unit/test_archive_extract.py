"""压缩包服务端解压(services/archive_extract)的单元测试。

覆盖:目录结构保留、zip-slip 与垃圾文件过滤、Windows 反斜杠分隔、
GBK 文件名还原、tar.gz 解包、条目数/总量上限、损坏包报错、包名推导。
"""

import io
import tarfile
import zipfile

import pytest

from services.archive_extract import (
    ArchiveError,
    _zip_entry_name,
    archive_stem,
    clean_relative,
    extract_entries,
    is_supported_archive,
)


def _make_zip(items: dict[str, bytes]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for name, content in items.items():
            zf.writestr(name, content)
    return buf.getvalue()


def test_zip_preserves_nested_structure():
    data = _make_zip({
        "docs/policy.txt": b"one",
        "docs/sub/deep.md": b"two",
        "root.csv": b"three",
    })
    entries = dict(extract_entries("bundle.zip", data))
    assert entries == {
        "docs/policy.txt": b"one",
        "docs/sub/deep.md": b"two",
        "root.csv": b"three",
    }


def test_zip_skips_slip_and_junk():
    data = _make_zip({
        "../evil.txt": b"x",
        "__MACOSX/meta.txt": b"x",
        "docs/.DS_Store": b"x",
        ".hidden/file.txt": b"x",
        "ok.txt": b"keep",
    })
    entries = dict(extract_entries("a.zip", data))
    assert entries == {"ok.txt": b"keep"}


def test_clean_relative_windows_separators_and_drive():
    assert clean_relative("dir\\sub\\file.txt") == "dir/sub/file.txt"
    assert clean_relative("C:\\evil.txt") is None
    assert clean_relative("/abs/rooted.txt") == "abs/rooted.txt"
    assert clean_relative("a/../b.txt") is None


def test_zip_gbk_filename_decoded():
    # 未标 UTF-8 的 zip:zipfile 按 cp437 解码,需还原字节再按 GBK 解
    info = zipfile.ZipInfo("中文文档.txt".encode("gbk").decode("cp437"))
    info.flag_bits = 0
    assert _zip_entry_name(info) == "中文文档.txt"

    utf8 = zipfile.ZipInfo("已是UTF8.txt")
    utf8.flag_bits = 0x800
    assert _zip_entry_name(utf8) == "已是UTF8.txt"


def test_targz_roundtrip():
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tf:
        payload = b"tar content"
        ti = tarfile.TarInfo("folder/inner.txt")
        ti.size = len(payload)
        tf.addfile(ti, io.BytesIO(payload))
    entries = dict(extract_entries("data.tar.gz", buf.getvalue()))
    assert entries == {"folder/inner.txt": b"tar content"}


def test_entry_count_cap(monkeypatch):
    import services.archive_extract as ax
    monkeypatch.setattr(ax, "MAX_ENTRIES", 2)
    data = _make_zip({f"f{i}.txt": b"x" for i in range(3)})
    with pytest.raises(ArchiveError, match="文件过多"):
        extract_entries("a.zip", data)


def test_total_size_cap(monkeypatch):
    import services.archive_extract as ax
    monkeypatch.setattr(ax, "MAX_TOTAL_BYTES", 10)
    data = _make_zip({"a.txt": b"12345678", "b.txt": b"12345678"})
    with pytest.raises(ArchiveError, match="总量"):
        extract_entries("a.zip", data)


def test_single_file_cap_matches_upload_limit():
    import services.archive_extract as ax
    from routes.local_upload import MAX_UPLOAD_BYTES

    assert ax.MAX_FILE_BYTES == MAX_UPLOAD_BYTES == 1_073_741_824  # 与上传上限同口径
    assert ax.MAX_TOTAL_BYTES == 2 * ax.MAX_FILE_BYTES
    assert ax._fmt_size(ax.MAX_FILE_BYTES) == "1GB"
    assert ax._fmt_size(ax.MAX_TOTAL_BYTES) == "2GB"
    assert ax._fmt_size(100 * 1024 * 1024) == "100MB"


def test_single_file_cap_boundary(monkeypatch):
    import services.archive_extract as ax

    data = _make_zip({"big.txt": b"12345"})
    monkeypatch.setattr(ax, "MAX_FILE_BYTES", 4)
    with pytest.raises(ArchiveError, match="单个文件超过"):
        extract_entries("a.zip", data)
    monkeypatch.setattr(ax, "MAX_FILE_BYTES", 5)   # 恰在上限内放行
    assert dict(extract_entries("a.zip", data)) == {"big.txt": b"12345"}


def test_inner_file_over_100mb_extracts():
    """回归:包内单文件超过旧 100MB 上限应能正常解出(现上限 1GB)。"""
    payload = bytes(110 * 1024 * 1024)   # 全零,tar.gz 压得很小
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tf:
        ti = tarfile.TarInfo("大文件/超百兆.bin")
        ti.size = len(payload)
        tf.addfile(ti, io.BytesIO(payload))
    entries = dict(extract_entries("big.tar.gz", buf.getvalue()))
    assert set(entries) == {"大文件/超百兆.bin"}
    assert len(entries["大文件/超百兆.bin"]) == len(payload)


def test_corrupt_archive_raises():
    with pytest.raises(ArchiveError, match="无法解析"):
        extract_entries("bad.zip", b"not a zip at all")


def test_supported_and_stem():
    assert is_supported_archive("语料.zip")
    assert is_supported_archive("x.tar.gz") and is_supported_archive("x.tgz")
    assert not is_supported_archive("x.rar") and not is_supported_archive("x.7z")
    assert archive_stem("语料包.zip") == "语料包"
    assert archive_stem("dump.tar.gz") == "dump"
