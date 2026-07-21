"""压缩包服务端解压(本地模式上传用)。

支持 zip / tar / tar.gz / tgz。纯函数,不落盘:extract_entries 把包内文件
展开为 (相对路径, 字节) 列表,由上传路由决定去重与落库。安全防线:
- zip-slip:含 `..`、绝对路径、盘符的条目直接丢弃;
- 解压炸弹:条目数、单文件、总量三重上限,超限报错而非静默截断;
- 垃圾文件:__MACOSX、.DS_Store、隐藏文件(点开头)跳过;
- 中文文件名:zip 未标 UTF-8 时按 cp437 还原字节再试 GBK(Windows
  压缩工具的常见编码),都失败保留 zipfile 的 cp437 解码。
嵌套压缩包不递归(按不支持类型跳过)。
"""

from __future__ import annotations

import io
import tarfile
import zipfile

MAX_ENTRIES = 500
MAX_FILE_BYTES = 100 * 1024 * 1024
MAX_TOTAL_BYTES = 500 * 1024 * 1024

_JUNK_BASENAMES = {".ds_store", "thumbs.db", "desktop.ini"}
_ARCHIVE_SUFFIXES = (".zip", ".tar", ".tar.gz", ".tgz")


class ArchiveError(Exception):
    """解压失败或超出安全上限,信息可直接展示给用户。"""


def is_supported_archive(filename: str) -> bool:
    return filename.lower().endswith(_ARCHIVE_SUFFIXES)


def archive_stem(filename: str) -> str:
    """去掉压缩扩展名的包名(解压目标文件夹名)。"""
    lower = filename.lower()
    for suffix in (".tar.gz", ".tgz", ".tar", ".zip"):
        if lower.endswith(suffix):
            return filename[: -len(suffix)]
    return filename


def clean_relative(name: str) -> str | None:
    """规范化包内条目路径;不安全或属于垃圾文件时返回 None。"""
    parts = []
    for part in name.replace("\\", "/").split("/"):
        if part in ("", "."):
            continue
        if part == ".." or ":" in part:
            return None  # zip-slip / 盘符
        if part == "__MACOSX" or part.startswith("."):
            return None
        if part.lower() in _JUNK_BASENAMES:
            return None
        parts.append(part)
    return "/".join(parts) or None


def _zip_entry_name(info: zipfile.ZipInfo) -> str:
    if info.flag_bits & 0x800:  # 已声明 UTF-8
        return info.filename
    try:
        raw = info.filename.encode("cp437")
    except UnicodeEncodeError:
        return info.filename
    for enc in ("gbk", "utf-8"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    return info.filename


def extract_entries(filename: str, data: bytes) -> list[tuple[str, bytes]]:
    """展开压缩包为 [(相对路径, 内容字节)];超限/损坏抛 ArchiveError。"""
    lower = filename.lower()
    try:
        if lower.endswith(".zip"):
            entries = _extract_zip(data)
        else:
            entries = _extract_tar(data)
    except ArchiveError:
        raise
    except (zipfile.BadZipFile, tarfile.TarError, EOFError, OSError) as e:
        raise ArchiveError(f"压缩包无法解析:{e}") from e
    return entries


def _check_caps(count: int, size: int, total: int) -> int:
    if count > MAX_ENTRIES:
        raise ArchiveError(f"压缩包内文件过多(上限 {MAX_ENTRIES} 个)")
    if size > MAX_FILE_BYTES:
        raise ArchiveError(f"包内单个文件超过 {MAX_FILE_BYTES // 1024 // 1024}MB 上限")
    total += size
    if total > MAX_TOTAL_BYTES:
        raise ArchiveError(f"解压总量超过 {MAX_TOTAL_BYTES // 1024 // 1024}MB 上限")
    return total


def _extract_zip(data: bytes) -> list[tuple[str, bytes]]:
    entries: list[tuple[str, bytes]] = []
    total = 0
    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        for info in zf.infolist():
            if info.is_dir():
                continue
            relative = clean_relative(_zip_entry_name(info))
            if not relative:
                continue
            total = _check_caps(len(entries) + 1, info.file_size, total)
            entries.append((relative, zf.read(info)))
    return entries


def _extract_tar(data: bytes) -> list[tuple[str, bytes]]:
    entries: list[tuple[str, bytes]] = []
    total = 0
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:*") as tf:
        for member in tf:
            if not member.isfile():
                continue  # 目录/符号链接/设备文件一律跳过
            relative = clean_relative(member.name)
            if not relative:
                continue
            total = _check_caps(len(entries) + 1, member.size, total)
            fh = tf.extractfile(member)
            if fh is None:
                continue
            entries.append((relative, fh.read()))
    return entries
