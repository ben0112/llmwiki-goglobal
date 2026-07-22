"""批量导入暴露的三类提取失败的回归测试。

- LibreOffice 并发互踩配置锁 → 每次调用独立 UserInstallation;
- 老式 .xls → xlrd 分支;
- opendataloader 挂掉的 PDF → pypdf 文本层兜底;
- 曾失败文档 → 启动 reconcile 重试一次。
"""

import asyncio
import sqlite3
import uuid
from pathlib import Path

import aiosqlite
import pytest

SCHEMA_PATH = Path(__file__).parents[2] / "shared" / "sqlite_schema.sql"
USER_ID = "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"

def _mini_pdf() -> bytes:
    """动态拼合法最小 PDF(一页,文本层 "Hello LLM"),xref 偏移实时计算。"""
    stream = b"BT /F1 24 Tf 72 700 Td (Hello LLM) Tj ET"
    objs = [
        b"<</Type/Catalog/Pages 2 0 R>>",
        b"<</Type/Pages/Kids[3 0 R]/Count 1>>",
        b"<</Type/Page/Parent 2 0 R/MediaBox[0 0 612 792]/Contents 4 0 R"
        b"/Resources<</Font<</F1 5 0 R>>>>>>",
        b"<</Length %d>>stream\n%s\nendstream" % (len(stream), stream),
        b"<</Type/Font/Subtype/Type1/BaseFont/Helvetica>>",
    ]
    out = bytearray(b"%PDF-1.4\n")
    offsets = []
    for i, body in enumerate(objs, 1):
        offsets.append(len(out))
        out += b"%d 0 obj\n%s\nendobj\n" % (i, body)
    xref_at = len(out)
    out += b"xref\n0 %d\n0000000000 65535 f \n" % (len(objs) + 1)
    for off in offsets:
        out += b"%010d 00000 n \n" % off
    out += (b"trailer<</Size %d/Root 1 0 R>>\nstartxref\n%d\n%%%%EOF\n"
            % (len(objs) + 1, xref_at))
    return bytes(out)


def test_xls_rows_read_via_xlrd(tmp_path):
    import xlwt

    from domain.local_processor import _read_sheet_rows

    wb = xlwt.Workbook()
    ws = wb.add_sheet("参与人员")
    ws.write(0, 0, "姓名")
    ws.write(0, 1, "岗位")
    ws.write(1, 0, "张三")
    ws.write(1, 1, "境外子公司总经理")
    path = tmp_path / "人员表.xls"
    wb.save(str(path))

    sheets = _read_sheet_rows(path)
    assert sheets[0][0] == "参与人员"
    assert sheets[0][1][0] == "姓名 | 岗位"
    assert "张三" in sheets[0][1][1]


def test_xlsx_rows_still_via_openpyxl(tmp_path):
    import openpyxl

    from domain.local_processor import _read_sheet_rows

    wb = openpyxl.Workbook()
    wb.active.title = "Sheet甲"
    wb.active.append(["a", "b"])
    path = tmp_path / "新表.xlsx"
    wb.save(str(path))

    sheets = _read_sheet_rows(path)
    assert sheets[0][0] == "Sheet甲" and sheets[0][1][0] == "a | b"


def test_pdf_fallback_to_pypdf(tmp_path, monkeypatch):
    import services.pdf_extract as px

    def boom(**kwargs):
        raise RuntimeError("模拟 opendataloader CLI 退出 1")

    monkeypatch.setattr(px.opendataloader_pdf, "convert", boom)
    pdf = tmp_path / "问题文件.pdf"
    pdf.write_bytes(_mini_pdf())

    pages = px.extract_pdf(str(pdf))
    assert len(pages) == 1
    num, md, images = pages[0]
    assert num == 1 and "Hello LLM" in md and images == []


def test_pdf_both_engines_fail_raises(tmp_path, monkeypatch):
    import services.pdf_extract as px

    monkeypatch.setattr(px.opendataloader_pdf, "convert",
                        lambda **kw: (_ for _ in ()).throw(RuntimeError("挂")))
    bad = tmp_path / "坏文件.pdf"
    bad.write_bytes(b"not a pdf at all")
    with pytest.raises(RuntimeError, match="兜底"):
        px.extract_pdf(str(bad))


async def test_office_conversion_uses_isolated_profile(tmp_path, monkeypatch):
    """并发防互踩:每次 LibreOffice 调用都带独立 UserInstallation 目录。"""
    import domain.local_processor as lp

    captured: list[list[str]] = []

    class FakeResult:
        returncode = 1
        stderr = b"Warning: failed to launch javaldx"

    def fake_run(cmd, **kwargs):
        captured.append(cmd)
        return FakeResult()

    monkeypatch.setattr(lp.shutil, "which", lambda name: "/usr/bin/libreoffice")
    monkeypatch.setattr(lp.subprocess, "run", fake_run)

    with pytest.raises(RuntimeError, match="LibreOffice conversion failed"):
        await lp._process_office(None, "doc-1", tmp_path / "a.docx", tmp_path)

    assert len(captured) == 1
    assert any(a.startswith("-env:UserInstallation=file://") for a in captured[0])


async def test_reconcile_retries_failed_docs(tmp_path, monkeypatch):
    import domain.local_processor as lp

    ws = tmp_path / "ws"
    (ws / ".llmwiki").mkdir(parents=True)
    db = await aiosqlite.connect(str(ws / ".llmwiki" / "index.db"))
    try:
        await db.executescript(SCHEMA_PATH.read_text(encoding="utf-8"))
        await db.execute(
            "INSERT INTO workspace (id, name, description, user_id) VALUES (?, 'w', '', ?)",
            (str(uuid.uuid4()), USER_ID))
        failed_id = str(uuid.uuid4())
        await db.execute(
            "INSERT INTO documents (id, user_id, filename, title, path, relative_path, "
            "source_kind, file_type, status, error_message, tags, version, document_number) "
            "VALUES (?, ?, 'a.docx', 'A', '/', 'a.docx', 'source', 'docx', 'failed', "
            "'LibreOffice conversion failed', '[]', 0, 1)",
            (failed_id, USER_ID))
        await db.commit()

        processed: list[str] = []

        async def fake_process(db_, doc_id, workspace_):
            processed.append(doc_id)

        monkeypatch.setattr(lp, "process_document", fake_process)
        await lp.reconcile_workspace(db, ws)

        assert processed == [failed_id]   # 失败文档被重试(且只此一条候选)
        cursor = await db.execute(
            "SELECT status, error_message FROM documents WHERE id = ?", (failed_id,))
        status, err = await cursor.fetchone()
        assert status == "pending" and err is None
    finally:
        await db.close()


def test_sniff_html_detects_disguised(tmp_path):
    from domain.local_processor import _sniff_html

    fake = tmp_path / "假装.pdf"
    fake.write_bytes(b"\n  <!DOCTYPE html>\n<html><body>404</body></html>")
    real = tmp_path / "真的.pdf"
    real.write_bytes(b"%PDF-1.4\n...")
    assert _sniff_html(fake) is True
    assert _sniff_html(real) is False


async def test_html_disguised_pdf_parsed_as_html(tmp_path):
    """爬虫把网页存成 .pdf:应走 HTML 解析入库,而非两个引擎双双失败。"""
    import domain.local_processor as lp

    ws = tmp_path / "ws"
    (ws / ".llmwiki").mkdir(parents=True)
    db = await aiosqlite.connect(str(ws / ".llmwiki" / "index.db"))
    try:
        await db.executescript(SCHEMA_PATH.read_text(encoding="utf-8"))
        await db.execute(
            "INSERT INTO workspace (id, name, description, user_id) VALUES (?, 'w', '', ?)",
            (str(uuid.uuid4()), USER_ID))
        (ws / "伪装网页.pdf").write_text(
            "<!DOCTYPE html><html><body><h1>境外投资办事指南</h1><p>正文内容。</p></body></html>",
            encoding="utf-8")
        doc_id = str(uuid.uuid4())
        await db.execute(
            "INSERT INTO documents (id, user_id, filename, title, path, relative_path, "
            "source_kind, file_type, status, tags, version, document_number) "
            "VALUES (?, ?, '伪装网页.pdf', 'X', '/', '伪装网页.pdf', 'source', 'pdf', "
            "'pending', '[]', 0, 1)", (doc_id, USER_ID))
        await db.commit()

        await lp.process_document(db, doc_id, ws)

        cursor = await db.execute(
            "SELECT status, parser, content FROM documents WHERE id = ?", (doc_id,))
        status, parser, content = await cursor.fetchone()
        assert status == "ready" and parser == "webmd"
        assert "境外投资办事指南" in (content or "")
    finally:
        await db.close()


# ── OCR 兜底启发式(不依赖 tesseract,CI 可跑)──────────────────

def _mk_pages(n, chars_per_page):
    return [(i + 1, "字" * chars_per_page, []) for i in range(n)]


def test_maybe_ocr_skips_high_yield(monkeypatch):
    import services.pdf_extract as px

    monkeypatch.setattr(px, "_ocr_available", lambda: True)
    called = []
    monkeypatch.setattr(px, "_ocr_single_page", lambda *a: called.append(1) or "x")
    pages = _mk_pages(10, 800)   # 正常产出:不应触发探针
    assert px._maybe_ocr("/tmp/x.pdf", pages) is pages
    assert called == []


def test_maybe_ocr_full_ocr_when_probe_wins(monkeypatch):
    import services.pdf_extract as px

    monkeypatch.setattr(px, "_ocr_available", lambda: True)
    monkeypatch.setattr(px, "_pdf_page_count", lambda p: 8)
    monkeypatch.setattr(px, "_ocr_single_page", lambda pdf, n, wd: "汉" * 900)
    pages = _mk_pages(8, 100)    # 文本层可疑(100/页)
    out = px._maybe_ocr("/tmp/x.pdf", pages)
    assert len(out) == 8 and all(md == "汉" * 900 for _, md, _ in out)


def test_maybe_ocr_keeps_primary_for_sparse_docs(monkeypatch):
    import services.pdf_extract as px

    monkeypatch.setattr(px, "_ocr_available", lambda: True)
    monkeypatch.setattr(px, "_pdf_page_count", lambda p: 8)
    # 图册类:OCR 采样同样稀疏 → 保留文本层结果
    monkeypatch.setattr(px, "_ocr_single_page", lambda pdf, n, wd: "图" * 120)
    pages = _mk_pages(8, 100)
    assert px._maybe_ocr("/tmp/x.pdf", pages) is pages


def test_maybe_ocr_noop_without_tools(monkeypatch):
    import services.pdf_extract as px

    monkeypatch.setattr(px, "_ocr_available", lambda: False)
    pages = _mk_pages(3, 10)
    assert px._maybe_ocr("/tmp/x.pdf", pages) is pages


async def test_reconcile_resets_stuck_processing(tmp_path, monkeypatch):
    """锁风暴/崩溃后卡在 processing 的文档:启动 reconcile 复位并重跑。"""
    import domain.local_processor as lp

    ws = tmp_path / "ws"
    (ws / ".llmwiki").mkdir(parents=True)
    db = await aiosqlite.connect(str(ws / ".llmwiki" / "index.db"))
    try:
        await db.executescript(SCHEMA_PATH.read_text(encoding="utf-8"))
        await db.execute(
            "INSERT INTO workspace (id, name, description, user_id) VALUES (?, 'w', '', ?)",
            (str(uuid.uuid4()), USER_ID))
        stuck_id = str(uuid.uuid4())
        await db.execute(
            "INSERT INTO documents (id, user_id, filename, title, path, relative_path, "
            "source_kind, file_type, status, tags, version, document_number) "
            "VALUES (?, ?, 'a.pdf', 'A', '/', 'a.pdf', 'source', 'pdf', 'processing', "
            "'[]', 0, 1)", (stuck_id, USER_ID))
        await db.commit()

        processed: list[str] = []

        async def fake_process(db_, doc_id, workspace_):
            processed.append(doc_id)

        monkeypatch.setattr(lp, "process_document", fake_process)
        await lp.reconcile_workspace(db, ws)

        assert stuck_id in processed
        cursor = await db.execute("SELECT status FROM documents WHERE id = ?", (stuck_id,))
        assert (await cursor.fetchone())[0] == "pending"   # 已复位(fake 未真正处理)
    finally:
        await db.close()


async def test_heavy_writes_serialized_by_gate(tmp_path, monkeypatch):
    """重量级入库写经进程级闸门串行:并发峰值必须为 1。"""
    import domain.local_processor as lp

    active = 0
    peak = 0
    real_store = lp._store_chunks

    async def tracking_store(db, doc_id, chunks):
        nonlocal active, peak
        active += 1
        peak = max(peak, active)
        try:
            await asyncio.sleep(0.02)
            return await real_store(db, doc_id, chunks)
        finally:
            active -= 1

    monkeypatch.setattr(lp, "_store_chunks", tracking_store)

    ws = tmp_path / "ws"
    (ws / ".llmwiki").mkdir(parents=True)
    db = await aiosqlite.connect(str(ws / ".llmwiki" / "index.db"))
    try:
        await db.executescript(SCHEMA_PATH.read_text(encoding="utf-8"))
        ids = []
        for i in range(4):
            doc_id = str(uuid.uuid4())
            await db.execute(
                "INSERT INTO documents (id, user_id, filename, title, path, relative_path, "
                "source_kind, file_type, status, tags, version, document_number) "
                "VALUES (?, ?, ?, 'T', '/', ?, 'source', 'txt', 'ready', '[]', 0, ?)",
                (doc_id, USER_ID, f"t{i}.txt", f"t{i}.txt", i + 1))
            ids.append(doc_id)
        await db.commit()

        await asyncio.gather(*(lp.chunk_text_document(db, d, "境外投资内容。" * 50) for d in ids))
        assert peak == 1   # 闸门生效:重写全程串行
    finally:
        await db.close()


def test_ocr_pages_run_in_parallel(monkeypatch):
    """页级并行:并发峰值 >1,结果按页号完整有序。"""
    import time as _time

    import services.pdf_extract as px

    active = 0
    peak = 0
    lock = __import__("threading").Lock()

    def fake_page(pdf, n, wd):
        nonlocal active, peak
        with lock:
            active += 1
            peak = max(peak, active)
        _time.sleep(0.05)
        with lock:
            active -= 1
        return f"第{n}页内容"

    monkeypatch.setattr(px, "_ocr_single_page", fake_page)
    monkeypatch.setattr(px, "OCR_WORKERS", 4)
    pages = px._extract_pdf_ocr("/tmp/x.pdf", 8)
    assert [p[0] for p in pages] == list(range(1, 9))          # 页序完整
    assert [p[1] for p in pages] == [f"第{i}页内容" for i in range(1, 9)]
    assert peak > 1                                             # 真并行
