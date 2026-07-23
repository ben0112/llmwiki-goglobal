"""批量导入暴露的三类提取失败的回归测试。

- LibreOffice 并发互踩配置锁 → 每次调用独立 UserInstallation;
- 老式 .xls → xlrd 分支;
- opendataloader 挂掉的 PDF → pypdf 文本层兜底;
- 曾失败文档 → 启动 reconcile 重试一次。
"""

import asyncio
import json
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
    monkeypatch.setattr(px, "_ocr_pool", None)   # 以补丁后的 OCR_WORKERS 重建全局池
    pages = px._extract_pdf_ocr("/tmp/x.pdf", 8)
    assert [p[0] for p in pages] == list(range(1, 9))          # 页序完整
    assert [p[1] for p in pages] == [f"第{i}页内容" for i in range(1, 9)]
    assert peak > 1                                             # 真并行


async def test_reconcile_rescues_pending_with_old_chunks(tmp_path, monkeypatch):
    """「重新识别」重置的源文档带旧提取结果(parser 已设、chunks 还在):
    停机丢失排队任务后,启动 reconcile 必须仍接住它 —— 曾经的盲区,
    表现为重启后 OCR 积压不恢复、CPU 闲置。"""
    import domain.local_processor as lp

    ws = tmp_path / "ws"
    (ws / ".llmwiki").mkdir(parents=True)
    db = await aiosqlite.connect(str(ws / ".llmwiki" / "index.db"))
    try:
        await db.executescript(SCHEMA_PATH.read_text(encoding="utf-8"))
        await db.execute(
            "INSERT INTO workspace (id, name, description, user_id) VALUES (?, 'w', '', ?)",
            (str(uuid.uuid4()), USER_ID))
        doc_id = str(uuid.uuid4())
        await db.execute(
            "INSERT INTO documents (id, user_id, filename, title, path, relative_path, "
            "source_kind, file_type, status, parser, tags, version, document_number) "
            "VALUES (?, ?, 'b.pdf', 'B', '/', 'b.pdf', 'source', 'pdf', 'pending', "
            "'opendataloader', '[]', 0, 1)", (doc_id, USER_ID))
        await db.execute(
            "INSERT INTO document_chunks (id, document_id, chunk_index, content, "
            "source_content, page, start_char, token_count, header_breadcrumb) "
            "VALUES (?, ?, 0, '旧内容', '旧内容', 1, 0, 2, '')",
            (str(uuid.uuid4()), doc_id))
        await db.commit()

        processed: list[str] = []

        async def fake_process(db_, d, workspace_):
            processed.append(d)

        monkeypatch.setattr(lp, "process_document", fake_process)
        await lp.reconcile_workspace(db, ws)

        assert processed == [doc_id]   # 接住且只入队一次
    finally:
        await db.close()


async def test_kick_pending_backlog_skips_inflight(tmp_path, monkeypatch):
    """sweep 自愈兜底:无主的 pending 文档重新入队,已在飞的跳过。"""
    import domain.local_processor as lp

    ws = tmp_path / "ws"
    (ws / ".llmwiki").mkdir(parents=True)
    db = await aiosqlite.connect(str(ws / ".llmwiki" / "index.db"))
    try:
        await db.executescript(SCHEMA_PATH.read_text(encoding="utf-8"))
        await db.execute(
            "INSERT INTO workspace (id, name, description, user_id) VALUES (?, 'w', '', ?)",
            (str(uuid.uuid4()), USER_ID))
        lost_id, inflight_id = str(uuid.uuid4()), str(uuid.uuid4())
        for n, doc_id in enumerate((lost_id, inflight_id), 1):
            await db.execute(
                "INSERT INTO documents (id, user_id, filename, title, path, relative_path, "
                "source_kind, file_type, status, tags, version, document_number) "
                "VALUES (?, ?, ?, 'D', '/', ?, 'source', 'pdf', 'pending', '[]', 0, ?)",
                (doc_id, USER_ID, f"d{n}.pdf", f"d{n}.pdf", n))
        await db.commit()

        processed: list[str] = []

        async def fake_process(db_, d, workspace_):
            processed.append(d)

        monkeypatch.setattr(lp, "process_document", fake_process)
        lp._inflight.add(inflight_id)
        try:
            kicked = await lp.kick_pending_backlog(db, ws)
            for _ in range(200):            # spawn 的任务真正跑完(≤4s)
                if processed:
                    break
                await asyncio.sleep(0.02)
        finally:
            lp._inflight.discard(inflight_id)

        assert kicked == 1
        assert processed == [lost_id]
    finally:
        await db.close()


def test_ocr_budget_shared_across_documents(monkeypatch):
    """全局 OCR 预算:两份文档同时 OCR,总并发不超过 OCR_WORKERS,
    不随文档数相乘(曾经 8 文档 × 4 页 = 32 路 tesseract 超订)。"""
    import concurrent.futures
    import threading
    import time as _time

    import services.pdf_extract as px

    active = 0
    peak = 0
    lock = threading.Lock()

    def fake_page(pdf, n, wd):
        nonlocal active, peak
        with lock:
            active += 1
            peak = max(peak, active)
        _time.sleep(0.05)
        with lock:
            active -= 1
        return f"{pdf}:{n}"

    monkeypatch.setattr(px, "_ocr_single_page", fake_page)
    monkeypatch.setattr(px, "OCR_WORKERS", 2)
    monkeypatch.setattr(px, "_ocr_pool", None)
    with concurrent.futures.ThreadPoolExecutor(2) as caller:
        f1 = caller.submit(px._ocr_pages_parallel, "a.pdf", [1, 2, 3, 4], Path("/tmp"))
        f2 = caller.submit(px._ocr_pages_parallel, "b.pdf", [1, 2, 3, 4], Path("/tmp"))
        r1, r2 = f1.result(), f2.result()
    assert set(r1) == {1, 2, 3, 4} and set(r2) == {1, 2, 3, 4}   # 都完整
    assert peak <= 2                                             # 共享预算,无乘性超订


async def test_save_local_images_batch_single_commit(tmp_path):
    """图片资产批量入库:一个事务一次 commit,编号连续,父 metadata 一次更新。"""
    import domain.local_processor as lp

    ws = tmp_path / "ws"
    (ws / ".llmwiki").mkdir(parents=True)
    db = await aiosqlite.connect(str(ws / ".llmwiki" / "index.db"))
    try:
        await db.executescript(SCHEMA_PATH.read_text(encoding="utf-8"))
        await db.execute(
            "INSERT INTO workspace (id, name, description, user_id) VALUES (?, 'w', '', ?)",
            (str(uuid.uuid4()), USER_ID))
        doc_id = str(uuid.uuid4())
        await db.execute(
            "INSERT INTO documents (id, user_id, filename, title, path, relative_path, "
            "source_kind, file_type, status, tags, version, document_number) "
            "VALUES (?, ?, '报告.pdf', 'R', '/', '报告.pdf', 'source', 'pdf', 'processing', "
            "'[]', 0, 7)", (doc_id, USER_ID))
        await db.commit()

        commits = 0
        orig_commit = db.commit

        async def counting_commit():
            nonlocal commits
            commits += 1
            await orig_commit()

        db.commit = counting_commit
        pages = [
            (1, "第一页", [{"id": "img_1_0.png", "bytes": b"PNG1", "format": "png"},
                          {"id": "img_1_1.png", "bytes": b"PNG2", "format": "png"}]),
            (2, "第二页", [{"id": "img_2_2.png", "bytes": b"PNG3", "format": "png"}]),
        ]
        page_elements = await lp._save_local_images(db, doc_id, ws, pages)
        db.commit = orig_commit

        assert commits == 1                       # 单事务单提交
        cursor = await db.execute(
            "SELECT document_number FROM documents WHERE source_kind = 'asset' "
            "ORDER BY document_number")
        numbers = [r[0] for r in await cursor.fetchall()]
        assert numbers == [8, 9, 10]              # 批量取号连续
        cursor = await db.execute("SELECT metadata FROM documents WHERE id = ?", (doc_id,))
        meta = json.loads((await cursor.fetchone())[0])
        assert len(meta["assets"]) == 3           # 父文档 metadata 一次写全
        assert page_elements                      # 页面元素照常返回
        pngs = list(ws.rglob("*.png"))
        assert len(pngs) == 3                     # 文件已落盘
    finally:
        await db.close()


async def test_failed_doc_quarantined_after_retry_limit(tmp_path, monkeypatch):
    """失败退避:extraction_attempts 达上限的文档不再随启动自动重试。"""
    import domain.local_processor as lp

    ws = tmp_path / "ws"
    (ws / ".llmwiki").mkdir(parents=True)
    db = await aiosqlite.connect(str(ws / ".llmwiki" / "index.db"))
    try:
        await db.executescript(SCHEMA_PATH.read_text(encoding="utf-8"))
        await db.execute(
            "INSERT INTO workspace (id, name, description, user_id) VALUES (?, 'w', '', ?)",
            (str(uuid.uuid4()), USER_ID))
        fresh_id, worn_id = str(uuid.uuid4()), str(uuid.uuid4())
        for n, (doc_id, attempts) in enumerate(((fresh_id, 1), (worn_id, 3)), 1):
            await db.execute(
                "INSERT INTO documents (id, user_id, filename, title, path, relative_path, "
                "source_kind, file_type, status, extraction_attempts, tags, version, "
                "document_number) "
                "VALUES (?, ?, ?, 'D', '/', ?, 'source', 'docx', 'failed', ?, '[]', 0, ?)",
                (doc_id, USER_ID, f"f{n}.docx", f"f{n}.docx", attempts, n))
        await db.commit()

        processed: list[str] = []

        async def fake_process(db_, d, workspace_):
            processed.append(d)

        monkeypatch.setattr(lp, "process_document", fake_process)
        await lp.reconcile_workspace(db, ws)

        assert processed == [fresh_id]            # 满 3 次的隔离,未满的重试
        cursor = await db.execute("SELECT status FROM documents WHERE id = ?", (worn_id,))
        assert (await cursor.fetchone())[0] == "failed"
    finally:
        await db.close()


async def test_failure_marking_increments_attempts(tmp_path):
    """每次提取失败 extraction_attempts +1(文件缺失路径)。"""
    import domain.local_processor as lp

    ws = tmp_path / "ws"
    (ws / ".llmwiki").mkdir(parents=True)
    db = await aiosqlite.connect(str(ws / ".llmwiki" / "index.db"))
    try:
        await db.executescript(SCHEMA_PATH.read_text(encoding="utf-8"))
        await db.execute(
            "INSERT INTO workspace (id, name, description, user_id) VALUES (?, 'w', '', ?)",
            (str(uuid.uuid4()), USER_ID))
        doc_id = str(uuid.uuid4())
        await db.execute(
            "INSERT INTO documents (id, user_id, filename, title, path, relative_path, "
            "source_kind, file_type, status, tags, version, document_number) "
            "VALUES (?, ?, 'ghost.pdf', 'G', '/', 'ghost.pdf', 'source', 'pdf', 'pending', "
            "'[]', 0, 1)", (doc_id, USER_ID))
        await db.commit()

        await lp.process_document(db, doc_id, ws)

        cursor = await db.execute(
            "SELECT status, extraction_attempts FROM documents WHERE id = ?", (doc_id,))
        status, attempts = await cursor.fetchone()
        assert status == "failed" and attempts == 1
    finally:
        await db.close()


async def test_extraction_attempts_migration_on_old_db(tmp_path):
    """旧库无 extraction_attempts 列:create_pool 启动时自动补列(默认 0)。"""
    from infra.db.sqlite import create_pool

    db_path = str(tmp_path / "old.db")
    conn = sqlite3.connect(db_path)
    old_schema = SCHEMA_PATH.read_text(encoding="utf-8").replace(
        "    extraction_attempts INTEGER NOT NULL DEFAULT 0,\n", "")
    assert "extraction_attempts" not in old_schema   # 确认真的模拟了旧库
    conn.executescript(old_schema)
    conn.execute(
        "INSERT INTO documents (id, user_id, filename, title, path, relative_path, "
        "source_kind, file_type, status, tags, version, document_number) "
        "VALUES ('d1', 'u1', 'a.pdf', 'A', '/', 'a.pdf', 'source', 'pdf', 'failed', '[]', 0, 1)")
    conn.commit()
    conn.close()

    db = await create_pool(db_path)
    try:
        cur = await db.execute(
            "SELECT extraction_attempts FROM documents WHERE id = 'd1'")
        assert (await cur.fetchone())[0] == 0        # 已补列,存量行默认 0
    finally:
        await db.close()


async def test_crash_looping_doc_quarantined_at_startup(tmp_path, monkeypatch):
    """反复把进程压垮的文档(异常没机会抛,靠 except 计数无效):
    reconcile 复位时计数,达上限转入失败隔离而非再次排队。"""
    import domain.local_processor as lp

    ws = tmp_path / "ws"
    (ws / ".llmwiki").mkdir(parents=True)
    db = await aiosqlite.connect(str(ws / ".llmwiki" / "index.db"))
    try:
        await db.executescript(SCHEMA_PATH.read_text(encoding="utf-8"))
        await db.execute(
            "INSERT INTO workspace (id, name, description, user_id) VALUES (?, 'w', '', ?)",
            (str(uuid.uuid4()), USER_ID))
        crasher_id, victim_id = str(uuid.uuid4()), str(uuid.uuid4())
        for n, (doc_id, attempts) in enumerate(((crasher_id, 2), (victim_id, 0)), 1):
            await db.execute(
                "INSERT INTO documents (id, user_id, filename, title, path, relative_path, "
                "source_kind, file_type, status, extraction_attempts, tags, version, "
                "document_number) "
                "VALUES (?, ?, ?, 'D', '/', ?, 'source', 'pdf', 'processing', ?, '[]', 0, ?)",
                (doc_id, USER_ID, f"c{n}.pdf", f"c{n}.pdf", attempts, n))
        await db.commit()

        processed: list[str] = []

        async def fake_process(db_, d, workspace_):
            processed.append(d)

        monkeypatch.setattr(lp, "process_document", fake_process)
        await lp.reconcile_workspace(db, ws)

        cursor = await db.execute(
            "SELECT status, extraction_attempts, error_message FROM documents WHERE id = ?",
            (crasher_id,))
        status, attempts, err = await cursor.fetchone()
        assert status == "failed" and attempts == 3 and "反复中断" in err   # 隔离
        assert crasher_id not in processed
        cursor = await db.execute(
            "SELECT status, extraction_attempts FROM documents WHERE id = ?", (victim_id,))
        assert (await cursor.fetchone()) == ("pending", 1)   # 无辜停机的:计 1 次并重跑
        assert victim_id in processed
    finally:
        await db.close()


async def test_save_local_images_evicts_watcher_squatter_rows(tmp_path):
    """落盘→入闸门的窗口内 watcher 可能把 .assets 图片抢注成普通文档行:
    批量插入前按 relative_path 清场,否则撞 UNIQUE 使提取确定性失败。"""
    import domain.local_processor as lp
    from services.extracted_assets import build_pdf_image_assets

    ws = tmp_path / "ws"
    (ws / ".llmwiki").mkdir(parents=True)
    db = await aiosqlite.connect(str(ws / ".llmwiki" / "index.db"))
    try:
        await db.executescript(SCHEMA_PATH.read_text(encoding="utf-8"))
        await db.execute(
            "INSERT INTO workspace (id, name, description, user_id) VALUES (?, 'w', '', ?)",
            (str(uuid.uuid4()), USER_ID))
        doc_id = str(uuid.uuid4())
        await db.execute(
            "INSERT INTO documents (id, user_id, filename, title, path, relative_path, "
            "source_kind, file_type, status, tags, version, document_number) "
            "VALUES (?, ?, '图册.pdf', 'T', '/', '图册.pdf', 'source', 'pdf', 'processing', "
            "'[]', 0, 1)", (doc_id, USER_ID))

        pages = [(1, "页", [{"id": "img_1_0.png", "bytes": b"PNG", "format": "png"}])]
        assets, _ = build_pdf_image_assets(doc_id, "图册.pdf", "/", pages)
        squat_rp = (assets[0].path.rstrip("/") + "/" + assets[0].filename).lstrip("/")
        await db.execute(
            "INSERT INTO documents (id, user_id, filename, title, path, relative_path, "
            "source_kind, file_type, status, tags, version, document_number) "
            "VALUES (?, ?, ?, 'S', '/', ?, 'source', 'png', 'pending', '[]', 0, 2)",
            (str(uuid.uuid4()), USER_ID, assets[0].filename, squat_rp))
        await db.commit()

        await lp._save_local_images(db, doc_id, ws, pages)   # 不得抛 IntegrityError

        cursor = await db.execute(
            "SELECT source_kind, status FROM documents WHERE relative_path = ?", (squat_rp,))
        rows = await cursor.fetchall()
        assert rows == [("asset", "ready")]                  # 抢注行已被资产行取代
    finally:
        await db.close()


async def test_exception_failure_increments_attempts(tmp_path, monkeypatch):
    """提取异常(生产最常见失败路径)每次 +1 —— 隔离阈值真实生效。"""
    import domain.local_processor as lp

    ws = tmp_path / "ws"
    (ws / ".llmwiki").mkdir(parents=True)
    (ws / "坏文档.docx").write_bytes(b"PK\x03\x04 not really a docx")
    db = await aiosqlite.connect(str(ws / ".llmwiki" / "index.db"))
    try:
        await db.executescript(SCHEMA_PATH.read_text(encoding="utf-8"))
        await db.execute(
            "INSERT INTO workspace (id, name, description, user_id) VALUES (?, 'w', '', ?)",
            (str(uuid.uuid4()), USER_ID))
        doc_id = str(uuid.uuid4())
        await db.execute(
            "INSERT INTO documents (id, user_id, filename, title, path, relative_path, "
            "source_kind, file_type, status, tags, version, document_number) "
            "VALUES (?, ?, '坏文档.docx', 'B', '/', '坏文档.docx', 'source', 'docx', "
            "'pending', '[]', 0, 1)", (doc_id, USER_ID))
        await db.commit()

        async def boom(db_, d, path_, ws_):
            raise RuntimeError("LibreOffice conversion failed: 模拟")

        monkeypatch.setattr(lp, "_process_office", boom)

        await lp.process_document(db, doc_id, ws)
        cursor = await db.execute(
            "SELECT status, extraction_attempts FROM documents WHERE id = ?", (doc_id,))
        assert (await cursor.fetchone()) == ("failed", 1)

        await db.execute("UPDATE documents SET status = 'pending' WHERE id = ?", (doc_id,))
        await db.commit()
        await lp.process_document(db, doc_id, ws)
        cursor = await db.execute(
            "SELECT extraction_attempts FROM documents WHERE id = ?", (doc_id,))
        assert (await cursor.fetchone())[0] == 2             # 逐次累计
    finally:
        await db.close()


async def test_save_local_images_respects_write_gate(tmp_path):
    """资产入库走 _db_write_gate:闸门被占时 DB 写必须等待(文件可先落盘)。"""
    import domain.local_processor as lp

    ws = tmp_path / "ws"
    (ws / ".llmwiki").mkdir(parents=True)
    db = await aiosqlite.connect(str(ws / ".llmwiki" / "index.db"))
    try:
        await db.executescript(SCHEMA_PATH.read_text(encoding="utf-8"))
        await db.execute(
            "INSERT INTO workspace (id, name, description, user_id) VALUES (?, 'w', '', ?)",
            (str(uuid.uuid4()), USER_ID))
        doc_id = str(uuid.uuid4())
        await db.execute(
            "INSERT INTO documents (id, user_id, filename, title, path, relative_path, "
            "source_kind, file_type, status, tags, version, document_number) "
            "VALUES (?, ?, '图.pdf', 'T', '/', '图.pdf', 'source', 'pdf', 'processing', "
            "'[]', 0, 1)", (doc_id, USER_ID))
        await db.commit()

        pages = [(1, "页", [{"id": "img_1_0.png", "bytes": b"PNG", "format": "png"}])]
        async with lp._db_write_gate:                        # 占住闸门
            task = asyncio.create_task(lp._save_local_images(db, doc_id, ws, pages))
            await asyncio.sleep(0.1)
            cursor = await db.execute(
                "SELECT COUNT(*) FROM documents WHERE source_kind = 'asset'")
            assert (await cursor.fetchone())[0] == 0         # DB 写被闸门挡住
        await task                                           # 释放后完成
        cursor = await db.execute(
            "SELECT COUNT(*) FROM documents WHERE source_kind = 'asset'")
        assert (await cursor.fetchone())[0] == 1
    finally:
        await db.close()


async def test_watcher_content_change_clears_quarantine(tmp_path, monkeypatch):
    """隔离文档被替换为新内容:watcher 重建索引时清零计数、解除隔离。"""
    import domain.local_processor as lp
    from domain import watcher

    ws = tmp_path / "ws"
    (ws / ".llmwiki").mkdir(parents=True)
    f = ws / "修复版.docx"
    f.write_bytes(b"new fixed content")
    db = await aiosqlite.connect(str(ws / ".llmwiki" / "index.db"))
    try:
        await db.executescript(SCHEMA_PATH.read_text(encoding="utf-8"))
        await db.execute(
            "INSERT INTO workspace (id, name, description, user_id) VALUES (?, 'w', '', ?)",
            (str(uuid.uuid4()), USER_ID))
        doc_id = str(uuid.uuid4())
        await db.execute(
            "INSERT INTO documents (id, user_id, filename, title, path, relative_path, "
            "source_kind, file_type, status, extraction_attempts, content_hash, tags, "
            "version, document_number) "
            "VALUES (?, ?, '修复版.docx', 'F', '/', '修复版.docx', 'source', 'docx', "
            "'failed', 3, 'old-hash-of-broken-file', '[]', 0, 1)", (doc_id, USER_ID))
        await db.commit()

        async def noop(workspace_, d):
            pass

        monkeypatch.setattr(lp, "process_document_isolated", noop)
        await watcher._index_file(db, ws, f)

        cursor = await db.execute(
            "SELECT status, extraction_attempts FROM documents WHERE id = ?", (doc_id,))
        assert (await cursor.fetchone()) == ("pending", 0)   # 隔离随内容变化解除
    finally:
        await db.close()


def _mini_docx(path: Path) -> None:
    """手工拼最小合法 docx(zip + word/document.xml,含表格段落)。"""
    import zipfile
    xml = (
        '<?xml version="1.0"?>'
        '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
        '<w:body>'
        '<w:p><w:r><w:t>境外投资备案指南</w:t></w:r></w:p>'
        '<w:p><w:r><w:t>第一章 总则</w:t></w:r><w:r><w:t>(续)</w:t></w:r></w:p>'
        '<w:tbl><w:tr><w:tc><w:p><w:r><w:t>表格内容甲</w:t></w:r></w:p></w:tc></w:tr></w:tbl>'
        '</w:body></w:document>'
    )
    with zipfile.ZipFile(path, 'w') as z:
        z.writestr('word/document.xml', xml)


def test_docx_zip_fallback_extracts_paragraphs(tmp_path):
    from domain.local_processor import _extract_docx_paragraphs

    f = tmp_path / '样例.docx'
    _mini_docx(f)
    paras = _extract_docx_paragraphs(f)
    assert paras == ['境外投资备案指南', '第一章 总则(续)', '表格内容甲']   # 表格段落也覆盖


async def test_process_office_falls_back_to_docx_xml(tmp_path, monkeypatch):
    """LibreOffice 崩溃(损坏/非标准 docx 常见)时走 zip 直读兜底入库。"""
    import domain.local_processor as lp

    ws = tmp_path / 'ws'
    (ws / '.llmwiki').mkdir(parents=True)
    _mini_docx(ws / '崩溃文档.docx')
    db = await aiosqlite.connect(str(ws / '.llmwiki' / 'index.db'))
    try:
        await db.executescript(SCHEMA_PATH.read_text(encoding='utf-8'))
        await db.execute(
            "INSERT INTO workspace (id, name, description, user_id) VALUES (?, 'w', '', ?)",
            (str(uuid.uuid4()), USER_ID))
        doc_id = str(uuid.uuid4())
        await db.execute(
            "INSERT INTO documents (id, user_id, filename, title, path, relative_path, "
            "source_kind, file_type, status, tags, version, document_number) "
            "VALUES (?, ?, '崩溃文档.docx', 'C', '/', '崩溃文档.docx', 'source', 'docx', "
            "'pending', '[]', 0, 1)", (doc_id, USER_ID))
        await db.commit()

        class Crashed:
            returncode = 134
            stderr = b'Fatal exception: Signal 6'
            stdout = b''

        monkeypatch.setattr(lp.shutil, 'which', lambda name: '/usr/bin/libreoffice')
        monkeypatch.setattr(lp.subprocess, 'run', lambda *a, **kw: Crashed())

        await lp.process_document(db, doc_id, ws)

        cursor = await db.execute(
            "SELECT status, parser, content, extraction_attempts FROM documents WHERE id = ?",
            (doc_id,))
        status, parser, content, attempts = await cursor.fetchone()
        assert status == 'ready' and parser == 'docx-xml' and attempts == 0
        assert '境外投资备案指南' in content and '表格内容甲' in content
    finally:
        await db.close()


async def test_doc_binary_has_no_fallback(tmp_path, monkeypatch):
    """.doc(OLE 二进制)没有廉价兜底:LibreOffice 失败即失败。"""
    import domain.local_processor as lp

    class Crashed:
        returncode = 134
        stderr = b'Fatal exception: Signal 6'
        stdout = b''

    monkeypatch.setattr(lp.shutil, 'which', lambda name: '/usr/bin/libreoffice')
    monkeypatch.setattr(lp.subprocess, 'run', lambda *a, **kw: Crashed())
    (tmp_path / 'old.doc').write_bytes(b'\xd0\xcf\x11\xe0 fake ole')
    with pytest.raises(RuntimeError, match='LibreOffice conversion failed'):
        await lp._process_office(None, 'doc-1', tmp_path / 'old.doc', tmp_path)
