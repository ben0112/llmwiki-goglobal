"""语料分类流水线(L3-P1):端点感知默认、配置优先级、mock 全链路、失败隔离。"""

import asyncio
import sqlite3
import uuid
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
SCHEMA = (REPO_ROOT / "shared" / "sqlite_schema.sql").read_text(encoding="utf-8")

POLICY_BODY = (
    "境外投资备案(核准)报告制度有关事项的通知。企业开展境外投资,应当按照本通知要求,"
    "通过全国一体化在线政务服务平台提交备案材料,商务主管部门在线出具企业境外投资证书电子证照。"
    "本通知自发布之日起施行,原有纸质材料报送要求同时废止。各地商务主管部门应当做好衔接工作,"
    "确保企业境外投资备案(核准)业务办理连续稳定,并加强事中事后监管与信息共享。" * 2
)


def _init_ws(tmp_path: Path) -> Path:
    ws = tmp_path / "ws"
    (ws / ".llmwiki").mkdir(parents=True)
    conn = sqlite3.connect(str(ws / ".llmwiki" / "index.db"))
    conn.executescript(SCHEMA)
    conn.execute("INSERT INTO workspace (id, name, description, user_id) VALUES ('w1', 't', '', 'u1')")
    conn.commit()
    conn.close()
    return ws


def _insert_source(ws: Path, filename: str, title: str, content: str) -> str:
    conn = sqlite3.connect(str(ws / ".llmwiki" / "index.db"))
    doc_id = str(uuid.uuid4())
    conn.execute(
        "INSERT INTO documents (id, user_id, filename, title, path, relative_path, "
        "source_kind, file_type, status, content) "
        "VALUES (?, 'u1', ?, ?, '/', ?, 'source', 'txt', 'ready', ?)",
        (doc_id, filename, title, filename, content),
    )
    conn.commit()
    conn.close()
    return doc_id


def _states(ws: Path) -> dict:
    conn = sqlite3.connect(str(ws / ".llmwiki" / "index.db"))
    rows = conn.execute("SELECT doc_id, state, attempts FROM corpus_pipeline").fetchall()
    conn.close()
    return {r[0]: (r[1], r[2]) for r in rows}


def test_endpoint_aware_batch_default():
    from corpus.llm import LLMConfig, default_batch_limit, is_local_endpoint

    assert is_local_endpoint("http://localhost:8000/v1")
    assert is_local_endpoint("http://host.docker.internal:8000/v1")
    assert is_local_endpoint("mock")
    assert not is_local_endpoint("https://api.deepseek.com/v1")
    assert default_batch_limit("http://127.0.0.1:1234/v1") == 20
    assert default_batch_limit("https://api.deepseek.com/v1") == 100
    assert LLMConfig(base_url="https://api.deepseek.com/v1", batch_limit=7).effective_batch_limit == 7


def test_resolve_config_precedence():
    from corpus.pipeline import resolve_config

    class Env:
        CORPUS_LLM_BASE_URL = "http://env:1/v1"
        CORPUS_LLM_MODEL = "env-model"
        CORPUS_LLM_API_KEY = "env-key"
        CORPUS_LLM_TIMEOUT = 60
        CORPUS_BATCH_LIMIT = 5

    stored = {"base_url": "http://ui:2/v1", "model": "ui-model"}
    cfg = resolve_config(stored, Env)
    assert cfg.base_url == "http://ui:2/v1"      # 设置页 > 环境变量
    assert cfg.model == "ui-model"
    assert cfg.api_key == "env-key"              # 未在 UI 配置的字段回退环境变量
    assert cfg.batch_limit == 5

    cfg2 = resolve_config({}, Env)
    assert cfg2.base_url == "http://env:1/v1"    # 环境变量 > 内置默认


def test_extract_flat_json_tolerates_noise():
    from corpus.llm import extract_flat_json

    assert extract_flat_json('<think>嗯</think>{"服务大类":"G1","阶段":"S2"}', ("服务大类",)) == {
        "服务大类": "G1", "阶段": "S2"}
    assert extract_flat_json('前言 {"收录":"是","理由":"ok",} 后记', ("收录",)) == {
        "收录": "是", "理由": "ok"}
    assert extract_flat_json("没有JSON", ("收录",)) is None


def test_run_batch_mock_end_to_end(tmp_path):
    from corpus.llm import LLMConfig
    from corpus.pipeline import run_batch, status_counts

    ws = _init_ws(tmp_path)
    good = _insert_source(ws, "0001_境外投资备案通知.txt", "境外投资备案(核准)无纸化通知", POLICY_BODY)
    short = _insert_source(ws, "0002_口号.txt", "欢迎关注", "关注我们,共创辉煌。")

    cfg = LLMConfig(base_url="mock")
    result = asyncio.run(run_batch(ws, cfg))
    assert (result.picked, result.imported, result.excluded, result.failed) == (2, 1, 1, 0)

    states = _states(ws)
    assert states[good][0] == "imported"
    assert states[short][0] == "excluded"

    # 条目落盘 + 入索引(货架路径),源文档保持不动
    conn = sqlite3.connect(str(ws / ".llmwiki" / "index.db"))
    entry = conn.execute(
        "SELECT relative_path, metadata FROM documents WHERE relative_path LIKE 'corpus/%'"
    ).fetchone()
    conn.close()
    assert entry is not None and entry[0].startswith("corpus/S2-G1/")
    assert (ws / entry[0]).exists()
    assert "境外投资备案" in (ws / entry[0]).read_text(encoding="utf-8")

    # 幂等:再跑一轮无候选、无重复
    result2 = asyncio.run(run_batch(ws, cfg))
    assert (result2.picked, result2.imported) == (0, 0)
    counts = sqlite3.connect(str(ws / ".llmwiki" / "index.db")).execute(
        "SELECT COUNT(*) FROM documents WHERE relative_path LIKE 'corpus/%'").fetchone()[0]
    assert counts == 1


def test_run_batch_failure_isolated_and_retried(tmp_path, monkeypatch):
    import corpus.pipeline as pl
    from corpus.llm import LLMConfig, LLMError

    ws = _init_ws(tmp_path)
    good = _insert_source(ws, "0001_境外投资备案通知.txt", "境外投资备案(核准)无纸化通知", POLICY_BODY)
    bad = _insert_source(ws, "0002_坏文档.txt", "会失败的文档", POLICY_BODY)

    real_classify = pl.classify

    async def flaky_classify(config, title, body, relpath):
        if "坏文档" in relpath:
            raise LLMError("模拟端点故障")
        return await real_classify(config, title, body, relpath)

    monkeypatch.setattr(pl, "classify", flaky_classify)

    cfg = LLMConfig(base_url="mock")
    result = asyncio.run(pl.run_batch(ws, cfg))
    assert result.imported == 1 and result.failed == 1     # 失败不拖累同批其他文档

    states = _states(ws)
    assert states[good][0] == "imported"
    assert states[bad] == ("failed", 1)

    # 失败条目下一轮重试,attempts 累加;到上限后不再入选
    asyncio.run(pl.run_batch(ws, cfg))
    asyncio.run(pl.run_batch(ws, cfg))
    assert _states(ws)[bad] == ("failed", 3)
    result4 = asyncio.run(pl.run_batch(ws, cfg))
    assert result4.picked == 0


def test_resolve_auto_precedence():
    from corpus.pipeline import resolve_auto

    class Env:
        CORPUS_AUTOCLASSIFY = True
        CORPUS_AUTO_INTERVAL = 300

    # 设置页显式关闭要压过环境变量的开启
    assert resolve_auto({"auto_enabled": False}, Env) == {"enabled": False, "interval": 300}
    assert resolve_auto({}, Env)["enabled"] is True
    assert resolve_auto({"auto_enabled": True, "auto_interval": 45}, Env) == {
        "enabled": True, "interval": 45}
    # 间隔下限 30s
    assert resolve_auto({"auto_interval": 5}, Env)["interval"] == 30


def test_run_batch_reports_progress_and_today_count(tmp_path):
    from corpus.llm import LLMConfig
    from corpus.pipeline import run_batch, status_counts, _connect

    ws = _init_ws(tmp_path)
    _insert_source(ws, "0001_境外投资备案通知.txt", "境外投资备案(核准)无纸化通知", POLICY_BODY)
    _insert_source(ws, "0002_口号.txt", "欢迎关注", "关注我们,共创辉煌。")

    seen = []
    asyncio.run(run_batch(ws, LLMConfig(base_url="mock"),
                          on_progress=lambda d, t: seen.append((d, t))))
    assert seen == [(1, 2), (2, 2)]

    conn = _connect(str(ws / ".llmwiki" / "index.db"))
    try:
        assert status_counts(conn)["imported_today"] == 1
    finally:
        conn.close()
