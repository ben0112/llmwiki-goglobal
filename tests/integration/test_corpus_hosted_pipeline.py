"""Hosted (Postgres) corpus pipeline (L3-P4): mock classify of KB source docs,
entry + chunks land transactionally, state machine + idempotent rerun."""

import asyncio
import os
import sys
import uuid
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

DB_URL = os.environ["DATABASE_URL"]
USER_ID = str(uuid.uuid4())
USER_EMAIL = "pipeline-admin@example.com"
KB_SLUG = "pipeline-corpus"

CN_POLICY = (
    "标题:境外投资备案无纸化管理通知\n\n"
    "为进一步优化境外投资管理服务,现就境外投资备案(核准)无纸化管理有关事项通知如下。"
    "企业应当通过商务部业务系统统一平台在线提交备案申请材料,实行全程无纸化办理,"
    "不再要求企业报送纸质材料。各级商务主管部门应当及时受理并按规定时限办结,"
    "同时加强事中事后监管,确保境外投资备案管理工作平稳有序开展。" * 2
)


@pytest.fixture
async def seeded_kb(pool):
    await pool.execute(
        "INSERT INTO users (id, email, display_name) VALUES ($1, $2, 'P') "
        "ON CONFLICT (id) DO NOTHING", USER_ID, USER_EMAIL)
    kb = await pool.fetchrow(
        "INSERT INTO knowledge_bases (user_id, name, slug) VALUES ($1, $2, $2) RETURNING id",
        uuid.UUID(USER_ID), KB_SLUG)
    kb_id = kb["id"]
    doc_id = uuid.uuid4()
    await pool.execute(
        "INSERT INTO documents (id, knowledge_base_id, user_id, filename, title, path, "
        "file_type, status, content) VALUES ($1, $2, $3, 'odi.txt', '境外投资备案通知', '/', "
        "'txt', 'ready', $4)", doc_id, kb_id, uuid.UUID(USER_ID), CN_POLICY)
    short_id = uuid.uuid4()
    await pool.execute(
        "INSERT INTO documents (id, knowledge_base_id, user_id, filename, title, path, "
        "file_type, status, content) VALUES ($1, $2, $3, 'ad.txt', '口号', '/', "
        "'txt', 'ready', '关注我们,共创辉煌。')", short_id, kb_id, uuid.UUID(USER_ID))
    yield {"kb_id": kb_id, "doc_id": doc_id, "short_id": short_id}
    await pool.execute("DELETE FROM knowledge_bases WHERE user_id = $1", uuid.UUID(USER_ID))
    await pool.execute("DELETE FROM users WHERE id = $1", uuid.UUID(USER_ID))


async def test_hosted_pipeline_mock_end_to_end(pool, seeded_kb):
    from corpus.llm import LLMConfig
    from corpus.pipeline import run_batch_hosted

    result = await run_batch_hosted(DB_URL, USER_EMAIL, KB_SLUG, LLMConfig(base_url="mock"))
    assert (result.picked, result.imported, result.excluded, result.failed) == (2, 1, 1, 0)

    entry = await pool.fetchrow(
        "SELECT path, filename, metadata->>'entry_id' AS eid, metadata->>'stage' AS stage "
        "FROM documents WHERE knowledge_base_id = $1 AND metadata->>'entry_id' IS NOT NULL",
        seeded_kb["kb_id"])
    assert entry is not None and entry["path"].startswith("/corpus/S2-G1/")
    assert entry["stage"] == "S2"
    # 检索分块已生成(标题从正文头提取)
    chunks = await pool.fetchval(
        "SELECT COUNT(*) FROM document_chunks sc JOIN documents d ON d.id = sc.document_id "
        "WHERE d.knowledge_base_id = $1 AND d.metadata->>'entry_id' IS NOT NULL",
        seeded_kb["kb_id"])
    assert chunks > 0

    states = {r["doc_id"]: r["state"] for r in await pool.fetch(
        "SELECT doc_id, state FROM corpus_pipeline")}
    assert states[seeded_kb["doc_id"]] == "imported"
    assert states[seeded_kb["short_id"]] == "excluded"

    # 幂等重跑:无候选、条目不重复
    result2 = await run_batch_hosted(DB_URL, USER_EMAIL, KB_SLUG, LLMConfig(base_url="mock"))
    assert result2.picked == 0
    n = await pool.fetchval(
        "SELECT COUNT(*) FROM documents WHERE knowledge_base_id = $1 "
        "AND metadata->>'entry_id' IS NOT NULL", seeded_kb["kb_id"])
    assert n == 1
