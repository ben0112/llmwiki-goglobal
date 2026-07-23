"""搜索的语料折叠与可信度标记:

- 已入库源文件的命中折叠为语料条目(条目命中时原文重复丢弃);
- 未入库 / 审核排除 / 分类失败的源文件命中带标记;
- 无分类流水线的普通工作区行为不变。
"""

import pytest


POLICY = (
    "为进一步优化境外投资管理服务,现就境外投资备案(核准)无纸化管理有关事项通知如下。"
    "企业应当通过商务部业务系统统一平台在线提交备案申请材料,实行全程无纸化办理,"
    "不再要求企业报送纸质材料。各级商务主管部门应当及时受理并按规定时限办结,"
    "同时加强事中事后监管,确保境外投资备案管理工作平稳有序开展。"
)
SLOGAN = (
    "欢迎关注我们的公众号,境外投资备案资讯每日更新,点赞收藏转发三连,"
    "更多境外投资备案精彩内容不容错过,谢谢大家的支持与厚爱,共创辉煌明天。" * 3
)
FRESH = (
    "刚拷入的境外投资备案新文件,还没有走完分类流水线的审核与八维标注流程,"
    "内容涉及境外投资备案申请的常见问题与材料准备清单,后续等待自动分类处理。"
    "这里补充足够的正文长度以便切块器不丢弃本段内容,保障全文检索可以命中。" * 3
)


def _handler(instance, kb_id):
    from tools.search import SearchHandler
    return SearchHandler(instance, {"id": kb_id, "slug": "test-workspace"})


async def _mk_docs(instance, kb_id):
    src = await instance.create_document(
        kb_id, "备案通知.md", "境外投资备案通知", "/", "md", POLICY, [])
    entry = await instance.create_document(
        kb_id, "entry.md", "境外投资备案通知(条目)", "/corpus/S2-G1/", "md", POLICY, [],
        metadata={"spec_version": "v2026.06", "entry_id": "E1", "stage": "S2",
                  "source_relpath": "备案通知.md"})
    slogan = await instance.create_document(
        kb_id, "口号.md", "口号文档", "/", "md", SLOGAN, [])
    fresh = await instance.create_document(
        kb_id, "新文件.md", "刚拷入的文件", "/", "md", FRESH, [])
    return src, entry, slogan, fresh


async def _mark_pipeline(instance, rows):
    db = instance._db_or_raise()
    for doc_id, state, error in rows:
        await db.execute(
            "INSERT INTO corpus_pipeline (doc_id, state, attempts, error, updated_at) "
            "VALUES (?, ?, 1, ?, datetime('now'))", (doc_id, state, error))
    await db.commit()


async def test_imported_source_folds_into_entry(fs):
    instance, kb_id = fs
    src, entry, slogan, fresh = await _mk_docs(instance, kb_id)
    await _mark_pipeline(instance, [(src["id"], "imported", None),
                                    (slogan["id"], "excluded", "口号/宣传,无语料价值")])

    out = await _handler(instance, kb_id).search_chunks("无纸化管理", "*", None, 10)
    assert "corpus/S2-G1/entry.md" in out       # 条目在场
    assert "**/备案通知.md**" not in out          # 原文重复被折叠掉
    assert "[语料条目]" not in out or "原文:" not in out  # 条目原生命中,无需折叠行


async def test_source_hit_rewritten_when_entry_not_hit(fs):
    """条目因分面/路径过滤等未命中时:原文命中改标为条目并保留原文指针。"""
    instance, kb_id = fs
    src, entry, slogan, fresh = await _mk_docs(instance, kb_id)
    await _mark_pipeline(instance, [(src["id"], "imported", None)])
    # 只搜源文件路径范围,条目被 path 过滤掉 → 折叠改标生效
    out = await _handler(instance, kb_id).search_chunks("无纸化管理", "备案*", None, 10)
    assert "[语料条目]" in out and "原文: /备案通知.md" in out


async def test_trust_markers_for_unvetted_sources(fs):
    instance, kb_id = fs
    src, entry, slogan, fresh = await _mk_docs(instance, kb_id)
    await _mark_pipeline(instance, [(src["id"], "imported", None),
                                    (slogan["id"], "excluded", "口号/宣传,无语料价值")])

    out = await _handler(instance, kb_id).search_chunks("境外投资备案", "*", None, 10)
    assert "[已排除:口号/宣传,无语料价值]" in out   # 审核拒收带理由
    assert "[未入库]" in out                        # 新文件未分类

    out2 = await _handler(instance, kb_id).search_chunks("常见问题与材料准备", "*", None, 10)
    assert "[未入库]" in out2


async def test_generic_workspace_unchanged(fs):
    """无流水线记录的普通工作区:不折叠、无标记。"""
    instance, kb_id = fs
    await _mk_docs(instance, kb_id)   # 不写任何 corpus_pipeline 行

    out = await _handler(instance, kb_id).search_chunks("境外投资备案", "*", None, 10)
    assert "[未入库]" not in out and "[已排除" not in out
    assert "**/备案通知.md**" in out               # 原文照常返回
