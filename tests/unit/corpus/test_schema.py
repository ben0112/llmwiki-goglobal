"""八维 entry parsing and validation."""

from datetime import date

import pytest

from corpus.codetable import load
from corpus.schema import ERROR, WARN, make_entry_id, normalize_headers, parse_row

TODAY = date(2026, 7, 20)


@pytest.fixture(scope="module")
def table():
    return load()


def flat_row(**overrides):
    """A well-formed pipeline-dialect (标注明细.csv) row."""
    row = {
        "entry_id": "",
        "relpath": "12_山东省走出去公共服务平台/odi_notice.txt",
        "source": "12_山东省走出去公共服务平台",
        "title": "境外投资备案(核准)无纸化管理通知",
        "url": "https://example.gov.cn/odi",
        "阶段": "S2",
        "服务大类": "G1(副C1)",
        "体裁": "政策法规",
        "隐性规则": "R0",
        "证据": "E1",
        "来源": "国内",
        "归口": "商务委",
        "国别区域": "通用",
        "行业形态": "通用/通用",
        "时效": "M2",
        "置信度": "高",
        "理由": "国家政策文件",
        "建议消费": "出海导办·ODI预审",
    }
    row.update(overrides)
    return row


def test_parse_flat_row(table):
    rec = parse_row(flat_row(), table, today=TODAY)
    assert rec.stage == "S2" and rec.stage_ext == []
    assert rec.domain == "G1" and rec.domain_ext == ["C1"]
    assert rec.genre == "政策法规"
    assert rec.rule_type == ["R0"]
    assert rec.evidence == "E1"
    assert rec.origin == "国内"
    assert rec.gov_dept == ["商务委"]
    assert rec.geo_scope == "通用" and rec.geo_country == []
    assert rec.industry == ["通用"] and rec.mode == []
    assert rec.timeliness == "M2"
    assert rec.review_due == "2026-10-20"  # M2 → +92 天
    assert rec.confidence == 0.9
    assert rec.consumer_agents == ["出海导办", "ODI预审"]
    assert not rec.needs_review
    assert rec.lifecycle_state == "已入库"
    # entry_id 自动生成: 阶段-大类-体裁短码-国别短码-流水
    assert rec.entry_id.startswith("S2-G1-政策-GEN-")
    assert not [i for i in rec.issues if i.level == ERROR]


def test_parse_decorated_row(table):
    """Curated 试标注样本 value style must parse identically."""
    rec = parse_row(flat_row(
        title="哥伦比亚公路项目",
        relpath="01_丝路基金/colombia_road.txt",
        阶段="S2④(副S3⑤)",
        服务大类="O1金融(副Z2)",
        体裁="案例经验",
        证据="E2/丝路基金(U0)",
        来源="国内",
        归口="",
        国别区域="拉美·哥伦比亚",
        行业形态="工程承包/产能",
        时效="M3/已发布",
        置信度="0.8",
    ), table, today=TODAY)
    assert rec.stage == "S2" and rec.stage_ext == ["S3"]
    assert rec.domain == "O1" and rec.domain_ext == ["Z2"]
    assert rec.evidence == "E2"
    assert rec.gov_dept == ["其他"]  # U0 embedded in the 证据 cell
    assert rec.geo_region == ["拉美"]
    assert rec.geo_country == ["COL"] and rec.geo_country_names == ["哥伦比亚"]
    assert rec.geo_scope == "单国"
    assert rec.industry == ["工程承包"] and rec.mode == ["产能出海"]
    assert rec.timeliness == "M3" and rec.lifecycle_state == "已发布"
    assert rec.confidence == 0.8
    assert rec.entry_id.startswith("S2-O1-案例-COL-")


def test_curated_header_dialect(table):
    row = {
        "标题(节选)": "印尼数据本地化新规影响分析",
        "来源/平台": "贸法通",
        "阶段(主/副)": "S3⑤",
        "服务大类(主/副)": "Z1(副C1)",
        "体裁F1": "国别研究",
        "隐性规则F2": "R1",
        "证据E/归口部门": "E2/网信办·数据局",
        "来源域": "目的地国",
        "国别·区域": "东盟·印尼",
        "行业/出海形态": "跨境电商/产品",
        "时效/状态": "M2/已发布",
        "置信": "0.85",
        "消费智能体": "国别市场·分析提炼",
    }
    assert "title" in normalize_headers(row)
    rec = parse_row(row, table, today=TODAY)
    assert rec.title == "印尼数据本地化新规影响分析"
    assert rec.domain == "Z1" and rec.rule_type == ["R1"]
    assert rec.origin == "目的地国"
    assert rec.gov_dept == ["网信办", "数据局"]  # 牵头+协同 双标签
    assert rec.geo_region == ["东盟"] and rec.geo_country == ["IDN"]
    assert rec.mode == ["产品出海"]


def test_fallback_row_goes_to_review(table):
    """公理一: nothing is unclassifiable — fallbacks + review flag instead."""
    rec = parse_row(flat_row(
        阶段="不知道",
        服务大类="X9",
        体裁="普通资讯",
        归口="",
        国别区域="火星",
        置信度="低",
    ), table, today=TODAY)
    assert rec.stage == "S0"  # 兜底
    assert rec.domain == "X9"
    assert rec.genre == "其他"  # alias 普通资讯 → 其他
    assert rec.gov_dept == ["其他"]
    assert rec.needs_review
    assert rec.lifecycle_state == "待复核"
    levels = {i.level for i in rec.issues}
    assert ERROR in levels and WARN in levels
    geo_issues = [i for i in rec.issues if i.field == "国别区域"]
    assert geo_issues and geo_issues[0].level == WARN


def test_invalid_entry_id_regenerated(table):
    rec = parse_row(flat_row(entry_id="not-a-valid-id"), table, today=TODAY)
    assert rec.entry_id.startswith("S2-G1-政策-GEN-")
    assert any(i.field == "entry_id" for i in rec.issues)


def test_valid_entry_id_kept(table):
    rec = parse_row(flat_row(entry_id="S2-G1-政策-GEN-0005"), table, today=TODAY)
    assert rec.entry_id == "S2-G1-政策-GEN-0005"


def test_entry_id_stable_across_runs(table):
    a = parse_row(flat_row(), table, today=TODAY)
    b = parse_row(flat_row(置信度="0.7"), table, today=TODAY)
    assert a.entry_id == b.entry_id  # serial keyed on relpath, 续跑不变


def test_make_entry_id_region_fallback(table):
    rec = parse_row(flat_row(国别区域="东盟"), table, today=TODAY)
    assert rec.geo_scope == "区域"
    assert make_entry_id(rec, rec.source_relpath, table).split("-")[3] == "ASN"


def test_business_view_columns(table):
    rec = parse_row(flat_row(**{
        "业务码": "B1.2", "业务场景": "", "业务需求类": "", "业务优先级": "", "业务待定": "",
    }), table, today=TODAY)
    assert rec.business_code == "B1.2"
    assert rec.business_scene == "市场主体设立流程"  # filled from 码表
    assert rec.business_class == "市场准入与主体设立类"
    assert rec.business_priority == "P2"
    assert not rec.business_pending
    assert "B1.2" in rec.to_tags()


def test_metadata_and_tags_shape(table):
    rec = parse_row(flat_row(), table, today=TODAY)
    meta = rec.to_metadata()
    for key in ("stage", "domain", "genre", "rule_type", "evidence", "origin",
                "gov_dept", "timeliness", "lifecycle_state", "review_due"):
        assert meta[key] not in (None, "", [])
    tags = rec.to_tags()
    assert "S2" in tags and "G1" in tags and "政策法规" in tags and "M2" in tags
