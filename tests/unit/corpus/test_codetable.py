"""码表 (code table) loading and normalization."""

import pytest

from corpus.codetable import DEFAULT_VERSION, load


@pytest.fixture(scope="module")
def table():
    return load(DEFAULT_VERSION)


def test_load_version(table):
    assert table.version == "v2026.06"
    assert table.stage_codes == {"S0", "S1", "S2", "S3", "S4"}
    assert len(table.domain_codes) == 20  # 19 大类 + X9 兜底
    assert "X9" in table.domain_codes
    assert len(table.genres) == 11
    assert len(table.rules) == 7
    assert len(table.evidence) == 5
    assert len(table.business_scenes) == 28  # 27 场景 + 待定


def test_unknown_version_lists_available():
    with pytest.raises(FileNotFoundError, match="v2026.06"):
        load("v1999.01")


def test_normalize_genre(table):
    assert table.normalize_genre("政策法规") == ("政策法规", False)
    assert table.normalize_genre("T1") == ("政策法规", False)
    assert table.normalize_genre("普通资讯") == ("其他", True)  # audit-taxonomy alias
    assert table.normalize_genre("实操指引") == ("实操指引方法论", True)
    assert table.normalize_genre("完全未知体裁") == (None, False)
    assert table.genre_short("政策法规") == "政策"
    assert table.genre_short("未知") == "其他"


def test_normalize_dept(table):
    assert table.normalize_dept("商务委") == "商务委"
    assert table.normalize_dept("U3") == "商务委"
    assert table.normalize_dept("市监") == "市场监管局"
    assert table.normalize_dept("知产局") == "知识产权局"
    assert table.normalize_dept("丝路基金") is None


def test_normalize_origin(table):
    assert table.normalize_origin("目的地国") == "目的地国"
    assert table.normalize_origin("O2") == "国际"
    assert table.normalize_origin("外太空") is None


def test_country_and_region_codes(table):
    assert table.country_code("印尼") == "IDN"
    assert table.country_code("印度尼西亚") == "IDN"
    assert table.country_code("UAE") == "ARE"  # pipeline alias → ISO 3166 alpha-3
    assert table.country_code("IDN") == "IDN"
    assert table.country_code("火星") is None
    assert table.region_code("东盟") == "ASN"
    assert table.region_code("一带一路") == "BNR"
    assert table.region_code("银河系") is None


def test_industry_and_mode(table):
    assert table.normalize_industry("跨境电商") == "跨境电商"
    assert table.normalize_industry("制造") == "制造业"
    assert table.normalize_industry("C") == "制造业"  # GB/T 4754 门类码
    assert table.normalize_industry("玄学") is None
    assert table.normalize_mode("产能") == "产能出海"
    assert table.normalize_mode("并购") == "并购"
    assert table.normalize_mode("资本出海") is None


def test_confidence(table):
    assert table.confidence_value("高") == 0.9
    assert table.confidence_value("0.8") == 0.8
    assert table.confidence_value("1.5") is None
    assert table.confidence_value("很有信心") is None


def test_review_days(table):
    assert table.review_days("M1") == 7
    assert table.review_days("M2") == 92
    assert table.review_days("M3") == 365
    assert table.review_days("M9") == 365  # 兜底 → M3


def test_business_class_of(table):
    assert table.business_class_of("B4.14") == ("B4", "本地化运营与合规管理类", "P1")
    assert table.business_class_of("待定") == ("待定", "待定", "-")


def test_code_extraction(table):
    assert table.extract_stages("S2④(副S3⑤)") == ["S2", "S3"]
    assert table.extract_domains("O1金融(副Z2)") == ["O1", "Z2"]
    assert table.extract_domains("X9") == ["X9"]
    assert table.extract_rules("R1+R5") == ["R1", "R5"]
    assert table.extract_evidence("E2/丝路基金(U0)") == "E2"
    assert table.extract_timeliness("M3/已发布") == "M3"
