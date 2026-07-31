import importlib.util
from pathlib import Path

from llmwiki_core.references import ReferenceEdge


def _references_module():
    spec = importlib.util.spec_from_file_location(
        "api_references_test",
        Path(__file__).resolve().parents[2] / "api" / "services" / "references.py",
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_parse_citation_preserves_hyphenated_version_suffix():
    parse_citation_filename = _references_module().parse_citation_filename
    assert parse_citation_filename("2501.12948v2-2.pdf, p.5") == (
        "2501.12948v2-2.pdf",
        5,
    )


def test_parse_citation_markdown_link_keeps_page_suffix():
    parse_citation_filename = _references_module().parse_citation_filename
    assert parse_citation_filename("[paper.pdf](https://example.com/paper), p.7") == (
        "paper.pdf",
        7,
    )


def test_extract_references_matches_hyphenated_version_source():
    extract_references = _references_module().extract_references
    source = {
        "id": "source-1",
        "filename": "2501.12948v2-2.pdf",
        "path": "/",
        "title": "DeepSeek-R1",
    }
    page = {
        "id": "page-1",
        "filename": "deepseek-r1.md",
        "path": "/wiki/",
        "title": "DeepSeek-R1",
    }

    edges = extract_references(
        "DeepSeek-R1 uses reinforcement learning.[^1]\n\n[^1]: 2501.12948v2-2.pdf, p.5",
        "page-1",
        "",
        {"2501.12948v2-2.pdf": source, "deepseek-r1.md": page},
        {"2501.12948v2-2": source, "deepseek-r1": page},
        {"deepseek-r1.md": page},
    )

    assert edges == [ReferenceEdge("source-1", "cites", 5)]


def test_parse_citation_chinese_article_suffix_strips_but_keeps_no_page():
    parse_citation_filename = _references_module().parse_citation_filename
    assert parse_citation_filename(
        "00033_《境外投资管理办法》(商务部令2014年第3号)_平台_0f0e072c0a.txt, 第2条"
    ) == ("00033_《境外投资管理办法》(商务部令2014年第3号)_平台_0f0e072c0a.txt", None)


def test_parse_citation_chinese_page_suffix_maps_to_page_number():
    parse_citation_filename = _references_module().parse_citation_filename
    assert parse_citation_filename("白皮书.pdf, 第12页") == ("白皮书.pdf", 12)


def test_parse_citation_chinese_suffix_fullwidth_comma_and_numerals():
    parse_citation_filename = _references_module().parse_citation_filename
    assert parse_citation_filename("条例.txt,第十三条") == ("条例.txt", None)
    assert parse_citation_filename("指南.txt, 第3章") == ("指南.txt", None)


def test_parse_citation_ascii_page_still_wins():
    parse_citation_filename = _references_module().parse_citation_filename
    assert parse_citation_filename("paper.pdf, p.3") == ("paper.pdf", 3)
