from llmwiki_core.references import (
    ReferenceEdge,
    build_lookup_maps,
    extract_references,
    parse_citation_filename,
)


def test_reference_extraction_deduplicates_edges():
    docs = [
        {"id": "src", "filename": "law.pdf", "title": "Law", "path": "/"},
        {"id": "page", "filename": "risk.md", "title": "Risk", "path": "/wiki/"},
    ]
    names, bases, paths = build_lookup_maps(docs)
    edges = extract_references(
        "Claim[^1].\n\n[^1]: law.pdf, p.3\n[^2]: law.pdf, p.4",
        "page",
        "",
        names,
        bases,
        paths,
    )
    assert edges == [ReferenceEdge("src", "cites", 3)]


def test_chinese_page_suffix_with_fullwidth_comma_is_parsed():
    assert parse_citation_filename("法规汇编.pdf，第 12 页") == ("法规汇编.pdf", 12)
