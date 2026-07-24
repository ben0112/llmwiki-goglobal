from llmwiki_core.references import ReferenceEdge
from llmwiki_core.wiki import WikiWriteBundle


def test_wiki_bundle_deduplicates_derived_edges():
    bundle = WikiWriteBundle.build(
        document_id="page",
        expected_version=4,
        filename="page.md",
        path="/wiki/",
        file_type="md",
        content="body",
        title="Page",
        tags=["risk"],
        date="2026-07-24",
        metadata={"description": "Page"},
        edges=[
            ReferenceEdge("src", "cites", 3),
            ReferenceEdge("src", "cites", 4),
        ],
    )

    assert bundle.tags == ("risk",)
    assert bundle.edges == (ReferenceEdge("src", "cites", 3),)
