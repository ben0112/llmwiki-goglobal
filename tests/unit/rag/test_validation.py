import json
from dataclasses import dataclass, replace
from time import perf_counter
from uuid import UUID

import pytest
from rag.prompts import RagDraft
from rag.validation import validate_draft

from llmwiki_core.documents import DocumentStatus
from llmwiki_core.rag import RagCitation, RagDomainError
from llmwiki_core.references import ReferenceEdge

DOCUMENT_ID = UUID("10000000-0000-0000-0000-000000000001")
SOURCE_ID = UUID("20000000-0000-0000-0000-000000000002")


@dataclass(frozen=True)
class _Evidence:
    document_id: UUID = SOURCE_ID
    document_version: int = 3
    chunk_index: int = 4
    page: int | None = 5
    filename: str = "source.pdf"
    path: str = "/corpus/"
    title: str = "Source"
    content: str = "Selected source text"
    status: str = "ready"
    archived: bool = False


def _citation() -> RagCitation:
    return RagCitation(SOURCE_ID, document_version=3, chunk_index=4, page=5)


def _content(
    *,
    title: str = "Launch risks",
    tags: str = "[launch, risk]",
    description: str = "A sourced launch risk summary.",
    date: str = "2026-07-27",
    visual: str = "```mermaid\ngraph TD\n  A --> B\n```",
    link: str = "[Overview](./overview.md)",
    citation: str = "Risk is documented.[^1]\n\n[^1]: source.pdf, p.5",
) -> str:
    return (
        "---\n"
        f"title: {title}\n"
        f"tags: {tags}\n"
        f"description: {description}\n"
        f"date: {date}\n"
        "---\n"
        f"# {title}\n\n{visual}\n\n{link}\n\n{citation}\n"
    )


def _draft(*, content: str | None = None, citations: tuple[RagCitation, ...] | None = None):
    return RagDraft(
        content=_content() if content is None else content,
        citations=(_citation(),) if citations is None else citations,
    )


def _add_frontmatter(raw: str) -> str:
    return _content().replace("date: 2026-07-27", f"date: 2026-07-27\n{raw}")


def _validate(draft: RagDraft, evidence=(_Evidence(),), **overrides):
    values = {
        "document_id": DOCUMENT_ID,
        "expected_version": 2,
        "target_path": "/wiki/launch/risks.md",
        "max_page_chars": 40_000,
    }
    values.update(overrides)
    return validate_draft(draft, evidence, **values)


def test_validate_draft_returns_complete_bundle_and_bounded_lint_summary():
    bundle, lint = _validate(_draft())

    assert bundle.document_id == str(DOCUMENT_ID)
    assert bundle.expected_version == 2
    assert bundle.filename == "risks.md"
    assert bundle.path == "/wiki/launch/"
    assert bundle.file_type == "md"
    assert bundle.content == _content()
    assert bundle.title == "Launch risks"
    assert bundle.tags == ("launch", "risk")
    assert bundle.date == "2026-07-27"
    assert bundle.metadata["description"] == "A sourced launch risk summary."
    assert bundle.edges == (ReferenceEdge(str(SOURCE_ID), "cites", 5),)
    assert lint == {
        "citation_count": 1,
        "internal_link_count": 1,
        "visual_count": 1,
        "warnings": 0,
    }
    assert len(json.dumps(lint, separators=(",", ":")).encode()) <= 16_384


def test_validate_draft_accepts_a_markdown_image_as_the_required_visual():
    content = _content(visual="![Risk matrix](/wiki/assets/risk.png)")
    bundle, lint = _validate(_draft(content=content))

    assert bundle.content == content
    assert lint["visual_count"] == 1


@pytest.mark.parametrize(
    "visual",
    [
        "![bad](javascript:alert(1))",
        "![bad](data:image/png;base64,private)",
        "![bad](/private/risk.png)",
        "![bad](../risk.png)",
        "![bad](/wiki/assets/risk.exe)",
        "![bad](//private.example/risk.png)",
    ],
)
def test_validate_draft_rejects_unsupported_image_uris(visual):
    with pytest.raises(RagDomainError) as caught:
        _validate(_draft(content=_content(visual=visual)))
    assert caught.value.code == "rag_link_invalid"


@pytest.mark.parametrize(
    "visual",
    [
        "![local](/wiki/assets/risk.png)",
        "![relative](./assets/risk.svg)",
        "![remote](https://static.example.test/risk.webp)",
    ],
)
def test_validate_draft_accepts_supported_image_uris(visual):
    _, lint = _validate(_draft(content=_content(visual=visual)))
    assert lint["visual_count"] == 1


def test_validate_draft_rejects_invented_citation():
    invented = RagCitation(UUID(int=99), document_version=1, chunk_index=0)
    with pytest.raises(RagDomainError) as caught:
        _validate(_draft(citations=(invented,)))
    assert caught.value.code == "rag_citation_invalid"


@pytest.mark.parametrize(
    "evidence",
    [
        (_Evidence(status="failed"),),
        (_Evidence(archived=True),),
        (_Evidence(document_version=4),),
        (),
    ],
)
def test_validate_draft_rejects_failed_archived_changed_or_unselected_evidence(evidence):
    with pytest.raises(RagDomainError) as caught:
        _validate(_draft(), evidence)
    assert caught.value.code == "rag_citation_invalid"


@pytest.mark.parametrize(
    "status",
    ["pending", "processing", "ready", DocumentStatus.PENDING, DocumentStatus.PROCESSING],
)
def test_validate_draft_accepts_every_known_nonfailed_evidence_status(status):
    bundle, _ = _validate(_draft(), (_Evidence(status=status),))
    assert bundle.edges == (ReferenceEdge(str(SOURCE_ID), "cites", 5),)


@pytest.mark.parametrize("status", ["failed", "unknown", True, None])
def test_validate_draft_rejects_failed_unknown_or_nonstring_evidence_status(status):
    with pytest.raises(RagDomainError) as caught:
        _validate(_draft(), (_Evidence(status=status),))
    assert caught.value.code == "rag_citation_invalid"


@pytest.mark.parametrize(
    "content",
    [
        "# no frontmatter\n\n```mermaid\ngraph TD\n```",
        _content(title=""),
        _content(tags="[]"),
        _content(tags="not-a-list"),
        _content(description=""),
        _content(date=""),
        _content(visual="plain text without a visual"),
        _content(visual="```mermaid\ngraph TD\n  A --> B"),
    ],
)
def test_validate_draft_rejects_missing_frontmatter_title_tags_description_date_or_visuals(content):
    with pytest.raises(RagDomainError) as caught:
        _validate(_draft(content=content))
    assert caught.value.code == "rag_invalid_draft"


def test_validate_draft_uses_safe_yaml_and_rejects_object_construction_tags():
    content = _content().replace(
        "title: Launch risks",
        "title: !!python/object/apply:os.system ['private-command']",
    )
    with pytest.raises(RagDomainError) as caught:
        _validate(_draft(content=content))
    assert caught.value.code == "rag_invalid_draft"
    assert "private-command" not in str(caught.value)


@pytest.mark.parametrize(
    "content",
    [
        _add_frontmatter("anchor: &private anchored\nalias: *private"),
        _add_frontmatter("tagged: !private value"),
        _add_frontmatter("merged:\n  <<: {private: value}"),
        _content().replace("title: Launch risks", "title: First\ntitle: Launch risks"),
        _add_frontmatter("nested:\n  duplicate: first\n  duplicate: second"),
    ],
)
def test_validate_draft_rejects_yaml_alias_tag_merge_and_duplicate_keys(content):
    with pytest.raises(RagDomainError) as caught:
        _validate(_draft(content=content))
    assert caught.value.code == "rag_invalid_draft"
    assert "private" not in str(caught.value)


@pytest.mark.parametrize("raw_date", ["not-a-date", "2026-7-2", "2026-07-27T12:00:00Z"])
def test_validate_draft_requires_an_exact_iso_calendar_date(raw_date):
    with pytest.raises(RagDomainError) as caught:
        _validate(_draft(content=_content(date=raw_date)))
    assert caught.value.code == "rag_invalid_draft"


def test_validate_draft_rejects_oversize_without_truncating_or_fixing_content():
    draft = _draft()
    with pytest.raises(RagDomainError) as caught:
        _validate(draft, max_page_chars=len(draft.content) - 1)
    assert caught.value.code == "rag_invalid_draft"
    assert draft.content == _content()


@pytest.mark.parametrize(
    ("link", "expected_code"),
    [
        ("[escape](../private.md)", "rag_link_invalid"),
        ("[escape](/private/page.md)", "rag_link_invalid"),
        ("[host](//private.example/page.md)", "rag_link_invalid"),
        ("[script](javascript:alert(1))", "rag_link_invalid"),
        ("[nul](bad\x00.md)", "rag_invalid_draft"),
    ],
)
def test_validate_draft_rejects_internal_links_that_do_not_normalize_under_wiki(link, expected_code):
    with pytest.raises(RagDomainError) as caught:
        _validate(_draft(content=_content(link=link)))
    assert caught.value.code == expected_code


def test_validate_draft_accepts_normalized_absolute_and_relative_wiki_links():
    content = _content(link=("[One](/wiki/shared/one.md) [Two](./two.md#details) [External](https://example.test)"))
    _, lint = _validate(_draft(content=content))
    assert lint["internal_link_count"] == 2


def test_validate_draft_accepts_a_safe_angle_bracket_link_destination():
    content = _content(link="[Two](<./two.md#details>)")
    _, lint = _validate(_draft(content=content))
    assert lint["internal_link_count"] == 1


@pytest.mark.parametrize(
    "link",
    [
        "[outer [inner](/private/page.md)](./safe.md)",
        "[outer [inner](javascript:alert(1))](./safe.md)",
        "[outer ![inner](/wiki/assets/safe.png)](./safe.md)",
    ],
)
def test_validate_draft_rejects_all_nested_inline_labels_fail_closed(link):
    with pytest.raises(RagDomainError) as caught:
        _validate(_draft(content=_content(link=link)))
    assert caught.value.code == "rag_link_invalid"


def test_validate_draft_still_accepts_a_plain_nonnested_label():
    _, lint = _validate(_draft(content=_content(link="[ordinary label](./safe.md)")))
    assert lint["internal_link_count"] == 1


@pytest.mark.parametrize("link", ["[escape](<../private.md>)", "[root](</private/page.md>)"])
def test_validate_draft_rejects_unsafe_angle_bracket_link_destinations(link):
    with pytest.raises(RagDomainError) as caught:
        _validate(_draft(content=_content(link=link)))
    assert caught.value.code == "rag_link_invalid"


def test_validate_draft_ignores_links_and_citations_inside_html_comments():
    content = _content(
        link="<!-- [escape](../private.md) --> [Safe](./safe.md)",
        citation=("Real.[^1]\n\n[^1]: source.pdf, p.5\n\n<!-- Fake.[^2]\n[^2]: invented.pdf, p.9 -->"),
    )
    _, lint = _validate(_draft(content=content))
    assert lint["internal_link_count"] == 1
    assert lint["citation_count"] == 1


def test_validate_draft_does_not_count_a_commented_image_as_a_visual():
    content = _content(visual="<!-- ![hidden](/wiki/assets/hidden.png) -->")
    with pytest.raises(RagDomainError) as caught:
        _validate(_draft(content=content))
    assert caught.value.code == "rag_invalid_draft"


@pytest.mark.parametrize("info", ["mermaid evil", "mermaid{theme:dark}", "mermaid-js"])
def test_validate_draft_requires_the_exact_mermaid_info_string(info):
    content = _content(visual=f"```{info}\ngraph TD\n  A --> B\n```")
    with pytest.raises(RagDomainError) as caught:
        _validate(_draft(content=content))
    assert caught.value.code == "rag_invalid_draft"


@pytest.mark.parametrize(
    "tag",
    [
        '<img src="javascript:private">',
        "<script>private</script>",
        '<iframe src="private">',
        '<object data="private">',
        '<embed src="private">',
        '<video src="private"></video>',
        '<audio src="private"></audio>',
        '<source src="private">',
        '<link rel="stylesheet" href="private">',
    ],
)
def test_validate_draft_rejects_active_html_asset_tags(tag):
    content = _content() + f"\n{tag}\n"
    with pytest.raises(RagDomainError) as caught:
        _validate(_draft(content=content))
    assert caught.value.code == "rag_link_invalid"
    assert "private" not in str(caught.value)


def test_validate_draft_ignores_active_html_tags_inside_comments():
    content = _content() + '\n<!-- <img src="javascript:private"> -->\n'
    bundle, _ = _validate(_draft(content=content))
    assert bundle.content == content


@pytest.mark.parametrize(
    "visual",
    [
        "```markdown\n![hidden](/wiki/assets/hidden.png)\n```",
        "~~~~ text\n![hidden](/wiki/assets/hidden.png)\n~~~~",
        "inline `![hidden](/wiki/assets/hidden.png)` code",
        "inline ``![hidden](/wiki/assets/hidden.png)`` code",
    ],
)
def test_validate_draft_does_not_count_fenced_or_inline_images_as_visuals(visual):
    with pytest.raises(RagDomainError) as caught:
        _validate(_draft(content=_content(visual=visual)))
    assert caught.value.code == "rag_invalid_draft"


def test_validate_draft_ignores_fenced_footnotes_and_html():
    content = _content(citation=('```text\nFake.[^2]\n[^2]: invented.pdf, p.9\n<img src="javascript:private">\n```'))
    bundle, lint = _validate(_draft(content=content, citations=()))
    assert bundle.edges == ()
    assert lint["citation_count"] == 0


def test_validate_draft_counts_real_evidence_outside_fenced_and_inline_code():
    content = _content(
        visual=(
            "```text\n![hidden](/wiki/assets/hidden.png)\n```\n"
            "`![also-hidden](/wiki/assets/hidden.png)`\n"
            "![real](/wiki/assets/real.png)"
        ),
        citation=("```text\nFake.[^2]\n[^2]: invented.pdf, p.9\n```\nReal.[^1]\n\n[^1]: source.pdf, p.5"),
    )
    _, lint = _validate(_draft(content=content))
    assert lint["visual_count"] == 1
    assert lint["citation_count"] == 1


def test_validate_draft_accepts_closed_exact_mermaid_with_long_matching_fence():
    content = _content(visual="````mermaid\ngraph TD\n  ```\n  A --> B\n````")
    _, lint = _validate(_draft(content=content))
    assert lint["visual_count"] == 1


def test_validate_draft_rejects_unclosed_inline_code_span_without_leaking_it():
    content = _content() + "\nunclosed `private <img src=javascript:private>\n"
    with pytest.raises(RagDomainError) as caught:
        _validate(_draft(content=content))
    assert caught.value.code == "rag_invalid_draft"
    assert "private" not in str(caught.value)


def test_validate_draft_ignores_raw_html_inside_inline_code():
    content = _content() + '\n`<img src="javascript:private">`\n'
    bundle, _ = _validate(_draft(content=content))
    assert bundle.content == content


def test_validate_draft_does_not_treat_escaped_backticks_as_inline_code():
    content = _content() + '\n\\`<img src="javascript:private">\\`\n'
    with pytest.raises(RagDomainError) as caught:
        _validate(_draft(content=content))
    assert caught.value.code == "rag_link_invalid"


def test_validate_draft_rejects_svg_image_html_and_all_raw_html_tags():
    content = _content() + '\n<svg><image href="javascript:private"></image></svg>\n'
    with pytest.raises(RagDomainError) as caught:
        _validate(_draft(content=content))
    assert caught.value.code == "rag_link_invalid"
    assert "private" not in str(caught.value)


def test_validate_draft_does_not_treat_plain_comparison_text_as_html():
    content = _content() + "\nCapacity is 3 < 5 and 9 > 4.\n"
    bundle, _ = _validate(_draft(content=content))
    assert bundle.content == content


@pytest.mark.parametrize(
    "reference",
    [
        "[asset]: javascript:private",
        "> [asset]: javascript:private",
        "![unsafe][asset]\n\n[asset]: javascript:private",
        "[unsafe][asset]\n\n[asset]: javascript:private",
        "![unsafe][]",
        "[unsafe][]",
        "![unsafe]",
        "[asset]\n\n[asset]: javascript:private",
    ],
)
def test_validate_draft_rejects_reference_style_links_images_and_definitions(reference):
    content = _content() + f"\n{reference}\n"
    with pytest.raises(RagDomainError) as caught:
        _validate(_draft(content=content))
    assert caught.value.code == "rag_link_invalid"
    assert "private" not in str(caught.value)


def test_validate_draft_does_not_count_backslash_escaped_image_syntax():
    content = _content(visual=r"\![escaped](/wiki/assets/escaped.png)")
    with pytest.raises(RagDomainError) as caught:
        _validate(_draft(content=content))
    assert caught.value.code == "rag_invalid_draft"


def test_validate_draft_does_not_validate_a_backslash_escaped_link():
    content = _content(link=r"\[escaped](/private/page.md)")
    _, lint = _validate(_draft(content=content))
    assert lint["internal_link_count"] == 0


def test_validate_draft_does_not_resolve_backslash_escaped_footnotes():
    content = _content(citation="Escaped.\\[^2]\n\n\\[^2]: invented.pdf, p.9")
    bundle, lint = _validate(_draft(content=content, citations=()))
    assert bundle.edges == ()
    assert lint["citation_count"] == 0


def test_validate_draft_preserves_even_backslash_image_semantics():
    content = _content(visual=r"\\![real](/wiki/assets/real.png)")
    _, lint = _validate(_draft(content=content))
    assert lint["visual_count"] == 1


def test_validate_draft_does_not_reject_a_backslash_escaped_raw_html_tag():
    content = _content() + "\n\\<svg\\> is shown as text.\n"
    bundle, _ = _validate(_draft(content=content))
    assert bundle.content == content


def test_validate_draft_preserves_private_use_link_text_without_escape_collision():
    private_use_filename = "\ue12e\ue12e\ue12fprivate.md"
    content = _content(link=f"[Private use](./{private_use_filename})")
    bundle, lint = _validate(_draft(content=content))
    assert bundle.content == content
    assert lint["internal_link_count"] == 1


@pytest.mark.parametrize(
    "visual",
    [
        "    ![hidden](/wiki/assets/hidden.png)",
        "\t![hidden](/wiki/assets/hidden.png)",
        "> ```text\n> ![hidden](/wiki/assets/hidden.png)\n> ```",
        "> > ~~~ text\n> > ![hidden](/wiki/assets/hidden.png)\n> > ~~~",
    ],
)
def test_validate_draft_does_not_count_indented_or_quoted_fenced_code_images(visual):
    with pytest.raises(RagDomainError) as caught:
        _validate(_draft(content=_content(visual=visual)))
    assert caught.value.code == "rag_invalid_draft"


def test_validate_draft_ignores_quoted_fenced_html_but_checks_quoted_normal_text():
    fenced = _content() + '\n> ```text\n> <svg><image href="javascript:private"></image></svg>\n> ```\n'
    bundle, _ = _validate(_draft(content=fenced))
    assert bundle.content == fenced

    active = _content(link="> [escape](/private/page.md)")
    with pytest.raises(RagDomainError) as caught:
        _validate(_draft(content=active))
    assert caught.value.code == "rag_link_invalid"


def test_validate_draft_ignores_footnote_tokens_on_blockquote_lines():
    content = _content(citation=("> Fake.[^2]\n> [^2]: invented.pdf, p.9\n\nReal.[^1]\n\n[^1]: source.pdf, p.5"))
    bundle, lint = _validate(_draft(content=content))
    assert bundle.edges == (ReferenceEdge(str(SOURCE_ID), "cites", 5),)
    assert lint["citation_count"] == 1


@pytest.mark.parametrize(
    ("visual", "link"),
    [
        ('![Nested](/wiki/assets/safe(foo).png "Title")', "[Nested](./safe(foo).md 'Title')"),
        (r"![Escaped](/wiki/assets/a_\(b\).png)", r"[Escaped](./a_\(b\).md (Title))"),
    ],
)
def test_validate_draft_accepts_balanced_nested_destinations_and_legal_titles(visual, link):
    _, lint = _validate(_draft(content=_content(visual=visual, link=link)))
    assert lint["visual_count"] == 1
    assert lint["internal_link_count"] == 1


def test_validate_draft_accepts_angle_destination_with_parentheses():
    content = _content(link="[Angle](<./a_(b).md>)")
    _, lint = _validate(_draft(content=content))
    assert lint["internal_link_count"] == 1


def test_validate_draft_rejects_traversal_after_balanced_destination_parentheses():
    content = _content(link="[Bad](safe(foo)/../../bad.md)")
    with pytest.raises(RagDomainError) as caught:
        _validate(_draft(content=content))
    assert caught.value.code == "rag_link_invalid"


@pytest.mark.parametrize(
    "link",
    [
        "[Bad](./unterminated.md",
        "![Bad](/wiki/assets/unterminated.png",
        "[Bad](./safe.md invalid-title)",
    ],
)
def test_validate_draft_rejects_malformed_suspected_inline_links_and_images(link):
    content = _content(link=link)
    with pytest.raises(RagDomainError) as caught:
        _validate(_draft(content=content))
    assert caught.value.code == "rag_link_invalid"


@pytest.mark.parametrize(
    ("content", "citations"),
    [
        (_content(citation="Claim.[^1]\n\n[^1]: invented.pdf, p.5"), (_citation(),)),
        (_content(citation="Claim.[^1]\n\n[^1]: source.pdf, p.4"), (_citation(),)),
        (_content(citation="Claim.[^2]\n\n[^1]: source.pdf, p.5"), (_citation(),)),
        (_content(citation="Claim without footnote."), (_citation(),)),
        (_content(citation="Claim.[^1]\n\n[^1]: source.pdf, p.5"), ()),
    ],
)
def test_validate_draft_rejects_unmapped_page_mismatched_or_orphan_footnotes(content, citations):
    with pytest.raises(RagDomainError) as caught:
        _validate(_draft(content=content, citations=citations))
    assert caught.value.code == "rag_citation_invalid"


def test_validate_draft_accepts_mapping_evidence_without_a_task7_dependency():
    evidence = {
        "document_id": SOURCE_ID,
        "document_version": 3,
        "chunk_index": 4,
        "page": 5,
        "filename": "source.pdf",
        "path": "/corpus/",
        "title": "Source",
        "content": "Selected source text",
        "status": "ready",
        "archived": False,
    }
    bundle, _ = _validate(_draft(), (evidence,))
    assert bundle.edges == (ReferenceEdge(str(SOURCE_ID), "cites", 5),)


def test_validate_draft_uses_identity_to_select_one_of_multiple_chunks_on_the_same_page():
    other = replace(_Evidence(), chunk_index=9)
    bundle, _ = _validate(_draft(), (_Evidence(), other))
    assert bundle.edges == (ReferenceEdge(str(SOURCE_ID), "cites", 5),)


def test_validate_draft_rejects_two_declared_chunks_for_one_ambiguous_footnote():
    other = replace(_Evidence(), chunk_index=9)
    other_citation = RagCitation(SOURCE_ID, document_version=3, chunk_index=9, page=5)
    content = _content(citation=("First.[^1] Second.[^2]\n\n[^1]: source.pdf, p.5\n[^2]: source.pdf, p.5"))
    with pytest.raises(RagDomainError) as caught:
        _validate(
            _draft(content=content, citations=(_citation(), other_citation)),
            (_Evidence(), other),
        )
    assert caught.value.code == "rag_citation_invalid"


@pytest.mark.parametrize(
    "values",
    [
        {"document_id": True},
        {"expected_version": True},
        {"expected_version": 0},
        {"expected_version": 2_147_483_647},
        {"target_path": "/wiki/launch/../private.md"},
        {"target_path": "/private/page.md"},
        {"target_path": "/wiki/launch/page.txt"},
        {"max_page_chars": True},
        {"max_page_chars": 0},
    ],
)
def test_validate_draft_rejects_invalid_explicit_bundle_inputs(values):
    with pytest.raises(RagDomainError) as caught:
        _validate(_draft(), **values)
    assert caught.value.code == "rag_invalid_draft"


def test_validate_draft_rejects_malformed_selected_evidence_without_leaking_it():
    private = replace(_Evidence(), filename="private\x00source.pdf")
    with pytest.raises(RagDomainError) as caught:
        _validate(_draft(), (private,))
    assert caught.value.code == "rag_citation_invalid"
    assert "private" not in str(caught.value)


@pytest.mark.parametrize("candidate", ["<a ", "[x"])
def test_validate_draft_rejects_large_repeated_markdown_candidates_quickly(candidate):
    def reject_repeated(repetitions: int) -> float:
        content = _content() + "\n" + candidate * repetitions
        assert len(content) <= 120_000
        started = perf_counter()
        with pytest.raises(RagDomainError) as caught:
            _validate(
                _draft(content=content),
                max_page_chars=120_000,
            )
        elapsed = perf_counter() - started
        assert caught.value.code == "rag_link_invalid"
        return elapsed

    reject_repeated(32)  # Warm caches before measuring the large public calls.
    smaller_repetitions = (50 * 1_024) // len(candidate)
    assert reject_repeated(smaller_repetitions) < 1.0
    assert reject_repeated(smaller_repetitions * 2) < 1.0
