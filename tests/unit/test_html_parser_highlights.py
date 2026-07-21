"""Unit tests for html_parser plaintext extraction + highlight locator (V2A).

The plaintext output has to match what TipTap's `doc.textBetween` would
produce on the parsed markdown — markdown punctuation removed, block
separation preserved, inline spans flattened.
"""

import socket

import pytest
from html_parser import Parser

# ── Plaintext extraction ──────────────────────────────────


def _plain(html: str) -> str:
    return Parser(html)._to_plaintext()


def test_plain_paragraph():
    assert _plain("<p>Hello world</p>") == "Hello world"


def test_plain_heading_and_paragraph_separated_by_blank_line():
    out = _plain("<h1>Title</h1><p>Body.</p>")
    assert out == "Title\n\nBody."


def test_plain_strips_inline_formatting():
    out = _plain('<p>The <strong>key</strong> insight is <em>that</em>.</p>')
    assert out == "The key insight is that."


def test_plain_keeps_link_text_drops_href():
    out = _plain('<p>Read <a href="https://x.com">the paper</a>.</p>')
    assert out == "Read the paper."


def test_plain_list_one_item_per_line():
    out = _plain("<ul><li>First</li><li>Second</li></ul>")
    assert out == "First\nSecond"


def test_plain_ordered_list_no_numbers():
    out = _plain("<ol><li>One</li><li>Two</li></ol>")
    assert out == "One\nTwo"


def test_plain_table_cells_space_separated():
    html = "<table><tr><th>A</th><th>B</th></tr><tr><td>x</td><td>y</td></tr></table>"
    assert _plain(html) == "A B\nx y"


def test_plain_blockquote():
    out = _plain("<blockquote><p>Quoted.</p></blockquote>")
    assert out == "Quoted."


def test_plain_image_emits_empty():
    """Images are leaf nodes in ProseMirror with no rendered text content.
    Plaintext drops them entirely so the client walker matches."""
    out = _plain('<p>See <img src="x.png" alt="diagram"/> for details.</p>')
    assert out == "See for details."


def test_plain_image_no_alt_also_empty():
    out = _plain('<p>Before <img src="x.png"/> after.</p>')
    assert out == "Before after."


def test_plain_br_is_newline():
    out = _plain("<p>line one<br/>line two</p>")
    assert out == "line one\nline two"


def test_plain_collapses_multiple_blank_lines():
    out = _plain("<div></div><p>One</p><div></div><p>Two</p>")
    assert out == "One\n\nTwo"


def test_plain_drops_script_and_style():
    html = "<p>Visible</p><script>alert(1)</script><style>p{color:red}</style>"
    assert _plain(html) == "Visible"


# ── Tagged HTML sanitization ──────────────────────────────


def test_sanitized_html_strips_executable_attributes_and_urls():
    html = """
    <html><head><base href="https://evil.example/"><meta http-equiv="refresh" content="0"></head>
    <body>
      <p onclick="alert(1)" style="background:url(javascript:alert(2))">
        <a href="javascript:alert(3)" target="_blank">bad link</a>
        <a href="https://example.com/safe" target="_blank">safe link</a>
        <img src="x.png" onerror="alert(4)" srcdoc="<script>alert(5)</script>" />
      </p>
      <iframe srcdoc="<script>alert(6)</script>"></iframe>
    </body></html>
    """
    parser = Parser(html, url="https://source.example/article", content_only=True)
    parser.parse()

    out = parser.html(sanitize=True)

    assert "onclick" not in out
    assert "onerror" not in out
    assert "style=" not in out
    assert "srcdoc" not in out
    assert "javascript:" not in out
    assert "<iframe" not in out
    assert "<base" not in out
    assert "<meta" not in out
    assert 'href="https://example.com/safe"' in out
    assert 'rel="noopener noreferrer"' in out
    assert 'src="https://source.example/x.png"' in out


def test_sanitized_html_allows_only_safe_image_data_urls():
    html = """
    <p>
      <img src="data:image/png;base64,AAAA" />
      <img src="data:image/svg+xml;base64,PHN2ZyBvbmxvYWQ9YWxlcnQoMSk+" />
      <img src="data:text/html,<script>alert(1)</script>" />
    </p>
    """
    parser = Parser(html)
    parser.parse()

    out = parser.html(sanitize=True)

    assert "data:image/png;base64,AAAA" in out
    assert "data:image/svg+xml" not in out
    assert "data:text/html" not in out


def test_sanitized_html_filters_srcset_entries():
    html = """
    <p>
      <img
        src="safe.png"
        srcset="safe-1.png 1x, javascript:alert(1) 2x, data:image/webp;base64,AAAA 3x, data:text/html,evil 4x"
      />
    </p>
    """
    parser = Parser(html, url="https://source.example/post")
    parser.parse()

    out = parser.html(sanitize=True)

    assert "srcset=" not in out
    assert "javascript:" not in out
    assert "AAAA 3x" not in out
    assert "data:text/html" not in out


def test_sanitized_html_keeps_safe_srcset_without_data_urls():
    html = '<p><img src="safe.png" srcset="safe-1.png 1x, https://cdn.example/safe-2.png 2x" /></p>'
    parser = Parser(html, url="https://source.example/post")
    parser.parse()

    out = parser.html(sanitize=True)

    assert 'srcset="https://source.example/safe-1.png 1x, https://cdn.example/safe-2.png 2x"' in out


def test_parser_prefers_srcset_candidates_over_placeholder_src():
    html = """
    <img
      src="https://cdn.example/image/-1x-1.webp"
      srcset="https://cdn.example/image/220x147.webp 220w,
              https://cdn.example/image/1200x801.webp 1200w"
      alt="Hero"
    />
    """

    result = Parser(html, url="https://source.example/post").parse()

    assert result.images[0].candidate_urls[0] == "https://cdn.example/image/1200x801.webp"
    assert result.images[0].candidate_urls[-1] == "https://cdn.example/image/-1x-1.webp"
    assert result.images[0].width == 1200
    assert result.images[0].height == 801


@pytest.mark.parametrize("ip", ["127.0.0.1", "169.254.169.254", "10.0.0.5", "192.168.1.1", "::1"])
def test_resolve_public_ip_rejects_internal_addresses(monkeypatch, ip):
    from infra import safe_fetch

    def fake_getaddrinfo(host, *args, **kwargs):
        family = socket.AF_INET6 if ":" in ip else socket.AF_INET
        return [(family, socket.SOCK_STREAM, 6, "", (ip, 0))]

    monkeypatch.setattr(safe_fetch.socket, "getaddrinfo", fake_getaddrinfo)
    assert safe_fetch.resolve_public_ip("evil.example") is None


def test_resolve_public_ip_allows_public_address(monkeypatch):
    from infra import safe_fetch

    def fake_getaddrinfo(host, *args, **kwargs):
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", 0))]

    monkeypatch.setattr(safe_fetch.socket, "getaddrinfo", fake_getaddrinfo)
    assert safe_fetch.resolve_public_ip("example.com") == "93.184.216.34"


def test_resolve_public_ip_rejects_when_any_resolved_address_is_internal(monkeypatch):
    from infra import safe_fetch

    def fake_getaddrinfo(host, *args, **kwargs):
        return [
            (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", 0)),
            (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("169.254.169.254", 0)),
        ]

    monkeypatch.setattr(safe_fetch.socket, "getaddrinfo", fake_getaddrinfo)
    assert safe_fetch.resolve_public_ip("rebind.example") is None


@pytest.mark.asyncio
async def test_build_pinned_request_targets_ip_but_keeps_host_and_sni():
    from urllib.parse import urlparse

    from infra import safe_fetch

    async with safe_fetch.httpx.AsyncClient() as client:
        parsed = urlparse("https://cdn.example/hero.png")
        request = safe_fetch.build_pinned_request(client, parsed, "93.184.216.34", {"Accept": "image/*"})

    assert request.url.host == "93.184.216.34"
    assert request.headers["host"] == "cdn.example"
    assert request.extensions.get("sni_hostname") == "cdn.example"


def test_parser_instances_are_single_use():
    parser = Parser(
        "<p>x</p>",
        url="https://source.example/post",
    )
    parser.parse()

    with pytest.raises(RuntimeError):
        parser.parse()


def test_locate_single_match():
    plaintext = "The quick brown fox jumps over the lazy dog."
    a = Parser._locate_highlight(plaintext, {"textContent": "brown fox"})
    assert a is not None
    assert plaintext[a.text_start:a.text_end] == "brown fox"


def test_locate_normalizes_whitespace_in_input():
    plaintext = "The quick brown fox."
    # Caller passes unnormalized text — locator should still find it
    a = Parser._locate_highlight(plaintext, {"textContent": "  quick   brown  "})
    assert a is not None
    assert a.text_content == "quick brown"
    assert plaintext[a.text_start:a.text_end] == "quick brown"


def test_locate_no_match_returns_none():
    plaintext = "Hello world."
    a = Parser._locate_highlight(plaintext, {"textContent": "absent text"})
    assert a is None


def test_locate_disambiguates_via_prefix():
    plaintext = "The cat sat on the mat. The cat ran away."
    a = Parser._locate_highlight(
        plaintext,
        {"textContent": "The cat", "prefix": "mat. ", "suffix": " ran"},
    )
    assert a is not None
    # Should match the second occurrence (after "mat. ")
    assert a.text_start == plaintext.index("The cat", 10)


def test_locate_no_context_picks_first():
    plaintext = "The cat sat on the mat. The cat ran away."
    a = Parser._locate_highlight(plaintext, {"textContent": "The cat"})
    assert a is not None
    assert a.text_start == 0


def test_locate_empty_text_content():
    plaintext = "Hello."
    a = Parser._locate_highlight(plaintext, {"textContent": ""})
    assert a is None


def test_locate_whitespace_only_text_content():
    plaintext = "Hello."
    a = Parser._locate_highlight(plaintext, {"textContent": "   \n  "})
    assert a is None


# ── End-to-end: parse with highlights ─────────────────────


def test_parse_with_highlights_returns_mapped():
    html = (
        "<html><body>"
        "<h1>Title</h1>"
        '<p>The <strong>key</strong> insight is that <a href="https://x.com">attention</a> matters.</p>'
        "</body></html>"
    )
    p = Parser(html)
    result = p.parse(highlights=[
        {"id": "h1", "anchor": {"textContent": "key insight"}},
        {"id": "h2", "anchor": {"textContent": "attention matters"}},
    ])

    assert result.plaintext == "Title\n\nThe key insight is that attention matters."
    assert len(result.highlights) == 2
    assert result.highlights[0].text_anchor is not None
    assert result.highlights[0].text_anchor.text_content == "key insight"
    assert result.highlights[1].text_anchor is not None
    assert result.highlights[1].text_anchor.text_content == "attention matters"


def test_parse_unmapped_highlight_keeps_payload():
    html = "<html><body><p>Hello world.</p></body></html>"
    p = Parser(html)
    result = p.parse(highlights=[
        {"id": "h1", "anchor": {"textContent": "missing phrase"}},
    ])
    assert len(result.highlights) == 1
    assert result.highlights[0].text_anchor is None
    assert result.highlights[0].payload["id"] == "h1"


def test_parse_no_highlights_returns_empty():
    p = Parser("<p>x</p>")
    result = p.parse()
    assert result.highlights == []
    assert result.plaintext == "x"


def test_content_only_preserves_article_header_title():
    html = """
    <html><body>
      <header role="banner"><nav>Site nav</nav></header>
      <article>
        <header>
          <p>Supported by</p>
          <h1>Trump Administration Sees Striking Exodus of Legal Talent</h1>
          <p>By Eileen Sullivan and Andrea Fuller</p>
        </header>
        <p>President Trump's upheaval of the federal government has led to an exodus.</p>
      </article>
    </body></html>
    """
    result = Parser(html, content_only=True).parse()

    assert "Site nav" not in result.content
    assert "# Trump Administration Sees Striking Exodus of Legal Talent" in result.content
    assert "President Trump's upheaval" in result.content


def test_parse_highlight_across_inline_tags():
    """Highlight spans a <strong> boundary in the source HTML.
    In plaintext the inline tag is gone, so the highlight matches cleanly."""
    html = "<p>The <strong>quick</strong> brown fox.</p>"
    p = Parser(html)
    result = p.parse(highlights=[
        {"id": "x", "anchor": {"textContent": "quick brown"}},
    ])
    a = result.highlights[0].text_anchor
    assert a is not None
    assert result.plaintext[a.text_start:a.text_end] == "quick brown"


def test_parse_handles_image_in_highlight_range():
    """User selected text that crossed an image. Image emits empty in
    plaintext, so the surrounding text concatenates and the highlight
    locates as expected."""
    html = '<p>Look at <img src="x.png" alt="diagram"/> here.</p>'
    p = Parser(html)
    result = p.parse(highlights=[
        {"id": "x", "anchor": {"textContent": "Look at here"}},
    ])
    a = result.highlights[0].text_anchor
    assert a is not None
    # Plaintext is "Look at  here." (two spaces where image was);
    # locator finds the normalized phrase
    assert "Look at" in result.plaintext[a.text_start:a.text_end]
    assert "here" in result.plaintext[a.text_start:a.text_end]


def test_parse_cross_block_highlight():
    """Selection spans paragraphs. range.toString() gives newline-collapsed
    text but plaintext keeps `\\n\\n` between paragraphs. The locator's
    normalize-with-index-map handles this."""
    html = "<p>End of first paragraph.</p><p>Start of second paragraph.</p>"
    p = Parser(html)
    result = p.parse(highlights=[
        # Extension might capture this as collapsed-whitespace text
        {"id": "x", "anchor": {"textContent": "first paragraph. Start of"}},
    ])
    a = result.highlights[0].text_anchor
    assert a is not None
    # Maps back to original plaintext offsets, which span the paragraph break
    span = result.plaintext[a.text_start:a.text_end]
    assert "first paragraph" in span
    assert "Start of" in span
    assert "\n\n" in span  # paragraph break preserved


def test_parse_short_highlight_no_context_unlocated():
    """Single common word with multiple occurrences and no prefix/suffix
    leaves the highlight unlocated rather than guessing."""
    html = "<p>The cat sat on the mat. The dog ran on the rug.</p>"
    p = Parser(html)
    result = p.parse(highlights=[
        {"id": "x", "anchor": {"textContent": "the"}},  # ambiguous, no context
    ])
    assert result.highlights[0].text_anchor is None


def test_parse_short_highlight_with_prefix_locates():
    """Same short text but with prefix becomes locatable."""
    html = "<p>The cat sat on the mat. The dog ran on the rug.</p>"
    p = Parser(html)
    result = p.parse(highlights=[
        {"id": "x", "anchor": {
            "textContent": "the",
            "prefix": "ran on ",
        }},
    ])
    a = result.highlights[0].text_anchor
    assert a is not None
    # Should land on "the" before "rug", not the first occurrence
    assert result.plaintext[a.text_start:a.text_start + 7] == "the rug"
