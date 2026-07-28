import base64
import json
from dataclasses import replace

import pytest

from llmwiki_core.read_cursor import CursorError, ReadCursor, decode_cursor, encode_cursor


def test_cursor_round_trip_preserves_bound_scope_and_key():
    cursor = ReadCursor(
        scope="documents.browse",
        revision=9,
        sort="name",
        direction="asc",
        key=("report.pdf", "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"),
    )

    assert decode_cursor(encode_cursor(cursor), expected_scope="documents.browse") == cursor


@pytest.mark.parametrize("encoded", ["", "%%%", "e30", "W10"])
def test_cursor_rejects_malformed_payload(encoded):
    with pytest.raises(CursorError):
        decode_cursor(encoded, expected_scope="documents.browse")


def test_cursor_rejects_cross_endpoint_reuse():
    encoded = encode_cursor(ReadCursor("wiki.pages", 1, "path", "asc", ("a", "id")))

    with pytest.raises(CursorError, match="scope"):
        decode_cursor(encoded, expected_scope="documents.browse")


def test_cursor_rejects_unsupported_version():
    current = ReadCursor("documents.browse", 1, "name", "asc", ("a", "id"))

    with pytest.raises(CursorError, match="version"):
        encode_cursor(replace(current, version=2))


def test_cursor_encoding_is_canonical_compact_json_without_padding():
    cursor = ReadCursor("documents.browse", 9, "name", "asc", ("report.pdf", "id"))

    encoded = encode_cursor(cursor)
    decoded = base64.urlsafe_b64decode(encoded + "=" * (-len(encoded) % 4)).decode()

    assert encoded.endswith("=") is False
    assert decoded == '{"d":"asc","k":["report.pdf","id"],"o":"name","r":9,"s":"documents.browse","v":1}'


def _encoded_payload(payload: object) -> str:
    raw = json.dumps(payload, separators=(",", ":")).encode()
    return base64.urlsafe_b64encode(raw).decode().rstrip("=")


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        ({"v": 1, "s": "documents.browse", "r": 1, "o": "name", "d": "asc", "k": ["a"], "x": 1}, "payload"),
        ({"v": 2, "s": "documents.browse", "r": 1, "o": "name", "d": "asc", "k": ["a"]}, "version"),
        ({"v": 1, "s": "documents.browse", "r": True, "o": "name", "d": "asc", "k": ["a"]}, "revision"),
        ({"v": 1, "s": "documents.browse", "r": 0, "o": "name", "d": "asc", "k": ["a"]}, "revision"),
        ({"v": 1, "s": "documents.browse", "r": 1, "o": "name", "d": "sideways", "k": ["a"]}, "direction"),
        ({"v": 1, "s": "documents.browse", "r": 1, "o": "name", "d": [], "k": ["a"]}, "direction"),
        ({"v": 1, "s": "", "r": 1, "o": "name", "d": "asc", "k": ["a"]}, "scope"),
        ({"v": 1, "s": "documents.browse", "r": 1, "o": "", "d": "asc", "k": ["a"]}, "sort"),
        ({"v": 1, "s": "documents.browse", "r": 1, "o": "name", "d": "asc", "k": []}, "key"),
        ({"v": 1, "s": "documents.browse", "r": 1, "o": "name", "d": "asc", "k": [""]}, "key"),
    ],
)
def test_cursor_decode_strictly_validates_payload(payload, message):
    with pytest.raises(CursorError, match=message):
        decode_cursor(_encoded_payload(payload), expected_scope="documents.browse")


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"revision": True}, "revision"),
        ({"revision": 0}, "revision"),
        ({"direction": "sideways"}, "direction"),
        ({"scope": ""}, "scope"),
        ({"sort": ""}, "sort"),
        ({"key": ()}, "key"),
        ({"key": ("",)}, "key"),
    ],
)
def test_cursor_encode_validates_the_same_contract(changes, message):
    cursor = ReadCursor("documents.browse", 1, "name", "asc", ("a", "id"))

    with pytest.raises(CursorError, match=message):
        encode_cursor(replace(cursor, **changes))


def test_cursor_rejects_oversized_encoded_input():
    with pytest.raises(CursorError, match="length"):
        decode_cursor("a" * 2049, expected_scope="documents.browse")


def test_read_model_sorts_are_closed():
    from api.services.read_models import ReadSort, SortDirection

    assert ReadSort("name") is ReadSort.NAME
    assert SortDirection("asc") is SortDirection.ASC
    with pytest.raises(ValueError):
        ReadSort("arbitrary-sql")
    with pytest.raises(ValueError):
        SortDirection("sideways")


def test_read_model_item_lists_are_bounded_to_200():
    from api.services.read_models import (
        BrowsePage,
        DocumentStatusPage,
        GraphSummary,
        ReadPage,
        UploadPreflightRequest,
        UploadPreflightResponse,
    )

    bounded_fields = [
        (ReadPage, "items"),
        (BrowsePage, "items"),
        (BrowsePage, "folders"),
        (GraphSummary, "cited_document_ids"),
        (DocumentStatusPage, "items"),
        (UploadPreflightRequest, "items"),
        (UploadPreflightResponse, "items"),
    ]
    for model, field_name in bounded_fields:
        metadata = model.model_fields[field_name].metadata
        assert any(getattr(constraint, "max_length", None) == 200 for constraint in metadata)


def test_base_read_page_accepts_endpoint_specific_lightweight_items():
    from api.services.read_models import ReadPage

    page = ReadPage(
        revision=3,
        items=[{"doc": {"id": "d1"}, "meta": {"stage": "S1"}}],
        total_count=1,
    )

    assert page.items[0]["meta"]["stage"] == "S1"


def test_etag_helpers_match_complete_http_list_tokens_only():
    from api.services.read_models import etag_for_revision, etag_matches

    assert etag_for_revision(9) == '"kb-read-9"'
    assert etag_matches(None, 9) is False
    assert etag_matches('"other", "kb-read-9"', 9) is True
    assert etag_matches('W/"kb-read-9"', 9) is True
    assert etag_matches("*", 9) is True
    assert etag_matches('"kb-read-90"', 9) is False
    assert etag_matches('prefix-"kb-read-9"-suffix', 9) is False
