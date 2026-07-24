import pytest

from llmwiki_core.documents import (
    DocumentIdentity,
    DocumentKind,
    DocumentStatus,
    InvalidStatusTransition,
    assert_status_transition,
    join_logical_path,
    normalize_directory_path,
)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [("", "/"), ("wiki", "/wiki/"), ("/wiki", "/wiki/"), ("/wiki/a/", "/wiki/a/")],
)
def test_normalize_directory_path(raw, expected):
    assert normalize_directory_path(raw) == expected


@pytest.mark.parametrize("raw", ["../x", "/wiki/../x", "a\x00b"])
def test_normalize_directory_path_rejects_unsafe_paths(raw):
    with pytest.raises(ValueError):
        normalize_directory_path(raw)


def test_join_logical_path_uses_one_canonical_form():
    assert join_logical_path("wiki/concepts", "risk.md") == "/wiki/concepts/risk.md"


@pytest.mark.parametrize(
    ("old", "new"),
    [
        (DocumentStatus.PENDING, DocumentStatus.PROCESSING),
        (DocumentStatus.PROCESSING, DocumentStatus.READY),
        (DocumentStatus.PENDING, DocumentStatus.FAILED),
        (DocumentStatus.PROCESSING, DocumentStatus.FAILED),
        (DocumentStatus.FAILED, DocumentStatus.PENDING),
    ],
)
def test_allowed_status_transitions(old, new):
    assert_status_transition(old, new)


def test_ready_cannot_skip_processing():
    with pytest.raises(InvalidStatusTransition):
        assert_status_transition(DocumentStatus.PENDING, DocumentStatus.READY)


def test_ready_can_only_return_to_pending_for_system_repair():
    with pytest.raises(InvalidStatusTransition):
        assert_status_transition(DocumentStatus.READY, DocumentStatus.PENDING)

    assert_status_transition(
        DocumentStatus.READY,
        DocumentStatus.PENDING,
        for_repair=True,
    )


def test_document_kinds_are_stable_wire_values():
    assert [kind.value for kind in DocumentKind] == ["source", "wiki", "asset"]


def test_document_identity_is_immutable_and_tenant_scoped():
    identity = DocumentIdentity(document_id="doc", knowledge_base_id="kb", user_id="user")
    assert identity.scope == ("user", "kb", "doc")
    with pytest.raises(AttributeError):
        identity.document_id = "other"
