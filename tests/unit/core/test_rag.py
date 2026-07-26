from uuid import UUID

import pytest

from llmwiki_core.rag import (
    RagBudget,
    RagCitation,
    RagCompletionReason,
    RagDomainError,
    RagPageState,
    RagRunConfig,
    RagStepStatus,
    RagStepType,
    RagUsage,
    RagWorkItem,
    remaining_work_items,
    validate_worklist,
)


def test_run_config_normalizes_the_approved_defaults():
    config = RagRunConfig.build(
        knowledge_base_id=UUID("00000000-0000-0000-0000-000000000001"),
        goal=" Build a launch wiki ",
        target_path_prefix="/wiki/launch",
        model_profile="primary",
    )

    assert config.target_path_prefix == "/wiki/launch/"
    assert config.retrieval_profile == "lexical"
    assert config.budget == RagBudget(
        max_pages=8,
        max_steps=96,
        max_model_tokens=64_000,
        max_context_chars=120_000,
        max_page_chars=40_000,
        per_call_timeout_seconds=60,
        max_page_attempts=2,
        max_conflict_retries=1,
    )


@pytest.mark.parametrize(
    "prefix",
    ["/", "/sources/", "/wiki/../private/", "wiki/no-leading-slash/"],
)
def test_run_config_rejects_paths_outside_wiki(prefix):
    with pytest.raises(ValueError, match="target path"):
        RagRunConfig.build(
            knowledge_base_id=UUID(int=1),
            goal="goal",
            target_path_prefix=prefix,
            model_profile="primary",
        )


def test_worklist_is_bounded_normalized_and_unique():
    items = (
        RagWorkItem.build(0, "/wiki/launch/overview.md", "Overview", "launch overview"),
        RagWorkItem.build(1, "/wiki/launch/risks.md", "Risks", "launch risks"),
    )

    validate_worklist(items, target_path_prefix="/wiki/launch/", max_pages=8)

    assert [item.ordinal for item in items] == [0, 1]
    assert remaining_work_items(items, last_committed_ordinal=0) == (items[1],)
    assert RagPageState.COMMITTED.value == "committed"
    assert RagStepType.CONFLICT.value == "conflict"
    assert RagStepStatus.FAILED.value == "failed"
    assert RagCompletionReason.BUDGET_EXHAUSTED.value == "budget_exhausted"


@pytest.mark.parametrize(
    ("field", "default", "cap"),
    [
        ("max_pages", 8, 32),
        ("max_steps", 96, 512),
        ("max_model_tokens", 64_000, 250_000),
        ("max_context_chars", 120_000, 240_000),
        ("max_page_chars", 40_000, 120_000),
        ("per_call_timeout_seconds", 60, 180),
        ("max_page_attempts", 2, 3),
        ("max_conflict_retries", 1, 3),
    ],
)
def test_budget_defaults_and_hard_caps_are_exact(field, default, cap):
    assert getattr(RagBudget(), field) == default
    assert getattr(RagBudget(**{field: cap}), field) == cap

    with pytest.raises(ValueError, match=field):
        RagBudget(**{field: cap + 1})


@pytest.mark.parametrize("value", [True, 1.0, 0, -1])
def test_budget_rejects_non_positive_or_non_integer_limits(value):
    with pytest.raises(ValueError, match="max_pages"):
        RagBudget(max_pages=value)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("goal", "x" * 4_001),
        ("model_profile", "x" * 101),
        ("retrieval_profile", "unknown"),
    ],
)
def test_run_config_rejects_invalid_text_and_profile_values(field, value):
    values = {
        "knowledge_base_id": UUID(int=1),
        "goal": "goal",
        "target_path_prefix": "/wiki/launch/",
        "model_profile": "primary",
    }
    values[field] = value

    with pytest.raises(ValueError):
        RagRunConfig.build(**values)


def test_run_config_and_work_item_accept_exact_text_hard_caps():
    config = RagRunConfig.build(
        knowledge_base_id=UUID(int=1),
        goal="g" * 4_000,
        target_path_prefix="/wiki/launch/",
        model_profile="m" * 100,
    )
    item = RagWorkItem.build(0, "/wiki/launch/page.md", "i" * 2_000, "q" * 2_000)

    assert len(config.goal) == 4_000
    assert len(config.model_profile) == 100
    assert len(item.intent) == 2_000
    assert len(item.query) == 2_000


@pytest.mark.parametrize(
    "field",
    [
        "max_pages",
        "max_steps",
        "max_model_tokens",
        "max_context_chars",
        "max_page_chars",
        "per_call_timeout_seconds",
        "max_page_attempts",
        "max_conflict_retries",
    ],
)
def test_run_config_rejects_bool_budget_overrides(field):
    values = {
        "knowledge_base_id": UUID(int=1),
        "goal": "goal",
        "target_path_prefix": "/wiki/launch/",
        "model_profile": "primary",
        field: True,
    }

    with pytest.raises(ValueError, match=field):
        RagRunConfig.build(**values)


@pytest.mark.parametrize(
    "raw_path",
    [
        "/wiki/launch/../private.md",
        "/wiki/launch/page.txt",
        "wiki/launch/page.md",
        "/wiki/launch/\x00page.md",
    ],
)
def test_work_item_rejects_unsafe_or_non_markdown_paths(raw_path):
    with pytest.raises(ValueError, match="path"):
        RagWorkItem.build(0, raw_path, "intent", "query")


def test_work_item_normalizes_unicode_path_without_character_loss():
    item = RagWorkItem.build(0, "/wiki//launch/数据.md", "intent", "query")

    assert item.path == "/wiki/launch/数据.md"


def test_work_item_rejects_paths_that_cannot_be_utf8_encoded():
    with pytest.raises(ValueError, match="UTF-8"):
        RagWorkItem.build(0, f"/wiki/launch/{chr(0xD800)}.md", "intent", "query")


@pytest.mark.parametrize("ordinal", [False, True, 0.0])
def test_work_item_rejects_non_exact_integer_ordinals(ordinal):
    with pytest.raises(ValueError, match="ordinal"):
        RagWorkItem(ordinal, "/wiki/launch/page.md", "intent", "query")


@pytest.mark.parametrize(
    ("field", "raw"),
    [
        ("path", "/wiki//launch/page.md"),
        ("intent", " intent "),
        ("query", " query "),
    ],
)
def test_work_item_direct_constructor_requires_canonical_public_fields(field, raw):
    values = {"ordinal": 0, "path": "/wiki/launch/page.md", "intent": "intent", "query": "query"}
    values[field] = raw

    with pytest.raises(ValueError, match=field):
        RagWorkItem(**values)


@pytest.mark.parametrize("unsafe", ["\x00", chr(0xD800)])
@pytest.mark.parametrize("field", ["goal", "model_profile"])
def test_run_config_rejects_nul_and_non_utf8_public_text(field, unsafe):
    values = {
        "knowledge_base_id": UUID(int=1),
        "goal": "goal",
        "target_path_prefix": "/wiki/launch/",
        "model_profile": "primary",
    }
    values[field] = f"safe{unsafe}text"

    with pytest.raises(ValueError):
        RagRunConfig.build(**values)


@pytest.mark.parametrize("unsafe", ["\x00", chr(0xD800)])
@pytest.mark.parametrize("field", ["intent", "query"])
def test_work_item_rejects_nul_and_non_utf8_public_text(field, unsafe):
    values = {"ordinal": 0, "path": "/wiki/launch/page.md", "intent": "intent", "query": "query"}
    values[field] = f"safe{unsafe}text"

    with pytest.raises(ValueError):
        RagWorkItem.build(**values)


@pytest.mark.parametrize("unsafe", ["\x00", chr(0xD800)])
def test_run_config_rejects_nul_and_non_utf8_target_path_prefix(unsafe):
    with pytest.raises(ValueError, match="target path"):
        RagRunConfig.build(
            knowledge_base_id=UUID(int=1),
            goal="goal",
            target_path_prefix=f"/wiki/launch{unsafe}/",
            model_profile="primary",
        )


@pytest.mark.parametrize("field", ["intent", "query"])
def test_work_item_rejects_text_over_hard_cap(field):
    values = {"ordinal": 0, "path": "/wiki/launch/page.md", "intent": "intent", "query": "query"}
    values[field] = "x" * 2_001

    with pytest.raises(ValueError, match=field):
        RagWorkItem.build(**values)


def test_worklist_rejects_duplicate_path_and_ordinal():
    first = RagWorkItem.build(0, "/wiki/launch/first.md", "first", "first")
    duplicate_path = RagWorkItem.build(1, "/wiki/launch/first.md", "second", "second")
    duplicate_ordinal = RagWorkItem.build(0, "/wiki/launch/second.md", "second", "second")

    with pytest.raises(ValueError, match="duplicate path"):
        validate_worklist((first, duplicate_path), target_path_prefix="/wiki/launch/", max_pages=8)
    with pytest.raises(ValueError, match="contiguous"):
        validate_worklist((first, duplicate_ordinal), target_path_prefix="/wiki/launch/", max_pages=8)


def test_worklist_rejects_a_path_that_bypasses_work_item_normalization():
    unchecked = RagWorkItem.build(0, "/wiki/launch/page.md", "intent", "query")
    object.__setattr__(unchecked, "path", "/wiki//launch/page.md")

    with pytest.raises(ValueError, match="normalized"):
        validate_worklist((unchecked,), target_path_prefix="/wiki/launch/", max_pages=8)


def test_worklist_revalidates_mutated_work_item_fields():
    unchecked = RagWorkItem.build(0, "/wiki/launch/page.md", "intent", "query")
    object.__setattr__(unchecked, "intent", " ")

    with pytest.raises(ValueError, match="intent"):
        validate_worklist((unchecked,), target_path_prefix="/wiki/launch/", max_pages=8)


def test_worklist_rejects_out_of_scope_and_too_many_items():
    item = RagWorkItem.build(0, "/wiki/other/page.md", "intent", "query")

    with pytest.raises(ValueError, match="target path"):
        validate_worklist((item,), target_path_prefix="/wiki/launch/", max_pages=8)


def test_worklist_rejects_more_items_than_a_valid_max_pages_limit():
    items = (
        RagWorkItem.build(0, "/wiki/launch/first.md", "first", "first"),
        RagWorkItem.build(1, "/wiki/launch/second.md", "second", "second"),
    )

    with pytest.raises(ValueError, match="max_pages"):
        validate_worklist(items, target_path_prefix="/wiki/launch/", max_pages=1)


def test_worklist_enforces_global_max_pages_cap_before_consuming_items():
    with pytest.raises(ValueError, match="max_pages"):
        validate_worklist((), target_path_prefix="/wiki/launch/", max_pages=33)


def test_worklist_stops_consuming_after_the_first_item_over_max_pages():
    consumed: list[int] = []

    def items():
        for ordinal in range(4):
            consumed.append(ordinal)
            if ordinal == 3:
                raise AssertionError("worklist was consumed beyond max_pages + 1")
            yield RagWorkItem.build(ordinal, f"/wiki/launch/{ordinal}.md", "intent", "query")

    with pytest.raises(ValueError, match="max_pages"):
        validate_worklist(items(), target_path_prefix="/wiki/launch/", max_pages=2)
    assert consumed == [0, 1, 2]


def test_run_config_rejects_non_string_retrieval_profiles_without_type_error():
    with pytest.raises(ValueError, match="retrieval profile"):
        RagRunConfig.build(
            knowledge_base_id=UUID(int=1),
            goal="goal",
            target_path_prefix="/wiki/launch/",
            model_profile="primary",
            retrieval_profile=[],
        )


@pytest.mark.parametrize(
    ("last_committed_ordinal", "expected_ordinals"),
    [(-1, [0, 1, 2]), (0, [1, 2]), (2, [])],
)
def test_remaining_work_items_returns_only_uncommitted_boundary_items(
    last_committed_ordinal, expected_ordinals
):
    items = tuple(
        RagWorkItem.build(ordinal, f"/wiki/launch/{ordinal}.md", "intent", "query")
        for ordinal in range(3)
    )

    assert [item.ordinal for item in remaining_work_items(items, last_committed_ordinal)] == expected_ordinals


def test_remaining_work_items_rejects_invalid_or_noncontiguous_ordinals():
    first = RagWorkItem.build(0, "/wiki/launch/first.md", "first", "first")
    second = RagWorkItem.build(1, "/wiki/launch/second.md", "second", "second")
    invalid = RagWorkItem.build(0, "/wiki/launch/invalid.md", "invalid", "invalid")
    object.__setattr__(invalid, "ordinal", False)
    gap = RagWorkItem.build(2, "/wiki/launch/gap.md", "gap", "gap")

    for items in ((invalid,), (second, first), (first, gap)):
        with pytest.raises(ValueError, match="ordinal"):
            remaining_work_items(items, last_committed_ordinal=-1)


def test_usage_reserves_and_commits_only_valid_provider_usage():
    budget = RagBudget(max_steps=1, max_model_tokens=10)
    usage = RagUsage()

    consumed = usage.consume_step(budget)
    assert consumed == RagUsage(steps=1)
    assert usage.reserve_model_call(budget, 10) == 10
    assert usage.commit_model_usage(budget, reservation=10, used_tokens=7) == RagUsage(model_tokens=7)

    with pytest.raises(RagDomainError, match="RAG budget") as caught:
        consumed.consume_step(budget)
    assert caught.value.code == "rag_budget_exhausted"
    with pytest.raises(RagDomainError, match="model response") as caught:
        usage.commit_model_usage(budget, reservation=5, used_tokens=6)
    assert caught.value.code == "rag_invalid_model_usage"


@pytest.mark.parametrize("reserved_tokens", [True, 0, -1])
def test_usage_rejects_invalid_model_reservation(reserved_tokens):
    with pytest.raises(ValueError, match="reserved model tokens"):
        RagUsage().reserve_model_call(RagBudget(), reserved_tokens)


def test_usage_rejects_reservation_and_committed_usage_over_remaining_budget():
    budget = RagBudget(max_model_tokens=10)
    usage = RagUsage(model_tokens=7)

    with pytest.raises(RagDomainError, match="RAG budget") as caught:
        usage.reserve_model_call(budget, 4)
    assert caught.value.code == "rag_budget_exhausted"

    with pytest.raises(RagDomainError, match="RAG budget") as caught:
        usage.commit_model_usage(budget, reservation=4, used_tokens=4)
    assert caught.value.code == "rag_budget_exhausted"


def test_budget_rejects_page_and_conflict_attempts_at_their_limits():
    budget = RagBudget(max_page_attempts=2, max_conflict_retries=1)

    assert budget.consume_page_attempt(1) == 2
    assert budget.consume_conflict_retry(0) == 1
    with pytest.raises(RagDomainError, match="page attempt"):
        budget.consume_page_attempt(2)
    with pytest.raises(RagDomainError, match="conflict retry"):
        budget.consume_conflict_retry(1)


def test_citation_and_domain_error_are_immutable_public_contracts():
    citation = RagCitation(UUID(int=1), document_version=1, chunk_index=0, page=2)
    error = RagDomainError("rag_invalid_draft", "The generated draft was invalid.")

    assert citation.document_id == UUID(int=1)
    assert error.code == "rag_invalid_draft"
    assert error.public_message == "The generated draft was invalid."
    assert error.retryable is False
    with pytest.raises(AttributeError):
        citation.page = 3
