import pytest

from llmwiki_core.models import EmbeddingProfile


def test_embedding_profile_is_immutable_and_has_stable_identity():
    profile = EmbeddingProfile(provider="openai_compatible", model="embed-v1", dimensions=3)

    assert profile.identity == ("openai_compatible", "embed-v1", 3)
    with pytest.raises((AttributeError, TypeError)):
        profile.model = "changed"  # type: ignore[misc]


@pytest.mark.parametrize(
    ("overrides", "match"),
    (
        ({"provider": ""}, "provider"),
        ({"provider": " unknown "}, "provider"),
        ({"model": ""}, "model"),
        ({"dimensions": 0}, "dimensions"),
        ({"dimensions": 4097}, "dimensions"),
        ({"dimensions": True}, "dimensions"),
    ),
)
def test_embedding_profile_rejects_invalid_identity_fields(overrides, match):
    values = {"provider": "openai_compatible", "model": "embed-v1", "dimensions": 3}
    values.update(overrides)

    with pytest.raises(ValueError, match=match):
        EmbeddingProfile(**values)


def test_embedding_profile_normalizes_identity_text():
    profile = EmbeddingProfile(provider=" openai_compatible ", model=" embed-v1 ", dimensions=4096)

    assert profile.identity == ("openai_compatible", "embed-v1", 4096)
