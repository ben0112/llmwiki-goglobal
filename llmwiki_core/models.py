"""Backend-neutral embedding model contracts."""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol


class EmbeddingError(Exception):
    """Base class for stable embedding boundary failures."""


class EmbeddingInputError(EmbeddingError, ValueError):
    """The caller supplied an invalid or oversized embedding request."""


class InvalidEmbeddingResponse(EmbeddingError):
    """The provider returned a response that violates the embedding contract."""


class EmbeddingUnavailable(EmbeddingError):
    """The embedding provider could not complete the request."""


@dataclass(frozen=True, slots=True)
class EmbeddingProfile:
    """Public identity of an embedding space; never contains credentials."""

    provider: str
    model: str
    dimensions: int

    def __post_init__(self) -> None:
        provider = _required_text(self.provider, label="embedding provider")
        if provider != "openai_compatible":
            raise ValueError("unsupported embedding provider")
        model = _required_text(self.model, label="embedding model")
        if type(self.dimensions) is not int or not 1 <= self.dimensions <= 4096:
            raise ValueError("embedding dimensions must be between 1 and 4096")
        object.__setattr__(self, "provider", provider)
        object.__setattr__(self, "model", model)

    @property
    def identity(self) -> tuple[str, str, int]:
        return (self.provider, self.model, self.dimensions)


class EmbeddingClient(Protocol):
    """Port implemented by embedding provider adapters."""

    @property
    def profile(self) -> EmbeddingProfile: ...

    async def embed(self, texts: Sequence[str]) -> tuple[tuple[float, ...], ...]: ...


def _required_text(value: object, *, label: str) -> str:
    if not isinstance(value, str) or not (normalized := value.strip()):
        raise ValueError(f"{label} must not be empty")
    return normalized


__all__ = [
    "EmbeddingClient",
    "EmbeddingError",
    "EmbeddingInputError",
    "EmbeddingProfile",
    "EmbeddingUnavailable",
    "InvalidEmbeddingResponse",
]
