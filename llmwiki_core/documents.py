"""Document identity, lifecycle, and logical-path contracts."""

from dataclasses import dataclass
from enum import StrEnum
from pathlib import PurePosixPath


class DocumentKind(StrEnum):
    SOURCE = "source"
    WIKI = "wiki"
    ASSET = "asset"


class DocumentStatus(StrEnum):
    PENDING = "pending"
    PROCESSING = "processing"
    READY = "ready"
    FAILED = "failed"


class InvalidStatusTransition(ValueError):
    """Raised when a document lifecycle transition violates the contract."""


@dataclass(frozen=True)
class DocumentIdentity:
    document_id: str
    knowledge_base_id: str
    user_id: str

    @property
    def scope(self) -> tuple[str, str, str]:
        return (self.user_id, self.knowledge_base_id, self.document_id)


_ALLOWED = {
    DocumentStatus.PENDING: {DocumentStatus.PROCESSING, DocumentStatus.FAILED},
    DocumentStatus.PROCESSING: {DocumentStatus.READY, DocumentStatus.FAILED},
    DocumentStatus.READY: set(),
    DocumentStatus.FAILED: {DocumentStatus.PENDING},
}


def assert_status_transition(
    old: DocumentStatus,
    new: DocumentStatus,
    *,
    for_repair: bool = False,
) -> None:
    """Validate a lifecycle transition.

    A ready document may only be invalidated back to pending by a system repair
    path after detecting missing or stale derived data.
    """
    if for_repair and old is DocumentStatus.READY and new is DocumentStatus.PENDING:
        return
    if new not in _ALLOWED[old]:
        raise InvalidStatusTransition(f"{old.value} -> {new.value}")


def normalize_directory_path(raw: str) -> str:
    """Return a canonical absolute POSIX directory path with a trailing slash."""
    if "\x00" in raw:
        raise ValueError("path contains NUL")
    parts = [part for part in raw.replace("\\", "/").split("/") if part and part != "."]
    if ".." in parts:
        raise ValueError("path traversal is not allowed")
    return "/" + "/".join(parts) + ("/" if parts else "")


def join_logical_path(directory: str, filename: str) -> str:
    """Join a logical directory and a single safe filename."""
    if not filename or filename in {".", ".."} or PurePosixPath(filename).name != filename:
        raise ValueError("filename must be a basename")
    return normalize_directory_path(directory) + filename
