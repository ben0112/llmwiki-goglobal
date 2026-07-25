"""API compatibility surface for the shared structured telemetry contract."""

from __future__ import annotations

import os

from llmwiki_core.telemetry import emit


def replica_role(default: str) -> str:
    """Return the explicit process role when deployment supplies one."""
    return os.getenv("LLMWIKI_REPLICA_ROLE", default)


__all__ = ["emit", "replica_role"]
