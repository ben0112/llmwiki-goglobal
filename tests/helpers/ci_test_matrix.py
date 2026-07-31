"""Deterministic, exhaustive test-file partitions for isolated CI processes."""

from __future__ import annotations

import argparse
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path

ROOT = Path(__file__).parents[2]
UNIT = ROOT / "tests/unit"
INTEGRATION = ROOT / "tests/integration"


def _relative(paths) -> tuple[str, ...]:
    return tuple(sorted(path.relative_to(ROOT).as_posix() for path in paths))


unit_files = set(UNIT.rglob("test_*.py"))
unit_core = set((UNIT / "core").rglob("test_*.py"))
unit_corpus = set((UNIT / "corpus").rglob("test_*.py"))
unit_mcp = set((UNIT / "mcp").rglob("test_*.py"))

integration_files = set(INTEGRATION.rglob("test_*.py"))
integration_mcp = set((INTEGRATION / "mcp").rglob("test_*.py"))
integration_mcp_postgres = {
    INTEGRATION / "mcp/test_mcp_isolation.py",
    INTEGRATION / "mcp/test_wiki_write_postgres.py",
}
integration_redis = {INTEGRATION / "test_tus_sessions_redis.py"}
integration_minio = {INTEGRATION / "test_s3_multipart.py"}
integration_scaled = {INTEGRATION / "test_scaled_compose.py"}
integration_rag = set(INTEGRATION.glob("test_rag_*.py")) | {INTEGRATION / "isolation/test_rag_api_isolation.py"}
integration_retrieval = {
    INTEGRATION / "test_chunk_embeddings_schema.py",
    INTEGRATION / "test_durable_embeddings.py",
    INTEGRATION / "test_hybrid_failure_matrix.py",
    INTEGRATION / "test_retrieval_evaluation.py",
    INTEGRATION / "test_vector_store.py",
}

SEGMENTS: dict[str, tuple[str, ...]] = {
    "unit-core": _relative(unit_core),
    "unit-api": _relative(unit_files - unit_core - unit_corpus - unit_mcp),
    "unit-mcp": _relative(unit_mcp),
    "unit-corpus": _relative(unit_corpus),
    "integration-api": _relative(
        integration_files
        - integration_mcp
        - integration_redis
        - integration_minio
        - integration_scaled
        - integration_rag
        - integration_retrieval
    ),
    "integration-rag": _relative(integration_rag),
    "integration-mcp": _relative(integration_mcp - integration_mcp_postgres),
    "integration-mcp-postgres": _relative(integration_mcp_postgres),
    "integration-retrieval": _relative(integration_retrieval),
    "integration-redis": _relative(integration_redis),
    "integration-minio": _relative(integration_minio),
    "integration-scaled": _relative(integration_scaled),
}

ISOLATED_SEGMENTS = frozenset({"integration-api", "integration-mcp-postgres", "integration-rag"})


def _segment_files(segment: str) -> tuple[str, ...]:
    """Generate a segment at execution time so failures cannot be hidden by a pipe."""
    return SEGMENTS[segment]


def _run(segment: str, command: Sequence[str]) -> int:
    try:
        files = tuple(_segment_files(segment))
    except Exception as exc:  # noqa: BLE001 - CLI boundary must fail closed.
        print(f"failed to generate CI test segment {segment}: {type(exc).__name__}", file=sys.stderr)
        return 2
    if not files:
        print(f"CI test segment {segment} is empty", file=sys.stderr)
        return 2
    if not command:
        print("CI test command must not be empty", file=sys.stderr)
        return 2

    commands = ((*command, test_file) for test_file in files) if segment in ISOLATED_SEGMENTS else ((*command, *files),)
    for child_command in commands:
        completed = subprocess.run(list(child_command), check=False)
        if completed.returncode != 0:
            return completed.returncode
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser()
    parser.add_argument("action_or_segment", choices=["run", *sorted(SEGMENTS)])
    parser.add_argument("arguments", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    if args.action_or_segment != "run":
        if args.arguments:
            parser.error("listing a segment does not accept a command")
        print("\n".join(_segment_files(args.action_or_segment)))
        return 0

    if not args.arguments:
        parser.error("run requires a segment and command")
    segment, *command = args.arguments
    if segment not in SEGMENTS:
        parser.error(f"unknown segment: {segment}")
    if command[:1] == ["--"]:
        command = command[1:]
    return _run(segment, command)


if __name__ == "__main__":
    raise SystemExit(main())
