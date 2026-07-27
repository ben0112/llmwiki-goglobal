"""REST-only CLI for hosted server-side RAG runs."""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Mapping, Sequence
from contextlib import suppress
from dataclasses import dataclass
from math import isfinite
from typing import NoReturn
from urllib.parse import urlencode, urlsplit, urlunsplit
from uuid import UUID

import httpx

from llmwiki_core.rag import (
    MAX_CONFLICT_RETRIES,
    MAX_CONTEXT_CHARS,
    MAX_GOAL_CHARS,
    MAX_MODEL_TOKENS,
    MAX_PAGE_ATTEMPTS,
    MAX_PAGE_CHARS,
    MAX_PAGES,
    MAX_PER_CALL_TIMEOUT_SECONDS,
    MAX_PROFILE_CHARS,
    MAX_STEPS,
    RAG_ERROR_CONTRACTS,
)

MAX_RESPONSE_BYTES = 1_048_576
HTTP_TIMEOUT_SECONDS = 30.0
_MAX_API_URL_CHARS = 2_048
_MAX_TOKEN_CHARS = 8_192
_MAX_IDEMPOTENCY_KEY_CHARS = 200
_MAX_TARGET_PREFIX_CHARS = 2_000
_MAX_JSON_DEPTH = 16
_PUBLIC_ERROR_CODES = frozenset(contract.code for contract in RAG_ERROR_CONTRACTS.values())
_BUDGET_FLAGS = (
    ("max_pages", MAX_PAGES),
    ("max_steps", MAX_STEPS),
    ("max_model_tokens", MAX_MODEL_TOKENS),
    ("max_context_chars", MAX_CONTEXT_CHARS),
    ("max_page_chars", MAX_PAGE_CHARS),
    ("per_call_timeout_seconds", MAX_PER_CALL_TIMEOUT_SECONDS),
    ("max_page_attempts", MAX_PAGE_ATTEMPTS),
    ("max_conflict_retries", MAX_CONFLICT_RETRIES),
)


@dataclass(frozen=True, slots=True, repr=False)
class Request:
    method: str
    api_url: str
    path: str
    headers: Mapping[str, str]
    body: Mapping[str, object] | None


@dataclass(frozen=True, slots=True, repr=False)
class ApiResponse:
    status_code: int
    payload: Mapping[str, object]


class _ParserExit(Exception):
    def __init__(self, status: int) -> None:
        self.status = status
        super().__init__()


class _SafeArgumentParser(argparse.ArgumentParser):
    def exit(self, status: int = 0, message: str | None = None) -> NoReturn:
        if status == 0 and message:
            self._print_message(message)
        raise _ParserExit(status)

    def error(self, message: str) -> NoReturn:
        del message
        raise _ParserExit(2)


def _emit_error(category: str, code: str) -> None:
    with suppress(Exception):
        sys.stderr.write(json.dumps({"category": category, "code": code}, sort_keys=True, separators=(",", ":")) + "\n")


def _parser() -> argparse.ArgumentParser:
    parser = _SafeArgumentParser(prog="python -m scripts.rag")
    commands = parser.add_subparsers(dest="command", required=True)

    build = commands.add_parser("build-wiki")
    build.add_argument("--knowledge-base", required=True)
    build.add_argument("--goal", required=True)
    build.add_argument("--target-prefix", required=True)
    build.add_argument("--model-profile", required=True)
    build.add_argument("--retrieval-profile", default="lexical")
    build.add_argument("--dry-run", action="store_true")
    build.add_argument("--idempotency-key", required=True)
    _add_budget_arguments(build)
    build.add_argument("--json", action="store_true")

    status_parser = commands.add_parser("status")
    status_parser.add_argument("run_id")
    status_parser.add_argument("--json", action="store_true")

    steps = commands.add_parser("steps")
    steps.add_argument("run_id")
    steps.add_argument("--after", default="0")
    steps.add_argument("--limit", default="50")
    steps.add_argument("--json", action="store_true")

    resume = commands.add_parser("resume")
    resume.add_argument("run_id")
    resume.add_argument("--idempotency-key", required=True)
    _add_budget_arguments(resume)
    resume.add_argument("--json", action="store_true")
    return parser


def _add_budget_arguments(parser: argparse.ArgumentParser) -> None:
    for field_name, _maximum in _BUDGET_FLAGS:
        parser.add_argument("--" + field_name.replace("_", "-"))


def _bounded_text(value: object, *, minimum: int, maximum: int, normalized: bool = False) -> str:
    if type(value) is not str or not minimum <= len(value) <= maximum or "\x00" in value:
        raise ValueError
    try:
        encoded = value.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise ValueError from exc
    if not encoded or (normalized and value.strip() != value):
        raise ValueError
    return value


def _canonical_uuid(value: object) -> str:
    raw = _bounded_text(value, minimum=36, maximum=36)
    try:
        parsed = UUID(raw)
    except ValueError as exc:
        raise ValueError from exc
    if str(parsed) != raw:
        raise ValueError
    return raw


def _bounded_integer(value: object, *, minimum: int, maximum: int) -> int:
    if type(value) is not str or not 1 <= len(value) <= 10 or not value.isascii() or not value.isdecimal():
        raise ValueError
    parsed = int(value)
    if not minimum <= parsed <= maximum:
        raise ValueError
    return parsed


def _idempotency_key(value: object) -> str:
    key = _bounded_text(
        value,
        minimum=1,
        maximum=_MAX_IDEMPOTENCY_KEY_CHARS,
        normalized=True,
    )
    if (
        not key.isascii()
        or len(key.encode("utf-8")) > 800
        or any(ord(character) < 32 or ord(character) == 127 for character in key)
    ):
        raise ValueError
    return key


def _configuration() -> tuple[str, str]:
    raw_url = _bounded_text(
        os.environ.get("LLMWIKI_API_URL"),
        minimum=1,
        maximum=_MAX_API_URL_CHARS,
        normalized=True,
    )
    if "\\" in raw_url or any(character.isspace() or ord(character) < 32 for character in raw_url):
        raise ValueError
    token = _bounded_text(
        os.environ.get("LLMWIKI_ACCESS_TOKEN"),
        minimum=1,
        maximum=_MAX_TOKEN_CHARS,
        normalized=True,
    )
    if not token.isascii() or any(
        character.isspace() or ord(character) < 32 or ord(character) == 127 for character in token
    ):
        raise ValueError
    parsed = urlsplit(raw_url)
    _ = parsed.port  # Access validates the numeric range before any network call.
    if (
        parsed.scheme not in {"http", "https"}
        or not parsed.netloc
        or not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
        or parsed.path not in {"", "/"}
    ):
        raise ValueError
    api_url = urlunsplit((parsed.scheme, parsed.netloc, "", "", ""))
    return api_url, token


def _budget(args: argparse.Namespace) -> dict[str, int]:
    result: dict[str, int] = {}
    for field_name, maximum in _BUDGET_FLAGS:
        raw = getattr(args, field_name)
        if raw is not None:
            result[field_name] = _bounded_integer(raw, minimum=1, maximum=maximum)
    return result


def _headers(token: str, *, idempotency_key: str | None = None, body: bool = False) -> dict[str, str]:
    headers = {"Accept": "application/json", "Authorization": f"Bearer {token}"}
    if body:
        headers["Content-Type"] = "application/json"
    if idempotency_key is not None:
        headers["Idempotency-Key"] = idempotency_key
    return headers


def _build_request(args: argparse.Namespace, api_url: str, token: str) -> Request:
    if args.command == "build-wiki":
        knowledge_base_id = _canonical_uuid(args.knowledge_base)
        goal = _bounded_text(args.goal, minimum=1, maximum=MAX_GOAL_CHARS)
        target_prefix = _bounded_text(args.target_prefix, minimum=1, maximum=_MAX_TARGET_PREFIX_CHARS)
        model_profile = _bounded_text(args.model_profile, minimum=1, maximum=MAX_PROFILE_CHARS, normalized=True)
        retrieval_profile = _bounded_text(
            args.retrieval_profile,
            minimum=1,
            maximum=MAX_PROFILE_CHARS,
            normalized=True,
        )
        key = _idempotency_key(args.idempotency_key)
        return Request(
            method="POST",
            api_url=api_url,
            path="/v1/rag/build-wiki",
            headers=_headers(token, idempotency_key=key, body=True),
            body={
                "knowledge_base_id": knowledge_base_id,
                "goal": goal,
                "target_path_prefix": target_prefix,
                "model_profile": model_profile,
                "retrieval_profile": retrieval_profile,
                "dry_run": args.dry_run,
                "budget": _budget(args),
            },
        )
    run_id = _canonical_uuid(args.run_id)
    if args.command == "status":
        return Request("GET", api_url, f"/v1/rag/runs/{run_id}", _headers(token), None)
    if args.command == "steps":
        after = _bounded_integer(args.after, minimum=0, maximum=MAX_STEPS)
        limit = _bounded_integer(args.limit, minimum=1, maximum=100)
        query = urlencode({"after": after, "limit": limit})
        return Request("GET", api_url, f"/v1/rag/runs/{run_id}/steps?{query}", _headers(token), None)
    if args.command == "resume":
        key = _idempotency_key(args.idempotency_key)
        return Request(
            "POST",
            api_url,
            f"/v1/rag/runs/{run_id}/resume",
            _headers(token, idempotency_key=key, body=True),
            {"budget": _budget(args)},
        )
    raise ValueError


def _reject_constant(_value: str) -> NoReturn:
    raise ValueError


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError
        result[key] = value
    return result


def _check_depth(value: object) -> None:
    stack = [(value, 1)]
    while stack:
        current, depth = stack.pop()
        if depth > _MAX_JSON_DEPTH:
            raise ValueError
        if type(current) is dict:
            stack.extend((item, depth + 1) for item in current.values())
        elif type(current) is list:
            stack.extend((item, depth + 1) for item in current)
        elif type(current) is float and not isfinite(current):
            raise ValueError


def _decode_json(raw: bytes) -> dict[str, object]:
    if type(raw) is not bytes or len(raw) > MAX_RESPONSE_BYTES:
        raise ValueError
    try:
        decoded = raw.decode("utf-8")
        value = json.loads(
            decoded,
            parse_constant=_reject_constant,
            object_pairs_hook=_unique_object,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, RecursionError) as exc:
        raise ValueError from exc
    if type(value) is not dict:
        raise ValueError
    _check_depth(value)
    return value


def _send(request: Request) -> ApiResponse:
    if type(request) is not Request:
        raise TypeError
    content = None
    if request.body is not None:
        content = json.dumps(
            dict(request.body),
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
        ).encode("utf-8")
    with (
        httpx.Client(follow_redirects=False, trust_env=False, timeout=HTTP_TIMEOUT_SECONDS) as client,
        client.stream(
            request.method,
            request.api_url + request.path,
            headers=dict(request.headers),
            content=content,
        ) as response,
    ):
        chunks = bytearray()
        for chunk in response.iter_bytes():
            if len(chunks) + len(chunk) > MAX_RESPONSE_BYTES:
                raise ValueError
            chunks.extend(chunk)
        payload = _decode_json(bytes(chunks))
        status_code = response.status_code
    if type(status_code) is not int or not 100 <= status_code <= 599:
        raise ValueError
    return ApiResponse(status_code=status_code, payload=payload)


def _api_error_code(response: ApiResponse) -> str:
    try:
        detail = response.payload.get("detail")
        code = detail.get("code") if type(detail) is dict else None
    except BaseException:  # noqa: BLE001 - response mappings are untrusted.
        return "rag_api_error"
    return code if type(code) is str and code in _PUBLIC_ERROR_CODES else "rag_api_error"


def _write_output(payload: Mapping[str, object], as_json: bool) -> None:
    if type(payload) is not dict:
        raise ValueError
    if as_json:
        output = json.dumps(payload, sort_keys=True, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
    else:
        lines = []
        for key in ("state", "run_id", "job_id", "id", "completion_reason", "next_cursor"):
            value = payload.get(key)
            if value is None or type(value) not in {str, int, bool}:
                continue
            lines.append(f"{key}={value}")
        items = payload.get("items")
        if type(items) is list:
            lines.append(f"items={len(items)}")
        output = "\n".join(lines) if lines else "ok"
    sys.stdout.write(output + "\n")


def main(argv: Sequence[str] | None = None) -> int:
    try:
        args = _parser().parse_args(argv)
    except _ParserExit as failure:
        if failure.status:
            _emit_error("arguments", "invalid_arguments")
        return failure.status
    try:
        api_url, token = _configuration()
    except (AttributeError, TypeError, ValueError):
        _emit_error("configuration", "invalid_configuration")
        return 2
    try:
        request = _build_request(args, api_url, token)
    except (AttributeError, TypeError, ValueError):
        _emit_error("arguments", "invalid_arguments")
        return 2
    try:
        response = _send(request)
    except Exception:  # noqa: BLE001 - transport and response details are private.
        _emit_error("transport", "request_failed")
        return 3
    if not 200 <= response.status_code < 300:
        _emit_error("api", _api_error_code(response))
        return 3
    try:
        _write_output(response.payload, args.json)
    except Exception:  # noqa: BLE001 - never expose output errors or linked context.
        _emit_error("output", "output_failed")
        return 4
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = ["ApiResponse", "MAX_RESPONSE_BYTES", "Request", "main"]
