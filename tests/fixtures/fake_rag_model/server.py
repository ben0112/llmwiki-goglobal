"""Deterministic, dependency-free OpenAI-compatible server for RAG tests."""

from __future__ import annotations

import json
import os
import re
import threading
import time
from contextlib import contextmanager
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

MAX_REQUEST_BYTES = 1024 * 1024
TIMEOUT_DELAY_SECONDS = 2.0
_CHAT_PATH = "/v1/chat/completions"
_UUID_RE = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}")
_TARGET_RE = re.compile(r'"(/wiki/(?:[^"/]+/)*)"')

PLAN_USAGE = {"prompt_tokens": 11, "completion_tokens": 7, "total_tokens": 18}
DRAFT_USAGE = {"prompt_tokens": 13, "completion_tokens": 11, "total_tokens": 24}
DRAFT_REQUESTED = threading.Event()
RELEASE_DRAFT = threading.Event()
REQUEST_KINDS: list[str] = []


def reset_controls() -> None:
    DRAFT_REQUESTED.clear()
    RELEASE_DRAFT.clear()
    REQUEST_KINDS.clear()


def _messages(request: dict[str, Any]) -> tuple[dict[str, str], ...]:
    raw = request.get("messages")
    if type(raw) is not list or not raw:
        raise ValueError
    messages = []
    for item in raw:
        if type(item) is not dict or set(item) != {"role", "content"}:
            raise ValueError
        if type(item["role"]) is not str or type(item["content"]) is not str:
            raise ValueError
        messages.append({"role": item["role"], "content": item["content"]})
    return tuple(messages)


def _block(messages: tuple[dict[str, str], ...], label: str) -> Any:
    opening = f"<{label} "
    closing = f"</{label}>"
    for message in messages:
        content = message["content"]
        start = content.find(opening)
        if start < 0:
            continue
        start = content.find(">\n", start)
        end = content.find(f"\n{closing}", start + 2)
        if start >= 0 and end >= 0:
            return json.loads(content[start + 2 : end])
    return None


def _plan(messages: tuple[dict[str, str], ...]) -> dict[str, Any]:
    joined = "\n".join(message["content"] for message in messages)
    target = _block(messages, "ALLOWED_TARGET_PATH_PREFIX")
    if type(target) is not str:
        match = _TARGET_RE.search(joined)
        target = match.group(1) if match else "/wiki/e2e/"
    if "[FAKE_MODEL_NO_WORK]" in joined:
        pages = []
    else:
        pages = [
            {
                "path": f"{target}overview.md",
                "intent": "Summarize the authoritative launch evidence",
                "query": "authoritative launch evidence",
            }
        ]
        if "[FAKE_MODEL_TWO_PAGES]" in joined:
            pages.append(
                {
                    "path": f"{target}details.md",
                    "intent": "Explain the authoritative launch details",
                    "query": "authoritative launch evidence",
                }
            )
    return {"pages": pages}


def _draft(messages: tuple[dict[str, str], ...]) -> dict[str, Any]:
    evidence = _block(messages, "UNTRUSTED_EVIDENCE")
    item = _block(messages, "WORK_ITEM")
    if type(evidence) is not list or not evidence or type(evidence[0]) is not dict:
        raise ValueError
    source = evidence[0]
    source_id = source.get("document_id")
    filename = source.get("filename")
    version = source.get("document_version")
    chunk_index = source.get("chunk_index")
    page = source.get("page")
    if (
        type(source_id) is not str
        or _UUID_RE.fullmatch(source_id) is None
        or type(filename) is not str
        or type(version) is not int
        or type(chunk_index) is not int
        or (page is not None and type(page) is not int)
    ):
        raise ValueError
    path = item.get("path") if type(item) is dict else "/wiki/e2e/overview.md"
    title = "Launch details" if type(path) is str and path.endswith("details.md") else "Launch overview"
    page_suffix = f", p.{page}" if page is not None else ""
    content = (
        "---\n"
        f"title: {title}\n"
        "tags: [launch, evidence]\n"
        "description: Deterministic cited launch guidance.\n"
        "date: 2026-07-27\n"
        "---\n"
        f"# {title}\n\n"
        "```mermaid\nflowchart TD\n  Evidence --> Guidance\n```\n\n"
        "The launch guidance is supported by immutable evidence.[^1]\n\n"
        f"[^1]: {filename}{page_suffix}\n"
    )
    citation = {
        "document_id": source_id,
        "document_version": version,
        "chunk_index": chunk_index,
    }
    if page is not None:
        citation["page"] = page
    return {"content": content, "citations": [citation]}


def _completion(request: dict[str, Any]) -> dict[str, Any]:
    messages = _messages(request)
    joined = "\n".join(message["content"] for message in messages)
    if "[FAKE_MODEL_TIMEOUT]" in joined:
        time.sleep(TIMEOUT_DELAY_SECONDS)
    planner = "bounded wiki worklist planner" in messages[0]["content"]
    REQUEST_KINDS.append("plan" if planner else "draft")
    if not planner and "[FAKE_MODEL_BLOCK_DRAFT]" in joined and not DRAFT_REQUESTED.is_set():
        DRAFT_REQUESTED.set()
        RELEASE_DRAFT.wait(timeout=5)
    elif not planner and "[FAKE_MODEL_BLOCK_DRAFT]" in joined and not RELEASE_DRAFT.is_set():
        time.sleep(1)
    payload = _plan(messages) if planner else _draft(messages)
    usage = PLAN_USAGE if planner else DRAFT_USAGE
    return {
        "choices": [{"message": {"content": json.dumps(payload, sort_keys=True, separators=(",", ":"))}}],
        "usage": usage,
    }


class FakeRagModelHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def do_POST(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API.
        if self.path != _CHAT_PATH:
            self.close_connection = True
            self._respond(HTTPStatus.NOT_FOUND, {"error": "not_found"})
            return
        raw_length = self.headers.get("Content-Length", "")
        if not raw_length.isascii() or not raw_length.isdigit():
            self.close_connection = True
            self._respond(HTTPStatus.LENGTH_REQUIRED, {"error": "length_required"})
            return
        length = int(raw_length)
        if length > MAX_REQUEST_BYTES:
            remaining = min(length, MAX_REQUEST_BYTES + 1)
            while remaining:
                chunk = self.rfile.read(min(64 * 1024, remaining))
                if not chunk:
                    break
                remaining -= len(chunk)
            self.close_connection = True
            self._respond(HTTPStatus.REQUEST_ENTITY_TOO_LARGE, {"error": "request_too_large"})
            return
        raw = self.rfile.read(length)
        try:
            request = json.loads(raw)
            if type(request) is not dict:
                raise ValueError
            response = _completion(request)
        except (UnicodeDecodeError, json.JSONDecodeError, TypeError, ValueError):
            self._respond(HTTPStatus.BAD_REQUEST, {"error": "invalid_request"})
            return
        self._respond(HTTPStatus.OK, response)

    def _respond(self, status: HTTPStatus, payload: dict[str, Any]) -> None:
        raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        try:
            self.send_response(status.value)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw)))
            if self.close_connection:
                self.send_header("Connection", "close")
            self.end_headers()
            self.wfile.write(raw)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def log_message(self, _format: str, *_args: object) -> None:
        return


@contextmanager
def running_server():
    server = ThreadingHTTPServer(("127.0.0.1", 0), FakeRagModelHandler)
    server.daemon_threads = True
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        host, port = server.server_address
        yield f"http://{host}:{port}"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def main() -> None:
    port = int(os.getenv("PORT", "8000"))
    server = ThreadingHTTPServer(("0.0.0.0", port), FakeRagModelHandler)
    server.serve_forever()


if __name__ == "__main__":
    main()
