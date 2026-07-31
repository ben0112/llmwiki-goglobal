"""Self-hosting S3 endpoint support: client kwargs + converter URL allowlist."""

import importlib.util
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]


# ---------------------------------------------------------------------------
# api/services/s3.py client kwargs
# ---------------------------------------------------------------------------

def _s3_module():
    from services import s3
    return s3


def test_default_kwargs_empty(monkeypatch):
    s3 = _s3_module()
    monkeypatch.setattr(s3.settings, "S3_ENDPOINT_URL", "")
    monkeypatch.setattr(s3.settings, "S3_FORCE_PATH_STYLE", False)
    assert s3.s3_client_kwargs() == {}


def test_endpoint_and_path_style(monkeypatch):
    s3 = _s3_module()
    monkeypatch.setattr(s3.settings, "S3_ENDPOINT_URL", "http://minio:9000")
    monkeypatch.setattr(s3.settings, "S3_FORCE_PATH_STYLE", True)
    kwargs = s3.s3_client_kwargs()
    assert kwargs["endpoint_url"] == "http://minio:9000"
    assert kwargs["config"].s3 == {"addressing_style": "path"}


# ---------------------------------------------------------------------------
# converter/main.py URL allowlist with a self-hosted endpoint
# ---------------------------------------------------------------------------

def _load_converter(monkeypatch, **env):
    monkeypatch.setenv("CONVERTER_SECRET", "test-secret")
    for key in ("S3_BUCKET", "S3_ENDPOINT", "S3_ENDPOINT_URL", "AWS_REGION"):
        monkeypatch.delenv(key, raising=False)
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    spec = importlib.util.spec_from_file_location(
        "converter_main_selfhost_test", REPO_ROOT / "converter" / "main.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        sys.modules.pop(spec.name, None)
    return module


def test_minio_path_style_accepted(monkeypatch):
    conv = _load_converter(monkeypatch, S3_BUCKET="llmwiki", S3_ENDPOINT="http://minio:9000")
    conv._validate_s3_url("http://minio:9000/llmwiki/user/doc/source.pdf?X-Amz-Signature=abc")


def test_minio_wrong_bucket_rejected(monkeypatch):
    from fastapi import HTTPException
    conv = _load_converter(monkeypatch, S3_BUCKET="llmwiki", S3_ENDPOINT="http://minio:9000")
    with pytest.raises(HTTPException):
        conv._validate_s3_url("http://minio:9000/other-bucket/user/doc/source.pdf")


def test_minio_wrong_host_or_port_rejected(monkeypatch):
    from fastapi import HTTPException
    conv = _load_converter(monkeypatch, S3_BUCKET="llmwiki", S3_ENDPOINT="http://minio:9000")
    with pytest.raises(HTTPException):
        conv._validate_s3_url("http://evil.example.com/llmwiki/user/doc/source.pdf")
    with pytest.raises(HTTPException):
        conv._validate_s3_url("http://minio:9001/llmwiki/user/doc/source.pdf")


def test_minio_vhost_style_accepted(monkeypatch):
    conv = _load_converter(monkeypatch, S3_BUCKET="llmwiki", S3_ENDPOINT="https://s3.example.internal")
    conv._validate_s3_url("https://llmwiki.s3.example.internal/user/doc/source.pdf")


def test_endpoint_url_alias_env(monkeypatch):
    """S3_ENDPOINT_URL (the API-side name) works as a fallback env var."""
    conv = _load_converter(monkeypatch, S3_BUCKET="llmwiki", S3_ENDPOINT_URL="http://minio:9000")
    conv._validate_s3_url("http://minio:9000/llmwiki/user/doc/source.pdf")


def test_aws_logic_unchanged_without_endpoint(monkeypatch):
    from fastapi import HTTPException
    conv = _load_converter(monkeypatch, S3_BUCKET="llmwiki", AWS_REGION="us-east-1")
    conv._validate_s3_url("https://llmwiki.s3.us-east-1.amazonaws.com/user/doc/source.pdf")
    with pytest.raises(HTTPException):
        conv._validate_s3_url("http://minio:9000/llmwiki/user/doc/source.pdf")
