"""Self-hosting S3 support: client kwargs, object primitives, and URL allowlist."""

import importlib.util
import sys
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest
from botocore.exceptions import ClientError

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


class _StreamingBody:
    def __init__(self, body: bytes):
        self._body = body

    async def read(self) -> bytes:
        return self._body


class _FakeS3Client:
    def __init__(self):
        self.calls: list[tuple[str, dict]] = []

    async def create_multipart_upload(self, **kwargs):
        self.calls.append(("create_multipart_upload", kwargs))
        return {"UploadId": "upload-123"}

    async def upload_part(self, **kwargs):
        self.calls.append(("upload_part", kwargs))
        return {"ETag": '"part-etag"'}

    async def complete_multipart_upload(self, **kwargs):
        self.calls.append(("complete_multipart_upload", kwargs))
        return {}

    async def abort_multipart_upload(self, **kwargs):
        self.calls.append(("abort_multipart_upload", kwargs))
        return {}

    async def head_object(self, **kwargs):
        self.calls.append(("head_object", kwargs))
        if kwargs["Key"] == "missing":
            raise ClientError(
                {"Error": {"Code": "NoSuchKey", "Message": "absent"}},
                "HeadObject",
            )
        if kwargs["Key"] == "forbidden":
            raise ClientError(
                {"Error": {"Code": "AccessDenied", "Message": "denied"}},
                "HeadObject",
            )
        return {
            "ContentLength": 12,
            "ETag": '"object-etag"',
            "ContentType": "text/plain",
        }

    async def get_object(self, **kwargs):
        self.calls.append(("get_object", kwargs))
        return {"Body": _StreamingBody(b"selected")}

    async def delete_object(self, **kwargs):
        self.calls.append(("delete_object", kwargs))
        return {}

    async def head_bucket(self, **kwargs):
        self.calls.append(("head_bucket", kwargs))
        return {}


class _FakeClientContext:
    def __init__(self, client):
        self._client = client

    async def __aenter__(self):
        return self._client

    async def __aexit__(self, exc_type, exc, traceback):
        return False


class _FakeSession:
    def __init__(self, client):
        self._client = client

    def client(self, service_name, **kwargs):
        assert service_name == "s3"
        return _FakeClientContext(self._client)


def _fake_service():
    s3 = _s3_module()
    client = _FakeS3Client()
    service = object.__new__(s3.S3Service)
    service._session = _FakeSession(client)
    service._bucket = "test-bucket"
    return s3, service, client


@pytest.mark.asyncio
async def test_multipart_calls_use_exact_s3_parameters_and_order_parts():
    s3, service, client = _fake_service()

    upload_id = await service.create_multipart("tenant/object.bin", "application/octet-stream")
    etag = await service.upload_part("tenant/object.bin", upload_id, 2, b"part-two")
    await service.complete_multipart(
        "tenant/object.bin",
        upload_id,
        [
            s3.MultipartPart(part_number=2, etag="part-two"),
            s3.MultipartPart(part_number=1, etag="part-one"),
        ],
    )
    await service.abort_multipart("tenant/abandoned.bin", "upload-456")

    assert upload_id == "upload-123"
    assert etag == "part-etag"
    assert client.calls == [
        (
            "create_multipart_upload",
            {
                "Bucket": "test-bucket",
                "Key": "tenant/object.bin",
                "ContentType": "application/octet-stream",
            },
        ),
        (
            "upload_part",
            {
                "Bucket": "test-bucket",
                "Key": "tenant/object.bin",
                "UploadId": "upload-123",
                "PartNumber": 2,
                "Body": b"part-two",
            },
        ),
        (
            "complete_multipart_upload",
            {
                "Bucket": "test-bucket",
                "Key": "tenant/object.bin",
                "UploadId": "upload-123",
                "MultipartUpload": {
                    "Parts": [
                        {"PartNumber": 1, "ETag": "part-one"},
                        {"PartNumber": 2, "ETag": "part-two"},
                    ]
                },
            },
        ),
        (
            "abort_multipart_upload",
            {
                "Bucket": "test-bucket",
                "Key": "tenant/abandoned.bin",
                "UploadId": "upload-456",
            },
        ),
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize("part_number", [0, 10_001])
async def test_upload_part_rejects_part_numbers_outside_s3_bounds(part_number):
    _, service, client = _fake_service()

    with pytest.raises(ValueError, match="1.*10000"):
        await service.upload_part("object.bin", "upload-123", part_number, b"body")

    assert client.calls == []


@pytest.mark.asyncio
@pytest.mark.parametrize("part_number", [1, 10_000])
async def test_upload_part_accepts_s3_boundary_part_numbers(part_number):
    _, service, client = _fake_service()

    assert await service.upload_part("object.bin", "upload-123", part_number, b"x") == "part-etag"
    assert client.calls[0][1]["PartNumber"] == part_number


@pytest.mark.asyncio
async def test_object_metadata_range_delete_and_bucket_contract():
    s3, service, client = _fake_service()

    metadata = await service.head_object("present")
    assert metadata == s3.ObjectMetadata(
        size=12,
        etag="object-etag",
        content_type="text/plain",
    )
    with pytest.raises(FrozenInstanceError):
        metadata.size = 13

    assert await service.read_range("present", 4, 10) == b"selected"
    await service.delete_object("present")
    await service.head_bucket()

    assert client.calls == [
        ("head_object", {"Bucket": "test-bucket", "Key": "present"}),
        (
            "get_object",
            {"Bucket": "test-bucket", "Key": "present", "Range": "bytes=4-10"},
        ),
        ("delete_object", {"Bucket": "test-bucket", "Key": "present"}),
        ("head_bucket", {"Bucket": "test-bucket"}),
    ]


@pytest.mark.asyncio
async def test_head_object_maps_only_explicit_not_found_errors_to_none():
    _, service, _ = _fake_service()

    assert await service.head_object("missing") is None
    with pytest.raises(ClientError) as exc_info:
        await service.head_object("forbidden")
    assert exc_info.value.response["Error"]["Code"] == "AccessDenied"


@pytest.mark.asyncio
@pytest.mark.parametrize(("start", "end"), [(-1, 0), (2, 1)])
async def test_read_range_rejects_invalid_inclusive_ranges(start, end):
    _, service, client = _fake_service()

    with pytest.raises(ValueError, match="range"):
        await service.read_range("present", start, end)

    assert client.calls == []


def test_multipart_part_is_immutable():
    s3 = _s3_module()
    part = s3.MultipartPart(part_number=1, etag="etag")
    with pytest.raises(FrozenInstanceError):
        part.etag = "changed"


@pytest.mark.parametrize("part_number", [0, 10_001])
def test_multipart_part_rejects_part_numbers_outside_s3_bounds(part_number):
    s3 = _s3_module()
    with pytest.raises(ValueError, match="1.*10000"):
        s3.MultipartPart(part_number=part_number, etag="etag")


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
