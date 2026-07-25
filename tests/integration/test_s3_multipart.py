"""Real S3-compatible multipart contract tests against opt-in MinIO."""

import asyncio
import os
from uuid import uuid4

import pytest
import pytest_asyncio
from botocore.exceptions import ClientError, EndpointConnectionError
from services import s3 as s3_module


def _client_error_code(exc: ClientError) -> str:
    return str(exc.response.get("Error", {}).get("Code", ""))


@pytest_asyncio.fixture
async def minio_service(monkeypatch):
    endpoint = os.getenv("S3_MULTIPART_TEST_ENDPOINT")
    if not endpoint:
        pytest.skip("S3_MULTIPART_TEST_ENDPOINT is not configured")

    bucket = os.getenv("S3_MULTIPART_TEST_BUCKET", "llmwiki-multipart-test")
    monkeypatch.setattr(s3_module.settings, "AWS_ACCESS_KEY_ID", os.environ["S3_MULTIPART_TEST_ACCESS_KEY"])
    monkeypatch.setattr(s3_module.settings, "AWS_SECRET_ACCESS_KEY", os.environ["S3_MULTIPART_TEST_SECRET_KEY"])
    monkeypatch.setattr(s3_module.settings, "AWS_REGION", "us-east-1")
    monkeypatch.setattr(s3_module.settings, "S3_BUCKET", bucket)
    monkeypatch.setattr(s3_module.settings, "S3_ENDPOINT_URL", endpoint)
    monkeypatch.setattr(s3_module.settings, "S3_FORCE_PATH_STYLE", True)
    service = s3_module.S3Service()

    last_error: Exception | None = None
    for _ in range(40):
        try:
            async with service._session.client("s3", **s3_module.s3_client_kwargs()) as client:
                try:
                    await client.head_bucket(Bucket=bucket)
                except ClientError as exc:
                    if _client_error_code(exc) not in {"404", "NoSuchBucket", "NotFound"}:
                        raise
                    await client.create_bucket(Bucket=bucket)
            break
        except EndpointConnectionError as exc:
            last_error = exc
            await asyncio.sleep(0.1)
    else:
        pytest.fail(f"MinIO did not become ready: {type(last_error).__name__}")

    yield service


@pytest.mark.asyncio
async def test_real_minio_multipart_lifecycle(minio_service):
    service = minio_service
    completed_key = f"multipart-tests/{uuid4()}/completed.bin"
    aborted_key = f"multipart-tests/{uuid4()}/aborted.bin"
    completed_upload_id: str | None = None
    aborted_upload_id: str | None = None

    try:
        await service.head_bucket()
        assert await service.head_object(completed_key) is None

        completed_upload_id = await service.create_multipart(
            completed_key,
            "application/octet-stream",
        )
        first_body = b"a" * (5 * 1024 * 1024)
        first_etag = await service.upload_part(
            completed_key,
            completed_upload_id,
            1,
            first_body,
        )
        second_etag = await service.upload_part(
            completed_key,
            completed_upload_id,
            2,
            b"tail",
        )
        assert '"' not in first_etag
        assert '"' not in second_etag

        await service.complete_multipart(
            completed_key,
            completed_upload_id,
            [
                s3_module.MultipartPart(part_number=2, etag=second_etag),
                s3_module.MultipartPart(part_number=1, etag=first_etag),
            ],
        )
        completed_upload_id = None

        metadata = await service.head_object(completed_key)
        assert metadata is not None
        assert metadata.size == len(first_body) + 4
        assert metadata.content_type == "application/octet-stream"
        assert metadata.etag
        assert '"' not in metadata.etag
        assert await service.read_range(completed_key, len(first_body) - 2, len(first_body) + 1) == b"aata"

        aborted_upload_id = await service.create_multipart(aborted_key, "application/octet-stream")
        await service.upload_part(aborted_key, aborted_upload_id, 1, b"discard")
        await service.abort_multipart(aborted_key, aborted_upload_id)
        with pytest.raises(ClientError) as exc_info:
            await service.upload_part(aborted_key, aborted_upload_id, 2, b"discard")
        assert _client_error_code(exc_info.value) == "NoSuchUpload"
        aborted_upload_id = None

        await service.delete_object(completed_key)
        assert await service.head_object(completed_key) is None
    finally:
        async with service._session.client("s3", **s3_module.s3_client_kwargs()) as client:
            if completed_upload_id is not None:
                try:
                    await client.abort_multipart_upload(
                        Bucket=service._bucket,
                        Key=completed_key,
                        UploadId=completed_upload_id,
                    )
                except ClientError as exc:
                    if _client_error_code(exc) != "NoSuchUpload":
                        raise
            if aborted_upload_id is not None:
                try:
                    await client.abort_multipart_upload(
                        Bucket=service._bucket,
                        Key=aborted_key,
                        UploadId=aborted_upload_id,
                    )
                except ClientError as exc:
                    if _client_error_code(exc) != "NoSuchUpload":
                        raise
            await client.delete_object(Bucket=service._bucket, Key=completed_key)
            await client.delete_object(Bucket=service._bucket, Key=aborted_key)
