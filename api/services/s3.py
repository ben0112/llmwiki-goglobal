import asyncio
import json
import logging
from dataclasses import dataclass
from pathlib import Path

import aioboto3
from botocore.config import Config as BotoConfig
from botocore.exceptions import ClientError
from config import settings

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class MultipartPart:
    part_number: int
    etag: str

    def __post_init__(self) -> None:
        if not 1 <= self.part_number <= 10_000:
            raise ValueError("part_number must be between 1 and 10000")


@dataclass(frozen=True, slots=True)
class ObjectMetadata:
    size: int
    etag: str
    content_type: str | None


def _normalize_etag(etag: str) -> str:
    if len(etag) >= 2 and etag.startswith('"') and etag.endswith('"'):
        return etag[1:-1]
    return etag


def s3_client_kwargs() -> dict:
    """Client kwargs honoring a self-hosted S3-compatible endpoint (MinIO)."""
    kwargs: dict = {}
    if settings.S3_ENDPOINT_URL:
        kwargs["endpoint_url"] = settings.S3_ENDPOINT_URL
    if settings.S3_FORCE_PATH_STYLE:
        kwargs["config"] = BotoConfig(s3={"addressing_style": "path"})
    return kwargs


class S3Service:
    def __init__(self):
        self._session = aioboto3.Session(
            aws_access_key_id=settings.AWS_ACCESS_KEY_ID,
            aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY,
            region_name=settings.AWS_REGION,
        )
        self._bucket = settings.S3_BUCKET

    async def upload_bytes(self, key: str, data: bytes, content_type: str = "application/octet-stream"):
        async with self._session.client("s3", **s3_client_kwargs()) as s3:
            await s3.put_object(Bucket=self._bucket, Key=key, Body=data, ContentType=content_type)

    async def upload_file(self, key: str, file_path: str, content_type: str = "application/octet-stream"):
        data = await asyncio.to_thread(Path(file_path).read_bytes)
        await self.upload_bytes(key, data, content_type)

    async def create_multipart(self, key: str, content_type: str) -> str:
        async with self._session.client("s3", **s3_client_kwargs()) as s3:
            response = await s3.create_multipart_upload(
                Bucket=self._bucket,
                Key=key,
                ContentType=content_type,
            )
        return response["UploadId"]

    async def upload_part(
        self,
        key: str,
        upload_id: str,
        part_number: int,
        body: bytes,
    ) -> str:
        if not 1 <= part_number <= 10_000:
            raise ValueError("part_number must be between 1 and 10000")
        async with self._session.client("s3", **s3_client_kwargs()) as s3:
            response = await s3.upload_part(
                Bucket=self._bucket,
                Key=key,
                UploadId=upload_id,
                PartNumber=part_number,
                Body=body,
            )
        return _normalize_etag(response["ETag"])

    async def complete_multipart(
        self,
        key: str,
        upload_id: str,
        parts: list[MultipartPart],
    ) -> None:
        ordered_parts = [
            {"PartNumber": part.part_number, "ETag": part.etag}
            for part in sorted(parts, key=lambda part: part.part_number)
        ]
        async with self._session.client("s3", **s3_client_kwargs()) as s3:
            await s3.complete_multipart_upload(
                Bucket=self._bucket,
                Key=key,
                UploadId=upload_id,
                MultipartUpload={"Parts": ordered_parts},
            )

    async def abort_multipart(self, key: str, upload_id: str) -> None:
        async with self._session.client("s3", **s3_client_kwargs()) as s3:
            await s3.abort_multipart_upload(
                Bucket=self._bucket,
                Key=key,
                UploadId=upload_id,
            )

    async def head_object(self, key: str) -> ObjectMetadata | None:
        async with self._session.client("s3", **s3_client_kwargs()) as s3:
            try:
                response = await s3.head_object(Bucket=self._bucket, Key=key)
            except ClientError as exc:
                code = str(exc.response.get("Error", {}).get("Code", ""))
                if code in {"404", "NoSuchKey", "NotFound"}:
                    return None
                raise
        return ObjectMetadata(
            size=response["ContentLength"],
            etag=_normalize_etag(response["ETag"]),
            content_type=response.get("ContentType"),
        )

    async def read_range(self, key: str, start: int, end: int) -> bytes:
        """Read the inclusive byte range ``start..end`` from one object."""
        if start < 0 or end < start:
            raise ValueError("inclusive byte range must satisfy 0 <= start <= end")
        async with self._session.client("s3", **s3_client_kwargs()) as s3:
            response = await s3.get_object(
                Bucket=self._bucket,
                Key=key,
                Range=f"bytes={start}-{end}",
            )
            return await response["Body"].read()

    async def delete_object(self, key: str) -> None:
        async with self._session.client("s3", **s3_client_kwargs()) as s3:
            await s3.delete_object(Bucket=self._bucket, Key=key)

    async def head_bucket(self) -> None:
        async with self._session.client("s3", **s3_client_kwargs()) as s3:
            await s3.head_bucket(Bucket=self._bucket)

    async def generate_presigned_get(self, key: str, expires_in: int = 3600) -> str:
        async with self._session.client("s3", **s3_client_kwargs()) as s3:
            return await s3.generate_presigned_url(
                "get_object",
                Params={"Bucket": self._bucket, "Key": key},
                ExpiresIn=expires_in,
            )

    async def generate_presigned_put(self, key: str, content_type: str = "application/pdf", expires_in: int = 3600) -> str:
        async with self._session.client("s3", **s3_client_kwargs()) as s3:
            return await s3.generate_presigned_url(
                "put_object",
                Params={"Bucket": self._bucket, "Key": key, "ContentType": content_type},
                ExpiresIn=expires_in,
            )

    async def delete_prefix(self, prefix: str) -> None:
        async with self._session.client("s3", **s3_client_kwargs()) as s3:
            paginator = s3.get_paginator("list_objects_v2")
            batch: list[dict] = []
            async for page in paginator.paginate(Bucket=self._bucket, Prefix=prefix):
                for obj in page.get("Contents", []):
                    batch.append({"Key": obj["Key"]})
                    if len(batch) == 1000:
                        await s3.delete_objects(Bucket=self._bucket, Delete={"Objects": batch})
                        batch = []
            if batch:
                await s3.delete_objects(Bucket=self._bucket, Delete={"Objects": batch})

    async def delete_key(self, key: str) -> None:
        """Delete one exact object for transaction compensation."""
        await self.delete_object(key)

    async def download_bytes(self, key: str) -> bytes:
        async with self._session.client("s3", **s3_client_kwargs()) as s3:
            resp = await s3.get_object(Bucket=self._bucket, Key=key)
            return await resp["Body"].read()

    async def download_to_file(self, key: str, file_path: str):
        data = await self.download_bytes(key)
        await asyncio.to_thread(Path(file_path).write_bytes, data)

    async def download_json(self, key: str) -> dict:
        body = await self.download_bytes(key)
        return json.loads(body)
