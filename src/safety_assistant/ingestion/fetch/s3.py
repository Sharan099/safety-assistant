"""S3-compatible blob store (AWS S3, MinIO, GCS/Azure via their S3 gateways).

Content-addressed keys, immutable objects (never overwritten), optional
prefix. Requires the `s3` extra (boto3). Credentials come from the standard
AWS chain (env, instance role, profile) — never from application config."""

from __future__ import annotations

from functools import cached_property
from typing import Any
from urllib.parse import urlsplit

from safety_assistant.ingestion.fetch.blobstore import sha256_bytes


class S3BlobStore:
    def __init__(self, bucket: str, prefix: str = "") -> None:
        self.bucket = bucket
        self.prefix = prefix.strip("/")

    @classmethod
    def from_uri(cls, uri: str) -> S3BlobStore:
        parts = urlsplit(uri)
        if parts.scheme != "s3" or not parts.netloc:
            raise ValueError(f"not an s3 uri: {uri}")
        return cls(parts.netloc, parts.path)

    @cached_property
    def _client(self) -> Any:
        try:
            import boto3
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("install the 's3' extra: uv sync --extra s3") from exc
        return boto3.client("s3")

    def _key(self, uri: str) -> str:
        expected = f"s3://{self.bucket}/"
        if not uri.startswith(expected):
            raise ValueError(f"uri does not belong to this store: {uri}")
        key = uri[len(expected) :]
        if ".." in key or key.startswith("/"):
            raise ValueError(f"unsafe key: {key}")
        return key

    def put(self, data: bytes, *, suffix: str = "") -> str:
        digest = sha256_bytes(data)
        key = "/".join(p for p in (self.prefix, "sha256", digest[:2], f"{digest}{suffix}") if p)
        uri = f"s3://{self.bucket}/{key}"
        if not self.exists(uri):
            self._client.put_object(
                Bucket=self.bucket,
                Key=key,
                Body=data,
                ContentType="application/pdf" if suffix == ".pdf" else "application/octet-stream",
                Metadata={"sha256": digest},
            )
        return uri

    def get(self, uri: str) -> bytes:
        body = self._client.get_object(Bucket=self.bucket, Key=self._key(uri))["Body"].read()
        return bytes(body)

    def exists(self, uri: str) -> bool:
        try:
            self._client.head_object(Bucket=self.bucket, Key=self._key(uri))
            return True
        except Exception as exc:  # noqa: BLE001 — botocore ClientError 404 / NoSuchKey
            if "404" in str(exc) or "NoSuchKey" in str(exc) or "Not Found" in str(exc):
                return False
            raise
