"""Content-addressed blob storage behind one small interface.

Keys are server-generated from the content hash — never from client-supplied
filenames (CLAUDE.md §14). ``FilesystemBlobStore`` is the development
backend; an S3-compatible adapter plugs in behind the same Protocol.
"""

from __future__ import annotations

import hashlib
import pathlib
from typing import Protocol


class BlobStore(Protocol):
    def put(self, data: bytes, *, suffix: str = "") -> str:
        """Store bytes, return an opaque storage URI. Idempotent on content."""
        ...

    def get(self, uri: str) -> bytes: ...

    def exists(self, uri: str) -> bool: ...


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


class FilesystemBlobStore:
    scheme = "file://"

    def __init__(self, root: pathlib.Path) -> None:
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, uri: str) -> pathlib.Path:
        if not uri.startswith(self.scheme):
            raise ValueError(f"not a filesystem blob uri: {uri}")
        rel = pathlib.PurePosixPath(uri[len(self.scheme) :])
        if ".." in rel.parts or rel.is_absolute():
            raise ValueError(f"unsafe blob uri: {uri}")
        return self.root / rel

    def put(self, data: bytes, *, suffix: str = "") -> str:
        digest = sha256_bytes(data)
        uri = f"{self.scheme}sha256/{digest[:2]}/{digest}{suffix}"
        path = self._path(uri)
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(path.suffix + ".tmp")
            tmp.write_bytes(data)
            tmp.replace(path)  # atomic publish; immutable thereafter
        return uri

    def get(self, uri: str) -> bytes:
        return self._path(uri).read_bytes()

    def exists(self, uri: str) -> bool:
        return self._path(uri).exists()


def blob_store_from_uri(uri: str) -> BlobStore:
    if uri.startswith("file://"):
        return FilesystemBlobStore(pathlib.Path(uri[len("file://") :]))
    if uri.startswith("s3://"):
        from safety_assistant.ingestion.fetch.s3 import S3BlobStore

        return S3BlobStore.from_uri(uri)
    raise NotImplementedError(f"blob store scheme not supported: {uri}")
