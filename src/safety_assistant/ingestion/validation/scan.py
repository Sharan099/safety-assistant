"""Malware scanning boundary for uploaded bytes.

Uploads are untrusted. The scanner runs in the worker's VALIDATING stage, before any parser touches
the bytes. Development defaults to `none`; deployments point `MALWARE_SCANNER=clamav` at a clamd
service (or a managed equivalent behind the same interface). An unreachable scanner is a retryable
failure, never a silent pass — a document is only READY after a clean verdict when scanning is on.
"""

from __future__ import annotations

import socket
import struct
from dataclasses import dataclass
from typing import Protocol


class ScannerUnavailable(RuntimeError):
    pass


@dataclass(frozen=True)
class ScanVerdict:
    clean: bool
    detail: str | None = None


class MalwareScanner(Protocol):
    name: str

    def scan(self, data: bytes) -> ScanVerdict: ...


class NoScanner:
    name = "none"

    def scan(self, data: bytes) -> ScanVerdict:
        return ScanVerdict(clean=True, detail="scanning disabled")


class ClamdScanner:
    """clamd INSTREAM protocol over TCP (stdlib only): chunks prefixed with a 4-byte big-endian length."""

    name = "clamav"

    def __init__(
        self, host: str = "localhost", port: int = 3310, timeout_s: float = 60.0, chunk: int = 1 << 20
    ) -> None:
        self.host, self.port, self.timeout_s, self.chunk = host, port, timeout_s, chunk

    def scan(self, data: bytes) -> ScanVerdict:
        try:
            with socket.create_connection((self.host, self.port), timeout=self.timeout_s) as s:
                s.sendall(b"zINSTREAM\0")
                for i in range(0, len(data), self.chunk):
                    part = data[i : i + self.chunk]
                    s.sendall(struct.pack("!I", len(part)) + part)
                s.sendall(struct.pack("!I", 0))
                reply = b""
                while not reply.endswith(b"\0"):
                    piece = s.recv(4096)
                    if not piece:
                        break
                    reply += piece
        except OSError as exc:
            raise ScannerUnavailable(f"clamd unreachable: {type(exc).__name__}") from exc
        text = reply.decode("utf-8", errors="replace").strip("\0\n ")
        if text.endswith("OK"):
            return ScanVerdict(clean=True)
        if text.endswith("FOUND"):
            return ScanVerdict(clean=False, detail=text.split(":", 1)[-1].strip())
        raise ScannerUnavailable(f"unexpected clamd reply: {text[:80]}")


def scanner_from_settings(kind: str, *, host: str, port: int) -> MalwareScanner:
    if kind == "clamav":
        return ClamdScanner(host=host, port=port)
    return NoScanner()
