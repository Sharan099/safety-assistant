"""SSRF-safe source fetcher (CLAUDE.md §14).

Rules, all enforced before a byte is read:
- https only; host must be on the allowlist derived from the registry;
- DNS is resolved *here* and every address must be public (no loopback,
  private, link-local, multicast, reserved) — defeats DNS-rebinding to
  internal services;
- redirects are followed manually, re-validated hop by hop, capped;
- response size is capped while streaming; content type must be PDF;
- conditional GET with ETag / If-Modified-Since so unchanged sources cost a 304.
"""

from __future__ import annotations

import datetime
import email.utils
import ipaddress
import socket
from dataclasses import dataclass
from urllib.parse import urlsplit

import httpx

MAX_REDIRECTS = 3
DEFAULT_TIMEOUT = 30.0


class FetchRefused(ValueError):
    """The URL failed a safety rule; nothing was downloaded."""


@dataclass(frozen=True)
class FetchResult:
    url: str
    final_url: str
    status: int  # 200 or 304
    data: bytes | None
    etag: str | None
    last_modified: datetime.datetime | None
    content_type: str | None


def _resolve_public(host: str) -> None:
    try:
        infos = socket.getaddrinfo(host, 443, proto=socket.IPPROTO_TCP)
    except socket.gaierror as exc:
        raise FetchRefused(f"cannot resolve {host}") from exc
    if not infos:
        raise FetchRefused(f"no addresses for {host}")
    for info in infos:
        ip = ipaddress.ip_address(info[4][0])
        if not ip.is_global or ip.is_multicast:
            raise FetchRefused(f"{host} resolves to non-public address {ip}")


def validate_url(url: str, allowed_hosts: frozenset[str]) -> str:
    parts = urlsplit(url)
    if parts.scheme != "https":
        raise FetchRefused(f"only https is allowed: {url}")
    if not parts.hostname or parts.username or parts.password or parts.port not in (None, 443):
        raise FetchRefused(f"malformed or non-standard authority: {url}")
    host = parts.hostname.lower()
    if host not in allowed_hosts and not any(host.endswith("." + h) for h in allowed_hosts):
        raise FetchRefused(f"host not on the source allowlist: {host}")
    _resolve_public(host)
    return host


def fetch_pdf(
    url: str,
    *,
    allowed_hosts: frozenset[str],
    max_bytes: int,
    etag: str | None = None,
    last_modified: datetime.datetime | None = None,
    timeout: float = DEFAULT_TIMEOUT,
    transport: httpx.BaseTransport | None = None,
) -> FetchResult:
    headers = {"User-Agent": "safety-assistant-ingest/0.2 (+registry allowlist)", "Accept": "application/pdf"}
    if etag:
        headers["If-None-Match"] = etag
    if last_modified:
        headers["If-Modified-Since"] = email.utils.format_datetime(last_modified.astimezone(datetime.UTC), usegmt=True)

    current = url
    with httpx.Client(timeout=timeout, follow_redirects=False, transport=transport) as client:
        for _hop in range(MAX_REDIRECTS + 1):
            validate_url(current, allowed_hosts)
            with client.stream("GET", current, headers=headers) as resp:
                if resp.status_code in (301, 302, 303, 307, 308):
                    location = resp.headers.get("location")
                    if not location:
                        raise FetchRefused("redirect without location")
                    current = str(httpx.URL(current).join(location))
                    continue
                if resp.status_code == 304:
                    return FetchResult(url, current, 304, None, etag, last_modified, None)
                if resp.status_code != 200:
                    raise FetchRefused(f"unexpected status {resp.status_code} from {current}")
                ctype = (resp.headers.get("content-type") or "").split(";")[0].strip().lower()
                if ctype not in ("application/pdf", "application/octet-stream"):
                    raise FetchRefused(f"unexpected content type {ctype!r}")
                declared = resp.headers.get("content-length")
                if declared and int(declared) > max_bytes:
                    raise FetchRefused(f"content-length {declared} exceeds cap {max_bytes}")
                buf = bytearray()
                for chunk in resp.iter_bytes():
                    buf.extend(chunk)
                    if len(buf) > max_bytes:
                        raise FetchRefused(f"response exceeded cap {max_bytes} bytes")
                lm = resp.headers.get("last-modified")
                return FetchResult(
                    url=url,
                    final_url=current,
                    status=200,
                    data=bytes(buf),
                    etag=resp.headers.get("etag"),
                    last_modified=email.utils.parsedate_to_datetime(lm) if lm else None,
                    content_type=ctype,
                )
    raise FetchRefused(f"too many redirects (> {MAX_REDIRECTS})")
