"""SSRF-safe fetcher: scheme/host allowlist, private-address refusal, redirect
re-validation, size cap, content-type, conditional GET."""

from __future__ import annotations

import datetime

import httpx
import pytest

from safety_assistant.ingestion.fetch.http import FetchRefused, fetch_pdf, validate_url

ALLOWED = frozenset({"unece.org"})


@pytest.fixture(autouse=True)
def public_dns(monkeypatch: pytest.MonkeyPatch):  # type: ignore[no-untyped-def]
    table = {
        "unece.org": "8.8.8.8",
        "www.unece.org": "8.8.8.8",
        "evil.example": "8.8.4.4",
        "internal.unece.org": "10.0.0.5",
    }

    def fake(host, port, proto=0):  # type: ignore[no-untyped-def]
        if host not in table:
            raise OSError("unknown host")
        return [(2, 1, 6, "", (table[host], port))]

    monkeypatch.setattr("safety_assistant.ingestion.fetch.http.socket.getaddrinfo", fake)


@pytest.mark.parametrize(
    "url",
    [
        "http://unece.org/x.pdf",  # not https
        "https://evil.example/x.pdf",  # not allowlisted
        "https://internal.unece.org/x.pdf",  # allowlisted suffix but resolves to 10/8
        "https://user:pw@unece.org/x.pdf",  # credentials in authority
        "https://unece.org:8443/x.pdf",  # non-standard port
        "https://127.0.0.1/x.pdf",
    ],
)
def test_unsafe_urls_are_refused_before_any_request(url: str) -> None:
    with pytest.raises(FetchRefused):
        validate_url(url, ALLOWED)


def test_redirect_to_unallowlisted_host_is_refused() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.host == "unece.org":
            return httpx.Response(302, headers={"location": "https://evil.example/x.pdf"})
        return httpx.Response(200, content=b"%PDF-1.4", headers={"content-type": "application/pdf"})

    with pytest.raises(FetchRefused, match="allowlist"):
        fetch_pdf(
            "https://unece.org/x.pdf", allowed_hosts=ALLOWED, max_bytes=10_000, transport=httpx.MockTransport(handler)
        )


def test_size_cap_and_content_type_enforced() -> None:
    big = httpx.MockTransport(
        lambda r: httpx.Response(200, content=b"%PDF-" + b"0" * 5000, headers={"content-type": "application/pdf"})
    )
    with pytest.raises(FetchRefused, match="cap"):
        fetch_pdf("https://unece.org/x.pdf", allowed_hosts=ALLOWED, max_bytes=1000, transport=big)
    html = httpx.MockTransport(lambda r: httpx.Response(200, content=b"<html>", headers={"content-type": "text/html"}))
    with pytest.raises(FetchRefused, match="content type"):
        fetch_pdf("https://unece.org/x.pdf", allowed_hosts=ALLOWED, max_bytes=1000, transport=html)


def test_conditional_get_and_metadata() -> None:
    seen: dict[str, str] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen.update(request.headers)
        if request.headers.get("if-none-match") == '"abc"':
            return httpx.Response(304)
        return httpx.Response(
            200,
            content=b"%PDF-1.4 data",
            headers={
                "content-type": "application/pdf",
                "etag": '"abc"',
                "last-modified": "Wed, 21 Oct 2015 07:28:00 GMT",
            },
        )

    first = fetch_pdf(
        "https://www.unece.org/r94.pdf", allowed_hosts=ALLOWED, max_bytes=1000, transport=httpx.MockTransport(handler)
    )
    assert (
        first.status == 200
        and first.etag == '"abc"'
        and first.last_modified == datetime.datetime(2015, 10, 21, 7, 28, tzinfo=datetime.UTC)
    )
    second = fetch_pdf(
        "https://www.unece.org/r94.pdf",
        allowed_hosts=ALLOWED,
        max_bytes=1000,
        etag=first.etag,
        last_modified=first.last_modified,
        transport=httpx.MockTransport(handler),
    )
    assert second.status == 304 and second.data is None
    assert "if-modified-since" in seen
