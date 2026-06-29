"""Official source domain allow-list per authority (FR-3)."""

from __future__ import annotations

from urllib.parse import urlparse

ALLOWED_DOMAINS: dict[str, list[str]] = {
    "UNECE": ["unece.org"],
    "Euro NCAP": ["euroncap.com", "cdn.euroncap.com"],
    "FMVSS": ["govinfo.gov", "www.govinfo.gov"],
    "NHTSA": ["nhtsa.gov", "www.nhtsa.gov", "static.nhtsa.gov"],
    "IIHS": ["iihs.org", "www.iihs.org"],
    "EU Regulations": ["eur-lex.europa.eu", "op.europa.eu"],
    "China C-NCAP": ["c-ncap.org", "www.c-ncap.org"],
}


class DomainNotAllowedError(ValueError):
    pass


def assert_url_allowed(url: str, authority: str) -> None:
    parsed = urlparse(url)
    if parsed.scheme != "https":
        raise DomainNotAllowedError(f"Non-HTTPS URL refused: {url}")
    host = (parsed.hostname or "").lower()
    allowed = ALLOWED_DOMAINS.get(authority, [])
    if not allowed:
        raise DomainNotAllowedError(f"No allow-list configured for authority: {authority}")
    if not any(host == d or host.endswith(f".{d}") for d in allowed):
        raise DomainNotAllowedError(
            f"Host {host!r} not in allow-list for {authority}: {allowed}"
        )
