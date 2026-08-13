"""Regression test for the CORS gap that broke apps/web with an opaque
"Failed to fetch" (no Access-Control-Allow-Origin header meant the browser
blocked every cross-origin request before it reached a route at all).
"""

from fastapi.testclient import TestClient

from apps.api.main import app
from packages.domain.db import get_settings

client = TestClient(app)


def test_frontend_origin_allowed_by_cors() -> None:
    frontend_origin = get_settings().cors_origins[0]
    resp = client.get("/api/v1/health", headers={"Origin": frontend_origin})
    assert resp.status_code == 200
    assert resp.headers["access-control-allow-origin"] == frontend_origin


def test_cors_preflight_succeeds() -> None:
    frontend_origin = get_settings().cors_origins[0]
    resp = client.options(
        "/api/v1/runs",
        headers={
            "Origin": frontend_origin,
            "Access-Control-Request-Method": "GET",
        },
    )
    assert resp.status_code == 200
    assert resp.headers["access-control-allow-origin"] == frontend_origin
