"""
OpenAI-compatible gateway endpoints.

Exposes the Intelligent Multi-LLM Gateway to external/agent clients using the
familiar OpenAI Chat Completions schema, so existing OpenAI SDK clients can point
at PSA AI and get automatic tiered routing, caching and failover.

Mounted under the API prefix:
    POST /api/v1/gateway/v1/chat/completions   (OpenAI-compatible)
    POST /api/v1/gateway/route-preview         (debug: routing decision only)
    GET  /api/v1/gateway/health                (config + provider availability)
"""

from __future__ import annotations

import time
import uuid
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from loguru import logger
from pydantic import BaseModel, Field

from backend.app.core.security import rate_limit
from backend.app.gateway import config as cfg
from backend.app.gateway.types import RoutingContext

router = APIRouter(prefix="/gateway", tags=["gateway"])


# ───────────────────────── OpenAI-compatible schema ─────────────────────────
class ChatMessage(BaseModel):
    role: str = Field(..., pattern="^(system|user|assistant)$")
    content: str


class ChatCompletionRequest(BaseModel):
    model: str | None = Field(default=None, max_length=128)  # "auto" => route
    messages: list[ChatMessage] = Field(..., min_length=1)
    temperature: float | None = None
    max_tokens: int | None = None
    user: str | None = Field(default=None, max_length=64)


def _join_prompt(messages: list[ChatMessage]) -> tuple[str, str]:
    """Return (prompt_for_routing, last_user_message)."""
    user_msgs = [m.content for m in messages if m.role == "user"]
    last_user = user_msgs[-1] if user_msgs else ""
    full = "\n\n".join(m.content for m in messages if m.role != "system")
    return full or last_user, last_user


@router.post("/v1/chat/completions")
async def chat_completions(req: ChatCompletionRequest, request: Request) -> dict:
    import asyncio

    rate_limit(request, "gateway_chat", limit=60, window_s=60)
    from backend.app.core.services import get_gateway

    gateway = get_gateway()
    if gateway is None:
        raise HTTPException(status_code=503, detail="Gateway is disabled")

    prompt, _last = _join_prompt(req.messages)
    depth = max(0, len([m for m in req.messages if m.role == "assistant"]))
    ctx = RoutingContext(
        prompt=prompt,
        query=_last,
        conversation_depth=depth,
        user_id=req.user,
    )
    try:
        result = await asyncio.to_thread(gateway.complete, prompt, ctx)
    except Exception as exc:
        logger.exception("Gateway chat/completions failed")
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    now = int(time.time())
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:24]}",
        "object": "chat.completion",
        "created": now,
        "model": result.model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": result.answer},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": result.prompt_tokens or 0,
            "completion_tokens": result.completion_tokens or 0,
            "total_tokens": (result.prompt_tokens or 0)
            + (result.completion_tokens or 0),
        },
        # Non-standard but useful PSA AI routing metadata.
        "psa_gateway": {
            "provider": result.provider,
            "tier": result.tier,
            "route_score": result.route_score,
            "route_reasons": result.route_reasons,
            "cache_hit": result.cache_hit,
            "fallback_used": result.fallback_used,
            "cost_usd": round(result.cost_usd, 6),
            "cost_saved_usd": round(result.cost_saved_usd, 6),
        },
    }


# ───────────────────────── debug / introspection ─────────────────────────
class RoutePreviewRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=8000)
    grounding_confidence: float | None = None
    conversation_depth: int = 0


@router.post("/route-preview")
async def route_preview(req: RoutePreviewRequest, request: Request) -> dict[str, Any]:
    rate_limit(request, "gateway_preview", limit=120, window_s=60)
    from backend.app.core.services import get_gateway

    gateway = get_gateway()
    grounding = (
        {"confidence": req.grounding_confidence}
        if req.grounding_confidence is not None
        else {}
    )
    ctx = RoutingContext(
        prompt=req.prompt,
        query=req.prompt,
        grounding=grounding,
        conversation_depth=req.conversation_depth,
    )
    if gateway is not None:
        decision = gateway.route_preview(ctx)
    else:
        from backend.app.gateway.classifier import classify

        decision = classify(ctx)
    return decision.to_dict()


@router.get("/health")
async def gateway_health() -> dict[str, Any]:
    from backend.app.core.services import get_gateway

    gateway = get_gateway()
    providers: dict[str, bool] = {}
    if gateway is not None:
        for key, provider in gateway._providers.items():  # noqa: SLF001
            try:
                providers[key] = provider.available()
            except Exception:
                providers[key] = False
    return {
        "enabled": cfg.ENABLE_GATEWAY,
        "shadow_mode": cfg.GATEWAY_SHADOW_MODE,
        "canary_pct": cfg.GATEWAY_CANARY_PCT,
        "cache_enabled": cfg.ENABLE_SEMANTIC_CACHE,
        "tiers": {
            t: {"provider": s.provider, "model": s.model, "purpose": s.purpose}
            for t, s in cfg.TIERS.items()
        },
        "providers_available": providers,
    }
