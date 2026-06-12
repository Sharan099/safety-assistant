"""
PSA AI Intelligent Multi-LLM Gateway (v3.0).

A self-contained, OpenAI-compatible routing layer that sits behind the LangGraph
`generate` node. It estimates query complexity (reusing the existing grounding
confidence), routes to the cheapest capable tier (Groq -> Claude Haiku ->
Claude Sonnet), serves a semantic cache backed by the existing BGE embeddings,
and provides timeout/retry/failover with full Prometheus observability.

Backward compatible: with ENABLE_GATEWAY=false the workflow keeps using GroqLLM.
"""

from backend.app.gateway.classifier import classify, score_only
from backend.app.gateway.config import ENABLE_GATEWAY, GatewaySettings
from backend.app.gateway.gateway import LLMGateway
from backend.app.gateway.types import (
    GatewayResult,
    ProviderResponse,
    RouteDecision,
    RoutingContext,
)

__all__ = [
    "LLMGateway",
    "RoutingContext",
    "RouteDecision",
    "GatewayResult",
    "ProviderResponse",
    "classify",
    "score_only",
    "ENABLE_GATEWAY",
    "GatewaySettings",
]
