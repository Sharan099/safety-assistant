"""Structured gateway routing / failover diagnostics."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from loguru import logger


@dataclass
class FailoverStep:
    provider_key: str
    model: str
    outcome: str  # success | retryable_error | fatal_error | skipped_unavailable
    detail: str = ""

    def to_dict(self) -> dict[str, str]:
        return {
            "provider": self.provider_key,
            "model": self.model,
            "outcome": self.outcome,
            "detail": self.detail[:200],
        }


@dataclass
class RoutingDiagnostic:
    intended_tier: int
    intended_provider: str
    intended_model: str
    served_tier: int | None = None
    served_provider: str | None = None
    served_model: str | None = None
    fallback_used: bool = False
    failover_steps: list[FailoverStep] = field(default_factory=list)
    route_reasons: list[str] = field(default_factory=list)
    capability_escalation: list[str] = field(default_factory=list)

    def log_summary(self) -> None:
        if self.fallback_used:
            last_fail = next(
                (s for s in reversed(self.failover_steps) if s.outcome != "success"),
                None,
            )
            why = last_fail.detail if last_fail else "unknown"
            logger.warning(
                f"Gateway FAILOVER: intended={self.intended_provider}/{self.intended_model} "
                f"(tier {self.intended_tier}) -> served={self.served_provider}/"
                f"{self.served_model} (tier {self.served_tier}); "
                f"trigger={why[:120]}"
            )
        else:
            logger.info(
                f"Gateway routing: tier={self.served_tier} model={self.served_model} "
                f"provider={self.served_provider} (no failover)"
            )
        if self.capability_escalation:
            logger.info(f"Gateway capability escalation: {self.capability_escalation}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "intended_tier": self.intended_tier,
            "intended_provider": self.intended_provider,
            "intended_model": self.intended_model,
            "served_tier": self.served_tier,
            "served_provider": self.served_provider,
            "served_model": self.served_model,
            "fallback_used": self.fallback_used,
            "failover_steps": [s.to_dict() for s in self.failover_steps],
            "route_reasons": self.route_reasons,
            "capability_escalation": self.capability_escalation,
        }
