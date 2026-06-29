"""Gateway routing configuration."""



from __future__ import annotations



import os



# Failover: Groq tiers first, then OpenRouter (paid) when Groq is rate-limited/exhausted.

# Direct anthropic_sonnet is omitted from the live chain on networks where api.anthropic.com

# is blocked — it only adds a ~3s connect timeout before evidence-only. Use openrouter_claude

# instead when OpenRouter egress works (routes to Anthropic upstream).

_OPENROUTER_FAILOVER = ["openrouter_llama", "openrouter_claude"]



FALLBACK_CHAINS: dict[str, list[str]] = {

    "groq": ["groq", "groq_fast", *_OPENROUTER_FAILOVER],

    "groq_power": ["groq_power", "groq", "groq_fast", *_OPENROUTER_FAILOVER],

    "groq_fast": ["groq_fast", "groq", *_OPENROUTER_FAILOVER],

}



DEFAULT_PRIMARY = os.getenv("GATEWAY_PRIMARY_MODEL", "groq")



ENABLE_GATEWAY = os.getenv("ENABLE_GATEWAY", "true").lower() == "true"

GATEWAY_SHADOW_MODE = os.getenv("GATEWAY_SHADOW_MODE", "false").lower() == "true"

