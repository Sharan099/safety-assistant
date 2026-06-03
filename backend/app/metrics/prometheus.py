"""Prometheus metrics: cost proxy, latency, tokens, error rate."""

from prometheus_client import Counter, Histogram, Gauge

# Latency histograms (seconds)
REQUEST_LATENCY = Histogram(
    "rag_request_duration_seconds",
    "End-to-end RAG request latency",
    ["endpoint", "status"],
    buckets=(0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0),
)

RETRIEVAL_LATENCY = Histogram(
    "rag_retrieval_duration_seconds",
    "Hybrid retrieval latency",
    buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0),
)

LLM_LATENCY = Histogram(
    "rag_llm_duration_seconds",
    "LLM generation latency",
    buckets=(0.5, 1.0, 2.0, 5.0, 10.0, 30.0),
)

# Token usage (estimated from Groq usage or char heuristic)
TOKENS_PROMPT = Counter(
    "rag_tokens_prompt_total",
    "Total prompt tokens (estimated)",
)

TOKENS_COMPLETION = Counter(
    "rag_tokens_completion_total",
    "Total completion tokens (estimated)",
)

# Cost proxy (USD estimate per request — configurable rate)
ESTIMATED_COST_USD = Counter(
    "rag_estimated_cost_usd_total",
    "Estimated LLM cost in USD",
)

# Errors
ERRORS_TOTAL = Counter(
    "rag_errors_total",
    "Total RAG errors",
    ["error_type"],
)

GUARDRAIL_BLOCKS = Counter(
    "rag_guardrail_blocks_total",
    "Queries blocked by guardrails",
    ["reason"],
)

ACTIVE_REQUESTS = Gauge(
    "rag_active_requests",
    "In-flight RAG requests",
)


def estimate_tokens(text: str) -> int:
    return max(1, int(len(text.split()) * 1.33))


def record_llm_usage(prompt: str, completion: str, latency_s: float) -> None:
    p_tok = estimate_tokens(prompt)
    c_tok = estimate_tokens(completion)
    TOKENS_PROMPT.inc(p_tok)
    TOKENS_COMPLETION.inc(c_tok)
    LLM_LATENCY.observe(latency_s)
    # Groq llama-3.3-70b rough proxy: $0.59/1M in, $0.79/1M out
    cost = (p_tok * 0.59 + c_tok * 0.79) / 1_000_000
    ESTIMATED_COST_USD.inc(cost)
