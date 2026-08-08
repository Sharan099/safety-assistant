"use client";

import { useCallback, useEffect, useState } from "react";
import { API_URL } from "@/lib/api";
import type { AggregateMetrics, TraceMetrics } from "@/lib/types";

type Props = {
  lastTraceId?: string | null;
  onClose?: () => void;
};

function money(n: number | undefined) {
  if (n == null || Number.isNaN(n)) return "—";
  if (n === 0) return "$0";
  if (n < 0.0001) return `$${n.toExponential(2)}`;
  return `$${n.toFixed(6)}`;
}

function pct(share: number | undefined) {
  if (share == null || Number.isNaN(share)) return "—";
  return `${(share * 100).toFixed(0)}%`;
}

export function MetricsPanel({ lastTraceId, onClose }: Props) {
  const [agg, setAgg] = useState<AggregateMetrics | null>(null);
  const [trace, setTrace] = useState<TraceMetrics | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  const load = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const aggRes = await fetch(`${API_URL}/metrics/aggregate`);
      if (!aggRes.ok) throw new Error(await aggRes.text());
      setAgg(await aggRes.json());
      if (lastTraceId) {
        const trRes = await fetch(`${API_URL}/metrics/${encodeURIComponent(lastTraceId)}`);
        if (trRes.ok) setTrace(await trRes.json());
        else setTrace(null);
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to load metrics");
    } finally {
      setLoading(false);
    }
  }, [lastTraceId]);

  useEffect(() => {
    void load();
  }, [load]);

  const providerEntries = Object.entries(agg?.by_provider || {});

  return (
    <aside className="metrics-panel" aria-label="Cost and latency metrics">
      <header className="metrics-panel__head">
        <div>
          <p className="metrics-eyebrow">Observability</p>
          <h2>Query metrics</h2>
        </div>
        <div className="metrics-panel__actions">
          <button type="button" className="ghost-btn" onClick={() => void load()} disabled={loading}>
            Refresh
          </button>
          {onClose ? (
            <button type="button" className="ghost-btn" onClick={onClose}>
              Close
            </button>
          ) : null}
        </div>
      </header>

      {error ? <div className="banner banner--error">{error}</div> : null}

      <section className="metrics-block">
        <h3>Aggregate</h3>
        {agg ? (
          <dl className="metrics-grid">
            <div>
              <dt>Queries</dt>
              <dd>{agg.n}</dd>
            </div>
            <div>
              <dt>Avg cost / query</dt>
              <dd>{money(agg.avg_cost_usd)}</dd>
            </div>
            <div>
              <dt>p95 latency</dt>
              <dd>{agg.p95_latency_ms?.toFixed?.(0) ?? "—"} ms</dd>
            </div>
            <div>
              <dt>Cost / passing query</dt>
              <dd>
                {money(agg.cost_per_passing_query_usd)}
                <span className="metrics-hint"> ({agg.n_passing} gated)</span>
              </dd>
            </div>
          </dl>
        ) : (
          <p className="muted">{loading ? "Loading…" : "No traces yet."}</p>
        )}
      </section>

      <section className="metrics-block">
        <h3>Provider share</h3>
        <p className="muted metrics-hint-block">
          Rising NVIDIA/Google share is an early signal Groq free-tier pressure (before hard 429s).
          Watch avg / p95 latency per provider — NIM thinking mode often shows 15s+ p95.
        </p>
        {providerEntries.length ? (
          <ul className="provider-share-list">
            {providerEntries.map(([name, row]) => (
              <li key={name}>
                <div className="provider-share-row">
                  <span className="provider-share-name">
                    {name}
                    {row.slow ? " ⚠ slow" : ""}
                  </span>
                  <span className="provider-share-meta">
                    {row.n} · {pct(row.share)} · {money(row.cost_usd)}
                    {row.n_cache_hit ? ` · ${row.n_cache_hit} cached` : ""}
                    {row.avg_latency_ms != null
                      ? ` · avg ${Math.round(row.avg_latency_ms)} ms`
                      : ""}
                    {row.p95_latency_ms != null
                      ? ` · p95 ${Math.round(row.p95_latency_ms)} ms`
                      : ""}
                  </span>
                </div>
                <div className="provider-share-bar" aria-hidden="true">
                  <span style={{ width: `${Math.max(2, Math.round((row.share || 0) * 100))}%` }} />
                </div>
              </li>
            ))}
          </ul>
        ) : (
          <p className="muted">No LLM calls recorded yet.</p>
        )}
      </section>

      <section className="metrics-block">
        <h3>Last question</h3>
        {trace ? (
          <>
            <p className="metrics-q">{trace.question}</p>
            <dl className="metrics-grid">
              <div>
                <dt>Provider</dt>
                <dd>
                  {trace.answer_provider || trace.provider || "—"}
                  {trace.target_index != null && trace.target_index > 0
                    ? ` (fallback #${trace.target_index})`
                    : ""}
                </dd>
              </div>
              <div>
                <dt>Model</dt>
                <dd>{trace.answer_model || trace.model || "—"}</dd>
              </div>
              <div>
                <dt>Tokens in</dt>
                <dd>{trace.input_tokens}</dd>
              </div>
              <div>
                <dt>Tokens out</dt>
                <dd>{trace.output_tokens}</dd>
              </div>
              <div>
                <dt>Cost</dt>
                <dd>{money(trace.cost_usd)}</dd>
              </div>
              <div>
                <dt>Latency</dt>
                <dd>{trace.latency_ms?.toFixed?.(0)} ms</dd>
              </div>
              <div>
                <dt>Cache</dt>
                <dd>
                  {(() => {
                    const status = (
                      trace.cache_status ||
                      (trace.served_from_cache || trace.answer_cached
                        ? "HIT"
                        : trace.cache_hit_kind) ||
                      ""
                    )
                      .toString()
                      .toUpperCase();
                    if (status.includes("HIT") || trace.served_from_cache || trace.answer_cached) {
                      return `hit${trace.cache_hit_kind ? `/${trace.cache_hit_kind}` : ""}`;
                    }
                    return status || "MISS";
                  })()}
                </dd>
              </div>
              <div>
                <dt>Context → LLM</dt>
                <dd>
                  {trace.context_chunks_to_llm ?? trace.chunk_ids?.length ?? "—"} chunks
                  {trace.context_tokens_est != null ? ` · ~${trace.context_tokens_est} tok` : ""}
                  {trace.context_budget_mode ? ` · ${trace.context_budget_mode}` : ""}
                </dd>
              </div>
              <div>
                <dt>Rerank</dt>
                <dd>
                  {trace.rerank_ran == null
                    ? "—"
                    : trace.rerank_ran
                      ? `yes (hybrid ${trace.n_hybrid_candidates ?? "?"} → ${trace.context_chunks_to_llm ?? "?"})`
                      : `cut only (hybrid ${trace.n_hybrid_candidates ?? "?"})`}
                </dd>
              </div>
              <div>
                <dt>Embed tokens</dt>
                <dd>{trace.embedding_tokens}</dd>
              </div>
            </dl>
            {trace.llm_calls?.length ? (
              <>
                <h4 className="metrics-sub">LLM calls</h4>
                <ul className="chunk-list">
                  {trace.llm_calls.map((c, i) => (
                    <li key={`${c.role}-${c.provider}-${i}`}>
                      <code>
                        {c.role}: {c.provider}/{c.model}
                      </code>
                      <span className="muted">
                        {" "}
                        · {c.input_tokens}/{c.output_tokens} tok · {money(c.cost_usd)} ·{" "}
                        {Math.round(c.latency_ms || 0)} ms · {c.cache_status || "—"}
                      </span>
                    </li>
                  ))}
                </ul>
              </>
            ) : (
              <p className="muted">
                {trace.rewrite_model ? `rewrite: ${trace.rewrite_provider || ""} ${trace.rewrite_model}` : ""}
              </p>
            )}
            <h4 className="metrics-sub">Retrieved chunks</h4>
            {trace.citations?.length ? (
              <ul className="chunk-list">
                {trace.citations.map((c) => (
                  <li key={c.chunk_id}>
                    <code>{c.citation || c.chunk_id}</code>
                    {c.score != null ? <span className="muted"> · {c.score.toFixed(3)}</span> : null}
                  </li>
                ))}
              </ul>
            ) : (
              <ul className="chunk-list">
                {(trace.chunk_ids || []).map((id) => (
                  <li key={id}>
                    <code>{id}</code>
                  </li>
                ))}
              </ul>
            )}
            <p className="metrics-trace-id">
              trace <code>{trace.trace_id}</code>
            </p>
          </>
        ) : (
          <p className="muted">Ask a question to see per-query cost & retrieved chunks.</p>
        )}
      </section>
    </aside>
  );
}
