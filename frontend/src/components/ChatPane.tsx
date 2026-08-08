"use client";

import { FormEvent, useEffect, useRef, useState } from "react";
import { CitationRichText } from "@/components/CitationRichText";
import { enrichCitation, fetchRegulations, runAgent, streamChat } from "@/lib/api";
import type { IndexedRegulation } from "@/lib/api";
import type { ChatMessage, Citation, ComplianceCriterion, QueryMetrics } from "@/lib/types";

const SUGGESTIONS = [
  "What is the HIC15 limit in R94?",
  "Compare R94 vs R95 chest deflection limits",
  "Gap analysis: our test setup vs R95 requirements",
];

type Props = {
  onOpenCitation: (c: Citation) => void;
  onTrace?: (traceId: string, metrics?: QueryMetrics) => void;
  regsRefreshKey?: number;
};

export function ChatPane({ onOpenCitation, onTrace, regsRefreshKey = 0 }: Props) {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [status, setStatus] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [agentMode, setAgentMode] = useState(false);
  const [indexedRegs, setIndexedRegs] = useState<IndexedRegulation[]>([]);
  const bottomRef = useRef<HTMLDivElement>(null);
  const conversationIdRef = useRef<string>(
    typeof crypto !== "undefined" && "randomUUID" in crypto
      ? crypto.randomUUID().replace(/-/g, "")
      : `c${Date.now().toString(16)}`
  );

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, loading]);

  useEffect(() => {
    let cancelled = false;
    void fetchRegulations().then((rows) => {
      if (!cancelled) setIndexedRegs(rows);
    });
    return () => {
      cancelled = true;
    };
  }, [regsRefreshKey]);

  async function send(question: string) {
    const q = question.trim();
    if (!q || loading) return;
    if (q.length < 3) {
      setError("Question is too short — ask a concrete regulation question.");
      return;
    }
    setError(null);
    setLoading(true);
    setStatus(agentMode ? "Agent planning…" : "Retrieving…");
    const userMsg: ChatMessage = {
      id: `u-${Date.now()}`,
      role: "user",
      content: q,
    };
    const assistantId = `a-${Date.now()}`;
    setMessages((m) => [
      ...m,
      userMsg,
      { id: assistantId, role: "assistant", content: "", citations: [], streaming: true },
    ]);
    setInput("");

    try {
      if (agentMode) {
        const result = await runAgent(q);
        const body =
          result.report_markdown ||
          (result.table_markdown
            ? `${result.answer}\n\n${result.table_markdown}`
            : result.answer);
        setMessages((m) =>
          m.map((msg) =>
            msg.id === assistantId
              ? {
                  ...msg,
                  content: body,
                  citations: result.sources,
                  streaming: false,
                  provider: `agent:${result.mode}`,
                  model: result.multi_step
                    ? `multi-step cov=${result.overall_citation_coverage.toFixed(2)}`
                    : `cov=${result.overall_citation_coverage.toFixed(2)}`,
                  trace_id: result.trace_id,
                  not_found: result.not_found,
                  query_intent: result.query_intent,
                  execution_layer: result.execution_layer,
                  multi_step: result.multi_step,
                  mode_disclaimer: result.mode_disclaimer,
                  mode_disclaimer_title: result.mode_disclaimer_title,
                }
              : msg
          )
        );
        if (result.trace_id) onTrace?.(result.trace_id);
        setStatus(null);
        return;
      }

      await streamChat(
        q,
        {
        onStatus: (stage) => setStatus(stage === "retrieving" ? "Retrieving passages…" : stage),
        onCitations: (citations) => {
          setStatus("Generating answer…");
          setMessages((m) =>
            m.map((msg) => (msg.id === assistantId ? { ...msg, citations } : msg))
          );
        },
        onToken: (text) => {
          setMessages((m) =>
            m.map((msg) =>
              msg.id === assistantId ? { ...msg, content: (msg.content || "") + text } : msg
            )
          );
        },
        onNotFound: (_message, _traceId, failureKind) => {
          setMessages((m) =>
            m.map((msg) =>
              msg.id === assistantId
                ? {
                    ...msg,
                    not_found: true,
                    failure_kind:
                      failureKind === "grounding_rejected" ||
                      failureKind === "retrieval_miss" ||
                      failureKind === "numeric_hallucination"
                        ? failureKind
                        : "retrieval_miss",
                  }
                : msg
            )
          );
        },
        onDone: (payload) => {
          if (payload.conversation_id) {
            conversationIdRef.current = payload.conversation_id;
          }
          setMessages((m) =>
            m.map((msg) =>
              msg.id === assistantId
                ? {
                    ...msg,
                    content: payload.answer || msg.content,
                    citations: payload.citations || msg.citations,
                    streaming: false,
                    model: payload.model,
                    provider: payload.provider,
                    trace_id: payload.trace_id,
                    metrics: payload.metrics,
                    not_found: payload.not_found,
                    compliance: payload.compliance || null,
                    query_intent: payload.query_intent || null,
                    execution_layer: payload.execution_layer || null,
                    multi_step: Boolean(payload.multi_step),
                    mode_disclaimer: payload.mode_disclaimer || null,
                    mode_disclaimer_title: payload.mode_disclaimer_title || null,
                    failure_kind:
                      payload.failure_kind === "grounding_rejected" ||
                      payload.failure_kind === "retrieval_miss" ||
                      payload.failure_kind === "numeric_hallucination"
                        ? payload.failure_kind
                        : payload.not_found
                          ? "retrieval_miss"
                          : null,
                  }
                : msg
            )
          );
          if (payload.trace_id) onTrace?.(payload.trace_id, payload.metrics);
          setStatus(null);
        },
        onError: (message) => {
          setError(message);
          setMessages((m) =>
            m.map((msg) =>
              msg.id === assistantId
                ? {
                    ...msg,
                    content: "Something went wrong generating an answer.",
                    streaming: false,
                    error: true,
                  }
                : msg
            )
          );
        },
        },
        { conversation_id: conversationIdRef.current }
      );
    } catch (err) {
      const message = err instanceof Error ? err.message : "Request failed";
      setError(message);
      setMessages((m) =>
        m.map((msg) =>
          msg.id === assistantId
            ? {
                ...msg,
                content: "Could not reach the API. Check that the backend is running.",
                streaming: false,
                error: true,
              }
            : msg
        )
      );
    } finally {
      setLoading(false);
      setStatus(null);
    }
  }

  function onSubmit(e: FormEvent) {
    e.preventDefault();
    void send(input);
  }

  async function handleCite(c: Citation) {
    const enriched = await enrichCitation(c);
    onOpenCitation(enriched);
  }

  return (
    <section className="chat-pane">
      <header className="chat-header">
        <div>
          <div className="brand-mark">Passive Safety</div>
          <h1>Regulation copilot</h1>
        </div>
        <div className="badges">
          <button
            type="button"
            className={`badge ${agentMode ? "badge--active" : "badge--soft"}`}
            onClick={() => setAgentMode((v) => !v)}
            title="Multi-step agent: compare / report / retrieve"
          >
            {agentMode ? "Agent on" : "Agent off"}
          </button>
        </div>
      </header>
      <div className="indexed-regs" aria-label="Indexed regulations">
        {indexedRegs.length === 0 ? (
          <span className="indexed-regs__empty">No regulations indexed yet</span>
        ) : (
          indexedRegs.map((r) => (
            <span
              key={`${r.regulation_id}:${r.revision}`}
              className="indexed-chip"
              title={`${r.regulation_id} · ${r.chunk_count} chunks`}
            >
              {r.label}
              <em>{r.chunk_count}</em>
            </span>
          ))
        )}
      </div>

      <div className="thread">
        {messages.length === 0 ? (
          <div className="empty-thread">
            <h2>Ask a grounded regulation question</h2>
            <p>Answers cite clause + page. Click a chip to open the PDF highlight.</p>
            <div className="suggestions">
              {SUGGESTIONS.map((s) => (
                <button key={s} type="button" className="suggestion" onClick={() => void send(s)}>
                  {s}
                </button>
              ))}
            </div>
          </div>
        ) : (
          messages.map((msg) => (
            <article
              key={msg.id}
              className={`bubble bubble--${msg.role}${msg.not_found ? " bubble--not-found" : ""}${
                msg.error ? " bubble--error" : ""
              }`}
            >
              <div className="bubble__role">{msg.role === "user" ? "You" : "Assistant"}</div>
              {msg.not_found ? (
                <div className="not-found-banner">
                  {msg.failure_kind === "grounding_rejected"
                    ? "Related content found — answer not confidently grounded"
                    : msg.failure_kind === "numeric_hallucination"
                      ? "Please confirm your measured value"
                      : "Not found in indexed regulations"}
                </div>
              ) : null}
              {msg.role === "assistant" && msg.mode_disclaimer ? (
                <aside className="mode-disclaimer" role="note">
                  <div className="mode-disclaimer__title">
                    {msg.mode_disclaimer_title || "Verify with your homologation authority"}
                  </div>
                  <p className="mode-disclaimer__body">{msg.mode_disclaimer}</p>
                </aside>
              ) : null}
              {msg.role === "assistant" ? (
                <CitationRichText
                  text={msg.content || (msg.streaming ? "…" : "")}
                  citations={msg.citations || []}
                  onCite={(c) => void handleCite(c)}
                />
              ) : (
                <p>{msg.content}</p>
              )}
              {msg.role === "assistant" && msg.compliance?.criteria?.length ? (
                <div className="compliance-panel">
                  <div className="compliance-panel__overall">
                    Overall: <strong>{msg.compliance.overall_verdict}</strong>
                  </div>
                  <table className="compliance-table">
                    <thead>
                      <tr>
                        <th>Criterion</th>
                        <th>Measured</th>
                        <th>Limit</th>
                        <th>Verdict</th>
                      </tr>
                    </thead>
                    <tbody>
                      {msg.compliance.criteria.map((row: ComplianceCriterion, idx: number) => (
                        <tr key={`${row.criterion}-${idx}`}>
                          <td>{row.criterion}</td>
                          <td>
                            {row.measured_display ?? row.measured}
                            {row.unit ? ` ${row.unit}` : ""}
                          </td>
                          <td>
                            {row.verdict === "LIMIT_NOT_FOUND"
                              ? "—"
                              : `${row.operator || "≤"} ${row.limit_display ?? row.limit ?? "—"}${
                                  row.unit ? ` ${row.unit}` : ""
                                }`}
                          </td>
                          <td>{row.verdict}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              ) : null}
              {msg.role === "assistant" && msg.citations && msg.citations.length > 0 ? (
                <div className="cite-row">
                  {msg.citations.slice(0, 6).map((c) => (
                    <button
                      key={c.chunk_id}
                      type="button"
                      className="cite-chip"
                      onClick={() => void handleCite(c)}
                    >
                      {c.label || c.citation}
                    </button>
                  ))}
                </div>
              ) : null}
              {msg.metrics ? (
                <div className="bubble__meta">
                  {msg.metrics.provider || msg.provider || "—"}
                  {msg.metrics.model || msg.model
                    ? ` · ${msg.metrics.model || msg.model}`
                    : ""}{" "}
                  · {msg.metrics.input_tokens}/{msg.metrics.output_tokens} tok · $
                  {Number(msg.metrics.cost_usd || 0).toFixed(6)} ·{" "}
                  {Math.round(msg.metrics.latency_ms || 0)} ms
                  {msg.metrics.served_from_cache ||
                  msg.metrics.answer_cached ||
                  (msg.metrics.cache_status || "").toUpperCase().includes("HIT")
                    ? ` · cache ${msg.metrics.cache_status || msg.metrics.cache_hit_kind || "HIT"}`
                    : msg.metrics.cache_status
                      ? ` · cache ${msg.metrics.cache_status}`
                      : ""}
                </div>
              ) : msg.provider ? (
                <div className="bubble__meta">
                  {msg.provider}
                  {msg.model ? ` · ${msg.model}` : ""}
                </div>
              ) : null}
            </article>
          ))
        )}
        <div ref={bottomRef} />
      </div>

      {error ? (
        <div className="banner banner--error" role="alert">
          {error}
          <button type="button" className="banner-dismiss" onClick={() => setError(null)}>
            Dismiss
          </button>
        </div>
      ) : null}
      {status ? <div className="banner">{status}</div> : null}

      <form className="composer" onSubmit={onSubmit}>
        <input
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder="Ask about injury criteria, belts, CRS…"
          disabled={loading}
          aria-label="Question"
          maxLength={4000}
        />
        <button type="submit" className="send" disabled={loading || !input.trim()}>
          Send
        </button>
      </form>
    </section>
  );
}
