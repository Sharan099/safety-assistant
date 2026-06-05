"use client";

import { useState } from "react";
import ReactMarkdown from "react-markdown";

type Doc = {
  id?: string;
  title?: string;
  regulation?: string;
  source?: string;
  rerank_score?: number;
};

type Citation = {
  marker: string;
  label: string;
  document?: string;
  full_title?: string;
  doc_type?: string;
  doc_type_label?: string;
  is_legal?: boolean;
  authority?: string;
  revision?: string;
  revision_verified?: boolean;
  page?: number | null;
  section?: string | null;
  snippet?: string;
};

type Flag = {
  type: string;
  regulation?: string;
  message: string;
};

type Grounding = {
  should_abstain?: boolean;
  confidence?: number;
  reason?: string;
};

type Message = {
  role: "user" | "assistant";
  content: string;
  docs?: Doc[];
  citations?: Citation[];
  flags?: Flag[];
  grounding?: Grounding;
  timing?: Record<string, number>;
  warnings?: string[];
};

function getApiBase() {
  if (process.env.NEXT_PUBLIC_API_URL) {
    return process.env.NEXT_PUBLIC_API_URL;
  }

  if (typeof window !== "undefined") {
    // Gateway mode: frontend and API are both served from :8080.
    if (window.location.port === "8080") {
      return "/api/v1";
    }
  }

  // Local dev mode: Next.js runs on :3000 and FastAPI runs on :8000.
  return "http://localhost:8000/api/v1";
}

export default function Home() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);

  function exportAnswer(msg: Message) {
    const lines: string[] = [msg.content, "", "Sources:"];
    (msg.citations || []).forEach((c) => {
      lines.push(
        `${c.marker} ${c.label} [${c.is_legal ? "Legal regulation" : c.doc_type_label || "Reference"}]`
      );
      if (c.snippet) lines.push(`    "${c.snippet}"`);
    });
    (msg.flags || []).forEach((f) => lines.push(`Note: ${f.message}`));
    navigator.clipboard?.writeText(lines.join("\n"));
  }

  async function send() {
    const q = input.trim();
    if (!q || loading) return;
    setInput("");
    setMessages((m) => [...m, { role: "user", content: q }]);
    setLoading(true);
    const apiBase = getApiBase();
    try {
      const res = await fetch(`${apiBase}/chat`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query: q }),
      });
      if (!res.ok) throw new Error(await res.text());
      const data = await res.json();
      setMessages((m) => [
        ...m,
        {
          role: "assistant",
          content: data.answer,
          docs: data.documents,
          citations: data.citations,
          flags: data.flags,
          grounding: data.grounding,
          timing: data.timing,
          warnings: data.warnings,
        },
      ]);
    } catch (e) {
      setMessages((m) => [
        ...m,
        {
          role: "assistant",
          content:
            e instanceof TypeError
              ? `Error: Could not reach the backend at ${apiBase}. Make sure FastAPI is running.`
              : `Error: ${e instanceof Error ? e.message : "Request failed"}`,
        },
      ]);
    } finally {
      setLoading(false);
    }
  }

  return (
    <main className="layout">
      <header className="header">
        <h1>PSA AI</h1>
        <p>Passive Safety Regulation Assistant — Hybrid RAG</p>
      </header>

      <section className="chat">
        {messages.length === 0 && (
          <div className="empty">
            <p>Ask about UN R14, UN R16, seat belt anchorages, crash tests…</p>
            <ul>
              <li>What are UN R14 anchorage strength requirements?</li>
              <li>Explain UN R16 restraint system approval tests.</li>
            </ul>
          </div>
        )}
        {messages.map((msg, i) => (
          <article
            key={i}
            className={`bubble ${msg.role === "user" ? "user" : "assistant"}`}
          >
            {msg.role === "assistant" ? (
              <ReactMarkdown>{msg.content}</ReactMarkdown>
            ) : (
              <p>{msg.content}</p>
            )}
            {msg.role === "assistant" &&
              msg.grounding?.should_abstain && (
                <div className="abstain">
                  Not answered from the corpus — retrieval confidence below
                  threshold
                  {msg.grounding.confidence != null &&
                    ` (${(msg.grounding.confidence * 100).toFixed(0)}%)`}
                  .
                </div>
              )}
            {msg.flags && msg.flags.length > 0 && (
              <div className="flags">
                {msg.flags.map((f, j) => (
                  <div key={j} className={`flag flag-${f.type}`}>
                    ⚑ {f.message}
                  </div>
                ))}
              </div>
            )}
            {msg.warnings && msg.warnings.length > 0 && (
              <div className="warnings">
                {msg.warnings.map((w, j) => (
                  <span key={j}>⚠ {w}</span>
                ))}
              </div>
            )}
            {msg.citations && msg.citations.length > 0 && (
              <details className="sources" open>
                <summary>Sources ({msg.citations.length})</summary>
                <ul className="citation-list">
                  {msg.citations.map((c, j) => (
                    <li key={j} className="citation">
                      <span className="cmarker">[{c.marker}]</span>
                      <span
                        className={`badge ${
                          c.is_legal ? "badge-legal" : "badge-rating"
                        }`}
                        title={c.doc_type_label}
                      >
                        {c.is_legal ? "Legal" : c.doc_type === "rating_protocol" ? "Rating" : "Ref"}
                      </span>
                      <span className="clabel">{c.label}</span>
                      {!c.revision_verified && (
                        <span className="badge badge-unverified" title="Revision not verified">
                          rev?
                        </span>
                      )}
                      {c.snippet && <p className="csnippet">{c.snippet}</p>}
                    </li>
                  ))}
                </ul>
              </details>
            )}
            {(!msg.citations || msg.citations.length === 0) &&
              msg.docs &&
              msg.docs.length > 0 && (
                <details className="sources">
                  <summary>Sources ({msg.docs.length})</summary>
                  <ul>
                    {msg.docs.map((d, j) => (
                      <li key={j}>
                        [{d.regulation}] {d.title} — {d.source}
                        {d.rerank_score != null &&
                          ` (score: ${d.rerank_score.toFixed(3)})`}
                      </li>
                    ))}
                  </ul>
                </details>
              )}
            {msg.role === "assistant" &&
              msg.citations &&
              msg.citations.length > 0 && (
                <button
                  className="export"
                  onClick={() => exportAnswer(msg)}
                  type="button"
                >
                  Copy answer + sources
                </button>
              )}
            {msg.timing && (
              <small className="timing">
                {msg.timing.total_ms != null && `${msg.timing.total_ms} ms total`}
                {msg.timing.retrieval_ms != null &&
                  ` · retrieval ${msg.timing.retrieval_ms} ms`}
                {msg.timing.llm_ms != null && ` · LLM ${msg.timing.llm_ms} ms`}
              </small>
            )}
          </article>
        ))}
        {loading && (
          <p className="loading">
            Retrieving regulations and calling Groq… (first request can take up to
            60s while models load)
          </p>
        )}
      </section>

      <footer className="composer">
        <textarea
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter" && !e.shiftKey) {
              e.preventDefault();
              send();
            }
          }}
          placeholder="Ask a passive safety question…"
          rows={2}
        />
        <button onClick={send} disabled={loading || !input.trim()}>
          Send
        </button>
      </footer>

      <style jsx>{`
        .layout {
          max-width: 900px;
          margin: 0 auto;
          min-height: 100vh;
          display: flex;
          flex-direction: column;
          padding: 1rem;
        }
        .header h1 {
          margin: 0;
          font-size: 1.5rem;
        }
        .header p {
          margin: 0.25rem 0 1rem;
          color: var(--muted);
          font-size: 0.9rem;
        }
        .chat {
          flex: 1;
          overflow-y: auto;
          display: flex;
          flex-direction: column;
          gap: 1rem;
        }
        .empty {
          color: var(--muted);
          padding: 2rem 0;
        }
        .empty ul {
          padding-left: 1.2rem;
        }
        .bubble {
          padding: 1rem 1.25rem;
          border-radius: 12px;
          line-height: 1.5;
        }
        .bubble.user {
          background: var(--accent);
          color: #fff;
          align-self: flex-end;
          max-width: 85%;
        }
        .bubble.assistant {
          background: var(--surface);
          border: 1px solid var(--border);
        }
        .warnings {
          margin-top: 0.75rem;
          display: flex;
          flex-direction: column;
          gap: 0.25rem;
          color: var(--warn);
          font-size: 0.85rem;
        }
        .abstain {
          margin-top: 0.75rem;
          padding: 0.5rem 0.75rem;
          border-radius: 8px;
          background: rgba(245, 158, 11, 0.12);
          border: 1px solid rgba(245, 158, 11, 0.4);
          color: var(--warn);
          font-size: 0.85rem;
        }
        .flags {
          margin-top: 0.75rem;
          display: flex;
          flex-direction: column;
          gap: 0.4rem;
        }
        .flag {
          padding: 0.5rem 0.75rem;
          border-radius: 8px;
          font-size: 0.82rem;
          background: rgba(59, 130, 246, 0.1);
          border: 1px solid rgba(59, 130, 246, 0.35);
        }
        .flag-mixed_doc_types {
          background: rgba(245, 158, 11, 0.12);
          border-color: rgba(245, 158, 11, 0.45);
        }
        .citation-list {
          list-style: none;
          padding: 0;
          margin: 0.5rem 0 0;
          display: flex;
          flex-direction: column;
          gap: 0.6rem;
        }
        .citation {
          padding: 0.5rem 0.6rem;
          border-radius: 8px;
          border: 1px solid var(--border);
          background: var(--surface);
        }
        .cmarker {
          font-weight: 700;
          margin-right: 0.4rem;
        }
        .badge {
          display: inline-block;
          padding: 0.05rem 0.4rem;
          border-radius: 6px;
          font-size: 0.7rem;
          font-weight: 700;
          margin-right: 0.4rem;
          vertical-align: middle;
        }
        .badge-legal {
          background: #16a34a;
          color: #fff;
        }
        .badge-rating {
          background: #f97316;
          color: #fff;
        }
        .badge-unverified {
          background: #6b7280;
          color: #fff;
        }
        .clabel {
          font-weight: 600;
        }
        .csnippet {
          margin: 0.4rem 0 0;
          color: var(--muted);
          font-size: 0.8rem;
          line-height: 1.4;
        }
        .export {
          margin-top: 0.6rem;
          padding: 0.35rem 0.7rem;
          font-size: 0.78rem;
          border: 1px solid var(--border);
          border-radius: 6px;
          background: transparent;
          color: var(--text);
          cursor: pointer;
        }
        .sources {
          margin-top: 0.75rem;
          font-size: 0.85rem;
          color: var(--muted);
        }
        .timing {
          display: block;
          margin-top: 0.5rem;
          color: var(--muted);
        }
        .loading {
          color: var(--muted);
          font-style: italic;
        }
        .composer {
          display: flex;
          gap: 0.5rem;
          padding-top: 1rem;
          border-top: 1px solid var(--border);
        }
        .composer textarea {
          flex: 1;
          resize: none;
          padding: 0.75rem;
          border-radius: 8px;
          border: 1px solid var(--border);
          background: var(--surface);
          color: var(--text);
        }
        .composer button {
          padding: 0 1.25rem;
          border: none;
          border-radius: 8px;
          background: var(--accent);
          color: #fff;
          font-weight: 600;
        }
        .composer button:disabled {
          opacity: 0.5;
        }
      `}</style>
    </main>
  );
}
