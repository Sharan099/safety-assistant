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

type Message = {
  role: "user" | "assistant";
  content: string;
  docs?: Doc[];
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
            {msg.warnings && msg.warnings.length > 0 && (
              <div className="warnings">
                {msg.warnings.map((w, j) => (
                  <span key={j}>⚠ {w}</span>
                ))}
              </div>
            )}
            {msg.docs && msg.docs.length > 0 && (
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
