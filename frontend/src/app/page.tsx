"use client";

import { useCallback, useEffect, useRef, useState } from "react";
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

type Flag = { type: string; regulation?: string; message: string };
type Grounding = { should_abstain?: boolean; confidence?: number; reason?: string };

type Feedback = {
  rating?: "up" | "down";
  panelOpen?: boolean;
  reasons: string[];
  comment: string;
  submitted?: boolean;
};

type Message = {
  role: "user" | "assistant";
  content: string;
  messageId?: string;
  docs?: Doc[];
  citations?: Citation[];
  flags?: Flag[];
  grounding?: Grounding;
  timing?: Record<string, number>;
  warnings?: string[];
  feedback?: Feedback;
};

type User = { user_id: string; username: string; session_id: string };
type Ready = {
  ready?: boolean;
  retrieval_ok?: boolean;
  llm_ok?: boolean;
  llm_configured?: boolean;
  detail?: string;
};

const PROBLEM_OPTIONS = [
  "Incorrect or made-up answer (hallucination)",
  "Missing or incomplete information",
  "Wrong, irrelevant, or missing sources / citations",
  "Not grounded in the regulation, or wrong revision",
  "Confusing, unclear, or badly formatted response",
];

const USER_KEY = "psa_user";

function getApiBase() {
  if (process.env.NEXT_PUBLIC_API_URL) return process.env.NEXT_PUBLIC_API_URL;
  if (typeof window !== "undefined" && window.location.port === "8080") {
    return "/api/v1";
  }
  return "http://localhost:8000/api/v1";
}

export default function Home() {
  const [user, setUser] = useState<User | null>(null);
  const [nameInput, setNameInput] = useState("");
  const [nameError, setNameError] = useState("");
  const [registering, setRegistering] = useState(false);

  const [ready, setReady] = useState<Ready | null>(null);
  const [readyError, setReadyError] = useState(false);

  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const chatEndRef = useRef<HTMLDivElement | null>(null);

  const apiBase = getApiBase();

  // Load saved user (session memory) on first paint.
  useEffect(() => {
    try {
      const raw = localStorage.getItem(USER_KEY);
      if (raw) setUser(JSON.parse(raw));
    } catch {
      /* ignore */
    }
  }, []);

  // Poll readiness until the backend self-test passes.
  const pollReady = useCallback(async () => {
    try {
      const res = await fetch(`${apiBase}/ready`);
      if (!res.ok) throw new Error("not ready");
      const data: Ready = await res.json();
      setReady(data);
      setReadyError(false);
      return Boolean(data.ready);
    } catch {
      setReadyError(true);
      return false;
    }
  }, [apiBase]);

  useEffect(() => {
    let active = true;
    let timer: ReturnType<typeof setTimeout>;
    const loop = async () => {
      const ok = await pollReady();
      if (active && !ok) timer = setTimeout(loop, 2500);
    };
    loop();
    return () => {
      active = false;
      clearTimeout(timer);
    };
  }, [pollReady]);

  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, loading]);

  async function registerUser() {
    const username = nameInput.trim();
    if (username.length < 2) {
      setNameError("Please enter at least 2 characters.");
      return;
    }
    setRegistering(true);
    setNameError("");
    try {
      const res = await fetch(`${apiBase}/users`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ username }),
      });
      if (!res.ok) {
        const t = await res.text();
        throw new Error(t || "Could not register");
      }
      const data = await res.json();
      const u: User = {
        user_id: data.user_id,
        username: data.username,
        session_id: data.session_id,
      };
      localStorage.setItem(USER_KEY, JSON.stringify(u));
      setUser(u);
    } catch (e) {
      setNameError(
        e instanceof TypeError
          ? "Could not reach the backend. Please try again in a moment."
          : "Could not register that name. Try a different one."
      );
    } finally {
      setRegistering(false);
    }
  }

  function resetUser() {
    localStorage.removeItem(USER_KEY);
    setUser(null);
    setMessages([]);
  }

  const canChat = Boolean(user) && Boolean(ready?.ready) && !loading;

  async function send() {
    const q = input.trim();
    if (!q || !canChat) return;
    setInput("");
    setMessages((m) => [...m, { role: "user", content: q }]);
    setLoading(true);
    try {
      const res = await fetch(`${apiBase}/chat`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          query: q,
          user_id: user?.user_id,
          session_id: user?.session_id,
        }),
      });
      if (!res.ok) throw new Error(await res.text());
      const data = await res.json();
      setMessages((m) => [
        ...m,
        {
          role: "assistant",
          content: data.answer,
          messageId: data.message_id,
          docs: data.documents,
          citations: data.citations,
          flags: data.flags,
          grounding: data.grounding,
          timing: data.timing,
          warnings: data.warnings,
          feedback: { reasons: [], comment: "" },
        },
      ]);
    } catch (e) {
      setMessages((m) => [
        ...m,
        {
          role: "assistant",
          content:
            e instanceof TypeError
              ? `Error: Could not reach the backend at ${apiBase}.`
              : `Error: ${e instanceof Error ? e.message : "Request failed"}`,
        },
      ]);
    } finally {
      setLoading(false);
    }
  }

  function updateFeedback(idx: number, patch: Partial<Feedback>) {
    setMessages((m) =>
      m.map((msg, i) =>
        i === idx
          ? { ...msg, feedback: { ...(msg.feedback || { reasons: [], comment: "" }), ...patch } }
          : msg
      )
    );
  }

  async function sendFeedback(idx: number, rating: "up" | "down") {
    const msg = messages[idx];
    const fb = msg.feedback || { reasons: [], comment: "" };
    try {
      await fetch(`${apiBase}/feedback`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          rating,
          message_id: msg.messageId,
          user_id: user?.user_id,
          session_id: user?.session_id,
          reasons: rating === "down" ? fb.reasons : [],
          comment: rating === "down" ? fb.comment : "",
          query: idx > 0 ? messages[idx - 1]?.content : "",
          answer: msg.content,
        }),
      });
      updateFeedback(idx, { submitted: true, panelOpen: false, rating });
    } catch {
      updateFeedback(idx, { submitted: true, panelOpen: false, rating });
    }
  }

  function onThumb(idx: number, rating: "up" | "down") {
    updateFeedback(idx, { rating });
    if (rating === "up") {
      sendFeedback(idx, "up");
    } else {
      updateFeedback(idx, { panelOpen: true });
    }
  }

  function toggleReason(idx: number, reason: string) {
    const fb = messages[idx].feedback || { reasons: [], comment: "" };
    const reasons = fb.reasons.includes(reason)
      ? fb.reasons.filter((r) => r !== reason)
      : [...fb.reasons, reason];
    updateFeedback(idx, { reasons });
  }

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

  // ───────────── username onboarding modal ─────────────
  if (!user) {
    return (
      <main className="onboard">
        <div className="card">
          <h1>PSA AI</h1>
          <p className="tag">Passive Safety Regulation Assistant — Hybrid RAG</p>
          <div className="testing">
            🧪 This system is in a <strong>testing phase</strong>. We are currently
            evaluating it with a small group of users. Your questions and feedback
            are stored to help us improve.
          </div>
          <label htmlFor="buddy">Pick a buddy name to get started</label>
          <input
            id="buddy"
            value={nameInput}
            onChange={(e) => setNameInput(e.target.value)}
            onKeyDown={(e) => e.key === "Enter" && registerUser()}
            placeholder="e.g. crash_buddy"
            maxLength={40}
            autoFocus
          />
          {nameError && <p className="err">{nameError}</p>}
          <button onClick={registerUser} disabled={registering}>
            {registering ? "Setting up…" : "Start chatting"}
          </button>
          <p className="fine">
            No email or password needed. We only store the name you choose, your
            chats, and any feedback you give — to improve the assistant.
          </p>
        </div>
        <style jsx>{onboardCss}</style>
      </main>
    );
  }

  const warming = !ready?.ready;

  return (
    <main className="layout">
      <header className="header">
        <div>
          <h1>PSA AI</h1>
          <p>Passive Safety Regulation Assistant — Hybrid RAG</p>
        </div>
        <div className="who">
          <span>👤 {user.username}</span>
          <button className="link" onClick={resetUser}>
            switch user
          </button>
        </div>
      </header>

      <div className="testing-banner">
        🧪 Testing phase — we&apos;re trialling this with a few users. Answers may
        be imperfect; please rate them so we can improve.
      </div>

      {warming && (
        <div className={`status ${readyError ? "status-err" : ""}`}>
          {readyError
            ? "Waiting for the backend… make sure the API is running. Retrying…"
            : "Warming up the model and running a self-test query… you can chat as soon as this turns green."}
        </div>
      )}
      {ready?.ready && ready.llm_configured === false && (
        <div className="status status-warn">
          Heads up: no LLM API key is configured, so answers may be limited to
          retrieved snippets.
        </div>
      )}

      <section className="chat">
        {messages.length === 0 && (
          <div className="empty">
            <h2>What this assistant does</h2>
            <p>
              It answers questions about passive-safety regulations (currently
              <strong> UN R14</strong> and <strong>UN R16</strong>) using only the
              indexed regulation text. Every answer shows its sources with the
              document, clause/section, and revision. If it isn&apos;t confident,
              it will say <em>&quot;I don&apos;t know&quot;</em> rather than guess.
            </p>
            <ul>
              <li>What are the UN R14 anchorage strength requirements?</li>
              <li>Explain UN R16 dynamic test requirements for belt assemblies.</li>
              <li>What test load applies to belt anchorages for M1 vehicles?</li>
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

            {msg.role === "assistant" && msg.grounding?.should_abstain && (
              <div className="abstain">
                Not answered from the corpus — retrieval confidence below threshold
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
                        className={`badge ${c.is_legal ? "badge-legal" : "badge-rating"}`}
                        title={c.doc_type_label}
                      >
                        {c.is_legal
                          ? "Legal"
                          : c.doc_type === "rating_protocol"
                          ? "Rating"
                          : "Ref"}
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

            {msg.timing && (
              <small className="timing">
                {msg.timing.total_ms != null && `${msg.timing.total_ms} ms total`}
                {msg.timing.retrieval_ms != null &&
                  ` · retrieval ${msg.timing.retrieval_ms} ms`}
                {msg.timing.llm_ms != null && ` · LLM ${msg.timing.llm_ms} ms`}
              </small>
            )}

            {/* Feedback */}
            {msg.role === "assistant" && msg.messageId && (
              <div className="feedback">
                {msg.feedback?.submitted ? (
                  <span className="fb-thanks">
                    ✓ Thanks for the feedback — it helps us improve.
                  </span>
                ) : (
                  <div className="fb-row">
                    <span className="fb-q">Was this answer helpful?</span>
                    <button
                      className={`thumb ${msg.feedback?.rating === "up" ? "active" : ""}`}
                      onClick={() => onThumb(i, "up")}
                      aria-label="Thumbs up"
                    >
                      👍
                    </button>
                    <button
                      className={`thumb ${msg.feedback?.rating === "down" ? "active" : ""}`}
                      onClick={() => onThumb(i, "down")}
                      aria-label="Thumbs down"
                    >
                      👎
                    </button>
                    <button className="export" onClick={() => exportAnswer(msg)}>
                      Copy answer + sources
                    </button>
                  </div>
                )}

                {msg.feedback?.panelOpen && !msg.feedback?.submitted && (
                  <div className="fb-panel">
                    <p className="fb-title">What went wrong? (select any)</p>
                    {PROBLEM_OPTIONS.map((opt) => (
                      <label key={opt} className="fb-opt">
                        <input
                          type="checkbox"
                          checked={msg.feedback?.reasons.includes(opt) || false}
                          onChange={() => toggleReason(i, opt)}
                        />
                        {opt}
                      </label>
                    ))}
                    <textarea
                      className="fb-comment"
                      placeholder="Tell us what you expected, or paste the correct answer…"
                      value={msg.feedback?.comment || ""}
                      maxLength={4000}
                      onChange={(e) => updateFeedback(i, { comment: e.target.value })}
                      rows={3}
                    />
                    <div className="fb-actions">
                      <button className="fb-cancel" onClick={() => updateFeedback(i, { panelOpen: false })}>
                        Cancel
                      </button>
                      <button className="fb-submit" onClick={() => sendFeedback(i, "down")}>
                        Submit feedback
                      </button>
                    </div>
                  </div>
                )}
              </div>
            )}
          </article>
        ))}

        {loading && (
          <p className="loading">
            Retrieving regulations and generating a grounded answer…
          </p>
        )}
        <div ref={chatEndRef} />
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
          placeholder={
            warming
              ? "Please wait — the assistant is warming up…"
              : "Ask a passive safety question…"
          }
          rows={2}
          disabled={!canChat}
        />
        <button onClick={send} disabled={!canChat || !input.trim()}>
          {warming ? "Warming up…" : "Send"}
        </button>
      </footer>

      <style jsx>{appCss}</style>
    </main>
  );
}

const onboardCss = `
  .onboard {
    min-height: 100vh;
    display: flex;
    align-items: center;
    justify-content: center;
    padding: 1rem;
  }
  .card {
    width: 100%;
    max-width: 440px;
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 2rem;
    display: flex;
    flex-direction: column;
    gap: 0.75rem;
  }
  .card h1 { margin: 0; font-size: 1.8rem; }
  .tag { margin: 0; color: var(--muted); font-size: 0.9rem; }
  .testing {
    margin: 0.5rem 0;
    padding: 0.75rem;
    border-radius: 10px;
    background: rgba(245, 158, 11, 0.12);
    border: 1px solid rgba(245, 158, 11, 0.4);
    color: var(--text);
    font-size: 0.85rem;
    line-height: 1.45;
  }
  label { font-size: 0.9rem; font-weight: 600; margin-top: 0.5rem; }
  input {
    padding: 0.75rem;
    border-radius: 8px;
    border: 1px solid var(--border);
    background: var(--bg);
    color: var(--text);
    font-size: 1rem;
  }
  .err { color: var(--danger); font-size: 0.85rem; margin: 0; }
  button {
    margin-top: 0.5rem;
    padding: 0.8rem;
    border: none;
    border-radius: 8px;
    background: var(--accent);
    color: #fff;
    font-weight: 600;
    font-size: 1rem;
  }
  button:disabled { opacity: 0.5; }
  .fine { color: var(--muted); font-size: 0.78rem; line-height: 1.4; margin: 0.25rem 0 0; }
`;

const appCss = `
  .layout {
    max-width: 920px;
    margin: 0 auto;
    min-height: 100vh;
    display: flex;
    flex-direction: column;
    padding: 1rem;
  }
  .header {
    display: flex;
    justify-content: space-between;
    align-items: flex-start;
  }
  .header h1 { margin: 0; font-size: 1.5rem; }
  .header p { margin: 0.25rem 0 0.5rem; color: var(--muted); font-size: 0.9rem; }
  .who { display: flex; flex-direction: column; align-items: flex-end; gap: 0.2rem; font-size: 0.85rem; }
  .link {
    background: none; border: none; color: var(--accent);
    font-size: 0.78rem; padding: 0; text-decoration: underline;
  }
  .testing-banner {
    margin-bottom: 0.5rem;
    padding: 0.5rem 0.75rem;
    border-radius: 8px;
    background: rgba(245, 158, 11, 0.12);
    border: 1px solid rgba(245, 158, 11, 0.4);
    font-size: 0.82rem;
  }
  .status {
    margin-bottom: 0.5rem;
    padding: 0.5rem 0.75rem;
    border-radius: 8px;
    background: rgba(59, 130, 246, 0.12);
    border: 1px solid rgba(59, 130, 246, 0.4);
    font-size: 0.83rem;
  }
  .status-err { background: rgba(239,68,68,0.12); border-color: rgba(239,68,68,0.45); }
  .status-warn { background: rgba(245,158,11,0.12); border-color: rgba(245,158,11,0.4); }
  .chat { flex: 1; overflow-y: auto; display: flex; flex-direction: column; gap: 1rem; }
  .empty { color: var(--muted); padding: 1rem 0; }
  .empty h2 { color: var(--text); font-size: 1.1rem; margin: 0 0 0.5rem; }
  .empty p { line-height: 1.5; }
  .empty ul { padding-left: 1.2rem; }
  .empty li { margin: 0.25rem 0; }
  .bubble { padding: 1rem 1.25rem; border-radius: 12px; line-height: 1.5; }
  .bubble.user { background: var(--accent); color: #fff; align-self: flex-end; max-width: 85%; }
  .bubble.assistant { background: var(--surface); border: 1px solid var(--border); }
  .abstain {
    margin-top: 0.75rem; padding: 0.5rem 0.75rem; border-radius: 8px;
    background: rgba(245,158,11,0.12); border: 1px solid rgba(245,158,11,0.4);
    color: var(--warn); font-size: 0.85rem;
  }
  .flags { margin-top: 0.75rem; display: flex; flex-direction: column; gap: 0.4rem; }
  .flag {
    padding: 0.5rem 0.75rem; border-radius: 8px; font-size: 0.82rem;
    background: rgba(59,130,246,0.1); border: 1px solid rgba(59,130,246,0.35);
  }
  .flag-mixed_doc_types { background: rgba(245,158,11,0.12); border-color: rgba(245,158,11,0.45); }
  .warnings { margin-top: 0.75rem; display: flex; flex-direction: column; gap: 0.25rem; color: var(--warn); font-size: 0.85rem; }
  .sources { margin-top: 0.75rem; font-size: 0.85rem; color: var(--muted); }
  .citation-list { list-style: none; padding: 0; margin: 0.5rem 0 0; display: flex; flex-direction: column; gap: 0.6rem; }
  .citation { padding: 0.5rem 0.6rem; border-radius: 8px; border: 1px solid var(--border); background: var(--bg); }
  .cmarker { font-weight: 700; margin-right: 0.4rem; }
  .badge { display: inline-block; padding: 0.05rem 0.4rem; border-radius: 6px; font-size: 0.7rem; font-weight: 700; margin-right: 0.4rem; }
  .badge-legal { background: #16a34a; color: #fff; }
  .badge-rating { background: #f97316; color: #fff; }
  .badge-unverified { background: #6b7280; color: #fff; }
  .clabel { font-weight: 600; }
  .csnippet { margin: 0.4rem 0 0; color: var(--muted); font-size: 0.8rem; line-height: 1.4; }
  .timing { display: block; margin-top: 0.5rem; color: var(--muted); }
  .feedback { margin-top: 0.85rem; border-top: 1px dashed var(--border); padding-top: 0.65rem; }
  .fb-row { display: flex; align-items: center; gap: 0.5rem; flex-wrap: wrap; }
  .fb-q { font-size: 0.83rem; color: var(--muted); }
  .thumb {
    background: var(--bg); border: 1px solid var(--border); border-radius: 8px;
    padding: 0.25rem 0.5rem; font-size: 1rem;
  }
  .thumb.active { border-color: var(--accent); }
  .fb-thanks { color: #22c55e; font-size: 0.85rem; }
  .fb-panel {
    margin-top: 0.6rem; padding: 0.75rem; border: 1px solid var(--border);
    border-radius: 10px; background: var(--bg); display: flex; flex-direction: column; gap: 0.4rem;
  }
  .fb-title { margin: 0 0 0.2rem; font-size: 0.85rem; font-weight: 600; }
  .fb-opt { display: flex; gap: 0.5rem; align-items: flex-start; font-size: 0.83rem; line-height: 1.3; }
  .fb-comment {
    margin-top: 0.3rem; padding: 0.5rem; border-radius: 8px; border: 1px solid var(--border);
    background: var(--surface); color: var(--text); resize: vertical;
  }
  .fb-actions { display: flex; gap: 0.5rem; justify-content: flex-end; }
  .fb-cancel { background: transparent; border: 1px solid var(--border); color: var(--text); border-radius: 8px; padding: 0.4rem 0.8rem; font-size: 0.82rem; }
  .fb-submit { background: var(--accent); border: none; color: #fff; border-radius: 8px; padding: 0.4rem 0.9rem; font-size: 0.82rem; font-weight: 600; }
  .export { background: transparent; border: 1px solid var(--border); color: var(--text); border-radius: 6px; padding: 0.25rem 0.6rem; font-size: 0.76rem; }
  .loading { color: var(--muted); font-style: italic; }
  .composer { display: flex; gap: 0.5rem; padding-top: 1rem; border-top: 1px solid var(--border); }
  .composer textarea {
    flex: 1; resize: none; padding: 0.75rem; border-radius: 8px;
    border: 1px solid var(--border); background: var(--surface); color: var(--text);
  }
  .composer textarea:disabled { opacity: 0.6; }
  .composer button { padding: 0 1.25rem; border: none; border-radius: 8px; background: var(--accent); color: #fff; font-weight: 600; }
  .composer button:disabled { opacity: 0.5; }
`;
