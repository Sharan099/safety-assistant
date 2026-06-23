"use client";

import Link from "next/link";
import { useCallback, useEffect, useRef, useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { apiChatStream, apiFetch, formatApiError } from "@/lib/api";
import { DocumentManager } from "@/components/DocumentManager";

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
  is_synthetic?: boolean;
  authority?: string;
  revision?: string;
  revision_verified?: boolean;
  page?: number | null;
  section?: string | null;
  snippet?: string;
};

type Flag = { type: string; regulation?: string; message: string };
type Grounding = {
  should_abstain?: boolean;
  confidence?: number;
  confidence_band?: "high" | "medium" | "low";
  reason?: string;
  generation_failed?: boolean;
};

type Gateway = {
  model?: string;
  provider?: string;
  tier?: number;
  cache_hit?: boolean;
  fallback_used?: boolean;
  route_score?: number;
  route_reasons?: string[];
  cost_usd?: number;
  cost_saved_usd?: number;
  prompt_tokens?: number;
  completion_tokens?: number;
};

const TIER_LABEL: Record<number, string> = {
  1: "Tier 1 · fast",
  2: "Tier 2 · balanced",
  3: "Tier 3 · advanced",
};

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
  gateway?: Gateway;
  timing?: Record<string, number>;
  warnings?: string[];
  generation_failed?: boolean;
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

const MODE_OPTIONS = [
  { id: "regulation_lookup", label: "Regulation lookup" },
  { id: "design_review", label: "Design review" },
  { id: "crash_investigation", label: "Crash investigation" },
  { id: "root_cause_analysis", label: "Root cause analysis" },
  { id: "test_preparation", label: "Test preparation" },
  { id: "post_test_analysis", label: "Post-test analysis" },
  { id: "knowledge_reuse", label: "Knowledge reuse" },
  { id: "management_view", label: "Management view" },
];

const MODE_FEEDBACK: Record<string, string[]> = {
  root_cause_analysis: [
    "Wrong causal chain",
    "Missing evidence step",
    "Incorrect conclusion",
    "Wrong or missing sources",
  ],
  management_view: [
    "Too much technical detail",
    "Missing risk summary",
    "Incorrect status",
    "Wrong or missing sources",
  ],
  default: PROBLEM_OPTIONS,
};

const USER_KEY = "psa_user";
const MODE_KEY = "psa_mode";
const ROLE_KEY = "psa_role";

const INDEXED_REGULATIONS = [
  { code: "UN R14", type: "Legal", topic: "Seat belt anchorages & strength" },
  { code: "UN R16", type: "Legal", topic: "Safety belts & restraint systems" },
  { code: "UN R17", type: "Legal", topic: "Seats, strength & head restraints" },
  { code: "UN R94", type: "Legal", topic: "Frontal impact occupant protection" },
  { code: "UN R95", type: "Legal", topic: "Side impact protection" },
  { code: "UN R135", type: "Legal", topic: "Pole side impact" },
  { code: "UN R137", type: "Legal", topic: "Full-width frontal impact" },
  { code: "FMVSS 208", type: "Legal", topic: "US frontal occupant protection" },
  { code: "Euro NCAP", type: "Rating", topic: "Frontal, side, rear & VRU protocols" },
  { code: "CAE Companion", type: "Reference", topic: "Crash simulation & virtual validation (handbook)" },
  { code: "Safety Companion", type: "Reference", topic: "Passive safety engineering guide (handbook)" },
];

const EXAMPLE_QUESTIONS = [
  "What are the UN R14 anchorage strength requirements?",
  "What test load applies to belt anchorages for M1 vehicles under UN R14?",
  "What tests apply to safety belts under UN R16?",
  "What frontal impact requirements apply under UN R94?",
  "What is the chest deflection limit under UN R94?",
  "How does Euro NCAP assess frontal crash protection?",
];

export default function Home() {
  const [user, setUser] = useState<User | null>(null);
  const [nameInput, setNameInput] = useState("");
  const [nameError, setNameError] = useState("");
  const [registering, setRegistering] = useState(false);

  const [ready, setReady] = useState<Ready | null>(null);
  const [readyError, setReadyError] = useState(false);

  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [mode, setMode] = useState("regulation_lookup");
  const [role, setRole] = useState<"engineer" | "manager">("engineer");
  const [loading, setLoading] = useState(false);
  const [loadingSec, setLoadingSec] = useState(0);
  const chatEndRef = useRef<HTMLDivElement | null>(null);
  const sendLockRef = useRef(false);

  // Load saved user (session memory) on first paint.
  useEffect(() => {
    try {
      const raw = localStorage.getItem(USER_KEY);
      if (raw) setUser(JSON.parse(raw));
      const savedMode = localStorage.getItem(MODE_KEY);
      const savedRole = localStorage.getItem(ROLE_KEY) as "engineer" | "manager" | null;
      if (savedRole === "manager") {
        setRole("manager");
        setMode(savedMode || "management_view");
      } else if (savedMode) {
        setMode(savedMode);
      }
    } catch {
      /* ignore */
    }
  }, []);

  // Poll readiness until the backend self-test passes.
  const pollReady = useCallback(async () => {
    try {
      const res = await apiFetch("/ready");
      if (!res.ok) throw new Error("not ready");
      const data: Ready = await res.json();
      setReady(data);
      setReadyError(false);
      return Boolean(data.ready);
    } catch {
      setReadyError(true);
      return false;
    }
  }, []);

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
      const res = await apiFetch("/users", {
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
          ? "Could not reach the backend — the API may be waking up (HF Space). Wait 30s and retry."
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
    if (!q || !canChat || sendLockRef.current) return;
    sendLockRef.current = true;
    setInput("");
    setMessages((m) => [...m, { role: "user", content: q }]);
    setLoading(true);
    setLoadingSec(0);
    try {
      const data = await apiChatStream(
        {
          query: q,
          user_id: user?.user_id,
          session_id: user?.session_id,
          mode,
          role,
        },
        (elapsed) => setLoadingSec(Math.round(elapsed)),
      );
      setMessages((m) => [
        ...m,
        {
          role: "assistant",
          content: data.answer,
          messageId: data.message_id ?? undefined,
          docs: data.documents as Doc[] | undefined,
          citations: data.citations as Citation[] | undefined,
          flags: data.flags as Flag[] | undefined,
          grounding: data.grounding as Grounding | undefined,
          gateway: data.gateway as Gateway | undefined,
          timing: data.timing,
          warnings: data.warnings,
          generation_failed: Boolean(data.generation_failed),
          feedback: { reasons: [], comment: "" },
        },
      ]);
    } catch (e) {
      setMessages((m) => [
        ...m,
        {
          role: "assistant",
          content: `Error: ${formatApiError(e)}`,
        },
      ]);
    } finally {
      setLoading(false);
      setLoadingSec(0);
      sendLockRef.current = false;
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
      const res = await apiFetch("/feedback", {
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
      if (!res.ok) throw new Error(await res.text());
      updateFeedback(idx, { submitted: true, panelOpen: false, rating });
    } catch {
      updateFeedback(idx, { panelOpen: rating === "down", submitted: false, rating });
      alert("Could not save feedback. Please try again.");
    }
  }

  function pickExample(question: string) {
    setInput(question);
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
          <div className="logo-badge">PSA</div>
          <h1>Passive Safety Assistant</h1>
          <p className="tag">
            Regulation-aware AI for crashworthiness, restraints, and occupant protection
          </p>
          <div className="testing">
            <strong>Testing phase</strong> — your questions and feedback are stored to
            help us improve answer quality and guardrails.
          </div>
          <section className="onboard-info">
            <h3>What&apos;s indexed</h3>
            <p className="muted">
              12 document families (~13,000 text chunks): UN R14/R16/R17/R94/R95/R135/R137,
              FMVSS 208, Euro NCAP protocols, CAE Companion, and Safety Companion.
            </p>
          </section>
          <label htmlFor="buddy">Choose a display name</label>
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
          <button className="primary" onClick={registerUser} disabled={registering}>
            {registering ? "Setting up…" : "Start chatting"}
          </button>
          <p className="fine">
            No email or password. We store your chosen name, chat history, and any
            thumbs-up/down feedback linked to your session.
          </p>
        </div>
        <style jsx>{onboardCss}</style>
      </main>
    );
  }

  const warming = !ready?.ready;

  return (
    <main className="app-shell">
      <aside className="sidebar">
        <div className="sidebar-brand">
          <span className="logo-badge sm">PSA</span>
          <div>
            <strong>Passive Safety</strong>
            <span className="sidebar-sub">Regulation Assistant</span>
          </div>
        </div>

        <section className="sidebar-section">
          <h2>Indexed regulations</h2>
          <p className="sidebar-hint">
            Answers are grounded only in these documents. Out-of-scope questions
            (e.g. general knowledge) are refused by guardrails.
          </p>
          <ul className="reg-list">
            {INDEXED_REGULATIONS.map((r) => (
              <li key={r.code}>
                <span className={`reg-type reg-type-${r.type.toLowerCase()}`}>{r.type}</span>
                <div>
                  <strong>{r.code}</strong>
                  <span>{r.topic}</span>
                </div>
              </li>
            ))}
          </ul>
        </section>

        <section className="sidebar-section">
          <h2>Use-case mode</h2>
          <label className="mode-label" htmlFor="mode-select">Active mode</label>
          <select
            id="mode-select"
            className="mode-select"
            value={mode}
            onChange={(e) => {
              const v = e.target.value;
              setMode(v);
              localStorage.setItem(MODE_KEY, v);
            }}
          >
            {MODE_OPTIONS.map((m) => (
              <option key={m.id} value={m.id}>{m.label}</option>
            ))}
          </select>
          <label className="mode-label" htmlFor="role-select">Role</label>
          <select
            id="role-select"
            className="mode-select"
            value={role}
            onChange={(e) => {
              const v = e.target.value as "engineer" | "manager";
              setRole(v);
              localStorage.setItem(ROLE_KEY, v);
              if (v === "manager") setMode("management_view");
            }}
          >
            <option value="engineer">Engineer</option>
            <option value="manager">Manager</option>
          </select>
        </section>

        <section className="sidebar-section">
          <h2>Good questions to ask</h2>
          <ul className="example-list">
            {EXAMPLE_QUESTIONS.map((q) => (
              <li key={q}>
                <button type="button" className="example-btn" onClick={() => pickExample(q)}>
                  {q}
                </button>
              </li>
            ))}
          </ul>
        </section>

        <section className="sidebar-section">
          <DocumentManager />
        </section>

        <section className="sidebar-section sidebar-foot">
          <p>
            <strong>How it works:</strong> hybrid retrieval (semantic + keyword) →
            reranking → grounded answer with citations. If evidence is weak, the
            assistant says it is not found in the regulations.
          </p>
          <Link href="/dashboard" className="dash-link">
            Feedback dashboard →
          </Link>
        </section>
      </aside>

      <div className="main-col">
      <header className="header">
        <div>
          <h1>Chat</h1>
          <p>Ask about UN/ECE regulations, FMVSS, Euro NCAP, and engineering reference handbooks</p>
        </div>
        <div className="who">
          <span className="user-pill">👤 {user.username}</span>
          <button className="link" onClick={resetUser}>
            Switch user
          </button>
        </div>
      </header>

      <div className="testing-banner">
        Testing phase — please rate answers with 👍/👎 so we can improve retrieval and guardrails.
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
            <h2>Welcome, {user.username}</h2>
            <p>
              This assistant answers <strong>passive safety</strong> questions using
              only the indexed regulation text — not general web knowledge. Every
              grounded answer shows sources (document, section, revision). If the
              corpus does not contain enough evidence, you will see an executive
              summary stating insufficient data in the knowledge base.
            </p>
            <div className="empty-grid">
              <div className="empty-card">
                <h3>You can ask about</h3>
                <ul>
                  <li>Test loads, procedures, and approval criteria (UN R14–R137)</li>
                  <li>US FMVSS 208 frontal protection requirements</li>
                  <li>Euro NCAP crash test protocols and scoring</li>
                  <li>CAE Companion &amp; Safety Companion handbooks (reference only)</li>
                  <li>CAE / Safety Companion engineering references</li>
                </ul>
              </div>
              <div className="empty-card">
                <h3>It will refuse</h3>
                <ul>
                  <li>General knowledge (geography, sports, etc.)</li>
                  <li>Prompt injection or jailbreak attempts</li>
                  <li>Unsafe or illegal instructions</li>
                  <li>Topics outside the indexed documents</li>
                </ul>
              </div>
            </div>
            <p className="empty-cta">Try an example from the sidebar, or type your own question below.</p>
          </div>
        )}

        {messages.map((msg, i) => {
          // On abstention (or when nothing was cited) there is no grounded
          // answer: render only the one-line answer — no sources, revision
          // banners, or warning blocks.
          const shouldAbstain = !!msg.grounding?.should_abstain;
          const generationFailed =
            !!msg.generation_failed || !!msg.grounding?.generation_failed;
          const hasCitations = !!(msg.citations && msg.citations.length > 0);
          const showExtras =
            msg.role === "assistant" &&
            !shouldAbstain &&
            !generationFailed &&
            hasCitations;

          // Revision/answer-level flags: render once per regulation as a single
          // compact banner (deduplicated by regulation id).
          const seenReg = new Set<string>();
          const dedupedFlags = (msg.flags || []).filter((f) => {
            const key = f.regulation || f.type;
            if (seenReg.has(key)) return false;
            seenReg.add(key);
            return true;
          });

          return (
          <article
            key={i}
            className={`bubble ${msg.role === "user" ? "user" : "assistant"}`}
          >
            {msg.role === "assistant" ? (
              mode === "management_view" && role === "manager" ? (
                <div className="mgmt-view">
                  <div className="mgmt-summary">
                    <ReactMarkdown remarkPlugins={[remarkGfm]}>{msg.content}</ReactMarkdown>
                  </div>
                  {showExtras && (
                    <details className="mgmt-detail">
                      <summary>Show clause detail</summary>
                      <ul className="citation-list compact">
                        {msg.citations!.map((c, j) => (
                          <li key={j}>{c.label}</li>
                        ))}
                      </ul>
                    </details>
                  )}
                </div>
              ) : mode === "root_cause_analysis" ? (
                <div className="rca-view">
                  <ReactMarkdown remarkPlugins={[remarkGfm]}>{msg.content}</ReactMarkdown>
                  {showExtras && (
                    <p className="rca-hint">Trace chain — each step should cite [S#] in the answer above.</p>
                  )}
                </div>
              ) : (
                <ReactMarkdown remarkPlugins={[remarkGfm]}>{msg.content}</ReactMarkdown>
              )
            ) : (
              <p>{msg.content}</p>
            )}

            {msg.role === "assistant" &&
              !shouldAbstain &&
              !generationFailed &&
              msg.grounding?.confidence_band && (
                <div className="confidence-row">
                  <span
                    className={`confidence-badge confidence-${msg.grounding.confidence_band}`}
                    title={`Grounding confidence: ${(
                      (msg.grounding.confidence ?? 0) * 100
                    ).toFixed(0)}%`}
                  >
                    {msg.grounding.confidence_band === "high"
                      ? "High confidence"
                      : msg.grounding.confidence_band === "medium"
                      ? "Medium confidence"
                      : "Low confidence"}
                  </span>
                  {msg.grounding.confidence_band === "low" && (
                    <span className="confidence-caution">
                      Low confidence — verify against the cited source.
                    </span>
                  )}
                </div>
              )}

            {showExtras && dedupedFlags.length > 0 && (
              <div className="flags">
                {dedupedFlags.map((f, j) => (
                  <div key={j} className={`flag flag-${f.type}`}>
                    ⚑ {f.message}
                  </div>
                ))}
              </div>
            )}

            {showExtras && msg.warnings && msg.warnings.length > 0 && (
              <div className="warnings">
                {msg.warnings.map((w, j) => (
                  <span key={j}>⚠ {w}</span>
                ))}
              </div>
            )}

            {showExtras && (
              <details className="sources" open>
                <summary>Sources ({msg.citations!.length})</summary>
                <ul className="citation-list">
                  {msg.citations!.map((c, j) => (
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
                      {c.is_synthetic && (
                        <span className="badge badge-synthetic" title="Synthetic test data">
                          SYNTHETIC DATA
                        </span>
                      )}
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

            {msg.role === "assistant" && msg.gateway?.model && (
              <div className="gateway">
                <span
                  className={`gw-badge gw-tier-${msg.gateway.tier ?? 0}`}
                  title={msg.gateway.route_reasons?.join(" · ") || "Routing decision"}
                >
                  🧠 {msg.gateway.model}
                  {msg.gateway.tier != null &&
                    ` · ${TIER_LABEL[msg.gateway.tier] || `Tier ${msg.gateway.tier}`}`}
                </span>
                {msg.gateway.cache_hit && (
                  <span className="gw-pill gw-cache" title="Served from the semantic cache">
                    ⚡ cached
                  </span>
                )}
                {msg.gateway.fallback_used && (
                  <span className="gw-pill gw-fallback" title="Primary provider failed; failed over">
                    ↪ failover
                  </span>
                )}
                {(msg.gateway.prompt_tokens != null ||
                  msg.gateway.completion_tokens != null) && (
                  <span className="gw-pill" title="Prompt / completion tokens">
                    🎟 {msg.gateway.prompt_tokens ?? 0} in / {msg.gateway.completion_tokens ?? 0} out
                  </span>
                )}
                {msg.gateway.route_score != null && (
                  <span className="gw-pill" title="Complexity score (0–10)">
                    score {msg.gateway.route_score.toFixed(1)}
                  </span>
                )}
                {msg.gateway.cost_saved_usd != null && msg.gateway.cost_saved_usd > 0 && (
                  <span className="gw-pill gw-saved" title="Estimated cost saved vs the most capable tier">
                    saved ${msg.gateway.cost_saved_usd.toFixed(5)}
                  </span>
                )}
              </div>
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
                    {(MODE_FEEDBACK[mode] || MODE_FEEDBACK.default).map((opt) => (
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
          );
        })}

        {loading && (
          <p className="loading">
            Retrieving, reranking, and generating answer…{" "}
            {loadingSec > 0 ? `(${loadingSec}s — still working)` : "(starting)"}
            . Slow CPU rerank can take 1–2 minutes — do not resend.
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
      </div>
    </main>
  );
}

const onboardCss = `
  .onboard {
    min-height: 100vh;
    display: flex;
    align-items: center;
    justify-content: center;
    padding: 1.5rem;
    background: linear-gradient(160deg, #fff7ed 0%, #ffffff 45%, #ffedd5 100%);
  }
  .card {
    width: 100%;
    max-width: 480px;
    background: #fff;
    border: 1px solid var(--border);
    border-radius: 20px;
    padding: 2rem;
    display: flex;
    flex-direction: column;
    gap: 0.75rem;
    box-shadow: var(--shadow);
  }
  .logo-badge {
    width: 48px; height: 48px; border-radius: 12px;
    background: linear-gradient(135deg, var(--accent), #fb923c);
    color: #fff; font-weight: 800; font-size: 0.95rem;
    display: flex; align-items: center; justify-content: center;
  }
  .card h1 { margin: 0; font-size: 1.65rem; color: var(--text); }
  .tag { margin: 0; color: var(--muted); font-size: 0.92rem; }
  .testing {
    margin: 0.25rem 0;
    padding: 0.85rem 1rem;
    border-radius: 12px;
    background: var(--accent-soft);
    border: 1px solid var(--border);
    font-size: 0.86rem;
    line-height: 1.45;
  }
  .onboard-info h3 { margin: 0.5rem 0 0.25rem; font-size: 0.95rem; }
  .muted { margin: 0; color: var(--muted); font-size: 0.84rem; line-height: 1.45; }
  label { font-size: 0.9rem; font-weight: 600; margin-top: 0.25rem; }
  input {
    padding: 0.8rem 0.9rem;
    border-radius: 10px;
    border: 1px solid var(--border);
    background: #fff;
    color: var(--text);
    font-size: 1rem;
  }
  input:focus { outline: 2px solid var(--accent-soft); border-color: var(--accent); }
  .err { color: var(--danger); font-size: 0.85rem; margin: 0; }
  .primary {
    margin-top: 0.35rem;
    padding: 0.85rem;
    border: none;
    border-radius: 10px;
    background: var(--accent);
    color: #fff;
    font-weight: 600;
    font-size: 1rem;
  }
  .primary:hover:not(:disabled) { background: var(--accent-hover); }
  .primary:disabled { opacity: 0.55; }
  .fine { color: var(--muted); font-size: 0.78rem; line-height: 1.45; margin: 0.25rem 0 0; }
`;

const appCss = `
  .app-shell {
    min-height: 100vh;
    display: grid;
    grid-template-columns: minmax(260px, 300px) 1fr;
    background: #fafaf9;
  }
  @media (max-width: 900px) {
    .app-shell { grid-template-columns: 1fr; }
    .sidebar { display: none; }
  }
  .sidebar {
    background: #fff;
    border-right: 1px solid var(--border);
    padding: 1.25rem 1rem 1.5rem;
    overflow-y: auto;
    max-height: 100vh;
    position: sticky;
    top: 0;
  }
  .sidebar-brand {
    display: flex; align-items: center; gap: 0.65rem;
    margin-bottom: 1.25rem; padding-bottom: 1rem;
    border-bottom: 1px solid var(--border);
  }
  .logo-badge.sm {
    width: 36px; height: 36px; border-radius: 10px; font-size: 0.75rem;
    background: linear-gradient(135deg, var(--accent), #fb923c);
    color: #fff; font-weight: 800;
    display: flex; align-items: center; justify-content: center;
  }
  .sidebar-brand strong { display: block; font-size: 0.95rem; }
  .sidebar-sub { color: var(--muted); font-size: 0.78rem; }
  .sidebar-section { margin-bottom: 1.25rem; }
  .sidebar-section h2 {
    margin: 0 0 0.35rem; font-size: 0.82rem; text-transform: uppercase;
    letter-spacing: 0.04em; color: var(--accent);
  }
  .sidebar-hint { margin: 0 0 0.6rem; font-size: 0.78rem; color: var(--muted); line-height: 1.4; }
  .reg-list { list-style: none; padding: 0; margin: 0; display: flex; flex-direction: column; gap: 0.45rem; }
  .reg-list li {
    display: flex; gap: 0.5rem; align-items: flex-start;
    padding: 0.45rem 0.5rem; border-radius: 8px; background: var(--surface);
    border: 1px solid transparent; font-size: 0.78rem;
  }
  .reg-list strong { display: block; font-size: 0.8rem; }
  .reg-list span { color: var(--muted); line-height: 1.3; }
  .reg-type {
    flex-shrink: 0; font-size: 0.62rem; font-weight: 700; text-transform: uppercase;
    padding: 0.15rem 0.35rem; border-radius: 4px; margin-top: 0.1rem;
  }
  .reg-type-legal { background: #dcfce7; color: #166534; }
  .reg-type-rating { background: #ffedd5; color: #c2410c; }
  .reg-type-reference { background: #f5f5f4; color: #57534e; }
  .example-list { list-style: none; padding: 0; margin: 0; display: flex; flex-direction: column; gap: 0.35rem; }
  .example-btn {
    width: 100%; text-align: left; padding: 0.5rem 0.6rem;
    border-radius: 8px; border: 1px solid var(--border);
    background: #fff; color: var(--text); font-size: 0.78rem; line-height: 1.35;
  }
  .example-btn:hover { border-color: var(--accent); background: var(--accent-soft); }
  .sidebar-foot p { margin: 0; font-size: 0.76rem; color: var(--muted); line-height: 1.45; }
  .doc-manager h3 { font-size: 0.85rem; margin: 0 0 0.5rem; }
  .doc-upload-form { display: flex; flex-direction: column; gap: 0.35rem; font-size: 0.78rem; }
  .doc-upload-form input, .doc-upload-form select, .doc-upload-form button { font-size: 0.78rem; }
  .ingest-status { font-size: 0.75rem; color: var(--muted); margin: 0.4rem 0; }
  .doc-list { list-style: none; padding: 0; margin: 0.5rem 0 0; font-size: 0.75rem; }
  .doc-list li { display: flex; gap: 0.35rem; align-items: center; margin-bottom: 0.25rem; flex-wrap: wrap; }
  .doc-cat { color: var(--muted); font-size: 0.7rem; }
  .dash-link {
    display: inline-block;
    margin-top: 0.65rem;
    font-size: 0.78rem;
    color: var(--accent);
    text-decoration: none;
    font-weight: 600;
  }
  .dash-link:hover { text-decoration: underline; }
  .main-col {
    max-width: 900px; width: 100%; margin: 0 auto;
    min-height: 100vh; display: flex; flex-direction: column; padding: 1rem 1.25rem;
  }
  .header {
    display: flex; justify-content: space-between; align-items: flex-start;
    gap: 1rem; margin-bottom: 0.5rem;
  }
  .header h1 { margin: 0; font-size: 1.45rem; }
  .header p { margin: 0.2rem 0 0; color: var(--muted); font-size: 0.88rem; max-width: 36rem; }
  .who { display: flex; flex-direction: column; align-items: flex-end; gap: 0.35rem; font-size: 0.85rem; }
  .user-pill {
    background: var(--surface); border: 1px solid var(--border);
    padding: 0.25rem 0.6rem; border-radius: 999px; font-size: 0.82rem;
  }
  .link {
    background: none; border: none; color: var(--accent);
    font-size: 0.78rem; padding: 0; text-decoration: underline;
  }
  .testing-banner {
    margin-bottom: 0.65rem;
    padding: 0.55rem 0.85rem;
    border-radius: 10px;
    background: var(--accent-soft);
    border: 1px solid var(--border);
    font-size: 0.82rem;
  }
  .status {
    margin-bottom: 0.5rem;
    padding: 0.55rem 0.85rem;
    border-radius: 10px;
    background: #fff7ed;
    border: 1px solid var(--border);
    font-size: 0.83rem;
  }
  .status-err { background: #fef2f2; border-color: #fecaca; color: var(--danger); }
  .status-warn { background: #fffbeb; border-color: #fde68a; color: var(--warn); }
  .chat { flex: 1; overflow-y: auto; display: flex; flex-direction: column; gap: 1rem; padding-bottom: 0.5rem; }
  .empty { padding: 0.5rem 0 1rem; }
  .empty h2 { font-size: 1.2rem; margin: 0 0 0.5rem; }
  .empty p { color: var(--muted); line-height: 1.55; margin: 0 0 1rem; }
  .empty-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 0.75rem; margin-bottom: 0.75rem; }
  @media (max-width: 640px) { .empty-grid { grid-template-columns: 1fr; } }
  .empty-card {
    background: #fff; border: 1px solid var(--border); border-radius: 12px;
    padding: 0.85rem 1rem;
  }
  .empty-card h3 { margin: 0 0 0.4rem; font-size: 0.9rem; color: var(--accent); }
  .empty-card ul { margin: 0; padding-left: 1.1rem; font-size: 0.84rem; color: var(--muted); }
  .empty-card li { margin: 0.2rem 0; }
  .empty-cta { font-size: 0.85rem; font-style: italic; margin: 0; }
  .bubble { padding: 1rem 1.15rem; border-radius: 14px; line-height: 1.55; }
  .bubble.user {
    background: linear-gradient(135deg, var(--accent), #f97316);
    color: #fff; align-self: flex-end; max-width: 85%;
    box-shadow: 0 2px 12px rgba(234, 88, 12, 0.2);
  }
  .bubble.assistant {
    background: #fff; border: 1px solid var(--border);
    box-shadow: 0 1px 8px rgba(0,0,0,0.04);
  }
  .bubble.assistant :global(table) {
    width: 100%; border-collapse: collapse; margin: 0.65rem 0;
    font-size: 0.82rem;
  }
  .bubble.assistant :global(th), .bubble.assistant :global(td) {
    border: 1px solid var(--border); padding: 0.4rem 0.55rem; text-align: left;
  }
  .bubble.assistant :global(th) { background: var(--surface); font-weight: 600; }
  .bubble.assistant :global(h2) { font-size: 1rem; margin: 1rem 0 0.35rem; }
  .flags { margin-top: 0.75rem; display: flex; flex-direction: column; gap: 0.4rem; }
  .flag {
    padding: 0.5rem 0.75rem; border-radius: 8px; font-size: 0.82rem;
    background: var(--accent-soft); border: 1px solid var(--border);
  }
  .flag-mixed_doc_types { background: #fffbeb; border-color: #fde68a; }
  .warnings { margin-top: 0.75rem; display: flex; flex-direction: column; gap: 0.25rem; color: var(--warn); font-size: 0.85rem; }
  .sources { margin-top: 0.75rem; font-size: 0.85rem; color: var(--muted); }
  .citation-list { list-style: none; padding: 0; margin: 0.5rem 0 0; display: flex; flex-direction: column; gap: 0.6rem; }
  .citation { padding: 0.55rem 0.65rem; border-radius: 10px; border: 1px solid var(--border); background: var(--surface); }
  .cmarker { font-weight: 700; margin-right: 0.4rem; color: var(--accent); }
  .badge { display: inline-block; padding: 0.05rem 0.4rem; border-radius: 6px; font-size: 0.7rem; font-weight: 700; margin-right: 0.4rem; }
  .badge-legal { background: #16a34a; color: #fff; }
  .badge-synthetic { background: #ea580c; color: #fff; margin-left: 0.25rem; }
  .mode-select { width: 100%; margin: 0.35rem 0 0.75rem; padding: 0.4rem; border-radius: 8px; border: 1px solid var(--border); background: var(--surface); color: var(--text); }
  .mode-label { font-size: 0.8rem; color: var(--muted); display: block; margin-top: 0.5rem; }
  .mgmt-detail { margin-top: 0.75rem; font-size: 0.9rem; }
  .rca-hint { font-size: 0.85rem; color: var(--muted); margin-top: 0.5rem; }
  .badge-rating { background: var(--accent); color: #fff; }
  .badge-unverified { background: #a8a29e; color: #fff; }
  .confidence-row { margin-top: 0.5rem; display: flex; flex-wrap: wrap; gap: 0.5rem; align-items: center; }
  .confidence-badge {
    display: inline-block;
    padding: 0.1rem 0.45rem;
    border-radius: 6px;
    font-size: 0.7rem;
    font-weight: 700;
  }
  .confidence-high { background: #16a34a; color: #fff; }
  .confidence-medium { background: #ca8a04; color: #fff; }
  .confidence-low { background: #dc2626; color: #fff; }
  .confidence-caution { font-size: 0.75rem; color: #b45309; font-weight: 600; }
  .clabel { font-weight: 600; }
  .csnippet { margin: 0.4rem 0 0; color: var(--muted); font-size: 0.8rem; line-height: 1.4; }
  .timing { display: block; margin-top: 0.5rem; color: var(--muted); font-size: 0.78rem; }
  .gateway { margin-top: 0.7rem; display: flex; flex-wrap: wrap; gap: 0.35rem; align-items: center; }
  .gw-badge {
    display: inline-flex; align-items: center; gap: 0.3rem;
    padding: 0.15rem 0.55rem; border-radius: 999px;
    font-size: 0.74rem; font-weight: 700; color: #fff;
    background: #78716c;
  }
  .gw-tier-1 { background: var(--accent); }
  .gw-tier-2 { background: #c2410c; }
  .gw-tier-3 { background: #9a3412; }
  .gw-pill {
    display: inline-flex; align-items: center; gap: 0.25rem;
    padding: 0.12rem 0.5rem; border-radius: 999px;
    font-size: 0.72rem; font-weight: 600;
    background: #fff; border: 1px solid var(--border); color: var(--muted);
  }
  .gw-cache { color: var(--success); border-color: #bbf7d0; }
  .gw-fallback { color: var(--warn); border-color: #fde68a; }
  .gw-saved { color: var(--success); border-color: #bbf7d0; }
  .feedback { margin-top: 0.85rem; border-top: 1px dashed var(--border); padding-top: 0.65rem; }
  .fb-row { display: flex; align-items: center; gap: 0.5rem; flex-wrap: wrap; }
  .fb-q { font-size: 0.83rem; color: var(--muted); }
  .thumb {
    background: #fff; border: 1px solid var(--border); border-radius: 8px;
    padding: 0.25rem 0.5rem; font-size: 1rem;
  }
  .thumb.active { border-color: var(--accent); background: var(--accent-soft); }
  .fb-thanks { color: var(--success); font-size: 0.85rem; }
  .fb-panel {
    margin-top: 0.6rem; padding: 0.75rem; border: 1px solid var(--border);
    border-radius: 12px; background: var(--surface); display: flex; flex-direction: column; gap: 0.4rem;
  }
  .fb-title { margin: 0 0 0.2rem; font-size: 0.85rem; font-weight: 600; }
  .fb-opt { display: flex; gap: 0.5rem; align-items: flex-start; font-size: 0.83rem; line-height: 1.3; }
  .fb-comment {
    margin-top: 0.3rem; padding: 0.5rem; border-radius: 8px; border: 1px solid var(--border);
    background: #fff; color: var(--text); resize: vertical;
  }
  .fb-actions { display: flex; gap: 0.5rem; justify-content: flex-end; }
  .fb-cancel { background: #fff; border: 1px solid var(--border); color: var(--text); border-radius: 8px; padding: 0.4rem 0.8rem; font-size: 0.82rem; }
  .fb-submit { background: var(--accent); border: none; color: #fff; border-radius: 8px; padding: 0.4rem 0.9rem; font-size: 0.82rem; font-weight: 600; }
  .export { background: #fff; border: 1px solid var(--border); color: var(--text); border-radius: 6px; padding: 0.25rem 0.6rem; font-size: 0.76rem; }
  .loading { color: var(--muted); font-style: italic; }
  .composer {
    display: flex; gap: 0.5rem; padding-top: 1rem;
    border-top: 1px solid var(--border); background: #fafaf9;
    position: sticky; bottom: 0;
  }
  .composer textarea {
    flex: 1; resize: none; padding: 0.8rem; border-radius: 12px;
    border: 1px solid var(--border); background: #fff; color: var(--text);
  }
  .composer textarea:focus { outline: 2px solid var(--accent-soft); border-color: var(--accent); }
  .composer textarea:disabled { opacity: 0.6; }
  .composer button {
    padding: 0 1.35rem; border: none; border-radius: 12px;
    background: var(--accent); color: #fff; font-weight: 600;
  }
  .composer button:hover:not(:disabled) { background: var(--accent-hover); }
  .composer button:disabled { opacity: 0.5; }
`;
