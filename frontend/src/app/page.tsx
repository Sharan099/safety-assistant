"use client";

import { FormEvent, useCallback, useEffect, useState } from "react";

const API_URL =
  process.env.NEXT_PUBLIC_API_URL || "http://127.0.0.1:8000/api/v1";

const fetchOpts: RequestInit = { credentials: "include" };

type Citation = {
  id: string;
  source_kind?: string;
  regulation_code?: string;
  document_name?: string;
  page_number?: number;
  section?: string;
  amendment?: string;
  snippet?: string;
  test_id?: string;
  confidential_tier?: boolean;
};

type ClauseCandidate = {
  regulation_code: string;
  section: string;
  document_name: string;
  linked_regulation_clause: string;
  snippet: string;
};

type Timing = Record<string, number | string | boolean | undefined>;

type ChatResult = {
  query: string;
  answer: string;
  citations: Citation[];
  gateway: {
    model_key?: string;
    model_id?: string;
    provider?: string;
    evidence_only?: boolean;
    latency_ms?: number;
    steps?: Array<{ model_key: string; outcome: string; latency_ms: number }>;
  };
  timing: Timing;
  evidence_only?: boolean;
};

type Message = {
  role: "user" | "assistant";
  content: string;
  meta?: ChatResult;
};

type AuthUser = {
  user_id: string;
  username: string;
};

export default function ChatPage() {
  const [ready, setReady] = useState<boolean | null>(null);
  const [readyDetail, setReadyDetail] = useState<string>("");
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [messages, setMessages] = useState<Message[]>([]);
  const [activeCitations, setActiveCitations] = useState<Citation[]>([]);
  const [error, setError] = useState<string | null>(null);

  const [authUser, setAuthUser] = useState<AuthUser | null>(null);
  const [authChecked, setAuthChecked] = useState(false);
  const [loginUser, setLoginUser] = useState("");
  const [loginPass, setLoginPass] = useState("");
  const [loginError, setLoginError] = useState<string | null>(null);
  const [loginLoading, setLoginLoading] = useState(false);
  const [guestMode, setGuestMode] = useState(false);

  const [clauseQuery, setClauseQuery] = useState("");
  const [clauseResults, setClauseResults] = useState<ClauseCandidate[]>([]);
  const [selectedClause, setSelectedClause] = useState<ClauseCandidate | null>(null);
  const [uploadFile, setUploadFile] = useState<File | null>(null);
  const [uploadStatus, setUploadStatus] = useState<string | null>(null);
  const [uploadError, setUploadError] = useState<string | null>(null);
  const [uploadLoading, setUploadLoading] = useState(false);

  const refreshAuth = useCallback(async () => {
    try {
      const res = await fetch(`${API_URL}/auth/me`, fetchOpts);
      if (res.ok) {
        setAuthUser(await res.json());
      } else {
        setAuthUser(null);
      }
    } catch {
      setAuthUser(null);
    } finally {
      setAuthChecked(true);
    }
  }, []);

  useEffect(() => {
    refreshAuth();
  }, [refreshAuth]);

  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const res = await fetch(`${API_URL}/ready`, fetchOpts);
        const data = await res.json();
        if (!cancelled) {
          setReady(Boolean(data.ready));
          setReadyDetail(
            data.ready
              ? `Corpus: ${data.chunks_in_corpus ?? "?"} chunks · ${data.embedding_model} @ ${data.embedding_dimension}d`
              : data.reason || "Backend not ready"
          );
        }
      } catch {
        if (!cancelled) {
          setReady(false);
          setReadyDetail("Cannot reach backend — check NEXT_PUBLIC_API_URL");
        }
      }
    })();
    return () => {
      cancelled = true;
    };
  }, []);

  const doLogin = async (e: FormEvent) => {
    e.preventDefault();
    setLoginError(null);
    setLoginLoading(true);
    try {
      const res = await fetch(`${API_URL}/auth/login`, {
        ...fetchOpts,
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ username: loginUser.trim(), password: loginPass }),
      });
      if (!res.ok) {
        const detail = (await res.json().catch(() => ({}))) as { detail?: string };
        throw new Error(detail.detail || "Login failed");
      }
      setAuthUser(await res.json());
      setGuestMode(false);
      setLoginPass("");
    } catch (err) {
      setLoginError(err instanceof Error ? err.message : "Login failed");
    } finally {
      setLoginLoading(false);
    }
  };

  const doLogout = async () => {
    await fetch(`${API_URL}/auth/logout`, { ...fetchOpts, method: "POST" });
    setAuthUser(null);
    setGuestMode(false);
    setSelectedClause(null);
    setUploadFile(null);
    setUploadStatus(null);
  };

  const searchClauses = async () => {
    setUploadError(null);
    const q = clauseQuery.trim();
    if (q.length < 2) return;
    const res = await fetch(
      `${API_URL}/clauses/search?q=${encodeURIComponent(q)}`,
      fetchOpts
    );
    if (!res.ok) {
      setUploadError("Clause search failed — sign in required");
      return;
    }
    setClauseResults(await res.json());
  };

  const submitUpload = async (e: FormEvent) => {
    e.preventDefault();
    if (!uploadFile || !selectedClause) return;
    setUploadLoading(true);
    setUploadError(null);
    setUploadStatus(null);
    try {
      const form = new FormData();
      form.append("file", uploadFile);
      form.append("upload_type", "structured_test_report");
      form.append("linked_regulation_clause", selectedClause.linked_regulation_clause);
      const res = await fetch(`${API_URL}/user-uploads`, {
        ...fetchOpts,
        method: "POST",
        body: form,
      });
      if (!res.ok) throw new Error(await res.text());
      const data = await res.json();
      setUploadStatus(`${data.status}${data.test_id ? ` — ${data.test_id}` : ""}`);
    } catch (err) {
      setUploadError(err instanceof Error ? err.message : "Upload failed");
    } finally {
      setUploadLoading(false);
    }
  };

  const submit = useCallback(
    async (e: FormEvent) => {
      e.preventDefault();
      const q = input.trim();
      if (!q || loading || !ready) return;
      setError(null);
      setLoading(true);
      setInput("");
      setMessages((m) => [...m, { role: "user", content: q }]);
      try {
        const res = await fetch(`${API_URL}/chat`, {
          ...fetchOpts,
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ query: q, top_k: 8 }),
        });
        if (res.status === 401) {
          throw new Error(
            "Login required for confidential harness data. Sign in or use regulation-only questions as guest."
          );
        }
        if (!res.ok) throw new Error(await res.text());
        const data: ChatResult = await res.json();
        setMessages((m) => [
          ...m,
          { role: "assistant", content: data.answer, meta: data },
        ]);
        setActiveCitations(data.citations || []);
      } catch (err) {
        const msg = err instanceof Error ? err.message : "Request failed";
        setError(msg);
      } finally {
        setLoading(false);
      }
    },
    [input, loading, ready]
  );

  const lastMeta = [...messages].reverse().find((m) => m.meta)?.meta;
  const showLoginGate = authChecked && !authUser && !guestMode;

  if (showLoginGate) {
    return (
      <div style={styles.shell}>
        <div style={styles.loginCard}>
          <h1 style={styles.title}>Passive Safety Assistant</h1>
          <p style={styles.subtitle}>
            Sign in to access confidential harness data and uploads (Phase A).
          </p>
          <form onSubmit={doLogin} style={styles.loginForm}>
            <label style={styles.label}>
              Username
              <input
                type="text"
                value={loginUser}
                onChange={(e) => setLoginUser(e.target.value)}
                autoComplete="username"
                style={styles.loginInput}
                required
              />
            </label>
            <label style={styles.label}>
              Password
              <input
                type="password"
                value={loginPass}
                onChange={(e) => setLoginPass(e.target.value)}
                autoComplete="current-password"
                style={styles.loginInput}
                required
              />
            </label>
            {loginError && <div style={styles.error}>{loginError}</div>}
            <button type="submit" disabled={loginLoading} style={styles.sendBtn}>
              {loginLoading ? "Signing in…" : "Sign in"}
            </button>
          </form>
          <button
            type="button"
            onClick={() => setGuestMode(true)}
            style={styles.guestBtn}
          >
            Continue as guest (regulation Q&amp;A only)
          </button>
          <p style={styles.loginHint}>
            Seed users: run <code>python scripts/seed_auth_users.py</code> — default
            password via AUTH_SEED_PASSWORD (changeme).
          </p>
        </div>
      </div>
    );
  }

  return (
    <div style={styles.shell}>
      <header style={styles.header}>
        <div>
          <h1 style={styles.title}>Passive Safety Assistant</h1>
          <p style={styles.subtitle}>EU UNECE regulation knowledge base</p>
        </div>
        <div style={styles.headerRight}>
          {authUser ? (
            <div style={styles.userRow}>
              <span style={styles.userBadge}>{authUser.username}</span>
              <button type="button" onClick={doLogout} style={styles.guestBtn}>
                Log out
              </button>
            </div>
          ) : (
            <span style={styles.guestBadge}>Guest — regulation only</span>
          )}
          <div style={statusPillStyle(ready)}>
            {ready === null ? "Checking…" : ready ? "Ready" : "Not ready"}
          </div>
        </div>
      </header>

      <p style={styles.readyLine}>{readyDetail}</p>

      <div style={styles.main}>
        <section style={styles.chatPanel}>
          <div style={styles.messages}>
            {messages.length === 0 && (
              <p style={styles.placeholder}>
                Ask about UN R94 chest limits, R14 ISOFIX Annex 6, R29 cab
                protection, child restraints, and more.
                {!authUser && (
                  <>
                    {" "}
                    Confidential harness queries require sign-in.
                  </>
                )}
              </p>
            )}
            {messages.map((m, i) => (
              <div
                key={i}
                style={
                  m.role === "user" ? styles.userBubble : styles.assistantBubble
                }
              >
                {m.meta?.evidence_only && (
                  <div style={styles.evidenceBanner}>
                    Evidence-only — LLM capacity limited; retrieved passages shown
                  </div>
                )}
                <div style={{ whiteSpace: "pre-wrap" }}>{m.content}</div>
              </div>
            ))}
            {loading && <div style={styles.loading}>Retrieving & generating…</div>}
            {error && <div style={styles.error}>{error}</div>}
          </div>

          <form onSubmit={submit} style={styles.form}>
            <textarea
              value={input}
              onChange={(e) => setInput(e.target.value)}
              placeholder={
                ready ? "Ask a passive-safety question…" : "Waiting for backend…"
              }
              disabled={!ready || loading}
              rows={2}
              style={styles.input}
            />
            <button
              type="submit"
              disabled={!ready || loading || !input.trim()}
              style={styles.sendBtn}
            >
              Ask
            </button>
          </form>
        </section>

        <aside style={styles.sidebar}>
          <h2 style={styles.sidebarTitle}>Sources & routing</h2>

          {authUser && (
            <div style={styles.metaBox}>
              <h3 style={{ fontSize: "0.9rem", marginBottom: 8 }}>Upload test report</h3>
              <form onSubmit={submitUpload}>
                <input
                  type="search"
                  placeholder="Search clause e.g. chest deflection"
                  value={clauseQuery}
                  onChange={(e) => setClauseQuery(e.target.value)}
                  style={styles.loginInput}
                />
                <button type="button" onClick={searchClauses} style={styles.guestBtn}>
                  Search clauses
                </button>
                {clauseResults.length > 0 && (
                  <ul style={styles.clauseList}>
                    {clauseResults.map((c) => (
                      <li key={c.linked_regulation_clause}>
                        <label style={{ cursor: "pointer", fontSize: "0.8rem" }}>
                          <input
                            type="radio"
                            name="clause"
                            checked={selectedClause?.linked_regulation_clause === c.linked_regulation_clause}
                            onChange={() => setSelectedClause(c)}
                          />{" "}
                          {c.linked_regulation_clause}
                        </label>
                        <div style={{ color: "var(--muted)", fontSize: "0.75rem" }}>{c.snippet}</div>
                      </li>
                    ))}
                  </ul>
                )}
                <input
                  type="file"
                  accept="application/pdf"
                  onChange={(e) => setUploadFile(e.target.files?.[0] ?? null)}
                  style={{ marginTop: 8, fontSize: "0.8rem" }}
                />
                <button
                  type="submit"
                  disabled={uploadLoading || !uploadFile || !selectedClause}
                  style={{ ...styles.sendBtn, marginTop: 8, width: "100%" }}
                >
                  {uploadLoading ? "Uploading…" : "Upload PDF"}
                </button>
              </form>
              {uploadStatus && <p style={{ color: "var(--success)", fontSize: "0.8rem" }}>{uploadStatus}</p>}
              {uploadError && <p style={styles.error}>{uploadError}</p>}
            </div>
          )}

          {lastMeta && (
            <div style={styles.metaBox}>
              <div style={styles.metaRow}>
                <span style={styles.metaLabel}>Model</span>
                <span>
                  {lastMeta.gateway?.model_key || "—"}
                  {lastMeta.gateway?.model_id
                    ? ` (${lastMeta.gateway.model_id})`
                    : ""}
                </span>
              </div>
              <div style={styles.metaRow}>
                <span style={styles.metaLabel}>Total latency</span>
                <span>{formatMs(lastMeta.timing?.total_ms)}</span>
              </div>
              {lastMeta.timing && (
                <details style={{ marginTop: 8 }}>
                  <summary style={{ cursor: "pointer", color: "var(--muted)" }}>
                    Per-step timing
                  </summary>
                  <ul style={styles.timingList}>
                    {Object.entries(lastMeta.timing)
                      .filter(([k, v]) => k.endsWith("_ms") && typeof v === "number")
                      .map(([k, v]) => (
                        <li key={k}>
                          {k.replace("_ms", "")}: {formatMs(v as number)}
                        </li>
                      ))}
                  </ul>
                </details>
              )}
            </div>
          )}

          <ul style={styles.citationList}>
            {activeCitations.length === 0 && (
              <li style={{ color: "var(--muted)", fontSize: 14 }}>
                Citations appear here after each answer.
              </li>
            )}
            {activeCitations.map((c) => (
              <li key={c.id} style={styles.citationItem}>
                <strong>[{c.id}]</strong>{" "}
                {c.source_kind === "harness_test" ? (
                  <span style={styles.harnessBadge}>Your uploaded report</span>
                ) : null}{" "}
                {c.regulation_code} · {c.section || "—"}
                <br />
                <span style={{ fontSize: 12, color: "var(--muted)" }}>
                  {c.document_name} · p.{c.page_number ?? "?"}
                  {c.amendment ? ` · ${c.amendment}` : ""}
                </span>
                {c.snippet && (
                  <p style={styles.snippet}>{c.snippet}…</p>
                )}
              </li>
            ))}
          </ul>
        </aside>
      </div>
    </div>
  );
}

function formatMs(v: unknown): string {
  if (typeof v !== "number") return "—";
  return `${v.toFixed(0)} ms`;
}

function statusPillStyle(ready: boolean | null): React.CSSProperties {
  return {
    padding: "0.35rem 0.75rem",
    borderRadius: 999,
    fontSize: "0.8rem",
    fontWeight: 600,
    background:
      ready === null
        ? "var(--surface-2)"
        : ready
          ? "rgba(46, 204, 113, 0.15)"
          : "rgba(230, 57, 70, 0.15)",
    color: ready ? "var(--success)" : ready === false ? "var(--accent)" : "var(--muted)",
    border: `1px solid ${ready ? "var(--success)" : ready === false ? "var(--accent)" : "var(--border)"}`,
  };
}

const styles: Record<string, React.CSSProperties> = {
  shell: {
    minHeight: "100vh",
    display: "flex",
    flexDirection: "column",
    padding: "1rem 1.5rem",
    gap: "0.75rem",
  },
  loginCard: {
    maxWidth: 420,
    margin: "4rem auto",
    padding: "2rem",
    background: "var(--surface)",
    borderRadius: 12,
    border: "1px solid var(--border)",
  },
  loginForm: {
    display: "flex",
    flexDirection: "column",
    gap: "0.75rem",
    marginTop: "1.25rem",
  },
  label: {
    display: "flex",
    flexDirection: "column",
    gap: 4,
    fontSize: "0.85rem",
    color: "var(--muted)",
  },
  loginInput: {
    background: "var(--bg)",
    color: "var(--text)",
    border: "1px solid var(--border)",
    borderRadius: 8,
    padding: "0.5rem 0.75rem",
    fontSize: "1rem",
  },
  loginHint: {
    marginTop: "1rem",
    fontSize: "0.78rem",
    color: "var(--muted)",
    lineHeight: 1.4,
  },
  guestBtn: {
    marginTop: "0.75rem",
    background: "transparent",
    color: "var(--muted)",
    border: "1px solid var(--border)",
    borderRadius: 8,
    padding: "0.5rem 1rem",
    cursor: "pointer",
    fontSize: "0.85rem",
  },
  headerRight: {
    display: "flex",
    alignItems: "center",
    gap: "0.75rem",
  },
  userRow: {
    display: "flex",
    alignItems: "center",
    gap: "0.5rem",
  },
  userBadge: {
    fontSize: "0.8rem",
    padding: "0.25rem 0.6rem",
    borderRadius: 999,
    background: "rgba(46, 204, 113, 0.12)",
    border: "1px solid var(--success)",
    color: "var(--success)",
  },
  guestBadge: {
    fontSize: "0.8rem",
    color: "var(--muted)",
  },
  header: {
    display: "flex",
    justifyContent: "space-between",
    alignItems: "center",
    borderBottom: "1px solid var(--border)",
    paddingBottom: "0.75rem",
  },
  title: { fontSize: "1.35rem", fontWeight: 700 },
  subtitle: { color: "var(--muted)", fontSize: "0.9rem", marginTop: 4 },
  readyLine: { color: "var(--muted)", fontSize: "0.85rem" },
  main: {
    flex: 1,
    display: "grid",
    gridTemplateColumns: "1fr 340px",
    gap: "1rem",
    minHeight: 0,
  },
  chatPanel: {
    display: "flex",
    flexDirection: "column",
    background: "var(--surface)",
    borderRadius: 12,
    border: "1px solid var(--border)",
    overflow: "hidden",
  },
  messages: {
    flex: 1,
    overflowY: "auto",
    padding: "1rem",
    display: "flex",
    flexDirection: "column",
    gap: "0.75rem",
    minHeight: 360,
  },
  placeholder: { color: "var(--muted)", fontSize: "0.95rem" },
  userBubble: {
    alignSelf: "flex-end",
    maxWidth: "85%",
    background: "var(--surface-2)",
    padding: "0.75rem 1rem",
    borderRadius: "12px 12px 4px 12px",
  },
  assistantBubble: {
    alignSelf: "flex-start",
    maxWidth: "92%",
    background: "#121a24",
    padding: "0.75rem 1rem",
    borderRadius: "12px 12px 12px 4px",
    border: "1px solid var(--border)",
  },
  evidenceBanner: {
    background: "rgba(69, 123, 157, 0.2)",
    border: "1px solid var(--evidence)",
    color: "#a8d0e6",
    padding: "0.4rem 0.6rem",
    borderRadius: 6,
    fontSize: "0.8rem",
    marginBottom: "0.5rem",
  },
  loading: { color: "var(--muted)", fontStyle: "italic" },
  error: { color: "var(--accent)", fontSize: "0.9rem" },
  form: {
    display: "flex",
    gap: "0.5rem",
    padding: "0.75rem",
    borderTop: "1px solid var(--border)",
  },
  input: {
    flex: 1,
    resize: "none",
    background: "var(--bg)",
    color: "var(--text)",
    border: "1px solid var(--border)",
    borderRadius: 8,
    padding: "0.6rem 0.75rem",
  },
  sendBtn: {
    background: "var(--accent)",
    color: "#fff",
    border: "none",
    borderRadius: 8,
    padding: "0 1.25rem",
    fontWeight: 600,
    cursor: "pointer",
  },
  sidebar: {
    background: "var(--surface)",
    borderRadius: 12,
    border: "1px solid var(--border)",
    padding: "1rem",
    overflowY: "auto",
  },
  sidebarTitle: { fontSize: "1rem", marginBottom: "0.75rem" },
  metaBox: {
    background: "var(--bg)",
    borderRadius: 8,
    padding: "0.75rem",
    marginBottom: "1rem",
    fontSize: "0.85rem",
  },
  metaRow: {
    display: "flex",
    justifyContent: "space-between",
    gap: "0.5rem",
    marginBottom: 4,
  },
  metaLabel: { color: "var(--muted)" },
  timingList: {
    listStyle: "none",
    marginTop: 6,
    fontSize: "0.8rem",
    color: "var(--muted)",
  },
  citationList: { listStyle: "none", display: "flex", flexDirection: "column", gap: "0.75rem" },
  citationItem: {
    background: "var(--bg)",
    borderRadius: 8,
    padding: "0.6rem 0.75rem",
    fontSize: "0.85rem",
    border: "1px solid var(--border)",
  },
  snippet: {
    marginTop: 6,
    fontSize: "0.78rem",
    color: "var(--muted)",
    lineHeight: 1.4,
  },
  clauseList: {
    listStyle: "none",
    marginTop: 8,
    display: "flex",
    flexDirection: "column",
    gap: 6,
    maxHeight: 140,
    overflowY: "auto",
  },
  harnessBadge: {
    fontSize: "0.72rem",
    padding: "2px 6px",
    borderRadius: 4,
    background: "rgba(230, 126, 34, 0.15)",
    border: "1px solid #e67e22",
    color: "#e67e22",
  },
};
