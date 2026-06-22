"use client";

import Link from "next/link";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { apiFetch } from "@/lib/api";

const KEY_STORAGE = "psa_dashboard_key";
const POLL_MS = 5000;

type Stats = {
  users: number;
  messages: number;
  thumbs_up: number;
  thumbs_down: number;
};

type UserProfile = {
  user_id: string;
  username: string;
  created_at: number;
  session_count: number;
  message_count: number;
  thumbs_up: number;
  thumbs_down: number;
};

type FeedbackItem = {
  id: string;
  message_id: string | null;
  session_id: string | null;
  user_id: string | null;
  username: string | null;
  rating: "up" | "down";
  reasons: string[];
  comment: string;
  query: string;
  answer: string;
  created_at: number;
};

type DashboardPayload = {
  server_time: number;
  stats: Stats;
  users: UserProfile[];
  feedback: FeedbackItem[];
};

function fmtTime(ts: number) {
  if (!ts) return "—";
  return new Date(ts * 1000).toLocaleString(undefined, {
    dateStyle: "medium",
    timeStyle: "short",
  });
}

function fmtRelative(ts: number) {
  const sec = Math.max(0, Math.floor(Date.now() / 1000 - ts));
  if (sec < 60) return `${sec}s ago`;
  if (sec < 3600) return `${Math.floor(sec / 60)}m ago`;
  return fmtTime(ts);
}

export default function FeedbackDashboard() {
  const [dashboardKey, setDashboardKey] = useState("");
  const [keyInput, setKeyInput] = useState("");
  const [authError, setAuthError] = useState("");
  const [loading, setLoading] = useState(false);
  const [stats, setStats] = useState<Stats | null>(null);
  const [users, setUsers] = useState<UserProfile[]>([]);
  const [feedback, setFeedback] = useState<FeedbackItem[]>([]);
  const [filterUserId, setFilterUserId] = useState<string | null>(null);
  const [filterRating, setFilterRating] = useState<"all" | "up" | "down">("all");
  const [expandedId, setExpandedId] = useState<string | null>(null);
  const [lastSync, setLastSync] = useState<number | null>(null);
  const [serverTime, setServerTime] = useState<number | null>(null);
  const [live, setLive] = useState(true);
  const newestTsRef = useRef(0);
  const feedbackMapRef = useRef<Map<string, FeedbackItem>>(new Map());

  useEffect(() => {
    const saved = sessionStorage.getItem(KEY_STORAGE);
    if (saved) setDashboardKey(saved);
  }, []);

  const mergeFeedback = useCallback((items: FeedbackItem[], replace = false) => {
    const map = replace ? new Map<string, FeedbackItem>() : new Map(feedbackMapRef.current);
    for (const item of items) {
      map.set(item.id, item);
      if (item.created_at > newestTsRef.current) newestTsRef.current = item.created_at;
    }
    feedbackMapRef.current = map;
    const sorted = Array.from(map.values()).sort((a, b) => b.created_at - a.created_at);
    setFeedback(sorted);
  }, []);

  const fetchDashboard = useCallback(
    async (incremental: boolean) => {
      if (!dashboardKey) return;
      const params = new URLSearchParams({ limit: "500" });
      if (filterUserId) params.set("user_id", filterUserId);
      if (incremental && newestTsRef.current > 0) {
        params.set("since", String(newestTsRef.current));
      }
      const res = await apiFetch(`/feedback/dashboard?${params}`, {
        headers: { "X-Dashboard-Key": dashboardKey },
      });
      if (res.status === 401) {
        sessionStorage.removeItem(KEY_STORAGE);
        setDashboardKey("");
        setAuthError("Invalid dashboard key.");
        throw new Error("unauthorized");
      }
      if (!res.ok) {
        const body = await res.json().catch(() => ({}));
        throw new Error(body.detail || `HTTP ${res.status}`);
      }
      const data: DashboardPayload = await res.json();
      setStats(data.stats);
      setUsers(data.users);
      setServerTime(data.server_time);
      setLastSync(Date.now());
      if (incremental && data.feedback.length > 0) {
        mergeFeedback(data.feedback, false);
      } else if (!incremental) {
        newestTsRef.current = 0;
        mergeFeedback(data.feedback, true);
      }
    },
    [dashboardKey, filterUserId, mergeFeedback],
  );

  useEffect(() => {
    if (!dashboardKey) return;
    let cancelled = false;
    (async () => {
      setLoading(true);
      setAuthError("");
      try {
        await fetchDashboard(false);
      } catch (e) {
        if (!cancelled && (e as Error).message !== "unauthorized") {
          setAuthError((e as Error).message || "Could not load dashboard");
        }
      } finally {
        if (!cancelled) setLoading(false);
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [dashboardKey, filterUserId, fetchDashboard]);

  useEffect(() => {
    if (!dashboardKey || !live) return;
    const id = window.setInterval(() => {
      fetchDashboard(true).catch(() => {});
    }, POLL_MS);
    return () => window.clearInterval(id);
  }, [dashboardKey, live, fetchDashboard]);

  const filteredFeedback = useMemo(() => {
    if (filterRating === "all") return feedback;
    return feedback.filter((f) => f.rating === filterRating);
  }, [feedback, filterRating]);

  const satisfaction = useMemo(() => {
    if (!stats) return null;
    const total = stats.thumbs_up + stats.thumbs_down;
    if (total === 0) return null;
    return Math.round((stats.thumbs_up / total) * 100);
  }, [stats]);

  function onUnlock(e: React.FormEvent) {
    e.preventDefault();
    const k = keyInput.trim();
    if (!k) return;
    sessionStorage.setItem(KEY_STORAGE, k);
    setDashboardKey(k);
    setKeyInput("");
    setAuthError("");
  }

  function signOut() {
    sessionStorage.removeItem(KEY_STORAGE);
    setDashboardKey("");
    feedbackMapRef.current = new Map();
    newestTsRef.current = 0;
    setFeedback([]);
    setStats(null);
    setUsers([]);
  }

  if (!dashboardKey) {
    return (
      <main className="gate">
        <div className="gate-card">
          <div className="logo-badge">PSA</div>
          <h1>Feedback dashboard</h1>
          <p className="muted">
            Real-time view of user ratings, profiles, and comments. Enter the admin key
            configured as <code>FEEDBACK_DASHBOARD_KEY</code> on the backend.
          </p>
          <form onSubmit={onUnlock}>
            <label htmlFor="dash-key">Dashboard key</label>
            <input
              id="dash-key"
              type="password"
              value={keyInput}
              onChange={(e) => setKeyInput(e.target.value)}
              placeholder="Enter admin key"
              autoComplete="off"
            />
            {authError && <p className="err">{authError}</p>}
            <button type="submit" className="primary">
              Unlock dashboard
            </button>
          </form>
          <Link href="/" className="back-link">
            ← Back to chat
          </Link>
        </div>
        <style jsx>{gateCss}</style>
      </main>
    );
  }

  return (
    <main className="dash">
      <header className="dash-header">
        <div>
          <div className="brand-row">
            <span className="logo-badge sm">PSA</span>
            <div>
              <h1>Feedback dashboard</h1>
              <p className="muted">Live user feedback and profiles</p>
            </div>
          </div>
        </div>
        <div className="header-actions">
          <span className={`live-pill ${live ? "on" : ""}`}>
            <span className="dot" />
            {live ? "Live" : "Paused"}
            {lastSync ? ` · synced ${fmtRelative(Math.floor(lastSync / 1000))}` : ""}
          </span>
          <button type="button" className="ghost" onClick={() => setLive((v) => !v)}>
            {live ? "Pause" : "Resume"}
          </button>
          <button
            type="button"
            className="ghost"
            onClick={() => fetchDashboard(false).catch(() => {})}
          >
            Refresh
          </button>
          <Link href="/" className="ghost link-btn">
            Chat
          </Link>
          <button type="button" className="ghost danger" onClick={signOut}>
            Lock
          </button>
        </div>
      </header>

      {authError && <div className="banner err-banner">{authError}</div>}
      {loading && !stats && <div className="banner">Loading…</div>}

      {stats && (
        <section className="stats-grid">
          <article className="stat-card">
            <span className="stat-label">Users</span>
            <strong className="stat-value">{stats.users}</strong>
          </article>
          <article className="stat-card">
            <span className="stat-label">Messages</span>
            <strong className="stat-value">{stats.messages}</strong>
          </article>
          <article className="stat-card up">
            <span className="stat-label">👍 Thumbs up</span>
            <strong className="stat-value">{stats.thumbs_up}</strong>
          </article>
          <article className="stat-card down">
            <span className="stat-label">👎 Thumbs down</span>
            <strong className="stat-value">{stats.thumbs_down}</strong>
          </article>
          {satisfaction !== null && (
            <article className="stat-card">
              <span className="stat-label">Satisfaction</span>
              <strong className="stat-value">{satisfaction}%</strong>
            </article>
          )}
        </section>
      )}

      <div className="dash-body">
        <aside className="user-panel">
          <div className="panel-head">
            <h2>Users</h2>
            <button
              type="button"
              className={`chip ${filterUserId === null ? "active" : ""}`}
              onClick={() => setFilterUserId(null)}
            >
              All
            </button>
          </div>
          <ul className="user-list">
            {users.map((u) => (
              <li key={u.user_id}>
                <button
                  type="button"
                  className={`user-row ${filterUserId === u.user_id ? "active" : ""}`}
                  onClick={() =>
                    setFilterUserId((cur) => (cur === u.user_id ? null : u.user_id))
                  }
                >
                  <span className="user-name">👤 {u.username}</span>
                  <span className="user-meta">Joined {fmtTime(u.created_at)}</span>
                  <span className="user-counts">
                    {u.message_count} msgs · 👍 {u.thumbs_up ?? 0} · 👎 {u.thumbs_down ?? 0}
                  </span>
                </button>
              </li>
            ))}
            {users.length === 0 && <li className="empty-note">No users yet.</li>}
          </ul>
        </aside>

        <section className="feed-panel">
          <div className="panel-head">
            <h2>Feedback</h2>
            <div className="filters">
              {(["all", "up", "down"] as const).map((r) => (
                <button
                  key={r}
                  type="button"
                  className={`chip ${filterRating === r ? "active" : ""}`}
                  onClick={() => setFilterRating(r)}
                >
                  {r === "all" ? "All" : r === "up" ? "👍 Up" : "👎 Down"}
                </button>
              ))}
            </div>
          </div>

          <ul className="feed">
            {filteredFeedback.map((item) => {
              const open = expandedId === item.id;
              return (
                <li key={item.id} className={`feed-item ${item.rating}`}>
                  <button
                    type="button"
                    className="feed-summary"
                    onClick={() => setExpandedId(open ? null : item.id)}
                  >
                    <span className={`rating-badge ${item.rating}`}>
                      {item.rating === "up" ? "👍" : "👎"}
                    </span>
                    <div className="feed-main">
                      <strong>{item.username || "Unknown user"}</strong>
                      <span className="feed-query">
                        {item.query
                          ? item.query.length > 120
                            ? `${item.query.slice(0, 120)}…`
                            : item.query
                          : "(no query stored)"}
                      </span>
                      {item.reasons.length > 0 && (
                        <span className="reason-tags">
                          {item.reasons.map((r) => (
                            <span key={r} className="tag">
                              {r}
                            </span>
                          ))}
                        </span>
                      )}
                      {item.comment && !open && (
                        <span className="comment-preview">
                          “{item.comment.length > 100 ? `${item.comment.slice(0, 100)}…` : item.comment}”
                        </span>
                      )}
                    </div>
                    <time>{fmtTime(item.created_at)}</time>
                  </button>
                  {open && (
                    <div className="feed-detail">
                      {item.comment && (
                        <p>
                          <strong>Comment:</strong> {item.comment}
                        </p>
                      )}
                      {item.query && (
                        <div>
                          <strong>Question</strong>
                          <pre>{item.query}</pre>
                        </div>
                      )}
                      {item.answer && (
                        <div>
                          <strong>Answer</strong>
                          <pre>{item.answer}</pre>
                        </div>
                      )}
                      <p className="ids">
                        ID {item.id.slice(0, 8)}… · user {item.user_id?.slice(0, 8) ?? "—"}… ·
                        session {item.session_id?.slice(0, 8) ?? "—"}…
                      </p>
                    </div>
                  )}
                </li>
              );
            })}
            {filteredFeedback.length === 0 && (
              <li className="empty-note">No feedback yet{filterUserId ? " for this user" : ""}.</li>
            )}
          </ul>
          {serverTime && (
            <p className="foot-note">Server time: {fmtTime(serverTime)} · polls every {POLL_MS / 1000}s</p>
          )}
        </section>
      </div>

      <style jsx>{dashCss}</style>
    </main>
  );
}

const gateCss = `
  .gate {
    min-height: 100vh;
    display: flex;
    align-items: center;
    justify-content: center;
    padding: 1.5rem;
    background: linear-gradient(160deg, #fff7ed 0%, #ffffff 45%, #ffedd5 100%);
  }
  .gate-card {
    width: 100%;
    max-width: 440px;
    background: #fff;
    border: 1px solid var(--border);
    border-radius: 20px;
    padding: 2rem;
    display: flex;
    flex-direction: column;
    gap: 0.65rem;
    box-shadow: var(--shadow);
  }
  .logo-badge {
    width: 48px; height: 48px; border-radius: 12px;
    background: linear-gradient(135deg, var(--accent), #fb923c);
    color: #fff; font-weight: 800; font-size: 0.95rem;
    display: flex; align-items: center; justify-content: center;
  }
  h1 { margin: 0; font-size: 1.5rem; }
  .muted { margin: 0; color: var(--muted); font-size: 0.88rem; line-height: 1.45; }
  code { font-size: 0.82rem; background: var(--surface); padding: 0.1rem 0.35rem; border-radius: 4px; }
  label { font-size: 0.88rem; font-weight: 600; margin-top: 0.35rem; }
  input {
    padding: 0.75rem 0.85rem;
    border-radius: 10px;
    border: 1px solid var(--border);
    font-size: 1rem;
  }
  .err { color: var(--danger); font-size: 0.85rem; margin: 0; }
  .primary {
    margin-top: 0.35rem;
    padding: 0.8rem;
    border: none;
    border-radius: 10px;
    background: var(--accent);
    color: #fff;
    font-weight: 600;
  }
  .back-link {
    margin-top: 0.5rem;
    color: var(--accent);
    font-size: 0.88rem;
    text-decoration: none;
  }
`;

const dashCss = `
  .dash {
    min-height: 100vh;
    background: #fafaf9;
    padding: 1rem 1.25rem 2rem;
  }
  .dash-header {
    display: flex;
    flex-wrap: wrap;
    align-items: flex-start;
    justify-content: space-between;
    gap: 1rem;
    margin-bottom: 1rem;
  }
  .brand-row { display: flex; gap: 0.75rem; align-items: center; }
  .logo-badge.sm {
    width: 40px; height: 40px; border-radius: 10px;
    background: linear-gradient(135deg, var(--accent), #fb923c);
    color: #fff; font-weight: 800; font-size: 0.82rem;
    display: flex; align-items: center; justify-content: center;
  }
  h1 { margin: 0; font-size: 1.35rem; }
  .muted { margin: 0; color: var(--muted); font-size: 0.82rem; }
  .header-actions { display: flex; flex-wrap: wrap; gap: 0.45rem; align-items: center; }
  .live-pill {
    display: inline-flex; align-items: center; gap: 0.35rem;
    font-size: 0.78rem; color: var(--muted);
    padding: 0.35rem 0.65rem; border-radius: 999px;
    background: #fff; border: 1px solid var(--border);
  }
  .live-pill.on .dot {
    width: 8px; height: 8px; border-radius: 50%;
    background: var(--success);
    box-shadow: 0 0 0 3px rgba(22, 163, 74, 0.2);
    animation: pulse 2s infinite;
  }
  @keyframes pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.5; }
  }
  .ghost {
    padding: 0.4rem 0.7rem;
    border-radius: 8px;
    border: 1px solid var(--border);
    background: #fff;
    font-size: 0.82rem;
  }
  .ghost.danger { color: var(--danger); }
  .link-btn { text-decoration: none; color: inherit; display: inline-flex; align-items: center; }
  .banner {
    padding: 0.65rem 0.9rem;
    border-radius: 10px;
    background: var(--surface);
    border: 1px solid var(--border);
    margin-bottom: 1rem;
    font-size: 0.88rem;
  }
  .err-banner { background: #fef2f2; border-color: #fecaca; color: var(--danger); }
  .stats-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(130px, 1fr));
    gap: 0.65rem;
    margin-bottom: 1rem;
  }
  .stat-card {
    background: #fff;
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 0.85rem 1rem;
    display: flex;
    flex-direction: column;
    gap: 0.2rem;
  }
  .stat-card.up { border-color: #bbf7d0; background: #f0fdf4; }
  .stat-card.down { border-color: #fecaca; background: #fef2f2; }
  .stat-label { font-size: 0.75rem; color: var(--muted); }
  .stat-value { font-size: 1.45rem; line-height: 1.1; }
  .dash-body {
    display: grid;
    grid-template-columns: minmax(240px, 280px) 1fr;
    gap: 1rem;
    align-items: start;
  }
  @media (max-width: 900px) {
    .dash-body { grid-template-columns: 1fr; }
  }
  .user-panel, .feed-panel {
    background: #fff;
    border: 1px solid var(--border);
    border-radius: 14px;
    overflow: hidden;
  }
  .panel-head {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 0.5rem;
    padding: 0.85rem 1rem;
    border-bottom: 1px solid var(--border);
    background: var(--surface);
  }
  .panel-head h2 { margin: 0; font-size: 0.95rem; }
  .filters { display: flex; gap: 0.35rem; flex-wrap: wrap; }
  .chip {
    padding: 0.25rem 0.55rem;
    border-radius: 999px;
    border: 1px solid var(--border);
    background: #fff;
    font-size: 0.75rem;
  }
  .chip.active { background: var(--accent); color: #fff; border-color: var(--accent); }
  .user-list, .feed { list-style: none; margin: 0; padding: 0; max-height: 70vh; overflow-y: auto; }
  .user-row {
    width: 100%;
    text-align: left;
    padding: 0.75rem 1rem;
    border: none;
    border-bottom: 1px solid #f5f5f4;
    background: #fff;
    display: flex;
    flex-direction: column;
    gap: 0.15rem;
  }
  .user-row:hover { background: var(--surface); }
  .user-row.active { background: var(--accent-soft); border-left: 3px solid var(--accent); }
  .user-name { font-weight: 600; font-size: 0.88rem; }
  .user-meta, .user-counts { font-size: 0.75rem; color: var(--muted); }
  .feed-item { border-bottom: 1px solid #f5f5f4; }
  .feed-item.down { background: #fffbfb; }
  .feed-summary {
    width: 100%;
    display: grid;
    grid-template-columns: auto 1fr auto;
    gap: 0.75rem;
    align-items: start;
    padding: 0.85rem 1rem;
    border: none;
    background: transparent;
    text-align: left;
  }
  .rating-badge {
    width: 36px; height: 36px;
    border-radius: 10px;
    display: flex; align-items: center; justify-content: center;
    font-size: 1.1rem;
    background: var(--surface);
  }
  .rating-badge.down { background: #fee2e2; }
  .rating-badge.up { background: #dcfce7; }
  .feed-main { display: flex; flex-direction: column; gap: 0.25rem; min-width: 0; }
  .feed-query { font-size: 0.84rem; color: var(--text); }
  .reason-tags { display: flex; flex-wrap: wrap; gap: 0.3rem; }
  .tag {
    font-size: 0.68rem;
    padding: 0.15rem 0.45rem;
    border-radius: 999px;
    background: var(--surface-2);
    color: var(--muted);
  }
  .comment-preview { font-size: 0.78rem; color: var(--muted); font-style: italic; }
  .feed-summary time { font-size: 0.72rem; color: var(--muted); white-space: nowrap; }
  .feed-detail {
    padding: 0 1rem 1rem 3.6rem;
    font-size: 0.84rem;
    display: flex;
    flex-direction: column;
    gap: 0.65rem;
  }
  .feed-detail pre {
    margin: 0.25rem 0 0;
    padding: 0.65rem;
    border-radius: 8px;
    background: #fafaf9;
    border: 1px solid #e7e5e4;
    white-space: pre-wrap;
    word-break: break-word;
    font-size: 0.8rem;
    max-height: 220px;
    overflow-y: auto;
  }
  .ids { font-size: 0.72rem; color: var(--muted); margin: 0; }
  .empty-note {
    padding: 1.5rem 1rem;
    color: var(--muted);
    font-size: 0.88rem;
    text-align: center;
  }
  .foot-note {
    margin: 0;
    padding: 0.65rem 1rem;
    font-size: 0.72rem;
    color: var(--muted);
    border-top: 1px solid var(--border);
    background: #fafaf9;
  }
`;
