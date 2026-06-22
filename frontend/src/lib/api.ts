/**
 * API routing for Vercel frontend + Hugging Face backend.
 *
 * Fast endpoints (/ready, /users, /feedback) → same-origin Vercel proxy (/api/v1).
 * Chat (/chat) → browser calls HF Space directly (CORS allowed) because Vercel's
 * proxy times out ~60s while Jina rerank + RAG can take 90–120s.
 */
const HF_BACKEND =
  process.env.NEXT_PUBLIC_HF_BACKEND_URL?.replace(/\/$/, "") ||
  "https://sharan099-passive-safety-assistant.hf.space";

const CHAT_TIMEOUT_MS = 180_000; // 3 min — Jina rerank on CPU can exceed 60s

function isVercelHost(): boolean {
  if (typeof window === "undefined") return false;
  const host = window.location.hostname;
  return host.endsWith(".vercel.app") || host === "safety-assistant-tan.vercel.app";
}

/** Proxied base for fast endpoints (ready, users, feedback). */
export function getApiBase(): string {
  if (typeof window !== "undefined") {
    if (isVercelHost()) return "/api/v1";
    if (window.location.port === "8080") return "/api/v1";
  }
  const env = process.env.NEXT_PUBLIC_API_URL?.replace(/\/$/, "");
  if (env && env.startsWith("/")) return env;
  if (env) return env;
  return "http://localhost:8000/api/v1";
}

/** Direct HF base for long-running chat (bypasses Vercel 60s proxy limit). */
export function getChatApiBase(): string {
  if (typeof window !== "undefined" && isVercelHost()) {
    return `${HF_BACKEND}/api/v1`;
  }
  return getApiBase();
}

export function getHfBackendUrl(): string {
  return HF_BACKEND;
}

function buildUrl(base: string, path: string): string {
  return `${base}${path.startsWith("/") ? path : `/${path}`}`;
}

/** Fetch with optional retry (for /ready, /users, etc.). */
export async function apiFetch(
  path: string,
  init?: RequestInit,
  retries = 1,
): Promise<Response> {
  const url = buildUrl(getApiBase(), path);
  let lastErr: unknown;
  for (let i = 0; i <= retries; i++) {
    try {
      return await fetch(url, init);
    } catch (err) {
      lastErr = err;
      if (i < retries) await new Promise((r) => setTimeout(r, 3000));
    }
  }
  throw lastErr;
}

/** Chat fetch — direct to HF on Vercel, long timeout for slow rerankers. */
export async function apiFetchChat(
  path: string,
  init?: RequestInit,
): Promise<Response> {
  const url = buildUrl(getChatApiBase(), path);
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), CHAT_TIMEOUT_MS);
  try {
    const res = await fetch(url, { ...init, signal: controller.signal });
    return res;
  } catch (err) {
    if (err instanceof DOMException && err.name === "AbortError") {
      throw new Error(
        "Request timed out after 3 minutes. The backend reranker is slow on CPU — " +
          "retry once, or set RERANKER_MODEL=BAAI/bge-reranker-v2-m3 on the HF Space.",
      );
    }
    throw err;
  } finally {
    clearTimeout(timer);
  }
}

export function formatApiError(err: unknown): string {
  if (err instanceof Error) return err.message;
  if (err instanceof TypeError) {
    return (
      "Could not reach the backend. If the Hugging Face Space was sleeping, " +
      "wait 30s and retry. Chat requests go directly to HF (not via Vercel proxy)."
    );
  }
  return "Request failed";
}
