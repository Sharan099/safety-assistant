/**
 * API base URL for the FastAPI backend.
 *
 * On Vercel (*.vercel.app) we use same-origin `/api/v1` proxied to Hugging Face
 * (see vercel.json + next.config.js). This avoids CORS errors and HF cold-start
 * quirks in the browser.
 */
const HF_BACKEND =
  process.env.NEXT_PUBLIC_HF_BACKEND_URL?.replace(/\/$/, "") ||
  "https://sharan099-passive-safety-assistant.hf.space";

export function getApiBase(): string {
  if (typeof window !== "undefined") {
    const host = window.location.hostname;
    if (host.endsWith(".vercel.app") || host === "safety-assistant-tan.vercel.app") {
      return "/api/v1";
    }
    if (window.location.port === "8080") {
      return "/api/v1";
    }
  }

  const env = process.env.NEXT_PUBLIC_API_URL?.replace(/\/$/, "");
  if (env) return env;

  return "http://localhost:8000/api/v1";
}

export function getHfBackendUrl(): string {
  return HF_BACKEND;
}

/** Fetch with one retry (helps when HF Space is waking from sleep). */
export async function apiFetch(
  path: string,
  init?: RequestInit,
  retries = 1,
): Promise<Response> {
  const url = `${getApiBase()}${path.startsWith("/") ? path : `/${path}`}`;
  let lastErr: unknown;
  for (let i = 0; i <= retries; i++) {
    try {
      const res = await fetch(url, init);
      return res;
    } catch (err) {
      lastErr = err;
      if (i < retries) await new Promise((r) => setTimeout(r, 3000));
    }
  }
  throw lastErr;
}
