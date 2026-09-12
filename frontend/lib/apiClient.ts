import type { AnswerResponse, AskRequest, Readiness, RegulationSummary } from "./types";
import { ApiError } from "./errors";

const API_BASE = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8010";
const TOKEN_KEY = "sa.token";

export function getToken(): string {
  if (typeof window === "undefined") return "";
  try {
    return window.localStorage.getItem(TOKEN_KEY) ?? process.env.NEXT_PUBLIC_DEV_API_KEY ?? "";
  } catch {
    return process.env.NEXT_PUBLIC_DEV_API_KEY ?? "";
  }
}

export function setToken(token: string): void {
  try {
    if (token) window.localStorage.setItem(TOKEN_KEY, token);
    else window.localStorage.removeItem(TOKEN_KEY);
  } catch {
    /* storage unavailable (private mode) — token lives for the session only */
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const token = getToken();
  const res = await fetch(`${API_BASE}${path}`, {
    ...init,
    headers: {
      "Content-Type": "application/json",
      ...(token ? { Authorization: `Bearer ${token}` } : {}),
      ...(init?.headers ?? {}),
    },
  });
  const requestId = res.headers.get("x-request-id");
  if (!res.ok) {
    let detail: unknown = res.statusText;
    try {
      detail = (await res.json()).detail ?? detail;
    } catch {
      /* non-JSON error body */
    }
    throw new ApiError(res.status, detail, requestId);
  }
  return (await res.json()) as T;
}

export const api = {
  ask: (body: AskRequest) => request<AnswerResponse>("/api/v1/ask", { method: "POST", body: JSON.stringify(body) }),
  regulations: () => request<RegulationSummary[]>("/api/v1/regulations"),
  ready: () => request<Readiness>("/health/ready"),
  evidenceUrl: (chunkId: string) => `${API_BASE}/api/v1/evidence/${chunkId}`,
  feedback: (trace_id: string, rating: 1 | -1, comment?: string) =>
    request<{ status: string }>("/api/v1/feedback", { method: "POST", body: JSON.stringify({ trace_id, rating, comment }) }),
};
