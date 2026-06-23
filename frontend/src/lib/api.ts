/**
 * API routing for Vercel frontend + Hugging Face backend.
 *
 * Fast endpoints → Vercel proxy (/api/v1).
 * Chat → direct HF + NDJSON stream (/chat/stream) with keepalive pings so the
 * HF gateway does not kill connections during 60–120s Jina rerank.
 */
const HF_BACKEND =
  process.env.NEXT_PUBLIC_HF_BACKEND_URL?.replace(/\/$/, "") ||
  "https://sharan099-passive-safety-assistant.hf.space";

const CHAT_TIMEOUT_MS = 180_000;

function isVercelHost(): boolean {
  if (typeof window === "undefined") return false;
  const host = window.location.hostname;
  return host.endsWith(".vercel.app") || host === "safety-assistant-tan.vercel.app";
}

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

export function getChatApiBase(): string {
  if (typeof window !== "undefined" && isVercelHost()) {
    return `${HF_BACKEND}/api/v1`;
  }
  return getApiBase();
}

function buildUrl(base: string, path: string): string {
  return `${base}${path.startsWith("/") ? path : `/${path}`}`;
}

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

export type ChatPayload = {
  query: string;
  answer: string;
  message_id?: string | null;
  documents?: unknown[];
  citations?: unknown[];
  flags?: unknown[];
  grounding?: Record<string, unknown>;
  generation_failed?: boolean;
  gateway?: Record<string, unknown>;
  timing?: Record<string, number>;
  warnings?: string[];
};

/** Stream chat with keepalive pings (fixes HF ~60s gateway timeout). */
export async function apiChatStream(
  body: { query: string; user_id?: string; session_id?: string; mode?: string; role?: string },
  onProgress?: (elapsedSec: number) => void,
): Promise<ChatPayload> {
  const url = buildUrl(getChatApiBase(), "/chat/stream");
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), CHAT_TIMEOUT_MS);

  try {
    const res = await fetch(url, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Accept: "application/x-ndjson",
      },
      body: JSON.stringify(body),
      signal: controller.signal,
    });
    if (!res.ok) {
      throw new Error((await res.text()) || `HTTP ${res.status}`);
    }
    if (!res.body) {
      throw new Error("Empty response body from chat stream");
    }

    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split("\n");
      buffer = lines.pop() ?? "";
      for (const line of lines) {
        if (!line.trim()) continue;
        const msg = JSON.parse(line) as {
          type: string;
          elapsed_s?: number;
          data?: ChatPayload;
          detail?: string;
        };
        if (msg.type === "ping" && msg.elapsed_s != null) {
          onProgress?.(msg.elapsed_s);
        } else if (msg.type === "result" && msg.data) {
          return msg.data;
        } else if (msg.type === "error") {
          throw new Error(msg.detail || "Chat stream failed");
        }
      }
    }
    throw new Error("Chat stream ended without a result");
  } catch (err) {
    if (err instanceof DOMException && err.name === "AbortError") {
      throw new Error(
        "Request timed out after 3 minutes. Try again or switch to BGE reranker on HF Space.",
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
      "Failed to fetch — connection dropped (HF gateway timeout or CORS). " +
      "Redeploy frontend + backend with /chat/stream support."
    );
  }
  return "Request failed";
}

export type DocumentMeta = {
  path?: string;
  name: string;
  category?: string;
};

export type IngestJob = {
  status: string;
  progress?: number;
  chunk_count?: number;
  error?: string;
  pdf_name?: string;
};

export async function apiListDocuments(): Promise<DocumentMeta[]> {
  const res = await apiFetch("/documents");
  if (!res.ok) throw new Error(`HTTP ${res.status}`);
  const data = (await res.json()) as { documents: DocumentMeta[] };
  return data.documents;
}

export async function apiUploadDocument(
  file: File,
  meta: { doc_type: string; authority?: string; region?: string; test_type?: string; revision?: string },
): Promise<{ job_id: string }> {
  const form = new FormData();
  form.append("file", file);
  form.append("doc_type", meta.doc_type);
  if (meta.authority) form.append("authority", meta.authority);
  if (meta.region) form.append("region", meta.region);
  if (meta.test_type) form.append("test_type", meta.test_type);
  if (meta.revision) form.append("revision", meta.revision);
  const res = await apiFetch("/documents", { method: "POST", body: form });
  if (!res.ok) throw new Error((await res.text()) || `HTTP ${res.status}`);
  return res.json() as Promise<{ job_id: string }>;
}

export async function apiIngestStatus(jobId: string): Promise<IngestJob> {
  const res = await apiFetch(`/documents/${jobId}`);
  if (!res.ok) throw new Error(`HTTP ${res.status}`);
  return res.json() as Promise<IngestJob>;
}

export async function apiDeleteDocument(docId: string): Promise<void> {
  const res = await apiFetch(`/documents/${encodeURIComponent(docId)}`, { method: "DELETE" });
  if (!res.ok) throw new Error((await res.text()) || `HTTP ${res.status}`);
}
