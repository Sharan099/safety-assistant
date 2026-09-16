// Typed API client. Same-origin (`/api/...` is rewritten to the backend in next.config.ts) so the
// HttpOnly session cookie is first-party; every mutating call carries the CSRF header the API requires.
import type {
  AnswerResponse,
  AskRequest,
  Conversation,
  ConversationDetail,
  DocumentDetail,
  DocumentList,
  IngestionJob,
  Me,
  MessageExchange,
  Preferences,
  Readiness,
  RegulationSummary,
  SourceScope,
  UploadResult,
} from "./types";
import { ApiError } from "./errors";

const API_BASE = process.env.NEXT_PUBLIC_API_URL ?? "";
const CSRF = { "X-Requested-With": "safety-assistant" };

async function request<T>(path: string, init: RequestInit = {}): Promise<T> {
  const isForm = init.body instanceof FormData;
  const res = await fetch(`${API_BASE}${path}`, {
    credentials: "include",
    ...init,
    headers: {
      ...(isForm ? {} : { "Content-Type": "application/json" }),
      ...(init.method && init.method !== "GET" ? CSRF : {}),
      ...(init.headers ?? {}),
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

const json = (body: unknown): RequestInit => ({ method: "POST", body: JSON.stringify(body) });
const patch = (body: unknown): RequestInit => ({ method: "PATCH", body: JSON.stringify(body) });

export const api = {
  // identity
  me: () => request<Me>("/api/v1/me"),
  devLogin: (email: string) => request<{ user_id: string; email: string }>("/api/v1/auth/dev-login", json({ email })),
  signup: (body: { email: string; display_name: string; password: string }) =>
    request<{ user_id: string; email: string }>("/api/v1/auth/signup", json(body)),
  login: (body: { email: string; password: string }) =>
    request<{ user_id: string; email: string }>("/api/v1/auth/login", json(body)),
  logout: () => request<{ ok: boolean }>("/api/v1/auth/logout", { method: "POST" }),
  authMethods: () => request<{ dev_login: boolean; oidc: boolean; password: boolean }>("/api/v1/auth/oidc/methods"),
  patchPreferences: (p: Partial<Preferences>) => request<Me>("/api/v1/me/preferences", patch(p)),
  // corpus / system
  ready: () => request<Readiness>("/health/ready"),
  regulations: () => request<RegulationSummary[]>("/api/v1/regulations"),
  ask: (body: AskRequest) => request<AnswerResponse>("/api/v1/ask", json(body)),
  evidenceUrl: (chunkId: string) => `${API_BASE}/api/v1/evidence/${chunkId}`,
  // conversations
  conversations: (q?: string, archived = false) =>
    request<{ items: Conversation[] }>(
      `/api/v1/conversations?archived=${archived}${q ? `&q=${encodeURIComponent(q)}` : ""}`,
    ),
  conversation: (id: string) => request<ConversationDetail>(`/api/v1/conversations/${id}`),
  createConversation: (body: { title?: string; workspace_id?: string | null; source_scope?: SourceScope }) =>
    request<Conversation>("/api/v1/conversations", json(body)),
  patchConversation: (id: string, body: { title?: string; archived?: boolean; source_scope?: SourceScope }) =>
    request<Conversation>(`/api/v1/conversations/${id}`, patch(body)),
  sendMessage: (id: string, body: { content: string; as_of?: string | null; k?: number }) =>
    request<MessageExchange>(`/api/v1/conversations/${id}/messages`, json(body)),
  // documents
  documents: (params: Record<string, string | undefined> = {}) => {
    const qs = Object.entries(params)
      .filter(([, v]) => v)
      .map(([k, v]) => `${k}=${encodeURIComponent(v as string)}`)
      .join("&");
    return request<DocumentList>(`/api/v1/documents${qs ? `?${qs}` : ""}`);
  },
  document: (id: string) => request<DocumentDetail>(`/api/v1/documents/${id}`),
  upload: (form: FormData) => request<UploadResult>("/api/v1/documents", { method: "POST", body: form }),
  archiveDocument: (id: string) => request<DocumentDetail>(`/api/v1/documents/${id}/archive`, { method: "POST" }),
  promoteDocument: (id: string) => request<DocumentDetail>(`/api/v1/documents/${id}/promote`, { method: "POST" }),
  job: (id: string) => request<IngestionJob>(`/api/v1/ingestion-jobs/${id}`),
  retryJob: (id: string) => request<IngestionJob>(`/api/v1/ingestion-jobs/${id}/retry`, { method: "POST" }),
  // admin
  ingestionRuns: () => request<IngestionRun[]>("/api/v1/admin/ingestion/runs"),
  adminIngest: (source_key: string) => request<{ ingestion_job_id: string }>("/api/v1/admin/ingest", json({ source_key })),
};

export interface IngestionRun {
  run_id: string;
  source_key: string;
  status: string;
  started_at: string;
  finished_at: string | null;
  attempt: number;
  error: string | null;
}
