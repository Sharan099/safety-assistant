import type {
  AgentRunResult,
  ComparabilitySummary,
  ConfigDiffEntry,
  EvidenceSummary,
  GlobalResponseComparison,
  HypothesisSummary,
  InvestigationSummary,
  QualityGateSummary,
  RetrievedChunk,
  RunDetail,
  RunSummary,
  SignalAnalysisResult,
} from "./types";

const API_BASE = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8010/api/v1";

export class ApiError extends Error {
  status: number;
  detail: unknown;

  constructor(status: number, detail: unknown) {
    super(typeof detail === "string" ? detail : JSON.stringify(detail));
    this.status = status;
    this.detail = detail;
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, {
    ...init,
    headers: { "Content-Type": "application/json", ...init?.headers },
  });
  if (!res.ok) {
    let detail: unknown;
    try {
      detail = (await res.json()).detail;
    } catch {
      detail = res.statusText;
    }
    throw new ApiError(res.status, detail);
  }
  if (res.status === 204) return undefined as T;
  return res.json() as Promise<T>;
}

export const api = {
  health: () => request<{ status: string }>("/health"),

  listRuns: () => request<RunSummary[]>("/runs"),
  getRun: (runId: string) => request<RunDetail>(`/runs/${encodeURIComponent(runId)}`),

  listInvestigations: () => request<InvestigationSummary[]>("/investigations"),
  createInvestigation: (body: {
    run_a_id: string;
    run_b_id: string;
    question: string;
    primary_metric?: string;
    title?: string;
  }) => request<InvestigationSummary>("/investigations", { method: "POST", body: JSON.stringify(body) }),
  getInvestigation: (id: string) => request<InvestigationSummary>(`/investigations/${id}`),

  computeQuality: (id: string) =>
    request<{ run_a: QualityGateSummary; run_b: QualityGateSummary }>(`/investigations/${id}/quality`, {
      method: "POST",
    }),
  computeGlobalResponse: (id: string) =>
    request<GlobalResponseComparison>(`/investigations/${id}/global-response`, { method: "POST" }),
  computeConfigurationDiff: (id: string) =>
    request<ConfigDiffEntry[]>(`/investigations/${id}/configuration-diff`, { method: "POST" }),
  computeComparability: (id: string) =>
    request<ComparabilitySummary>(`/investigations/${id}/comparability`, { method: "POST" }),
  analyzeSignal: (id: string, signalName: string) =>
    request<SignalAnalysisResult>(`/investigations/${id}/signals/${encodeURIComponent(signalName)}/analyze`, {
      method: "POST",
    }),
  getSignalTimeseries: (id: string, signalName: string) =>
    request<{ signal: string; time_s: number[]; run_a: number[]; run_b: number[]; unit: string | null }>(
      `/investigations/${id}/signals/${encodeURIComponent(signalName)}/timeseries`,
    ),

  runAgent: (id: string) => request<AgentRunResult>(`/investigations/${id}/run-agent`, { method: "POST" }),
  listEvidence: (id: string) => request<EvidenceSummary[]>(`/investigations/${id}/evidence`),
  listHypotheses: (id: string) => request<HypothesisSummary[]>(`/investigations/${id}/hypotheses`),
  submitReview: (id: string, body: { decision: string; comment?: string }) =>
    request<{ review_id: string; investigation_state: string }>(`/investigations/${id}/review`, {
      method: "POST",
      body: JSON.stringify(body),
    }),

  searchKnowledge: (q: string, params?: { source_type?: string; authority_level?: string; limit?: number }) => {
    const search = new URLSearchParams({ q, ...(params as Record<string, string>) });
    return request<RetrievedChunk[]>(`/knowledge/search?${search.toString()}`);
  },
};
