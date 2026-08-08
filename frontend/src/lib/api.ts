import type { Citation, QueryMetrics } from "./types";

export const API_URL =
  process.env.NEXT_PUBLIC_API_URL?.replace(/\/$/, "") || "http://127.0.0.1:8000";

export type IndexedRegulation = {
  regulation_id: string;
  revision: string;
  chunk_count: number;
  label: string;
};

export async function fetchRegulations(): Promise<IndexedRegulation[]> {
  try {
    const res = await fetch(`${API_URL}/regulations`);
    if (!res.ok) return [];
    const data = await res.json();
    return (data.regulations as IndexedRegulation[]) || [];
  } catch {
    return [];
  }
}

export type IngestJob = {
  job_id: string;
  status: "queued" | "parsing" | "chunking" | "embedding" | "done" | "failed" | string;
  regulation_id: string;
  revision: string;
  error?: string | null;
  chunk_count?: number | null;
  original_filename?: string;
};

export type UploadResponse = {
  job_id: string;
  status: string;
  regulation_id: string;
  revision: string;
  detected?: { regulation_id: string; revision: string; source: string };
  original_filename?: string;
};

export async function detectRegulationMeta(file: File): Promise<{
  regulation_id: string;
  revision: string;
  source: string;
  page1_preview?: string;
}> {
  const body = new FormData();
  body.append("file", file);
  const res = await fetch(`${API_URL}/regulations/detect`, { method: "POST", body });
  if (!res.ok) {
    const detail = await res.text();
    throw new Error(detail || `Detect failed (${res.status})`);
  }
  return res.json();
}

export async function uploadRegulation(
  file: File,
  opts?: { regulation_id?: string; revision?: string }
): Promise<UploadResponse> {
  const body = new FormData();
  body.append("file", file);
  if (opts?.regulation_id) body.append("regulation_id", opts.regulation_id);
  if (opts?.revision) body.append("revision", opts.revision);
  const res = await fetch(`${API_URL}/regulations/upload`, { method: "POST", body });
  if (!res.ok) {
    let detail = `Upload failed (${res.status})`;
    try {
      const data = await res.json();
      detail = data.detail || detail;
    } catch {
      /* ignore */
    }
    throw new Error(typeof detail === "string" ? detail : JSON.stringify(detail));
  }
  return res.json();
}

export async function fetchIngestJob(jobId: string): Promise<IngestJob> {
  const id = (jobId || "").trim();
  if (!id) {
    throw new Error("Missing job_id — cannot poll ingest status");
  }
  const res = await fetch(`${API_URL}/regulations/jobs/${encodeURIComponent(id)}`);
  if (!res.ok) throw new Error(`Job status failed (${res.status})`);
  return res.json();
}

export function pdfUrl(regulationId: string): string {
  return `${API_URL}/pdf/${encodeURIComponent(regulationId)}`;
}

export function citationUrl(chunkId: string): string {
  return `${API_URL}/citation/${encodeURIComponent(chunkId)}`;
}

export async function enrichCitation(c: Citation): Promise<Citation> {
  if (c.bounding_box?.length >= 4 && c.page_number != null) return c;
  try {
    const res = await fetch(citationUrl(c.chunk_id));
    if (!res.ok) return c;
    const data = await res.json();
    return {
      ...c,
      page_number: data.page_number ?? c.page_number,
      bounding_box: data.bounding_box?.length ? data.bounding_box : c.bounding_box,
      coord_origin: data.coord_origin || c.coord_origin || "BOTTOMLEFT",
      text: data.text || c.text,
      regulation_id: data.regulation_id || c.regulation_id,
      section_number: data.section_number || c.section_number,
    };
  } catch {
    return c;
  }
}

type StreamHandlers = {
  onStatus?: (stage: string) => void;
  onCitations?: (citations: Citation[]) => void;
  onToken?: (text: string) => void;
  onNotFound?: (
    message: string,
    traceId?: string,
    failureKind?: string | null
  ) => void;
  onDone?: (payload: {
    answer: string;
    citations: Citation[];
    model?: string;
    provider?: string;
    trace_id?: string;
    metrics?: QueryMetrics;
    not_found?: boolean;
    failure_kind?: string | null;
    answer_cached?: boolean;
    served_from_cache?: boolean;
    cache_hit_kind?: string;
    conversation_id?: string;
    condensed_question?: string;
    condensation_applied?: boolean;
    compliance?: import("./types").CompliancePayload | null;
    query_intent?: string | null;
    execution_layer?: string | null;
    multi_step?: boolean;
    mode_disclaimer?: string | null;
    mode_disclaimer_title?: string | null;
  }) => void;
  onError?: (message: string) => void;
};

export async function streamChat(
  question: string,
  handlers: StreamHandlers,
  opts?: { regulation_id?: string; conversation_id?: string }
): Promise<void> {
  let res: Response;
  try {
    res = await fetch(`${API_URL}/chat`, {
      method: "POST",
      headers: { "Content-Type": "application/json", Accept: "text/event-stream" },
      body: JSON.stringify({
        question,
        regulation_id: opts?.regulation_id || null,
        conversation_id: opts?.conversation_id || null,
      }),
    });
  } catch {
    throw new Error("API unreachable — is the backend running on port 8000?");
  }
  if (!res.ok || !res.body) {
    let detail = await res.text();
    try {
      const j = JSON.parse(detail);
      detail = j.detail || detail;
    } catch {
      /* keep text */
    }
    throw new Error(detail || `HTTP ${res.status}`);
  }

  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    const chunks = buffer.split("\n\n");
    buffer = chunks.pop() || "";
    for (const block of chunks) {
      const lines = block.split("\n");
      let event = "message";
      let data = "";
      for (const line of lines) {
        if (line.startsWith("event:")) event = line.slice(6).trim();
        if (line.startsWith("data:")) data += line.slice(5).trim();
      }
      if (!data) continue;
      let payload: Record<string, unknown>;
      try {
        payload = JSON.parse(data);
      } catch {
        continue;
      }
      if (event === "status" && handlers.onStatus) {
        handlers.onStatus(String(payload.stage || ""));
      } else if (event === "citations" && handlers.onCitations) {
        handlers.onCitations((payload.citations as Citation[]) || []);
      } else if (event === "token" && handlers.onToken) {
        handlers.onToken(String(payload.text || ""));
      } else if (event === "not_found" && handlers.onNotFound) {
        handlers.onNotFound(
          String(payload.message || ""),
          payload.trace_id as string | undefined,
          (payload.failure_kind as string | undefined) || null
        );
      } else if (event === "done" && handlers.onDone) {
        handlers.onDone({
          answer: String(payload.answer || ""),
          citations: (payload.citations as Citation[]) || [],
          model: payload.model as string | undefined,
          provider: payload.provider as string | undefined,
          trace_id: payload.trace_id as string | undefined,
          metrics: payload.metrics as QueryMetrics | undefined,
          not_found: Boolean(payload.not_found),
          failure_kind: (payload.failure_kind as string | undefined) || null,
          answer_cached: Boolean(payload.answer_cached),
          served_from_cache: Boolean(payload.served_from_cache),
          cache_hit_kind: payload.cache_hit_kind as string | undefined,
          conversation_id: payload.conversation_id as string | undefined,
          condensed_question: payload.condensed_question as string | undefined,
          condensation_applied: Boolean(payload.condensation_applied),
          compliance: (payload.compliance as import("./types").CompliancePayload) || null,
          query_intent: (payload.query_intent as string | undefined) || null,
          execution_layer: (payload.execution_layer as string | undefined) || null,
          multi_step: Boolean(payload.multi_step),
          mode_disclaimer: (payload.mode_disclaimer as string | undefined) || null,
          mode_disclaimer_title:
            (payload.mode_disclaimer_title as string | undefined) || null,
        });
      } else if (event === "error" && handlers.onError) {
        handlers.onError(String(payload.message || "chat error"));
      }
    }
  }
}

export async function runAgent(task: string): Promise<{
  answer: string;
  mode: string;
  table_markdown?: string;
  report_markdown?: string;
  trace_id: string;
  overall_citation_coverage: number;
  ungrounded_claim_count: number;
  steps: Array<{ tool: string; citation_coverage: number; preview?: string }>;
  sources: Citation[];
  not_found?: boolean;
  query_intent?: string | null;
  execution_layer?: string | null;
  multi_step?: boolean;
  mode_disclaimer?: string | null;
  mode_disclaimer_title?: string | null;
}> {
  let res: Response;
  try {
    res = await fetch(`${API_URL}/agent`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ task }),
    });
  } catch {
    throw new Error("API unreachable — is the backend running on port 8000?");
  }
  if (!res.ok) {
    let detail = await res.text();
    try {
      const j = JSON.parse(detail);
      detail = j.detail || detail;
    } catch {
      /* keep */
    }
    throw new Error(detail || `HTTP ${res.status}`);
  }
  const data = await res.json();
  return {
    answer: data.answer || data.report_markdown || data.table_markdown || "",
    mode: data.mode || "qa",
    table_markdown: data.table_markdown,
    report_markdown: data.report_markdown,
    trace_id: data.trace_id,
    overall_citation_coverage: data.overall_citation_coverage ?? 1,
    ungrounded_claim_count: data.ungrounded_claim_count ?? 0,
    steps: (data.steps || []).map((s: Record<string, unknown>) => ({
      tool: String(s.tool || ""),
      citation_coverage: Number(s.citation_coverage ?? 1),
      preview: String(s.output_preview || s.preview || ""),
    })),
    sources: (data.sources || []).map((s: Record<string, unknown>) => ({
      chunk_id: String(s.chunk_id || ""),
      regulation_id: String(s.regulation_id || ""),
      section_number: String(s.section_number || ""),
      section_title: String(s.section_title || ""),
      page_number: (s.page_number as number | null) ?? null,
      bounding_box: (s.bounding_box as number[]) || [],
      citation: String(s.citation || ""),
      label: String(s.citation || ""),
      text: String(s.text || ""),
      score: Number(s.score || 0),
    })),
    not_found: Boolean(data.not_found),
    query_intent: data.query_intent ?? null,
    execution_layer: data.execution_layer ?? null,
    multi_step: Boolean(data.multi_step),
    mode_disclaimer: data.mode_disclaimer ?? null,
    mode_disclaimer_title: data.mode_disclaimer_title ?? null,
  };
}
