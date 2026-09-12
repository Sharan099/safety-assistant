// Mirrors safety_assistant.generation.schemas / retrieval.context (API contract).

export type AnswerMode = "GENERATED" | "EVIDENCE_ONLY" | "ABSTAINED";

export interface Claim {
  text: string;
  evidence_ids: string[];
  kind: "REQUIREMENT" | "INTERPRETATION";
}

export interface Citation {
  evidence_id: string;
  label: string;
  regulation_key: string;
  version_label: string;
  section_path: string;
  page_start: number | null;
  page_end: number | null;
  source_sha256: string;
  source_uri: string | null;
  valid_from: string | null;
  valid_to: string | null;
  version_status: string;
}

export interface RelatedSection {
  path: string;
  citation_label: string;
  excerpt: string;
  via: string;
}

export interface Evidence {
  evidence_id: string;
  chunk_id: string;
  regulation_key: string;
  regulation_title: string;
  kind: string;
  jurisdiction: string;
  authority_level: string;
  version_id: string;
  version_label: string;
  version_status: string;
  valid_from: string | null;
  valid_to: string | null;
  published_at: string | null;
  section_path: string;
  section_number: string | null;
  section_title: string | null;
  annex: string | null;
  normative: boolean | null;
  chunk_type: string;
  page_start: number | null;
  page_end: number | null;
  citation_label: string;
  content: string;
  parent_context: string | null;
  related: RelatedSection[];
  ranks: { dense: number | null; sparse: number | null; exact: number | null; fused_score: number; rerank_score: number | null };
  source_sha256: string;
  source_uri: string | null;
  storage_uri: string;
}

export interface AnswerResponse {
  trace_id: string;
  query: string;
  mode: AnswerMode;
  answer: string | null;
  claims: Claim[];
  citations: Citation[];
  warnings: string[];
  abstain_reason: string | null;
  scope: { regulation_keys: string[]; as_of: string | null; intent: string; historical: boolean };
  evidence: Evidence[];
  validation: { ok: boolean; dropped_claims: number } | null;
  versions: Record<string, unknown>;
  latency_ms: Record<string, number>;
}

export interface RegulationSummary {
  regulation_key: string;
  title: string;
  kind: string;
  jurisdiction: string;
  authority_level: string;
  versions: { id: string; label: string; status: string; published_at: string | null; valid_from: string | null; valid_to: string | null }[];
}

export interface Readiness {
  status: "ready" | "not_ready";
  app_env: string;
  deps: Record<string, { ok: boolean; [k: string]: unknown }>;
}

export interface AskRequest {
  query: string;
  as_of?: string | null;
  regulation_keys?: string[];
  include_superseded?: boolean;
  k?: number;
}
