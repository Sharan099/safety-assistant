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

export type SourceScopeName = "AUTHORITATIVE_ORG" | "WORKSPACE" | "PRIVATE_USER";

export interface SourceScope {
  scopes: SourceScopeName[];
  workspace_ids: string[];
  document_ids: string[];
}

export interface AskRequest {
  query: string;
  as_of?: string | null;
  regulation_keys?: string[];
  include_superseded?: boolean;
  k?: number;
  source_scope?: SourceScope;
}

// ---- identity (api/routes/me.py)

export interface Preferences {
  default_workspace_id: string | null;
  answer_density: "concise" | "standard" | "detailed";
  preferred_language: string;
  ui_theme: "light" | "dark" | "system";
}

export interface Me {
  user: { id: string; email: string; display_name: string; roles: string[]; scopes: string[] };
  organizations: { id: string; role: string }[];
  workspaces: { id: string; name: string; organization_id: string }[];
  preferences: Preferences;
}

// ---- conversations (api/routes/conversations.py)

export interface Conversation {
  id: string;
  title: string;
  title_locked: boolean;
  workspace_id: string | null;
  source_scope: SourceScope;
  created_at: string;
  updated_at: string;
  archived_at: string | null;
}

export interface MessageCitation {
  order: number;
  label: string;
  chunk_id: string | null;
  version_id: string | null;
  regulation_key: string;
  version_label: string;
  section_path: string;
  page_start: number | null;
  page_end: number | null;
  source_sha256: string;
  quote_excerpt: string | null;
  evidence_available: boolean;
}

export interface Message {
  id: string;
  role: "user" | "assistant";
  content: string;
  answer_mode: AnswerMode | null;
  trace_id: string | null;
  warnings: string[];
  created_at: string;
  citations: MessageCitation[];
}

export interface ConversationDetail extends Conversation {
  messages: Message[];
}

export interface MessageExchange {
  conversation: Conversation;
  user_message: Message;
  assistant_message: Message;
  answer: AnswerResponse;
}

// ---- documents (api/routes/documents.py, domain/documents.py)

export type DisplayStatus =
  | "UPLOADED"
  | "VALIDATING"
  | "PARSING"
  | "CHUNKING"
  | "EMBEDDING"
  | "INDEXING"
  | "VERIFYING"
  | "READY"
  | "FAILED"
  | "QUARANTINED"
  | "ARCHIVED";

export const STAGES: DisplayStatus[] = [
  "UPLOADED",
  "VALIDATING",
  "PARSING",
  "CHUNKING",
  "EMBEDDING",
  "INDEXING",
  "VERIFYING",
  "READY",
];

export interface IngestionJob {
  id: string;
  document_version_id: string;
  document_id?: string;
  status: "QUEUED" | "RUNNING" | "SUCCEEDED" | "FAILED" | "QUARANTINED" | "CANCELLED";
  stage: DisplayStatus;
  attempt: number;
  max_attempts: number;
  error_code: string | null;
  error_public_message: string | null;
  diagnostic_reference: string | null;
  created_at: string;
  started_at: string | null;
  completed_at: string | null;
}

export interface DocumentSummary {
  id: string;
  document_key: string;
  title: string;
  document_type: string;
  scope: SourceScopeName;
  authority_level: string;
  organization_id: string;
  workspace_id: string | null;
  owner_user_id: string | null;
  status: DisplayStatus;
  version: {
    id: string;
    label: string;
    status: string;
    valid_from: string | null;
    valid_to: string | null;
    page_count: number | null;
    activated_at: string | null;
    source_sha256: string | null;
  } | null;
  latest_job: IngestionJob | null;
  created_at: string;
  archived_at: string | null;
  notes: string | null;
}

export interface DocumentDetail extends DocumentSummary {
  extraction_report?: { status?: string; processed_page_count?: number; failed_pages?: number[] } | null;
}

export interface DocumentList {
  items: DocumentSummary[];
  stages: DisplayStatus[];
}

export interface UploadResult {
  document_id: string;
  document_version_id: string;
  ingestion_job_id: string;
  status: string;
  duplicate: boolean;
  document: DocumentSummary;
}
