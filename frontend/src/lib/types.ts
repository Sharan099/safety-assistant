export type Citation = {
  chunk_id: string;
  regulation_id: string;
  section_number: string;
  section_title?: string;
  page_number: number | null;
  bounding_box: number[];
  coord_origin?: "BOTTOMLEFT" | "TOPLEFT";
  citation: string;
  label: string;
  text?: string;
  score?: number;
};

export type LlmCallMetrics = {
  role?: string;
  provider?: string;
  model?: string;
  input_tokens?: number;
  output_tokens?: number;
  cost_usd?: number;
  cache_status?: string;
  cached?: boolean;
  target_index?: number | null;
  latency_ms?: number;
  retry_attempts?: number;
};

export type QueryMetrics = {
  trace_id: string;
  input_tokens: number;
  output_tokens: number;
  embedding_tokens?: number;
  rerank_calls?: number;
  cost_usd: number;
  latency_ms: number;
  model?: string;
  provider?: string;
  cache_status?: string;
  llm_calls?: LlmCallMetrics[];
  chunk_ids?: string[];
  not_found?: boolean;
  answer_cached?: boolean;
  served_from_cache?: boolean;
  cache_hit_kind?: string;
  prompt_cache_hit?: boolean;
};

export type TraceMetrics = QueryMetrics & {
  question?: string;
  rewrite_model?: string;
  answer_model?: string;
  rewrite_provider?: string;
  answer_provider?: string;
  target_index?: number | null;
  llm_cost_usd?: number;
  context_chunks_to_llm?: number;
  context_tokens_est?: number;
  context_budget_mode?: string;
  n_hybrid_candidates?: number;
  rerank_ran?: boolean;
  citations?: Array<{
    chunk_id: string;
    citation?: string;
    score?: number;
  }>;
  faithfulness_passed?: boolean | null;
};

export type ProviderShare = {
  n: number;
  n_cache_hit?: number;
  cost_usd?: number;
  share?: number;
  avg_latency_ms?: number;
  p95_latency_ms?: number;
  slow?: boolean;
};

export type AggregateMetrics = {
  n: number;
  avg_cost_usd: number;
  avg_latency_ms: number;
  p95_latency_ms: number;
  n_passing: number;
  cost_per_passing_query_usd: number;
  total_cost_usd?: number;
  by_provider?: Record<string, ProviderShare>;
};

export type FailureKind =
  | "retrieval_miss"
  | "grounding_rejected"
  | "numeric_hallucination";

export type ComplianceCriterion = {
  criterion: string;
  measured: number;
  measured_display?: string;
  limit?: number | null;
  limit_display?: string | null;
  unit?: string;
  operator?: string;
  verdict: string;
  source_chunk_id?: string;
  section_number?: string;
  regulation_id?: string;
  note?: string;
};

export type CompliancePayload = {
  overall_verdict: string;
  summary_prose?: string;
  criteria: ComplianceCriterion[];
  regulation_id?: string | null;
};

export type ChatMessage = {
  id: string;
  role: "user" | "assistant";
  content: string;
  citations?: Citation[];
  streaming?: boolean;
  model?: string;
  provider?: string;
  trace_id?: string;
  metrics?: QueryMetrics;
  not_found?: boolean;
  failure_kind?: FailureKind | null;
  compliance?: CompliancePayload | null;
  /** Intent label from the query router (e.g. RETEST_SCOPE). */
  query_intent?: string | null;
  /** Hybrid layer: fast (Layer 1–2) vs multi_step (Layer 3–5). */
  execution_layer?: string | null;
  multi_step?: boolean;
  /** Mode honesty banner body (e.g. retest / homologation disclaimer). */
  mode_disclaimer?: string | null;
  mode_disclaimer_title?: string | null;
  error?: boolean;
};

export type ActiveCitation = Citation & {
  pdfUrl: string;
};
