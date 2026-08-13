// Mirrors apps/api/schemas.py and packages/analysis/models.py.
// Kept as plain interfaces (not generated) — the API surface is still
// moving; regenerate/codegen once it stabilizes.

export type Status = "PASS" | "WARNING" | "FAIL" | "UNKNOWN";
export type ComparabilityStatus =
  | "COMPARABLE"
  | "CONDITIONAL"
  | "NOT_COMPARABLE"
  | "NOT_ESTABLISHED"
  | "UNKNOWN";
export type ChangeStatus = "SAME" | "CHANGED" | "UNKNOWN";
export type ChangeClassification = "INTENTIONAL" | "DEPENDENCY" | "UNINTENTIONAL" | "UNKNOWN";

export interface RunSummary {
  id: string;
  run_id: string;
  vehicle_name: string;
  model_version: string;
  solver: string | null;
  solver_version: string | null;
  impact_type: string | null;
  impact_speed: number | null;
  quality_status: string;
  created_at: string;
}

export interface RunDetail extends RunSummary {
  dummy_version: string | null;
  seat_configuration: Record<string, unknown> | null;
  restraint_configuration: Record<string, unknown> | null;
  result_processing_version: string | null;
  metadata: Record<string, unknown> | null;
}

export interface InvestigationSummary {
  id: string;
  title: string;
  question: string;
  primary_metric: string | null;
  state: string;
  decision: string | null;
  run_a_id: string;
  run_b_id: string;
  created_at: string;
}

export interface Provenance {
  algorithm: string;
  algorithm_version: string;
  parameters: Record<string, unknown>;
}

export interface QualityCheckResult {
  check_type: string;
  status: Status;
  value: Record<string, unknown> | null;
  threshold: Record<string, unknown> | null;
  explanation: string;
  provenance: Provenance;
}

export interface QualityGateSummary {
  run_id: string;
  overall_status: Status;
  checks: QualityCheckResult[];
}

export interface ComparabilityResult {
  dimension: string;
  status: ComparabilityStatus;
  explanation: string;
  evidence: string[];
}

export interface ComparabilitySummary {
  run_a_id: string;
  run_b_id: string;
  overall_status: ComparabilityStatus;
  dimensions: ComparabilityResult[];
}

export interface GlobalResponseMetric {
  name: string;
  run_a_value: number | null;
  run_b_value: number | null;
  delta: number | null;
  delta_pct: number | null;
  materially_different: boolean;
  provenance: Provenance;
}

export interface GlobalResponseComparison {
  run_a_id: string;
  run_b_id: string;
  metrics: GlobalResponseMetric[];
  material_difference_detected: boolean;
}

export interface ConfigDiffEntry {
  path: string;
  run_a_value: unknown;
  run_b_value: unknown;
  change_status: ChangeStatus;
  change_classification: ChangeClassification;
}

export interface SignalFeatureSet {
  signal: string;
  run_id: string;
  peak: number;
  time_to_peak_ms: number;
  rise_time_ms: number | null;
  duration_ms: number | null;
  integral: number;
  provenance: Provenance;
}

export interface DivergenceEvent {
  signal: string;
  time_ms: number;
  threshold: Record<string, unknown>;
  window: Record<string, unknown>;
  alignment_method: string;
  source_run_a: string;
  source_run_b: string;
  provenance: Provenance;
}

export interface SignalAnalysisResult {
  signal: string;
  run_a_id: string;
  run_b_id: string;
  run_a_features: SignalFeatureSet;
  run_b_features: SignalFeatureSet;
  correlation: number;
  divergence: DivergenceEvent | null;
  provenance: Provenance;
}

export interface EvidenceSummary {
  id: string;
  evidence_type: string;
  source_type: string;
  content: string | null;
  created_at: string;
}

export interface HypothesisSummary {
  id: string;
  title: string;
  description: string | null;
  status: string;
  confidence_basis: string | null;
  created_at: string;
}

export interface AgentRunResult {
  investigation_id: string;
  state: string;
  blocked_reason: string | null;
  hypotheses: HypothesisSummary[];
  evidence_count: number;
}

export interface RetrievedChunk {
  chunk_id: string;
  content: string;
  document_key: string;
  document_title: string;
  authority_level: string;
  source_type: string;
  revision_label: string;
  section_title: string | null;
  section_content: string | null;
  page_start: number | null;
  page_end: number | null;
  fused_score: number;
  matched_fts: boolean;
  matched_vector: boolean;
}
