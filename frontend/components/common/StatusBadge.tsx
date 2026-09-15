import { CheckCircle2, CircleDashed, Loader2, OctagonAlert, XCircle, Archive } from "lucide-react";

import { Badge } from "@/components/ui/badge";
import type { AnswerMode, DisplayStatus, SourceScopeName } from "@/lib/types";
import { cn } from "@/lib/utils";

// Status is always text + icon + colour (never colour alone).

const STATUS: Record<DisplayStatus, { label: string; cls: string; Icon: typeof CheckCircle2; spin?: boolean }> = {
  UPLOADED: { label: "Uploaded", cls: "bg-secondary text-foreground", Icon: CircleDashed },
  VALIDATING: { label: "Validating", cls: "bg-primary-soft text-primary", Icon: Loader2, spin: true },
  PARSING: { label: "Parsing", cls: "bg-primary-soft text-primary", Icon: Loader2, spin: true },
  CHUNKING: { label: "Chunking", cls: "bg-primary-soft text-primary", Icon: Loader2, spin: true },
  EMBEDDING: { label: "Embedding", cls: "bg-primary-soft text-primary", Icon: Loader2, spin: true },
  INDEXING: { label: "Indexing", cls: "bg-primary-soft text-primary", Icon: Loader2, spin: true },
  VERIFYING: { label: "Verifying", cls: "bg-primary-soft text-primary", Icon: Loader2, spin: true },
  READY: { label: "Ready", cls: "bg-success-soft text-success", Icon: CheckCircle2 },
  FAILED: { label: "Failed", cls: "bg-destructive-soft text-destructive", Icon: XCircle },
  QUARANTINED: { label: "Quarantined", cls: "bg-destructive-soft text-destructive", Icon: OctagonAlert },
  ARCHIVED: { label: "Archived", cls: "bg-secondary text-muted-foreground", Icon: Archive },
};

export function DocumentStatus({ status }: { status: DisplayStatus }) {
  const s = STATUS[status] ?? STATUS.FAILED;
  return (
    <Badge variant="secondary" className={cn("gap-1 font-medium", s.cls)} data-testid="doc-status" data-status={status}>
      <s.Icon className={cn("size-3.5", s.spin && "animate-spin")} aria-hidden />
      {s.label}
    </Badge>
  );
}

const SCOPE: Record<SourceScopeName, { label: string; cls: string }> = {
  AUTHORITATIVE_ORG: { label: "Verified regulation", cls: "bg-evidence-soft text-evidence" },
  WORKSPACE: { label: "Workspace document", cls: "bg-primary-soft text-primary" },
  PRIVATE_USER: { label: "Private document", cls: "bg-warning-soft text-warning" },
};

const VERIFIED_KIND: Record<string, string> = {
  REGULATION: "Verified regulation",
  STANDARD: "Verified standard / protocol",
  MANUAL: "Verified manual",
  TECHNICAL_REPORT: "Verified reference",
};

/** Verified sources are labelled by what they are (a CAE manual is not a regulation); uploads by who can see them. */
export function ScopeBadge({ scope, kind }: { scope: SourceScopeName; kind?: string }) {
  const s = SCOPE[scope];
  const label = scope === "AUTHORITATIVE_ORG" && kind ? (VERIFIED_KIND[kind] ?? "Verified source") : s.label;
  return (
    <Badge variant="secondary" className={cn("font-medium", s.cls)} data-testid="scope-badge">
      {label}
    </Badge>
  );
}

const MODE: Record<AnswerMode, { label: string; cls: string }> = {
  GENERATED: { label: "Grounded", cls: "bg-success-soft text-success" },
  EVIDENCE_ONLY: { label: "Evidence only", cls: "bg-warning-soft text-warning" },
  ABSTAINED: { label: "Insufficient evidence", cls: "bg-destructive-soft text-destructive" },
};

export function AnswerModeBadge({ mode }: { mode: AnswerMode }) {
  const m = MODE[mode];
  return (
    <Badge variant="secondary" className={cn("font-medium", m.cls)} data-testid="answer-mode" data-mode={mode}>
      {m.label}
    </Badge>
  );
}
