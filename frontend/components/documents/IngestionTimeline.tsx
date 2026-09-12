"use client";
import { Check, Circle, Loader2, OctagonAlert, XCircle } from "lucide-react";

import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Button } from "@/components/ui/button";
import { type DisplayStatus, type IngestionJob, STAGES } from "@/lib/types";
import { cn } from "@/lib/utils";

const LABEL: Record<DisplayStatus, string> = {
  UPLOADED: "Uploaded",
  VALIDATING: "Validating",
  PARSING: "Parsing",
  CHUNKING: "Chunking",
  EMBEDDING: "Embedding",
  INDEXING: "Indexing",
  VERIFYING: "Verifying",
  READY: "Ready",
  FAILED: "Failed",
  QUARANTINED: "Quarantined",
  ARCHIVED: "Archived",
};

/** Real backend stages only — no fake percentages (03_UI_UX "Upload experience", step 3). */
export function IngestionTimeline({
  status,
  job,
  onRetry,
  canRetry,
}: {
  status: DisplayStatus;
  job: IngestionJob | null;
  onRetry?: () => void;
  canRetry?: boolean;
}) {
  const failed = status === "FAILED" || status === "QUARANTINED";
  const current = failed ? STAGES.findIndex((s) => s === (job?.stage ?? "UPLOADED")) : STAGES.indexOf(status);
  return (
    <div data-testid="ingestion-timeline" data-status={status}>
      <ol className="flex flex-wrap gap-x-4 gap-y-2" aria-label="Processing stages">
        {STAGES.map((s, i) => {
          const done = !failed && i < current;
          const active = !failed && i === current && status !== "READY";
          const reached = failed && i <= current;
          return (
            <li key={s} className={cn("flex items-center gap-1.5 text-sm", done || status === "READY" ? "text-success" : active ? "text-primary" : reached ? "text-destructive" : "text-muted-foreground")}>
              {status === "READY" || done ? (
                <Check className="size-4" aria-hidden />
              ) : active && job?.status === "RUNNING" ? (
                <Loader2 className="size-4 animate-spin" aria-hidden />
              ) : failed && i === current ? (
                <XCircle className="size-4" aria-hidden />
              ) : (
                <Circle className="size-3.5" aria-hidden />
              )}
              <span aria-current={active ? "step" : undefined}>{LABEL[s]}</span>
            </li>
          );
        })}
      </ol>
      {job && (job.status === "QUEUED" || job.status === "RUNNING") && (
        <p className="mt-2 text-xs text-text-secondary" role="status" aria-live="polite">
          {job.status === "QUEUED" ? "Waiting for a worker" : "Processing"} · attempt {Math.max(job.attempt, 1)} of {job.max_attempts}
        </p>
      )}
      {failed && job && (
        <Alert variant="destructive" className="mt-3" data-testid="ingestion-error">
          <OctagonAlert className="size-4" aria-hidden />
          <AlertTitle>{status === "QUARANTINED" ? "Document quarantined" : "Processing failed"}</AlertTitle>
          <AlertDescription>
            <p>{job.error_public_message ?? "The document could not be processed."}</p>
            {job.diagnostic_reference && (
              <p className="mt-1 font-mono text-xs">Diagnostic reference: {job.diagnostic_reference}</p>
            )}
            <p className="mt-1 text-xs">
              {status === "QUARANTINED"
                ? "Quarantined files are never indexed. Fix the file and upload it again."
                : "Failed documents are excluded from retrieval until a retry succeeds."}
            </p>
            {status === "FAILED" && canRetry && onRetry && (
              <Button size="sm" variant="outline" className="mt-2" onClick={onRetry} data-testid="retry-job">
                Retry processing
              </Button>
            )}
          </AlertDescription>
        </Alert>
      )}
    </div>
  );
}
