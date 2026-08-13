import { StatusBadge } from "./StatusBadge";
import type { HypothesisSummary } from "@/lib/types";

// UI_UX_DESIGN_BRIEF.md Section 17.
export function HypothesisCard({ hypothesis }: { hypothesis: HypothesisSummary }) {
  return (
    <div className="rounded border border-neutral-800 bg-neutral-900/40 p-3">
      <div className="mb-1 flex items-center justify-between gap-3">
        <span className="text-sm font-medium text-neutral-200">{hypothesis.title}</span>
        <StatusBadge status={hypothesis.status} />
      </div>
      {hypothesis.description && hypothesis.description !== hypothesis.title && (
        <p className="text-sm text-neutral-400">{hypothesis.description}</p>
      )}
      {hypothesis.confidence_basis && (
        <p className="mt-1 text-xs text-neutral-600">basis: {hypothesis.confidence_basis}</p>
      )}
    </div>
  );
}
