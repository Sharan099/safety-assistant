import type { EvidenceSummary } from "@/lib/types";

const TYPE_STYLES: Record<string, string> = {
  OBSERVED: "border-sky-800 text-sky-300",
  CALCULATED: "border-violet-800 text-violet-300",
  DOCUMENTARY: "border-emerald-800 text-emerald-300",
  HISTORICAL: "border-amber-800 text-amber-300",
  INFERRED: "border-neutral-600 text-neutral-300",
};

// UI_UX_DESIGN_BRIEF.md Section 16.
export function EvidenceCard({ evidence }: { evidence: EvidenceSummary }) {
  return (
    <div className="rounded border border-neutral-800 p-3">
      <div className="mb-1 flex items-center gap-2 text-xs">
        <span
          className={`rounded border px-1.5 py-0.5 font-mono ${TYPE_STYLES[evidence.evidence_type] ?? "border-neutral-600 text-neutral-400"}`}
        >
          {evidence.evidence_type}
        </span>
        <span className="text-neutral-600">{evidence.source_type}</span>
      </div>
      <p className="text-sm text-neutral-300">{evidence.content}</p>
    </div>
  );
}
