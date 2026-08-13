import { StatusBadge } from "./StatusBadge";
import type { QualityGateSummary } from "@/lib/types";

// UI_UX_DESIGN_BRIEF.md Section 10.
export function QualityCheckTable({ label, summary }: { label: string; summary: QualityGateSummary }) {
  return (
    <div className="rounded border border-neutral-800">
      <div className="flex items-center justify-between border-b border-neutral-800 bg-neutral-900/50 px-3 py-2">
        <span className="text-sm font-medium text-neutral-200">{label}</span>
        <StatusBadge status={summary.overall_status} />
      </div>
      <table className="w-full text-left text-sm">
        <tbody>
          {summary.checks.map((check) => (
            <tr key={check.check_type} className="border-t border-neutral-800">
              <td className="px-3 py-1.5 text-neutral-300">{check.check_type.replaceAll("_", " ")}</td>
              <td className="px-3 py-1.5">
                <StatusBadge status={check.status} />
              </td>
              <td className="px-3 py-1.5 text-neutral-500">{check.explanation}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
