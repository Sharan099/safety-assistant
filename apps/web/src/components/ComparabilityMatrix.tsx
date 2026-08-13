import { StatusBadge } from "./StatusBadge";
import type { ComparabilitySummary } from "@/lib/types";

// UI_UX_DESIGN_BRIEF.md Section 11.
export function ComparabilityMatrix({ summary }: { summary: ComparabilitySummary }) {
  return (
    <div className="rounded border border-neutral-800">
      <div className="flex items-center justify-between border-b border-neutral-800 bg-neutral-900/50 px-3 py-2">
        <span className="text-sm font-medium text-neutral-200">Comparability</span>
        <StatusBadge status={summary.overall_status} />
      </div>
      <table className="w-full text-left text-sm">
        <tbody>
          {summary.dimensions.map((dim) => (
            <tr key={dim.dimension} className="border-t border-neutral-800">
              <td className="px-3 py-1.5 text-neutral-300">{dim.dimension.replaceAll("_", " ")}</td>
              <td className="px-3 py-1.5">
                <StatusBadge status={dim.status} />
              </td>
              <td className="px-3 py-1.5 text-neutral-500">{dim.explanation}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
