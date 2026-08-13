"use client";

import Link from "next/link";
import { useQuery } from "@tanstack/react-query";
import { api } from "@/lib/api";
import { StatusBadge } from "@/components/StatusBadge";

// UI_UX_DESIGN_BRIEF.md Section 8: Run Browser columns.
export default function RunsPage() {
  const runs = useQuery({ queryKey: ["runs"], queryFn: api.listRuns });

  return (
    <div className="space-y-4">
      <h1 className="text-lg font-semibold text-neutral-100">Runs</h1>
      <div className="overflow-x-auto rounded border border-neutral-800">
        <table className="w-full text-left text-sm">
          <thead className="bg-neutral-900 text-neutral-400">
            <tr>
              <th className="px-3 py-2 font-medium">Run ID</th>
              <th className="px-3 py-2 font-medium">Vehicle</th>
              <th className="px-3 py-2 font-medium">Model</th>
              <th className="px-3 py-2 font-medium">Impact</th>
              <th className="px-3 py-2 font-medium">Solver</th>
              <th className="px-3 py-2 font-medium">Quality</th>
              <th className="px-3 py-2 font-medium">Created</th>
            </tr>
          </thead>
          <tbody>
            {runs.isLoading && (
              <tr>
                <td colSpan={7} className="px-3 py-6 text-center text-neutral-500">
                  Loading…
                </td>
              </tr>
            )}
            {runs.isError && (
              <tr>
                <td colSpan={7} className="px-3 py-6 text-center text-red-400">
                  Could not reach the API ({(runs.error as Error).message}). Is the backend running?
                </td>
              </tr>
            )}
            {runs.data?.map((run) => (
              <tr key={run.id} className="border-t border-neutral-800 hover:bg-neutral-900">
                <td className="px-3 py-2 font-mono text-xs">
                  <Link href={`/runs/${encodeURIComponent(run.run_id)}`} className="text-sky-400 hover:underline">
                    {run.run_id}
                  </Link>
                </td>
                <td className="px-3 py-2">{run.vehicle_name}</td>
                <td className="px-3 py-2 tabular-nums">{run.model_version}</td>
                <td className="px-3 py-2">
                  {run.impact_type} {run.impact_speed ? `@ ${run.impact_speed} km/h` : ""}
                </td>
                <td className="px-3 py-2">
                  {run.solver} {run.solver_version}
                </td>
                <td className="px-3 py-2">
                  <StatusBadge status={run.quality_status} />
                </td>
                <td className="px-3 py-2 text-neutral-500">{new Date(run.created_at).toLocaleDateString()}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
