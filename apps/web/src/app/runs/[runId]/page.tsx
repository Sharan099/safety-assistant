"use client";

import { useParams } from "next/navigation";
import { useQuery } from "@tanstack/react-query";
import { api } from "@/lib/api";
import { StatusBadge } from "@/components/StatusBadge";

// UI_UX_DESIGN_BRIEF.md Section 9: Run Manifest — a property list, source on click
// deferred (V1: values only, no click-through provenance yet).
export default function RunDetailPage() {
  const params = useParams<{ runId: string }>();
  const run = useQuery({ queryKey: ["run", params.runId], queryFn: () => api.getRun(params.runId) });

  if (run.isLoading) return <p className="text-neutral-500">Loading…</p>;
  if (run.isError || !run.data) return <p className="text-red-400">Run not found.</p>;

  const r = run.data;
  const rows: [string, React.ReactNode][] = [
    ["Vehicle", r.vehicle_name],
    ["Model version", r.model_version],
    ["Solver", `${r.solver ?? "—"} ${r.solver_version ?? ""}`],
    ["Dummy", r.dummy_version ?? "—"],
    ["Impact", `${r.impact_type ?? "—"}${r.impact_speed ? ` @ ${r.impact_speed} km/h` : ""}`],
    ["Result processing version", r.result_processing_version ?? "—"],
    ["Quality status", <StatusBadge key="q" status={r.quality_status} />],
    ["Created", new Date(r.created_at).toLocaleString()],
  ];

  return (
    <div className="space-y-6">
      <h1 className="font-mono text-lg font-semibold text-neutral-100">{r.run_id}</h1>

      <div className="max-w-2xl overflow-hidden rounded border border-neutral-800">
        <table className="w-full text-left text-sm">
          <tbody>
            {rows.map(([label, value]) => (
              <tr key={label} className="border-t border-neutral-800 first:border-t-0">
                <td className="w-64 bg-neutral-900/50 px-3 py-2 text-neutral-400">{label}</td>
                <td className="px-3 py-2 text-neutral-100">{value}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {r.seat_configuration && (
        <ConfigBlock title="Seat configuration" data={r.seat_configuration} />
      )}
      {r.restraint_configuration && (
        <ConfigBlock title="Restraint configuration" data={r.restraint_configuration} />
      )}
    </div>
  );
}

function ConfigBlock({ title, data }: { title: string; data: Record<string, unknown> }) {
  return (
    <div>
      <h2 className="mb-2 text-sm font-semibold text-neutral-300">{title}</h2>
      <pre className="max-w-2xl overflow-x-auto rounded border border-neutral-800 bg-neutral-900/50 p-3 text-xs text-neutral-300">
        {JSON.stringify(data, null, 2)}
      </pre>
    </div>
  );
}
