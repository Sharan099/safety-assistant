"use client";

import Link from "next/link";
import { useQuery } from "@tanstack/react-query";
import { api } from "@/lib/api";
import { StatusBadge } from "@/components/StatusBadge";

export default function Dashboard() {
  const runs = useQuery({ queryKey: ["runs"], queryFn: api.listRuns });
  const investigations = useQuery({ queryKey: ["investigations"], queryFn: api.listInvestigations });

  const failedRuns = runs.data?.filter((r) => r.quality_status === "FAIL") ?? [];
  const openInvestigations = investigations.data?.filter((i) => !["DECISION", "CLOSED"].includes(i.state)) ?? [];

  return (
    <div className="space-y-8">
      <div>
        <h1 className="text-lg font-semibold text-neutral-100">Dashboard</h1>
        <p className="mt-1 text-sm text-neutral-400">
          Engineering investigation workstation — not a chat assistant. Evidence stays visible, uncertainty stays
          explicit.
        </p>
      </div>

      <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-4">
        <StatCard label="Open investigations" value={openInvestigations.length} loading={investigations.isLoading} />
        <StatCard label="Total runs" value={runs.data?.length ?? 0} loading={runs.isLoading} />
        <StatCard label="Quality failures" value={failedRuns.length} loading={runs.isLoading} tone="warn" />
        <StatCard label="Total investigations" value={investigations.data?.length ?? 0} loading={investigations.isLoading} />
      </div>

      <div className="flex gap-3">
        <Link
          href="/investigations/new"
          className="rounded bg-sky-600 px-4 py-2 text-sm font-medium text-white hover:bg-sky-500"
        >
          New Investigation
        </Link>
        <Link
          href="/runs"
          className="rounded border border-neutral-700 px-4 py-2 text-sm font-medium text-neutral-200 hover:bg-neutral-900"
        >
          Browse Runs
        </Link>
      </div>

      <section>
        <h2 className="mb-2 text-sm font-semibold text-neutral-300">Recent investigations</h2>
        <div className="overflow-x-auto rounded border border-neutral-800">
          <table className="w-full text-left text-sm">
            <thead className="bg-neutral-900 text-neutral-400">
              <tr>
                <th className="px-3 py-2 font-medium">Title</th>
                <th className="px-3 py-2 font-medium">Question</th>
                <th className="px-3 py-2 font-medium">State</th>
                <th className="px-3 py-2 font-medium">Created</th>
              </tr>
            </thead>
            <tbody>
              {investigations.data?.slice(0, 10).map((inv) => (
                <tr key={inv.id} className="border-t border-neutral-800 hover:bg-neutral-900">
                  <td className="px-3 py-2">
                    <Link href={`/investigations/${inv.id}`} className="text-sky-400 hover:underline">
                      {inv.title}
                    </Link>
                  </td>
                  <td className="max-w-md truncate px-3 py-2 text-neutral-400">{inv.question}</td>
                  <td className="px-3 py-2">
                    <StatusBadge status={inv.state} />
                  </td>
                  <td className="px-3 py-2 text-neutral-500">{new Date(inv.created_at).toLocaleString()}</td>
                </tr>
              ))}
              {investigations.data?.length === 0 && (
                <tr>
                  <td colSpan={4} className="px-3 py-6 text-center text-neutral-500">
                    No investigations yet — start one from the button above.
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
      </section>
    </div>
  );
}

function StatCard({
  label,
  value,
  loading,
  tone = "default",
}: {
  label: string;
  value: number;
  loading: boolean;
  tone?: "default" | "warn";
}) {
  return (
    <div className="rounded border border-neutral-800 bg-neutral-900/50 p-4">
      <div className="text-xs text-neutral-500">{label}</div>
      <div className={`mt-1 text-2xl font-semibold tabular-nums ${tone === "warn" && value > 0 ? "text-amber-400" : "text-neutral-100"}`}>
        {loading ? "—" : value}
      </div>
    </div>
  );
}
