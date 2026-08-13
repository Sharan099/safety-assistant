"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";
import { useMutation, useQuery } from "@tanstack/react-query";
import { api, ApiError } from "@/lib/api";

// APP_FLOW.md Section 4: New Investigation — Run A, Run B, question, primary
// metric. "The user should not need to write a long AI prompt."
const PRIMARY_METRICS = [
  "chest_deflection",
  "chest_acceleration",
  "belt_force",
  "pelvis_acceleration",
];

export default function NewInvestigationPage() {
  const router = useRouter();
  const runs = useQuery({ queryKey: ["runs"], queryFn: api.listRuns });

  const [runAId, setRunAId] = useState("");
  const [runBId, setRunBId] = useState("");
  const [question, setQuestion] = useState("");
  const [primaryMetric, setPrimaryMetric] = useState(PRIMARY_METRICS[0]);
  const [error, setError] = useState<string | null>(null);

  const create = useMutation({
    mutationFn: () =>
      api.createInvestigation({ run_a_id: runAId, run_b_id: runBId, question, primary_metric: primaryMetric }),
    onSuccess: (investigation) => router.push(`/investigations/${investigation.id}`),
    onError: (err) => setError(err instanceof ApiError ? String(err.detail) : String(err)),
  });

  const canSubmit = runAId && runBId && runAId !== runBId && question.trim().length > 0;

  return (
    <div className="max-w-xl space-y-6">
      <h1 className="text-lg font-semibold text-neutral-100">New Investigation</h1>

      <form
        className="space-y-4"
        onSubmit={(e) => {
          e.preventDefault();
          setError(null);
          create.mutate();
        }}
      >
        <Field label="Run A (baseline)">
          <RunSelect value={runAId} onChange={setRunAId} runs={runs.data ?? []} />
        </Field>
        <Field label="Run B (comparison)">
          <RunSelect value={runBId} onChange={setRunBId} runs={runs.data ?? []} />
        </Field>
        <Field label="Engineering question">
          <textarea
            className="w-full rounded border border-neutral-700 bg-neutral-900 px-3 py-2 text-sm text-neutral-100"
            rows={3}
            placeholder="Why did chest deflection increase between Run A and Run B?"
            value={question}
            onChange={(e) => setQuestion(e.target.value)}
          />
        </Field>
        <Field label="Primary metric">
          <select
            className="w-full rounded border border-neutral-700 bg-neutral-900 px-3 py-2 text-sm text-neutral-100"
            value={primaryMetric}
            onChange={(e) => setPrimaryMetric(e.target.value)}
          >
            {PRIMARY_METRICS.map((m) => (
              <option key={m} value={m}>
                {m}
              </option>
            ))}
          </select>
        </Field>

        {error && <p className="text-sm text-red-400">{error}</p>}

        <button
          type="submit"
          disabled={!canSubmit || create.isPending}
          className="rounded bg-sky-600 px-4 py-2 text-sm font-medium text-white hover:bg-sky-500 disabled:opacity-50"
        >
          {create.isPending ? "Creating…" : "Create investigation"}
        </button>
      </form>
    </div>
  );
}

function Field({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <label className="block">
      <span className="mb-1 block text-sm text-neutral-400">{label}</span>
      {children}
    </label>
  );
}

function RunSelect({
  value,
  onChange,
  runs,
}: {
  value: string;
  onChange: (v: string) => void;
  runs: { run_id: string; vehicle_name: string; quality_status: string }[];
}) {
  return (
    <select
      className="w-full rounded border border-neutral-700 bg-neutral-900 px-3 py-2 text-sm text-neutral-100"
      value={value}
      onChange={(e) => onChange(e.target.value)}
    >
      <option value="">Select a run…</option>
      {runs.map((r) => (
        <option key={r.run_id} value={r.run_id}>
          {r.run_id} — {r.vehicle_name} ({r.quality_status})
        </option>
      ))}
    </select>
  );
}
