"use client";

import { useState } from "react";
import { useParams } from "next/navigation";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { api, ApiError } from "@/lib/api";
import { StatusBadge } from "@/components/StatusBadge";
import { QualityCheckTable } from "@/components/QualityCheckTable";
import { ComparabilityMatrix } from "@/components/ComparabilityMatrix";
import { HypothesisCard } from "@/components/HypothesisCard";
import { EvidenceCard } from "@/components/EvidenceCard";
import { SignalChart } from "@/components/SignalChart";

const SIGNAL_OPTIONS = [
  "chest_deflection",
  "chest_acceleration",
  "chest_velocity",
  "belt_force",
  "pelvis_acceleration",
  "torso_rotation",
  "airbag_pressure",
  "vehicle_pulse",
];

const REVIEW_DECISIONS = [
  "ACCEPT",
  "REJECT",
  "MODIFY",
  "REQUEST_SIGNAL",
  "REQUEST_SOURCE",
  "REQUEST_CONTROLLED_COMPARISON",
  "MARK_INCONCLUSIVE",
];

export default function InvestigationWorkspace() {
  const params = useParams<{ id: string }>();
  const id = params.id;
  const qc = useQueryClient();

  const investigation = useQuery({ queryKey: ["investigation", id], queryFn: () => api.getInvestigation(id) });
  const evidence = useQuery({ queryKey: ["evidence", id], queryFn: () => api.listEvidence(id), enabled: !!id });
  const hypotheses = useQuery({ queryKey: ["hypotheses", id], queryFn: () => api.listHypotheses(id), enabled: !!id });

  const [quality, setQuality] = useState<Awaited<ReturnType<typeof api.computeQuality>> | null>(null);
  const [comparability, setComparability] = useState<Awaited<ReturnType<typeof api.computeComparability>> | null>(null);
  const [configDiff, setConfigDiff] = useState<Awaited<ReturnType<typeof api.computeConfigurationDiff>> | null>(null);
  const [selectedSignal, setSelectedSignal] = useState(SIGNAL_OPTIONS[0]);
  const [signalData, setSignalData] = useState<Awaited<ReturnType<typeof api.analyzeSignal>> | null>(null);
  const [timeseries, setTimeseries] = useState<Awaited<ReturnType<typeof api.getSignalTimeseries>> | null>(null);
  const [reviewDecision, setReviewDecision] = useState(REVIEW_DECISIONS[0]);
  const [reviewComment, setReviewComment] = useState("");
  const [actionError, setActionError] = useState<string | null>(null);

  const refreshAll = () => {
    qc.invalidateQueries({ queryKey: ["investigation", id] });
    qc.invalidateQueries({ queryKey: ["evidence", id] });
    qc.invalidateQueries({ queryKey: ["hypotheses", id] });
  };

  const runFullAgent = useMutation({
    mutationFn: () => api.runAgent(id),
    onSuccess: refreshAll,
    onError: (e) => setActionError(e instanceof ApiError ? String(e.detail) : String(e)),
  });
  const runQuality = useMutation({
    mutationFn: () => api.computeQuality(id),
    onSuccess: (data) => {
      setQuality(data);
      refreshAll();
    },
    onError: (e) => setActionError(e instanceof ApiError ? String(e.detail) : String(e)),
  });
  const runComparability = useMutation({
    mutationFn: () => api.computeComparability(id),
    onSuccess: (data) => {
      setComparability(data);
      refreshAll();
    },
    onError: (e) => setActionError(e instanceof ApiError ? String(e.detail) : String(e)),
  });
  const runConfigDiff = useMutation({
    mutationFn: () => api.computeConfigurationDiff(id),
    onSuccess: (data) => {
      setConfigDiff(data);
      refreshAll();
    },
    onError: (e) => setActionError(e instanceof ApiError ? String(e.detail) : String(e)),
  });
  const runSignal = useMutation({
    mutationFn: async () => {
      const [analysis, ts] = await Promise.all([
        api.analyzeSignal(id, selectedSignal),
        api.getSignalTimeseries(id, selectedSignal),
      ]);
      return { analysis, ts };
    },
    onSuccess: ({ analysis, ts }) => {
      setSignalData(analysis);
      setTimeseries(ts);
      refreshAll();
    },
    onError: (e) => setActionError(e instanceof ApiError ? String(e.detail) : String(e)),
  });
  const submitReview = useMutation({
    mutationFn: () => api.submitReview(id, { decision: reviewDecision, comment: reviewComment || undefined }),
    onSuccess: refreshAll,
    onError: (e) => setActionError(e instanceof ApiError ? String(e.detail) : String(e)),
  });

  if (investigation.isLoading) return <p className="text-neutral-500">Loading…</p>;
  if (investigation.isError || !investigation.data) return <p className="text-red-400">Investigation not found.</p>;

  const inv = investigation.data;

  return (
    <div className="max-w-5xl space-y-8">
      <div>
        <div className="flex items-center gap-3">
          <h1 className="text-lg font-semibold text-neutral-100">{inv.title}</h1>
          <StatusBadge status={inv.state} />
        </div>
        <p className="mt-1 text-sm text-neutral-400">{inv.question}</p>
        <p className="mt-1 text-xs text-neutral-600">
          Run A: <span className="font-mono">{inv.run_a_id}</span> · Run B:{" "}
          <span className="font-mono">{inv.run_b_id}</span>
          {inv.primary_metric && <> · primary metric: {inv.primary_metric}</>}
        </p>
      </div>

      <div className="flex flex-wrap items-center gap-2 border-b border-neutral-800 pb-6">
        <button
          onClick={() => {
            setActionError(null);
            runFullAgent.mutate();
          }}
          disabled={runFullAgent.isPending}
          className="rounded bg-sky-600 px-4 py-2 text-sm font-medium text-white hover:bg-sky-500 disabled:opacity-50"
        >
          {runFullAgent.isPending ? "Running agent…" : "Run full agent"}
        </button>
        <SecondaryButton onClick={() => runQuality.mutate()} pending={runQuality.isPending}>
          Quality
        </SecondaryButton>
        <SecondaryButton onClick={() => runConfigDiff.mutate()} pending={runConfigDiff.isPending}>
          Configuration diff
        </SecondaryButton>
        <SecondaryButton onClick={() => runComparability.mutate()} pending={runComparability.isPending}>
          Comparability
        </SecondaryButton>
        {actionError && <span className="text-sm text-red-400">{actionError}</span>}
      </div>

      {quality && (
        <section className="grid grid-cols-1 gap-4 md:grid-cols-2">
          <QualityCheckTable label={`Run A (${inv.run_a_id})`} summary={quality.run_a} />
          <QualityCheckTable label={`Run B (${inv.run_b_id})`} summary={quality.run_b} />
        </section>
      )}

      {comparability && (
        <section>
          <ComparabilityMatrix summary={comparability} />
        </section>
      )}

      {configDiff && (
        <section>
          <h2 className="mb-2 text-sm font-semibold text-neutral-300">Configuration diff</h2>
          <div className="overflow-hidden rounded border border-neutral-800">
            <table className="w-full text-left text-sm">
              <thead className="bg-neutral-900 text-neutral-400">
                <tr>
                  <th className="px-3 py-2 font-medium">Path</th>
                  <th className="px-3 py-2 font-medium">Run A</th>
                  <th className="px-3 py-2 font-medium">Run B</th>
                  <th className="px-3 py-2 font-medium">Status</th>
                </tr>
              </thead>
              <tbody>
                {configDiff
                  .filter((d) => d.change_status !== "SAME")
                  .map((d) => (
                    <tr key={d.path} className="border-t border-neutral-800">
                      <td className="px-3 py-1.5 font-mono text-xs text-neutral-300">{d.path}</td>
                      <td className="px-3 py-1.5 text-neutral-400">{JSON.stringify(d.run_a_value)}</td>
                      <td className="px-3 py-1.5 text-neutral-400">{JSON.stringify(d.run_b_value)}</td>
                      <td className="px-3 py-1.5">
                        <StatusBadge status={d.change_status} />
                      </td>
                    </tr>
                  ))}
                {configDiff.every((d) => d.change_status === "SAME") && (
                  <tr>
                    <td colSpan={4} className="px-3 py-4 text-center text-neutral-500">
                      No configuration differences detected.
                    </td>
                  </tr>
                )}
              </tbody>
            </table>
          </div>
        </section>
      )}

      <section>
        <h2 className="mb-2 text-sm font-semibold text-neutral-300">Signal workspace</h2>
        <div className="mb-3 flex items-center gap-2">
          <select
            className="rounded border border-neutral-700 bg-neutral-900 px-3 py-1.5 text-sm text-neutral-100"
            value={selectedSignal}
            onChange={(e) => setSelectedSignal(e.target.value)}
          >
            {SIGNAL_OPTIONS.map((s) => (
              <option key={s} value={s}>
                {s}
              </option>
            ))}
          </select>
          <SecondaryButton onClick={() => runSignal.mutate()} pending={runSignal.isPending}>
            Analyze
          </SecondaryButton>
        </div>

        {timeseries && (
          <div className="rounded border border-neutral-800 p-3">
            <SignalChart
              timeS={timeseries.time_s}
              runA={timeseries.run_a}
              runB={timeseries.run_b}
              unit={timeseries.unit}
              divergenceMs={signalData?.divergence?.time_ms}
            />
          </div>
        )}

        {signalData && (
          <div className="mt-3 grid grid-cols-2 gap-4 md:grid-cols-4">
            <Metric label="Run A peak" value={signalData.run_a_features.peak.toFixed(1)} />
            <Metric label="Run B peak" value={signalData.run_b_features.peak.toFixed(1)} />
            <Metric label="Correlation" value={signalData.correlation.toFixed(3)} />
            <Metric
              label="First divergence"
              value={signalData.divergence ? `${signalData.divergence.time_ms.toFixed(1)} ms` : "none detected"}
            />
          </div>
        )}
      </section>

      <section>
        <h2 className="mb-2 text-sm font-semibold text-neutral-300">Evidence ({evidence.data?.length ?? 0})</h2>
        <div className="space-y-2">
          {evidence.data?.map((e) => (
            <EvidenceCard key={e.id} evidence={e} />
          ))}
          {evidence.data?.length === 0 && (
            <p className="text-sm text-neutral-500">
              No evidence yet — run the agent or the individual analysis steps above.
            </p>
          )}
        </div>
      </section>

      <section>
        <h2 className="mb-2 text-sm font-semibold text-neutral-300">Hypotheses ({hypotheses.data?.length ?? 0})</h2>
        <div className="space-y-2">
          {hypotheses.data?.map((h) => (
            <HypothesisCard key={h.id} hypothesis={h} />
          ))}
        </div>
      </section>

      <section className="rounded border border-neutral-800 p-4">
        <h2 className="mb-3 text-sm font-semibold text-neutral-300">Engineer review</h2>
        <div className="flex flex-wrap items-end gap-3">
          <label className="block">
            <span className="mb-1 block text-xs text-neutral-500">Decision</span>
            <select
              className="rounded border border-neutral-700 bg-neutral-900 px-3 py-1.5 text-sm text-neutral-100"
              value={reviewDecision}
              onChange={(e) => setReviewDecision(e.target.value)}
            >
              {REVIEW_DECISIONS.map((d) => (
                <option key={d} value={d}>
                  {d}
                </option>
              ))}
            </select>
          </label>
          <label className="block flex-1">
            <span className="mb-1 block text-xs text-neutral-500">Comment</span>
            <input
              className="w-full rounded border border-neutral-700 bg-neutral-900 px-3 py-1.5 text-sm text-neutral-100"
              value={reviewComment}
              onChange={(e) => setReviewComment(e.target.value)}
              placeholder="Optional"
            />
          </label>
          <button
            onClick={() => submitReview.mutate()}
            disabled={submitReview.isPending}
            className="rounded bg-emerald-700 px-4 py-2 text-sm font-medium text-white hover:bg-emerald-600 disabled:opacity-50"
          >
            Submit review
          </button>
        </div>
      </section>
    </div>
  );
}

function SecondaryButton({
  children,
  onClick,
  pending,
}: {
  children: React.ReactNode;
  onClick: () => void;
  pending: boolean;
}) {
  return (
    <button
      onClick={onClick}
      disabled={pending}
      className="rounded border border-neutral-700 px-3 py-2 text-sm text-neutral-200 hover:bg-neutral-900 disabled:opacity-50"
    >
      {pending ? "…" : children}
    </button>
  );
}

function Metric({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded border border-neutral-800 bg-neutral-900/40 p-3">
      <div className="text-xs text-neutral-500">{label}</div>
      <div className="mt-1 font-mono text-sm text-neutral-100">{value}</div>
    </div>
  );
}
