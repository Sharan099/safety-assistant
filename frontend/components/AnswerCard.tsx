"use client";
import { useState } from "react";
import type { AnswerResponse } from "@/lib/types";
import { CitationPanel } from "./CitationPanel";
import { api } from "@/lib/apiClient";

const MODE_LABEL: Record<AnswerResponse["mode"], string> = {
  GENERATED: "Answer grounded in the cited evidence",
  EVIDENCE_ONLY: "Evidence only: no generated answer (LLM disabled or unavailable)",
  ABSTAINED: "No answer given",
};

export function AnswerCard({ response }: { response: AnswerResponse }) {
  const [highlight, setHighlight] = useState<string | null>(null);
  const [rated, setRated] = useState<1 | -1 | null>(null);
  const scope = response.scope;
  return (
    <article data-testid="answer" data-mode={response.mode} className="space-y-4 rounded-lg border border-zinc-800 p-4">
      <header className="flex flex-wrap items-center gap-2 text-xs text-zinc-400">
        <span
          className={`rounded px-2 py-0.5 font-medium ${
            response.mode === "GENERATED" ? "bg-emerald-900/50 text-emerald-200" : response.mode === "ABSTAINED" ? "bg-rose-900/50 text-rose-200" : "bg-zinc-800 text-zinc-200"
          }`}
        >
          {MODE_LABEL[response.mode]}
        </span>
        <span>intent: {scope.intent}</span>
        <span>scope: {scope.regulation_keys.length ? scope.regulation_keys.join(", ") : "all active regulations"}</span>
        <span>{scope.as_of ? `as of ${scope.as_of} (historical)` : "currently in force"}</span>
        <span>{response.latency_ms.total ? `${Math.round(response.latency_ms.total)} ms` : ""}</span>
        <span className="font-mono">trace {response.trace_id.slice(0, 8)}</span>
      </header>

      {response.mode === "ABSTAINED" && (
        <p data-testid="abstain" className="text-sm text-rose-200">
          <strong>{response.abstain_reason?.replace(/_/g, " ")}:</strong> {response.answer}
        </p>
      )}

      {response.mode === "GENERATED" && (
        <div className="space-y-3">
          <p className="whitespace-pre-wrap text-sm leading-relaxed">{response.answer}</p>
          <ul className="space-y-1">
            {response.claims.map((c, i) => (
              <li key={i} className="flex flex-wrap items-start gap-2 text-sm">
                <span className={`mt-0.5 rounded px-1.5 text-xs ${c.kind === "REQUIREMENT" ? "bg-sky-900/50 text-sky-200" : "bg-zinc-800 text-zinc-300"}`}>
                  {c.kind === "REQUIREMENT" ? "regulation text" : "interpretation"}
                </span>
                <span>{c.text}</span>
                <span className="flex gap-1">
                  {c.evidence_ids.map((id) => (
                    <button
                      key={id}
                      className="rounded border border-sky-700 px-1 font-mono text-xs text-sky-300"
                      onMouseEnter={() => setHighlight(id)}
                      onMouseLeave={() => setHighlight(null)}
                      onClick={() => setHighlight(id)}
                    >
                      {id}
                    </button>
                  ))}
                </span>
              </li>
            ))}
          </ul>
          {response.validation && !response.validation.ok && (
            <p className="text-xs text-amber-200">
              {response.validation.dropped_claims} claim(s) were removed because they could not be verified against the evidence.
            </p>
          )}
        </div>
      )}

      {response.warnings.length > 0 && (
        <ul data-testid="warnings" className="space-y-1 rounded bg-amber-950/40 p-2 text-xs text-amber-200">
          {response.warnings.map((w, i) => (
            <li key={i}>⚠ {w}</li>
          ))}
        </ul>
      )}

      <CitationPanel citations={response.citations} evidence={response.evidence} highlighted={highlight} />

      <footer className="flex items-center gap-3 text-xs text-zinc-500">
        <span>Was this useful?</span>
        {([1, -1] as const).map((r) => (
          <button
            key={r}
            disabled={rated !== null}
            onClick={() => api.feedback(response.trace_id, r).then(() => setRated(r)).catch(() => undefined)}
            className={`rounded border px-2 py-0.5 ${rated === r ? "border-sky-400 text-sky-300" : "border-zinc-700"}`}
          >
            {r === 1 ? "yes" : "no"}
          </button>
        ))}
      </footer>
    </article>
  );
}
