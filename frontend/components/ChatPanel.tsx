"use client";
import { useEffect, useState } from "react";
import { useChat } from "@/hooks/useChat";
import { api } from "@/lib/apiClient";
import type { RegulationSummary } from "@/lib/types";
import { AnswerCard } from "./AnswerCard";

const EXAMPLES = [
  "What is the tibia index limit in UN R94?",
  "Compare the head performance criterion in R94 and R95",
  "What does UN R94 paragraph 5.2.1.8 require?",
  "What was the R94 tibia index limit as of 2015?",
];

export function ChatPanel() {
  const { turns, ask } = useChat();
  const [query, setQuery] = useState("");
  const [asOf, setAsOf] = useState("");
  const [regulation, setRegulation] = useState("");
  const [regulations, setRegulations] = useState<RegulationSummary[]>([]);

  useEffect(() => {
    api.regulations().then(setRegulations).catch(() => setRegulations([]));
  }, []);

  const submit = (q: string) => {
    if (!q.trim()) return;
    ask({ query: q.trim(), as_of: asOf || null, regulation_keys: regulation ? [regulation] : [], k: 8 });
    setQuery("");
  };

  return (
    <div className="space-y-6">
      <form
        onSubmit={(e) => {
          e.preventDefault();
          submit(query);
        }}
        className="space-y-3 rounded-lg border border-zinc-800 p-4"
        aria-label="Ask a regulatory question"
      >
        <label className="block text-sm text-zinc-300" htmlFor="q">
          Question
        </label>
        <textarea
          id="q"
          data-testid="query"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          rows={3}
          maxLength={2000}
          className="w-full rounded border border-zinc-700 bg-zinc-950 p-2 text-sm"
          placeholder="e.g. What is the thorax compression criterion limit in UN R94?"
        />
        <div className="flex flex-wrap items-end gap-3 text-sm">
          <label className="flex flex-col gap-1">
            <span className="text-xs text-zinc-400">As of date (historical scope)</span>
            <input type="date" value={asOf} onChange={(e) => setAsOf(e.target.value)} className="rounded border border-zinc-700 bg-zinc-950 p-1" data-testid="as-of" />
          </label>
          <label className="flex flex-col gap-1">
            <span className="text-xs text-zinc-400">Regulation</span>
            <select value={regulation} onChange={(e) => setRegulation(e.target.value)} className="rounded border border-zinc-700 bg-zinc-950 p-1" data-testid="regulation">
              <option value="">all</option>
              {regulations.map((r) => (
                <option key={r.regulation_key} value={r.regulation_key}>
                  {r.regulation_key} · {r.versions.find((v) => v.status === "ACTIVE")?.label ?? r.versions[0]?.label}
                </option>
              ))}
            </select>
          </label>
          <button type="submit" data-testid="ask" className="rounded bg-sky-600 px-4 py-1.5 font-medium text-white hover:bg-sky-500">
            Ask
          </button>
        </div>
        <div className="flex flex-wrap gap-2">
          {EXAMPLES.map((ex) => (
            <button key={ex} type="button" onClick={() => submit(ex)} className="rounded-full border border-zinc-700 px-3 py-1 text-xs text-zinc-300 hover:border-zinc-500">
              {ex}
            </button>
          ))}
        </div>
      </form>

      <div className="space-y-6">
        {[...turns].reverse().map((t, i) => (
          <div key={turns.length - i} className="space-y-2">
            <p className="text-sm text-zinc-400">
              <span className="text-zinc-500">Q:</span> {t.request.query}
              {t.request.as_of ? <span className="ml-2 text-xs">(as of {t.request.as_of})</span> : null}
            </p>
            {t.pending && <p className="text-sm text-zinc-500" data-testid="pending">Retrieving evidence…</p>}
            {t.error && <p className="text-sm text-rose-300" data-testid="error">{t.error}</p>}
            {t.response && <AnswerCard response={t.response} />}
          </div>
        ))}
      </div>
    </div>
  );
}
