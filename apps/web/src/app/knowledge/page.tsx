"use client";

import { useState } from "react";
import { useMutation } from "@tanstack/react-query";
import { api } from "@/lib/api";
import type { RetrievedChunk } from "@/lib/types";

// UI_UX_DESIGN_BRIEF.md Section 19: Knowledge Viewer.
export default function KnowledgePage() {
  const [query, setQuery] = useState("");
  const [sourceType, setSourceType] = useState("");
  const [results, setResults] = useState<RetrievedChunk[]>([]);

  const search = useMutation({
    mutationFn: () => api.searchKnowledge(query, sourceType ? { source_type: sourceType } : undefined),
    onSuccess: setResults,
  });

  return (
    <div className="max-w-3xl space-y-6">
      <h1 className="text-lg font-semibold text-neutral-100">Knowledge</h1>
      <p className="text-sm text-neutral-400">
        Search regulations and solver documentation actually ingested into this system — see{" "}
        <code className="text-xs">knowledge/00_registry/source_manifest.yaml</code>. No source is shown here unless
        it was retrieved (UI_UX_DESIGN_BRIEF.md Section 21).
      </p>

      <form
        className="flex gap-2"
        onSubmit={(e) => {
          e.preventDefault();
          if (query.trim()) search.mutate();
        }}
      >
        <input
          className="flex-1 rounded border border-neutral-700 bg-neutral-900 px-3 py-2 text-sm text-neutral-100"
          placeholder="e.g. frontal collision occupant protection"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
        />
        <select
          className="rounded border border-neutral-700 bg-neutral-900 px-3 py-2 text-sm text-neutral-100"
          value={sourceType}
          onChange={(e) => setSourceType(e.target.value)}
        >
          <option value="">All sources</option>
          <option value="REGULATION">Regulation</option>
          <option value="OFFICIAL_DOCUMENTATION">Official documentation</option>
        </select>
        <button
          type="submit"
          disabled={search.isPending}
          className="rounded bg-sky-600 px-4 py-2 text-sm font-medium text-white hover:bg-sky-500 disabled:opacity-50"
        >
          {search.isPending ? "Searching…" : "Search"}
        </button>
      </form>

      <div className="space-y-3">
        {results.map((r) => (
          <div key={r.chunk_id} className="rounded border border-neutral-800 p-3">
            <div className="mb-1 flex flex-wrap items-center gap-2 text-xs text-neutral-500">
              <span className="rounded border border-neutral-700 px-1.5 py-0.5 font-mono text-neutral-300">
                {r.authority_level}
              </span>
              <span className="font-medium text-neutral-300">{r.document_title}</span>
              {r.page_start != null && (
                <span>
                  p.{r.page_start}
                  {r.page_end && r.page_end !== r.page_start ? `–${r.page_end}` : ""}
                </span>
              )}
              {r.section_title && <span>· {r.section_title}</span>}
            </div>
            <p className="whitespace-pre-line text-sm text-neutral-300">{r.content}</p>
          </div>
        ))}
        {search.isSuccess && results.length === 0 && (
          <p className="text-sm text-neutral-500">No results.</p>
        )}
      </div>
    </div>
  );
}
