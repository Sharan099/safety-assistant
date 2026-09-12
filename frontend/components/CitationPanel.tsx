"use client";
import { useState } from "react";
import type { Citation, Evidence } from "@/lib/types";
import { api } from "@/lib/apiClient";

function Validity({ from, to, status }: { from: string | null; to: string | null; status: string }) {
  const stale = status !== "ACTIVE";
  return (
    <span className={`rounded px-1.5 py-0.5 text-xs ${stale ? "bg-amber-900/60 text-amber-200" : "bg-emerald-900/50 text-emerald-200"}`}>
      {status} · in force {from ?? "unknown"} → {to ?? "open"}
    </span>
  );
}

export function CitationPanel({
  citations,
  evidence,
  highlighted,
}: {
  citations: Citation[];
  evidence: Evidence[];
  highlighted: string | null;
}) {
  const [open, setOpen] = useState<string | null>(null);
  const byId = Object.fromEntries(evidence.map((e) => [e.evidence_id, e]));
  const list = citations.length ? citations : evidence.map((e) => ({
    evidence_id: e.evidence_id, label: e.citation_label, regulation_key: e.regulation_key, version_label: e.version_label,
    section_path: e.section_path, page_start: e.page_start, page_end: e.page_end, source_sha256: e.source_sha256,
    source_uri: e.source_uri, valid_from: e.valid_from, valid_to: e.valid_to, version_status: e.version_status,
  }));
  if (!list.length) return null;
  return (
    <section aria-label="Evidence and citations" className="space-y-2">
      <h3 className="text-sm font-semibold text-zinc-300">Evidence ({list.length})</h3>
      <ul className="space-y-2">
        {list.map((c) => {
          const e = byId[c.evidence_id];
          const isOpen = open === c.evidence_id;
          return (
            <li
              key={c.evidence_id}
              data-testid="citation"
              className={`rounded border p-3 text-sm ${highlighted === c.evidence_id ? "border-sky-400" : "border-zinc-800"}`}
            >
              <div className="flex flex-wrap items-center gap-2">
                <span className="font-mono text-xs text-sky-300">{c.evidence_id}</span>
                <span className="font-medium">{c.label}</span>
                <Validity from={c.valid_from} to={c.valid_to} status={c.version_status} />
                {e?.normative === true && <span className="rounded bg-zinc-800 px-1.5 text-xs">normative</span>}
                {e?.normative === false && <span className="rounded bg-zinc-800 px-1.5 text-xs">informative</span>}
                {e?.chunk_type === "TABLE" && <span className="rounded bg-zinc-800 px-1.5 text-xs">table</span>}
              </div>
              <div className="mt-1 text-xs text-zinc-400">
                {c.regulation_key} · {c.version_label} · section {c.section_path}
                {c.page_start ? ` · p. ${c.page_start}${c.page_end && c.page_end !== c.page_start ? `–${c.page_end}` : ""}` : ""}
                {" · "}sha256 {c.source_sha256.slice(0, 12)}…
                {c.source_uri && (
                  <>
                    {" · "}
                    <a className="underline" href={c.source_uri} target="_blank" rel="noreferrer">official source</a>
                  </>
                )}
                {e && (
                  <>
                    {" · "}
                    <a className="underline" href={api.evidenceUrl(e.chunk_id)} target="_blank" rel="noreferrer">open record</a>
                  </>
                )}
              </div>
              {e && (
                <button
                  className="mt-2 text-xs text-zinc-300 underline"
                  onClick={() => setOpen(isOpen ? null : c.evidence_id)}
                  aria-expanded={isOpen}
                >
                  {isOpen ? "hide text" : "show text"}
                </button>
              )}
              {isOpen && e && (
                <div className="mt-2 space-y-2">
                  <pre className="whitespace-pre-wrap rounded bg-zinc-900 p-2 text-xs text-zinc-200">{e.content}</pre>
                  {e.parent_context && (
                    <details className="text-xs text-zinc-400">
                      <summary>parent section</summary>
                      <pre className="mt-1 whitespace-pre-wrap">{e.parent_context}</pre>
                    </details>
                  )}
                  {e.related.map((r) => (
                    <details key={r.path} className="text-xs text-zinc-400">
                      <summary>cross-reference: {r.citation_label} (via “{r.via}”)</summary>
                      <pre className="mt-1 whitespace-pre-wrap">{r.excerpt}</pre>
                    </details>
                  ))}
                </div>
              )}
            </li>
          );
        })}
      </ul>
    </section>
  );
}
