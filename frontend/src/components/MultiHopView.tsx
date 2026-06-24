"use client";

import type { MultiHopPayload } from "@/lib/api";

const ROLE_LABELS: Record<string, string> = {
  binding: "Binding requirement",
  measured: "Measured / observed",
  advisory: "Advisory",
};

type Props = {
  multiHop?: MultiHopPayload;
};

export function MultiHopView({ multiHop }: Props) {
  if (!multiHop?.hops?.length || multiHop.hops.length < 2) return null;

  return (
    <div className="multi-hop" role="region" aria-label="Cross-document reasoning path">
      <h4>Reasoning path (multi-hop)</h4>
      <ol className="hop-list">
        {multiHop.hops.map((hop) => (
          <li key={hop.hop_id} className={hop.abstained ? "hop-abstain" : "hop-ok"}>
            <div className="hop-head">
              <span className="hop-num">Hop {hop.hop_id}</span>
              <strong>{hop.label}</strong>
              <span className={`hop-role hop-role-${hop.authority_role || "advisory"}`}>
                {ROLE_LABELS[hop.authority_role || "advisory"] || hop.authority_role}
              </span>
            </div>
            <p className="hop-query">{hop.query}</p>
            {hop.abstained ? (
              <p className="hop-abstain-msg">{hop.abstain_reason || "Not found in uploaded documents"}</p>
            ) : (
              <p className="hop-found">{hop.document_count ?? 0} source(s) retrieved</p>
            )}
          </li>
        ))}
      </ol>
      <style jsx>{`
        .multi-hop {
          margin-top: 0.75rem;
          padding: 0.75rem;
          background: var(--surface-2);
          border: 1px solid var(--border);
          border-radius: var(--radius);
        }
        .multi-hop h4 {
          margin: 0 0 0.5rem;
          font-size: 0.82rem;
          color: var(--text);
        }
        .hop-list {
          margin: 0;
          padding-left: 1.1rem;
          display: flex;
          flex-direction: column;
          gap: 0.55rem;
        }
        .hop-head {
          display: flex;
          flex-wrap: wrap;
          gap: 0.35rem;
          align-items: center;
          font-size: 0.8rem;
        }
        .hop-num {
          font-family: "Fira Code", monospace;
          font-size: 0.72rem;
          color: var(--muted);
        }
        .hop-role {
          font-size: 0.68rem;
          font-weight: 700;
          padding: 0.1rem 0.35rem;
          border-radius: 4px;
        }
        .hop-role-binding { background: var(--legal); color: #fff; }
        .hop-role-measured { background: var(--historical); color: #fff; }
        .hop-role-advisory { background: var(--engref); color: #fff; }
        .hop-query {
          margin: 0.2rem 0 0;
          font-size: 0.75rem;
          color: var(--muted);
        }
        .hop-abstain-msg {
          margin: 0.25rem 0 0;
          font-size: 0.78rem;
          color: var(--danger);
          font-weight: 600;
        }
        .hop-found {
          margin: 0.25rem 0 0;
          font-size: 0.75rem;
          color: var(--success);
        }
        .hop-abstain { border-left: 3px solid var(--danger); padding-left: 0.5rem; }
        .hop-ok { border-left: 3px solid var(--success); padding-left: 0.5rem; }
      `}</style>
    </div>
  );
}
