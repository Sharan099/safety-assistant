"use client";

import { apiClearSession, artifactDocUrl, artifactSessionUrl, type SessionDocument } from "@/lib/api";

type Props = {
  sessionId: string;
  documents: SessionDocument[];
};

export function ArtifactsPanel({ sessionId, documents }: Props) {
  const ready = documents.filter((d) => d.status === "ready");

  async function clearSession() {
    if (!confirm("Clear this session and delete all uploaded documents?")) return;
    await apiClearSession(sessionId);
    window.location.reload();
  }

  if (!documents.length) return null;

  return (
    <section className="artifacts-panel">
      <h3>Artifacts</h3>
      <p className="artifacts-hint">Download extracted markdown, chunk JSON, and manifest.</p>
      <div className="artifacts-actions">
        <a className="btn-artifact" href={artifactSessionUrl(sessionId)} download>
          Download session (.zip)
        </a>
        {ready.map((d) => (
          <a
            key={d.doc_id}
            className="btn-artifact btn-artifact-sm"
            href={artifactDocUrl(sessionId, d.doc_id)}
            download
          >
            {d.filename}
          </a>
        ))}
      </div>
      <button type="button" className="btn-clear-session" onClick={clearSession}>
        Clear session
      </button>
      <style jsx>{`
        .artifacts-panel {
          margin-top: 1rem;
          padding-top: 0.75rem;
          border-top: 1px solid var(--border);
        }
        .artifacts-panel h3 {
          margin: 0 0 0.25rem;
          font-size: 0.85rem;
        }
        .artifacts-hint {
          margin: 0 0 0.5rem;
          font-size: 0.72rem;
          color: var(--muted);
        }
        .artifacts-actions {
          display: flex;
          flex-wrap: wrap;
          gap: 0.35rem;
        }
        .btn-artifact {
          display: inline-block;
          padding: 0.35rem 0.6rem;
          background: var(--surface-2);
          border: 1px solid var(--border);
          border-radius: 6px;
          font-size: 0.75rem;
          color: var(--text-body);
          text-decoration: none;
        }
        .btn-artifact:hover {
          border-color: var(--primary-light);
          background: var(--accent-soft);
        }
        .btn-artifact-sm {
          max-width: 100%;
          overflow: hidden;
          text-overflow: ellipsis;
          white-space: nowrap;
        }
        .btn-clear-session {
          margin-top: 0.5rem;
          background: none;
          border: none;
          color: var(--danger);
          font-size: 0.72rem;
          text-decoration: underline;
          padding: 0;
        }
      `}</style>
    </section>
  );
}
