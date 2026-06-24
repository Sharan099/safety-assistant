"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import {
  apiConfirmAuthorityTier,
  apiIngestStatus,
  apiSessionDocuments,
  apiUploadSessionDocument,
  AUTHORITY_TIERS,
  type IngestJob,
  type SessionDocument,
} from "@/lib/api";
import { ArtifactsPanel } from "@/components/ArtifactsPanel";

const STAGE_LABELS: Record<string, string> = {
  queued: "Queued",
  extracting: "Extracting (OCR)",
  chunking: "Chunking",
  classifying: "Classifying metadata",
  pending_tier: "Awaiting tier confirmation",
  embedding: "Embedding",
  indexing: "Building index",
  ready: "Ready",
  failed: "Failed",
};

function tierBadgeClass(tier?: string): string {
  switch (tier) {
    case "legal_binding":
      return "tier-legal";
    case "rating_protocol":
      return "tier-rating";
    case "engineering_ref":
      return "tier-engref";
    case "oem_internal":
      return "tier-oem";
    case "historical_data":
      return "tier-historical";
    default:
      return "tier-default";
  }
}

type Props = {
  sessionId: string;
  onIndexedChange?: (hasReady: boolean) => void;
};

export function WorkspacePanel({ sessionId, onIndexedChange }: Props) {
  const [docs, setDocs] = useState<SessionDocument[]>([]);
  const [dragOver, setDragOver] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [error, setError] = useState("");
  const [activeJobs, setActiveJobs] = useState<Record<string, IngestJob>>({});
  const [tierDraft, setTierDraft] = useState<Record<string, string>>({});
  const fileRef = useRef<HTMLInputElement>(null);

  const refresh = useCallback(async () => {
    try {
      const list = await apiSessionDocuments(sessionId);
      setDocs(list);
      const ready = list.some((d) => d.status === "ready" && d.tier_confirmed);
      onIndexedChange?.(ready);
    } catch {
      /* session may be new */
    }
  }, [sessionId, onIndexedChange]);

  useEffect(() => {
    refresh();
    const t = setInterval(refresh, 5000);
    return () => clearInterval(t);
  }, [refresh]);

  useEffect(() => {
    const pending = Object.entries(activeJobs).filter(
      ([, j]) => !["ready", "failed", "pending_tier"].includes(j.status),
    );
    if (!pending.length) return;

    const t = setInterval(async () => {
      for (const [jobId] of pending) {
        try {
          const s = await apiIngestStatus(jobId);
          setActiveJobs((prev) => ({ ...prev, [jobId]: s }));
          if (["ready", "failed", "pending_tier"].includes(s.status)) {
            refresh();
          }
        } catch {
          /* ignore poll errors */
        }
      }
    }, 2500);
    return () => clearInterval(t);
  }, [activeJobs, refresh]);

  async function uploadFiles(files: FileList | File[]) {
    setError("");
    setUploading(true);
    try {
      for (const file of Array.from(files)) {
        if (file.type !== "application/pdf") {
          setError("Only PDF files are supported.");
          continue;
        }
        const { job_id } = await apiUploadSessionDocument(sessionId, file);
        setActiveJobs((prev) => ({
          ...prev,
          [job_id]: { status: "queued", stage: "queued", progress: 5 },
        }));
      }
      refresh();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Upload failed");
    } finally {
      setUploading(false);
    }
  }

  async function confirmTier(doc: SessionDocument) {
    const tier = tierDraft[doc.doc_id] ?? doc.proposed_authority_tier ?? "engineering_ref";
    try {
      await apiConfirmAuthorityTier(sessionId, doc.doc_id, tier);
      refresh();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Could not confirm tier");
    }
  }

  function onDrop(e: React.DragEvent) {
    e.preventDefault();
    setDragOver(false);
    if (e.dataTransfer.files.length) uploadFiles(e.dataTransfer.files);
  }

  return (
    <section className="workspace-panel">
      <h2>Document workspace</h2>
      <p className="workspace-hint">
        Upload regulations, crash reports, RCA docs, or OEM specs. Each file is processed in
        your session only — nothing is shared across users.
      </p>

      <div
        className={`drop-zone${dragOver ? " drop-zone-active" : ""}`}
        onDragOver={(e) => {
          e.preventDefault();
          setDragOver(true);
        }}
        onDragLeave={() => setDragOver(false)}
        onDrop={onDrop}
        onClick={() => fileRef.current?.click()}
        role="button"
        tabIndex={0}
        onKeyDown={(e) => e.key === "Enter" && fileRef.current?.click()}
      >
        <input
          ref={fileRef}
          type="file"
          accept="application/pdf"
          multiple
          hidden
          onChange={(e) => e.target.files && uploadFiles(e.target.files)}
        />
        <span className="drop-title">
          {uploading ? "Uploading…" : "Drop PDFs here or click to browse"}
        </span>
        <span className="drop-sub">Multi-file upload supported</span>
      </div>

      {error && <p className="workspace-error">{error}</p>}

      <ul className="doc-workspace-list">
        {docs.map((doc) => {
          const job = Object.values(activeJobs).find((j) => j.doc_id === doc.doc_id);
          const stage = job?.stage || doc.stage || doc.status || "queued";
          const progress = job?.progress ?? doc.progress ?? 0;
          const pendingTier = doc.status === "pending_tier" || stage === "pending_tier";

          return (
            <li key={doc.doc_id} className="doc-workspace-item">
              <div className="doc-workspace-head">
                <strong className="doc-filename" title={doc.filename}>
                  {doc.filename}
                </strong>
                <span className={`tier-badge ${tierBadgeClass(doc.authority_tier)}`}>
                  {(doc.authority_tier || "pending").replace(/_/g, " ").toUpperCase()}
                </span>
              </div>
              <div className="doc-meta-row">
                <span>{doc.doc_type || "—"}</span>
                <span>{doc.chunk_count ?? 0} chunks</span>
                <span>{doc.regulation || "auto-detected"}</span>
              </div>
              {!pendingTier && doc.status !== "ready" && (
                <div className="stage-progress">
                  <div className="stage-bar" style={{ width: `${progress}%` }} />
                  <span className="stage-label">
                    {STAGE_LABELS[stage] || stage} ({progress}%)
                  </span>
                </div>
              )}
              {pendingTier && (
                <div className="tier-confirm">
                  <label htmlFor={`tier-${doc.doc_id}`}>Confirm authority tier</label>
                  <select
                    id={`tier-${doc.doc_id}`}
                    value={tierDraft[doc.doc_id] ?? doc.proposed_authority_tier ?? "engineering_ref"}
                    onChange={(e) =>
                      setTierDraft((prev) => ({ ...prev, [doc.doc_id]: e.target.value }))
                    }
                  >
                    {AUTHORITY_TIERS.map((t) => (
                      <option key={t.value} value={t.value}>
                        {t.label}
                      </option>
                    ))}
                  </select>
                  <button type="button" className="btn-confirm-tier" onClick={() => confirmTier(doc)}>
                    Confirm &amp; index
                  </button>
                  <p className="tier-warn">
                    Never treat uploads as binding law without your confirmation.
                  </p>
                </div>
              )}
              {doc.status === "ready" && (
                <p className="doc-ready-note">
                  Indexed source: <code>{doc.filename}</code>
                  {doc.tier_confirmed ? " · tier confirmed" : ""}
                </p>
              )}
              {doc.error && <p className="workspace-error">{doc.error}</p>}
            </li>
          );
        })}
      </ul>

      <ArtifactsPanel sessionId={sessionId} documents={docs} />

      <style jsx>{`
        .workspace-panel h2 {
          margin: 0 0 0.35rem;
          font-size: 0.95rem;
          color: var(--text);
        }
        .workspace-hint {
          margin: 0 0 0.75rem;
          font-size: 0.78rem;
          color: var(--muted);
          line-height: 1.45;
        }
        .drop-zone {
          border: 2px dashed var(--border);
          border-radius: var(--radius);
          padding: 1rem;
          text-align: center;
          background: var(--surface);
          cursor: pointer;
          margin-bottom: 0.75rem;
        }
        .drop-zone:hover,
        .drop-zone-active {
          border-color: var(--primary-light);
          background: var(--accent-soft);
        }
        .drop-title {
          display: block;
          font-size: 0.85rem;
          font-weight: 600;
        }
        .drop-sub {
          font-size: 0.75rem;
          color: var(--muted);
        }
        .workspace-error {
          color: var(--danger);
          font-size: 0.8rem;
          margin: 0.25rem 0;
        }
        .doc-workspace-list {
          list-style: none;
          padding: 0;
          margin: 0;
          display: flex;
          flex-direction: column;
          gap: 0.65rem;
        }
        .doc-workspace-item {
          background: var(--surface);
          border: 1px solid var(--border);
          border-radius: var(--radius);
          padding: 0.65rem 0.75rem;
        }
        .doc-workspace-head {
          display: flex;
          justify-content: space-between;
          gap: 0.5rem;
          align-items: flex-start;
        }
        .doc-filename {
          font-size: 0.82rem;
          overflow: hidden;
          text-overflow: ellipsis;
          white-space: nowrap;
          max-width: 70%;
        }
        .tier-badge {
          font-size: 0.65rem;
          font-weight: 700;
          padding: 0.15rem 0.4rem;
          border-radius: 4px;
          white-space: nowrap;
        }
        .tier-legal { background: var(--legal); color: #fff; }
        .tier-rating { background: var(--rating); color: #fff; }
        .tier-engref { background: var(--engref); color: #fff; }
        .tier-oem { background: var(--oem); color: #fff; }
        .tier-historical { background: var(--historical); color: #fff; }
        .tier-default { background: var(--surface-2); color: var(--muted); }
        .doc-meta-row {
          display: flex;
          gap: 0.65rem;
          font-size: 0.72rem;
          color: var(--muted);
          margin-top: 0.25rem;
        }
        .stage-progress {
          margin-top: 0.45rem;
          background: var(--surface-2);
          border-radius: 6px;
          height: 1.25rem;
          position: relative;
          overflow: hidden;
        }
        .stage-bar {
          height: 100%;
          background: linear-gradient(90deg, var(--primary), var(--primary-light));
          transition: width var(--transition);
        }
        .stage-label {
          position: absolute;
          inset: 0;
          display: flex;
          align-items: center;
          justify-content: center;
          font-size: 0.68rem;
          font-weight: 500;
        }
        .tier-confirm {
          margin-top: 0.5rem;
          display: flex;
          flex-direction: column;
          gap: 0.35rem;
        }
        .tier-confirm label {
          font-size: 0.75rem;
          font-weight: 600;
        }
        .tier-confirm select {
          padding: 0.35rem;
          border-radius: 6px;
          border: 1px solid var(--border);
          font-size: 0.8rem;
        }
        .btn-confirm-tier {
          padding: 0.4rem 0.65rem;
          background: var(--primary);
          color: #fff;
          border: none;
          border-radius: 6px;
          font-size: 0.8rem;
          font-weight: 600;
        }
        .btn-confirm-tier:hover { background: var(--accent-hover); }
        .tier-warn {
          margin: 0;
          font-size: 0.72rem;
          color: var(--warn);
        }
        .doc-ready-note {
          margin: 0.35rem 0 0;
          font-size: 0.72rem;
          color: var(--muted);
        }
      `}</style>
    </section>
  );
}
