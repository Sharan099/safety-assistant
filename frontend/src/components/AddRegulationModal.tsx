"use client";

import { FormEvent, useCallback, useEffect, useRef, useState } from "react";
import {
  detectRegulationMeta,
  fetchIngestJob,
  uploadRegulation,
  type IngestJob,
} from "@/lib/api";

type Props = {
  open: boolean;
  onClose: () => void;
  onComplete: (job: IngestJob) => void;
};

const STAGES = ["queued", "parsing", "chunking", "embedding", "done"] as const;

export function AddRegulationModal({ open, onClose, onComplete }: Props) {
  const [file, setFile] = useState<File | null>(null);
  const [dragOver, setDragOver] = useState(false);
  const [regulationId, setRegulationId] = useState("");
  const [revision, setRevision] = useState("");
  const [detectHint, setDetectHint] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [job, setJob] = useState<IngestJob | null>(null);
  const inputRef = useRef<HTMLInputElement>(null);
  const pollRef = useRef<number | null>(null);

  const clearPoll = useCallback(() => {
    if (pollRef.current != null) {
      window.clearInterval(pollRef.current);
      pollRef.current = null;
    }
  }, []);

  useEffect(() => {
    if (!open) {
      clearPoll();
      setFile(null);
      setRegulationId("");
      setRevision("");
      setDetectHint(null);
      setBusy(false);
      setError(null);
      setJob(null);
    }
    return clearPoll;
  }, [open, clearPoll]);

  async function onPickFile(next: File | null) {
    if (!next) return;
    if (!next.name.toLowerCase().endsWith(".pdf")) {
      setError("Please choose a PDF file.");
      return;
    }
    setFile(next);
    setError(null);
    setDetectHint("Detecting cover metadata…");
    try {
      const meta = await detectRegulationMeta(next);
      if (meta.regulation_id) setRegulationId(meta.regulation_id);
      if (meta.revision) setRevision(meta.revision);
      setDetectHint(
        meta.source && meta.source !== "none"
          ? `Detected via ${meta.source}`
          : "Could not auto-detect — enter ids manually"
      );
    } catch {
      setDetectHint("Could not auto-detect — enter ids manually");
    }
  }

  async function onSubmit(e: FormEvent) {
    e.preventDefault();
    if (!file || busy) return;
    setBusy(true);
    setError(null);
    try {
      const uploaded = await uploadRegulation(file, {
        regulation_id: regulationId.trim() || undefined,
        revision: revision.trim() || undefined,
      });
      const jobId = (uploaded.job_id || "").trim();
      if (!jobId) {
        throw new Error("Upload succeeded but returned no job_id");
      }
      setRegulationId(uploaded.regulation_id);
      setRevision(uploaded.revision);
      setJob({
        job_id: jobId,
        status: uploaded.status,
        regulation_id: uploaded.regulation_id,
        revision: uploaded.revision,
      });
      clearPoll();
      pollRef.current = window.setInterval(async () => {
        try {
          const status = await fetchIngestJob(jobId);
          setJob(status);
          if (status.status === "done") {
            clearPoll();
            setBusy(false);
            onComplete(status);
          } else if (status.status === "failed") {
            clearPoll();
            setBusy(false);
            setError(status.error || "Ingestion failed");
          }
        } catch (err) {
          clearPoll();
          setBusy(false);
          setError(err instanceof Error ? err.message : "Polling failed");
        }
      }, 1500);
    } catch (err) {
      setBusy(false);
      setError(err instanceof Error ? err.message : "Upload failed");
    }
  }

  if (!open) return null;

  const stageIdx = job
    ? Math.max(0, STAGES.indexOf(job.status as (typeof STAGES)[number]))
    : -1;

  return (
    <div className="modal-backdrop" role="presentation" onClick={onClose}>
      <div
        className="modal"
        role="dialog"
        aria-modal="true"
        aria-labelledby="add-reg-title"
        onClick={(ev) => ev.stopPropagation()}
      >
        <header className="modal__head">
          <h2 id="add-reg-title">Add regulation</h2>
          <button type="button" className="modal__close" onClick={onClose} aria-label="Close">
            ×
          </button>
        </header>

        <form className="modal__body" onSubmit={onSubmit}>
          <div
            className={`dropzone ${dragOver ? "is-over" : ""} ${file ? "has-file" : ""}`}
            onDragOver={(ev) => {
              ev.preventDefault();
              setDragOver(true);
            }}
            onDragLeave={() => setDragOver(false)}
            onDrop={(ev) => {
              ev.preventDefault();
              setDragOver(false);
              const f = ev.dataTransfer.files?.[0] || null;
              void onPickFile(f);
            }}
            onClick={() => inputRef.current?.click()}
          >
            <input
              ref={inputRef}
              type="file"
              accept="application/pdf,.pdf"
              hidden
              onChange={(ev) => void onPickFile(ev.target.files?.[0] || null)}
            />
            {file ? (
              <>
                <strong>{file.name}</strong>
                <span>{(file.size / (1024 * 1024)).toFixed(1)} MB</span>
              </>
            ) : (
              <>
                <strong>Drop a UNECE PDF here</strong>
                <span>or click to browse</span>
              </>
            )}
          </div>

          <label className="field">
            <span>Regulation ID</span>
            <input
              value={regulationId}
              onChange={(ev) => setRegulationId(ev.target.value)}
              placeholder="UN-ECE-R95"
              required
              disabled={busy && !!job}
            />
          </label>
          <label className="field">
            <span>Revision</span>
            <input
              value={revision}
              onChange={(ev) => setRevision(ev.target.value)}
              placeholder="Rev.3"
              disabled={busy && !!job}
            />
          </label>
          {detectHint ? <p className="field-hint">{detectHint}</p> : null}

          {job ? (
            <div className="ingest-progress" aria-live="polite">
              <div className="ingest-progress__stages">
                {STAGES.filter((s) => s !== "done").map((s, i) => (
                  <span
                    key={s}
                    className={
                      job.status === "failed"
                        ? "is-failed"
                        : stageIdx > i || job.status === "done"
                          ? "is-done"
                          : stageIdx === i
                            ? "is-active"
                            : ""
                    }
                  >
                    {s}
                  </span>
                ))}
              </div>
              <p className="ingest-progress__status">
                {job.status === "done"
                  ? `Indexed ${job.chunk_count ?? "?"} chunks`
                  : job.status === "failed"
                    ? "Failed"
                    : `Status: ${job.status}…`}
              </p>
            </div>
          ) : null}

          {error ? <p className="modal__error">{error}</p> : null}

          <footer className="modal__actions">
            <button type="button" className="btn-ghost" onClick={onClose} disabled={busy && !!job && job.status !== "failed"}>
              {job?.status === "done" ? "Close" : "Cancel"}
            </button>
            <button type="submit" className="btn-primary" disabled={!file || busy}>
              {busy ? "Ingesting…" : "Start ingest"}
            </button>
          </footer>
        </form>
      </div>
    </div>
  );
}
