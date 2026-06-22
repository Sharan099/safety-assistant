"use client";

import { useCallback, useEffect, useState } from "react";
import {
  apiDeleteDocument,
  apiIngestStatus,
  apiListDocuments,
  apiUploadDocument,
  type DocumentMeta,
  type IngestJob,
} from "@/lib/api";

export function DocumentManager() {
  const [docs, setDocs] = useState<DocumentMeta[]>([]);
  const [file, setFile] = useState<File | null>(null);
  const [docType, setDocType] = useState("reference");
  const [testType, setTestType] = useState("general");
  const [revision, setRevision] = useState("");
  const [jobId, setJobId] = useState<string | null>(null);
  const [job, setJob] = useState<IngestJob | null>(null);
  const [error, setError] = useState("");

  const refresh = useCallback(async () => {
    try {
      setDocs(await apiListDocuments());
    } catch {
      /* backend may be offline */
    }
  }, []);

  useEffect(() => {
    refresh();
  }, [refresh]);

  useEffect(() => {
    if (!jobId) return;
    const t = setInterval(async () => {
      try {
        const s = await apiIngestStatus(jobId);
        setJob(s);
        if (s.status === "complete" || s.status === "failed") {
          clearInterval(t);
          refresh();
        }
      } catch {
        clearInterval(t);
      }
    }, 3000);
    return () => clearInterval(t);
  }, [jobId, refresh]);

  async function onUpload(e: React.FormEvent) {
    e.preventDefault();
    if (!file) return;
    setError("");
    try {
      const { job_id } = await apiUploadDocument(file, {
        doc_type: docType,
        test_type: testType,
        revision,
      });
      setJobId(job_id);
      setJob({ status: "uploaded", progress: 5 });
    } catch (err) {
      setError(err instanceof Error ? err.message : "Upload failed");
    }
  }

  async function onDelete(name: string) {
    if (!confirm(`Remove ${name} from index?`)) return;
    try {
      await apiDeleteDocument(name);
      refresh();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Delete failed");
    }
  }

  return (
    <section className="doc-manager">
      <h3>Document manager</h3>
      <form onSubmit={onUpload} className="doc-upload-form">
        <input
          type="file"
          accept="application/pdf"
          onChange={(e) => setFile(e.target.files?.[0] ?? null)}
        />
        <select value={docType} onChange={(e) => setDocType(e.target.value)}>
          <option value="legal">Legal regulation</option>
          <option value="rating">Rating protocol</option>
          <option value="reference">Reference handbook</option>
        </select>
        <select value={testType} onChange={(e) => setTestType(e.target.value)}>
          <option value="general">General</option>
          <option value="frontal">Frontal</option>
          <option value="side">Side</option>
          <option value="rear">Rear</option>
          <option value="belt">Belt</option>
          <option value="seat">Seat</option>
        </select>
        <input
          placeholder="Revision (optional)"
          value={revision}
          onChange={(e) => setRevision(e.target.value)}
        />
        <button type="submit" disabled={!file}>
          Upload &amp; index
        </button>
      </form>
      {job && (
        <p className="ingest-status">
          Status: {job.status}
          {job.progress != null ? ` (${job.progress}%)` : ""}
          {job.chunk_count != null ? ` — ${job.chunk_count} chunks` : ""}
          {job.error ? ` — ${job.error}` : ""}
        </p>
      )}
      {error && <p className="error-text">{error}</p>}
      <ul className="doc-list">
        {docs.map((d) => (
          <li key={d.name}>
            <span>{d.name}</span>
            <span className="doc-cat">{d.category}</span>
            <button type="button" onClick={() => onDelete(d.name)}>
              Delete
            </button>
          </li>
        ))}
      </ul>
    </section>
  );
}
