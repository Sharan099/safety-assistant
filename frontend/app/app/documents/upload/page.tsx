"use client";
import Link from "next/link";
import { useState } from "react";

import { ErrorState, LoadingState } from "@/components/common/States";
import { DocumentStatus } from "@/components/common/StatusBadge";
import { IngestionTimeline } from "@/components/documents/IngestionTimeline";
import { UploadForm } from "@/components/documents/UploadForm";
import { Button } from "@/components/ui/button";
import { useDocument, useMe, useRetryJob } from "@/features/queries";
import { errorMessage } from "@/lib/errors";
import type { UploadResult } from "@/lib/types";

// Steps 1–2 (choose + describe) live in UploadForm; steps 3–4 (processing, ready) render here from
// the live document record polled by useDocument.
export default function UploadPage() {
  const me = useMe();
  const [result, setResult] = useState<UploadResult | null>(null);
  const doc = useDocument(result?.document_id ?? null);
  const retry = useRetryJob();

  if (me.isPending) return <div className="p-6"><LoadingState rows={3} /></div>;
  if (me.isError) return <div className="p-6"><ErrorState message={errorMessage(me.error)} /></div>;
  const status = doc.data?.status ?? result?.document.status;

  return (
    <div className="mx-auto max-w-3xl space-y-6 px-4 py-6">
      <h1 className="text-xl font-semibold">Upload a document</h1>
      {!result ? (
        <UploadForm me={me.data} onUploaded={setResult} />
      ) : (
        <section className="space-y-4 rounded-xl border bg-card p-4" data-testid="upload-progress">
          <div className="flex flex-wrap items-center gap-2">
            <h2 className="text-sm font-semibold">3. Processing</h2>
            {status && <DocumentStatus status={status} />}
            {result.duplicate && <span className="text-xs text-text-secondary">Identical file already uploaded — showing the existing document.</span>}
          </div>
          <p className="text-sm text-text-secondary">
            <span className="font-medium text-foreground">{result.document.title}</span> · job {result.ingestion_job_id.slice(0, 8)}
          </p>
          {status && (
            <IngestionTimeline
              status={status}
              job={doc.data?.latest_job ?? result.document.latest_job}
              canRetry
              onRetry={() => doc.data?.latest_job && retry.mutate(doc.data.latest_job.id)}
            />
          )}
          {status === "READY" && (
            <div className="flex flex-wrap gap-2 border-t pt-3" data-testid="upload-ready-actions">
              <h2 className="sr-only">4. Ready</h2>
              <Button nativeButton={false} size="sm" render={<Link href={`/app/chat?document=${result.document_id}`} data-testid="ask-document" />}>Ask this document</Button>
              <Button nativeButton={false} size="sm" variant="outline" render={<Link href={`/app/documents/${result.document_id}`} data-testid="view-details" />}>View details</Button>
              <Button size="sm" variant="ghost" onClick={() => setResult(null)}>Upload another</Button>
            </div>
          )}
          {(status === "FAILED" || status === "QUARANTINED") && (
            <div className="flex gap-2 border-t pt-3">
              <Button size="sm" variant="outline" onClick={() => setResult(null)} data-testid="upload-replace">Upload a corrected file</Button>
              <Button nativeButton={false} size="sm" variant="ghost" render={<Link href={`/app/documents/${result.document_id}`} data-testid="view-details" />}>View details</Button>
            </div>
          )}
        </section>
      )}
    </div>
  );
}
