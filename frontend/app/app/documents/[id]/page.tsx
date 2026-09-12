"use client";
import Link from "next/link";
import { useParams } from "next/navigation";
import { toast } from "sonner";

import { ErrorState, LoadingState } from "@/components/common/States";
import { DocumentStatus, ScopeBadge } from "@/components/common/StatusBadge";
import { IngestionTimeline } from "@/components/documents/IngestionTimeline";
import { ownerLabel } from "@/components/documents/DocumentTable";
import { Button } from "@/components/ui/button";
import { useDocument, useDocumentAction, useMe, useRetryJob } from "@/features/queries";
import { ApiError, errorMessage } from "@/lib/errors";

function Row({ k, v }: { k: string; v: React.ReactNode }) {
  return (
    <div className="grid grid-cols-[10rem_1fr] gap-2 py-1.5 text-sm">
      <dt className="text-text-secondary">{k}</dt>
      <dd className="min-w-0 break-words">{v ?? "—"}</dd>
    </div>
  );
}

export default function DocumentDetailPage() {
  const { id } = useParams<{ id: string }>();
  const me = useMe();
  const doc = useDocument(id);
  const action = useDocumentAction(id);
  const retry = useRetryJob();

  if (doc.isPending || me.isPending) return <div className="p-6"><LoadingState rows={6} label="Loading document" /></div>;
  if (doc.isError || me.isError) {
    const notFound = doc.error instanceof ApiError && doc.error.status === 404;
    return (
      <div className="p-6" data-testid="document-error">
        <ErrorState message={notFound ? "This document does not exist or you do not have access to it." : errorMessage(doc.error ?? me.error)} onRetry={notFound ? undefined : () => doc.refetch()} />
      </div>
    );
  }
  const d = doc.data;
  const mine = d.owner_user_id === me.data.user.id;
  const canPromote = me.data.user.scopes.includes("document:promote") && d.scope !== "AUTHORITATIVE_ORG" && d.status === "READY";
  const canArchive = !d.archived_at && (mine || me.data.user.scopes.includes("system:admin")) && d.scope !== "AUTHORITATIVE_ORG";

  return (
    <div className="mx-auto max-w-4xl space-y-6 px-4 py-6">
      <div className="flex flex-wrap items-start gap-3">
        <div className="min-w-0 flex-1">
          <h1 className="text-xl font-semibold" data-testid="document-title">{d.title}</h1>
          <div className="mt-1 flex flex-wrap items-center gap-2">
            <ScopeBadge scope={d.scope} />
            <DocumentStatus status={d.status} />
            <span className="font-mono text-xs text-muted-foreground">{d.document_key}</span>
          </div>
        </div>
        <div className="flex flex-wrap gap-2">
          {d.status === "READY" && (
            <Button nativeButton={false} size="sm" render={<Link href={`/app/chat?document=${d.id}`} data-testid="ask-document" />}>Ask this document</Button>
          )}
          {canPromote && (
            <Button size="sm" variant="outline" data-testid="promote" onClick={() => action.mutate("promote", { onSuccess: () => toast.success("Promoted to verified organization source"), onError: (e) => toast.error(errorMessage(e)) })}>
              Promote to verified
            </Button>
          )}
          {canArchive && (
            <Button size="sm" variant="ghost" data-testid="archive-document" onClick={() => action.mutate("archive", { onSuccess: () => toast.success("Archived"), onError: (e) => toast.error(errorMessage(e)) })}>
              Archive
            </Button>
          )}
        </div>
      </div>

      <section className="rounded-xl border bg-card p-4">
        <h2 className="mb-2 text-sm font-semibold">Processing</h2>
        <IngestionTimeline status={d.status} job={d.latest_job} canRetry={mine || me.data.user.scopes.includes("document:ingest")} onRetry={() => d.latest_job && retry.mutate(d.latest_job.id, { onError: (e) => toast.error(errorMessage(e)) })} />
      </section>

      <section className="rounded-xl border bg-card p-4">
        <h2 className="mb-2 text-sm font-semibold">Provenance</h2>
        <dl className="divide-y">
          <Row k="Type" v={d.document_type.toLowerCase().replace("_", " ")} />
          <Row k="Owner / workspace" v={ownerLabel(d, me.data)} />
          <Row k="Version" v={d.version?.label} />
          <Row k="Effective" v={d.version?.valid_from ? `${d.version.valid_from}${d.version.valid_to ? ` → ${d.version.valid_to}` : ""}` : "not set"} />
          <Row k="Pages" v={d.version?.page_count ?? d.extraction_report?.processed_page_count} />
          <Row k="Activated" v={d.version?.activated_at ? new Date(d.version.activated_at).toLocaleString() : null} />
          <Row k="Source hash" v={d.version?.source_sha256 ? <span className="font-mono text-xs">{d.version.source_sha256}</span> : null} />
          <Row k="Authority level" v={d.authority_level.toLowerCase().replace("_", " ")} />
          <Row k="Extraction QA" v={d.extraction_report?.status ? `${d.extraction_report.status}${d.extraction_report.failed_pages?.length ? ` · failed pages: ${d.extraction_report.failed_pages.join(", ")}` : ""}` : null} />
          <Row k="Notes" v={d.notes} />
          <Row k="Uploaded" v={new Date(d.created_at).toLocaleString()} />
        </dl>
      </section>
    </div>
  );
}
