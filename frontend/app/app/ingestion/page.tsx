"use client";
import Link from "next/link";

import { EmptyState, ErrorState, LoadingState } from "@/components/common/States";
import { DocumentStatus, ScopeBadge } from "@/components/common/StatusBadge";
import { useDocuments, useMe } from "@/features/queries";
import { errorMessage } from "@/lib/errors";
import { isTerminal } from "@/features/queries";

// Ingestion overview for the signed-in user: what is processing, what failed, what is ready.
export default function IngestionPage() {
  const me = useMe();
  const docs = useDocuments({ include_archived: "false" });
  if (docs.isPending || me.isPending) return <div className="p-6"><LoadingState rows={5} /></div>;
  if (docs.isError || me.isError) return <div className="p-6"><ErrorState message={errorMessage(docs.error ?? me.error)} onRetry={() => docs.refetch()} /></div>;
  const mine = docs.data.items.filter((d) => d.scope !== "AUTHORITATIVE_ORG");
  const groups = [
    { title: "Processing", items: mine.filter((d) => !isTerminal(d.status)) },
    { title: "Needs attention", items: mine.filter((d) => d.status === "FAILED" || d.status === "QUARANTINED") },
    { title: "Ready", items: mine.filter((d) => d.status === "READY") },
  ];
  return (
    <div className="mx-auto max-w-4xl space-y-6 px-4 py-6">
      <h1 className="text-xl font-semibold">Ingestion</h1>
      {mine.length === 0 && <EmptyState title="Nothing uploaded yet" hint="Documents you upload show their processing state here." />}
      {groups.map((g) =>
        g.items.length ? (
          <section key={g.title} className="space-y-2">
            <h2 className="text-sm font-semibold">{g.title} ({g.items.length})</h2>
            <ul className="divide-y rounded-xl border bg-card">
              {g.items.map((d) => (
                <li key={d.id} className="flex flex-wrap items-center gap-2 px-4 py-2.5 text-sm">
                  <Link href={`/app/documents/${d.id}`} className="font-medium text-primary hover:underline">{d.title}</Link>
                  <ScopeBadge scope={d.scope} kind={d.document_type} />
                  <span className="ml-auto"><DocumentStatus status={d.status} /></span>
                  {d.latest_job?.error_public_message && <span className="basis-full text-xs text-destructive">{d.latest_job.error_public_message}</span>}
                </li>
              ))}
            </ul>
          </section>
        ) : null,
      )}
    </div>
  );
}
