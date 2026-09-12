"use client";
import Link from "next/link";

import { EmptyState, ErrorState, LoadingState } from "@/components/common/States";
import { DocumentStatus, ScopeBadge } from "@/components/common/StatusBadge";
import { Button } from "@/components/ui/button";
import { isTerminal, useConversations, useDocuments, useMe, useReady, useRegulations } from "@/features/queries";
import { errorMessage } from "@/lib/errors";

function Card({ title, children, action }: { title: string; children: React.ReactNode; action?: React.ReactNode }) {
  return (
    <section className="rounded-xl border bg-card p-4">
      <div className="mb-3 flex items-center justify-between gap-2">
        <h2 className="text-sm font-semibold">{title}</h2>
        {action}
      </div>
      {children}
    </section>
  );
}

// Orientation, not vanity metrics (03_UI_UX "Home/dashboard").
export default function HomePage() {
  const me = useMe();
  const convs = useConversations();
  const docs = useDocuments({});
  const regs = useRegulations();
  const ready = useReady();
  if (me.isPending) return <div className="p-6"><LoadingState rows={4} /></div>;
  if (me.isError) return <div className="p-6"><ErrorState message={errorMessage(me.error)} /></div>;
  const processing = docs.data?.items.filter((d) => !isTerminal(d.status)) ?? [];
  const failed = docs.data?.items.filter((d) => d.status === "FAILED" || d.status === "QUARANTINED") ?? [];
  const recentDocs = (docs.data?.items ?? []).slice(0, 5);

  return (
    <div className="mx-auto max-w-6xl space-y-6 px-4 py-6">
      <div>
        <h1 className="text-xl font-semibold">Welcome back, {me.data.user.display_name}</h1>
        <p className="text-sm text-text-secondary">Resume an investigation, check knowledge readiness, or add a document.</p>
      </div>
      <div className="flex flex-wrap gap-2">
        <Button nativeButton={false} render={<Link href="/app/chat" data-testid="home-new" />}>New investigation</Button>
        <Button nativeButton={false} variant="outline" render={<Link href="/app/documents/upload" />}>Upload PDF</Button>
        <Button nativeButton={false} variant="outline" render={<Link href="/app/documents" />}>Browse documents</Button>
      </div>
      <div className="grid gap-4 lg:grid-cols-2">
        <Card title="Resume work" action={<Link href="/app/chat" className="text-xs text-primary hover:underline">All investigations</Link>}>
          {convs.isPending ? (
            <LoadingState rows={3} />
          ) : convs.isError ? (
            <ErrorState message={errorMessage(convs.error)} />
          ) : convs.data.items.length === 0 ? (
            <EmptyState title="No investigations yet" />
          ) : (
            <ul className="divide-y" data-testid="recent-conversations">
              {convs.data.items.slice(0, 6).map((c) => (
                <li key={c.id}>
                  <Link href={`/app/chat/${c.id}`} className="block py-2 text-sm hover:text-primary">
                    {c.title}
                    <span className="block text-xs text-muted-foreground">{new Date(c.updated_at).toLocaleString()}</span>
                  </Link>
                </li>
              ))}
            </ul>
          )}
        </Card>
        <Card title="Knowledge readiness">
          <dl className="space-y-1.5 text-sm">
            <div className="flex justify-between">
              <dt className="text-text-secondary">System</dt>
              <dd data-testid="home-ready">{ready.isError ? "API unreachable" : ready.data?.status === "ready" ? "ready" : "not ready"}</dd>
            </div>
            <div className="flex justify-between">
              <dt className="text-text-secondary">Verified regulations</dt>
              <dd>{regs.data ? `${regs.data.length} active` : "…"}</dd>
            </div>
            <div className="flex justify-between">
              <dt className="text-text-secondary">Uploads processing</dt>
              <dd>{processing.length}</dd>
            </div>
            <div className="flex justify-between">
              <dt className="text-text-secondary">Uploads needing attention</dt>
              <dd className={failed.length ? "text-destructive" : ""}>{failed.length}</dd>
            </div>
          </dl>
          {failed.length > 0 && (
            <p className="mt-3 text-xs text-text-secondary">
              <Link href="/app/ingestion" className="text-primary hover:underline">Review failed ingestion</Link>
            </p>
          )}
        </Card>
      </div>
      <Card title="Recent documents" action={<Link href="/app/documents" className="text-xs text-primary hover:underline">Library</Link>}>
        {docs.isPending ? (
          <LoadingState rows={3} />
        ) : recentDocs.length === 0 ? (
          <EmptyState title="No documents" />
        ) : (
          <ul className="divide-y">
            {recentDocs.map((d) => (
              <li key={d.id} className="flex flex-wrap items-center gap-2 py-2 text-sm">
                <Link href={`/app/documents/${d.id}`} className="font-medium hover:text-primary">{d.title}</Link>
                <ScopeBadge scope={d.scope} />
                <span className="text-xs text-muted-foreground">{d.version?.label}</span>
                <span className="ml-auto"><DocumentStatus status={d.status} /></span>
              </li>
            ))}
          </ul>
        )}
      </Card>
    </div>
  );
}
