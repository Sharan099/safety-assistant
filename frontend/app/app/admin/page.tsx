"use client";
import { useState } from "react";
import { toast } from "sonner";

import { EmptyState, ErrorState, LoadingState, PermissionDenied } from "@/components/common/States";
import { DocumentStatus, ScopeBadge } from "@/components/common/StatusBadge";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { useDocuments, useIngestionRuns, useMe, useRegulations } from "@/features/queries";
import { api } from "@/lib/api";
import { errorMessage } from "@/lib/errors";

// Role-gated admin (03_UI_UX IA: /app/admin/{corpus,ingestion,audit}) — one page, three tabs; desktop-first.
export default function AdminPage() {
  const me = useMe();
  const canAudit = !!me.data?.user.scopes.includes("audit:read");
  const canIngest = !!me.data?.user.scopes.includes("document:ingest");
  const regs = useRegulations();
  const runs = useIngestionRuns(canAudit);
  const uploads = useDocuments({ include_archived: "true" });
  const [sourceKey, setSourceKey] = useState("");

  if (me.isPending) return <div className="p-6"><LoadingState rows={3} /></div>;
  if (me.isError) return <div className="p-6"><ErrorState message={errorMessage(me.error)} /></div>;
  if (!canAudit && !canIngest) return <div className="p-6"><PermissionDenied what="administration" /></div>;

  return (
    <div className="mx-auto max-w-6xl space-y-4 px-4 py-6">
      <h1 className="text-xl font-semibold">Administration</h1>
      <Tabs defaultValue="corpus">
        <TabsList>
          <TabsTrigger value="corpus">Corpus</TabsTrigger>
          <TabsTrigger value="ingestion">Ingestion runs</TabsTrigger>
          <TabsTrigger value="audit">Promotion & audit</TabsTrigger>
        </TabsList>

        <TabsContent value="corpus" className="space-y-4">
          {canIngest && (
            <form
              className="flex flex-wrap items-end gap-2 rounded-xl border bg-card p-4"
              onSubmit={(e) => {
                e.preventDefault();
                api.adminIngest(sourceKey.trim())
                  .then((r) => toast.success(`Queued job ${r.ingestion_job_id.slice(0, 8)}`))
                  .catch((err) => toast.error(errorMessage(err)));
              }}
            >
              <div className="space-y-1.5">
                <label htmlFor="source-key" className="text-sm font-medium">Queue a registry source</label>
                <Input id="source-key" value={sourceKey} onChange={(e) => setSourceKey(e.target.value)} placeholder="source_key from knowledge/00_registry/sources.yaml" className="w-96" />
              </div>
              <Button type="submit" size="sm" disabled={!sourceKey.trim()}>Enqueue</Button>
            </form>
          )}
          {regs.isPending ? <LoadingState rows={4} /> : regs.isError ? <ErrorState message={errorMessage(regs.error)} /> : (
            <div className="overflow-x-auto rounded-xl border bg-card">
              <Table>
                <TableHeader>
                  <TableRow><TableHead>Regulation</TableHead><TableHead>Kind</TableHead><TableHead>Authority</TableHead><TableHead>Versions</TableHead></TableRow>
                </TableHeader>
                <TableBody>
                  {regs.data.map((r) => (
                    <TableRow key={r.regulation_key}>
                      <TableCell><span className="font-medium">{r.regulation_key}</span><div className="text-xs text-muted-foreground">{r.title}</div></TableCell>
                      <TableCell className="text-sm">{r.kind.toLowerCase()}</TableCell>
                      <TableCell className="text-sm">{r.authority_level.toLowerCase()}</TableCell>
                      <TableCell className="text-sm">{r.versions.map((v) => `${v.label} (${v.status.toLowerCase()}${v.valid_from ? `, from ${v.valid_from}` : ""})`).join("; ")}</TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </div>
          )}
        </TabsContent>

        <TabsContent value="ingestion">
          {!canAudit ? <PermissionDenied what="ingestion runs" /> : runs.isPending ? <LoadingState rows={4} /> : runs.isError ? <ErrorState message={errorMessage(runs.error)} /> : runs.data.length === 0 ? <EmptyState title="No runs yet" /> : (
            <div className="overflow-x-auto rounded-xl border bg-card">
              <Table>
                <TableHeader>
                  <TableRow><TableHead>Source</TableHead><TableHead>Status</TableHead><TableHead>Started</TableHead><TableHead>Attempt</TableHead><TableHead>Error</TableHead></TableRow>
                </TableHeader>
                <TableBody>
                  {runs.data.map((r) => (
                    <TableRow key={r.run_id}>
                      <TableCell className="font-mono text-xs">{r.source_key}</TableCell>
                      <TableCell className="text-sm">{r.status}</TableCell>
                      <TableCell className="text-sm">{new Date(r.started_at).toLocaleString()}</TableCell>
                      <TableCell className="text-sm">{r.attempt}</TableCell>
                      <TableCell className="max-w-md truncate text-xs text-destructive" title={r.error ?? undefined}>{r.error?.split("\n")[0] ?? ""}</TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </div>
          )}
        </TabsContent>

        <TabsContent value="audit" className="space-y-3">
          <p className="text-sm text-text-secondary">Uploaded documents visible to you. Promotion happens from a document’s detail page and is recorded as an audit event.</p>
          {uploads.isPending || me.isPending ? <LoadingState rows={3} /> : uploads.isError ? <ErrorState message={errorMessage(uploads.error)} /> : (
            <ul className="divide-y rounded-xl border bg-card">
              {uploads.data.items.filter((d) => d.document_key.startsWith("DOC-")).map((d) => (
                <li key={d.id} className="flex flex-wrap items-center gap-2 px-4 py-2 text-sm">
                  <a href={`/app/documents/${d.id}`} className="font-medium text-primary hover:underline">{d.title}</a>
                  <ScopeBadge scope={d.scope} kind={d.document_type} />
                  <span className="ml-auto"><DocumentStatus status={d.status} /></span>
                </li>
              ))}
              {uploads.data.items.filter((d) => d.document_key.startsWith("DOC-")).length === 0 && <li className="px-4 py-3 text-sm text-text-secondary">No uploaded documents visible.</li>}
            </ul>
          )}
        </TabsContent>
      </Tabs>
    </div>
  );
}
