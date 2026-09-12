"use client";
import Link from "next/link";
import { Suspense, useState } from "react";

import { EmptyState, ErrorState, LoadingState } from "@/components/common/States";
import { DocumentTable } from "@/components/documents/DocumentTable";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { useDocuments, useMe } from "@/features/queries";
import { errorMessage } from "@/lib/errors";

const ANY = "__any";

function Library() {
  const me = useMe();
  const [scope, setScope] = useState(ANY);
  const [status, setStatus] = useState(ANY);
  const [q, setQ] = useState("");
  const docs = useDocuments({ scope: scope === ANY ? undefined : scope, status: status === ANY ? undefined : status, q: q || undefined });

  return (
    <div className="mx-auto max-w-6xl space-y-4 px-4 py-6">
      <div className="flex flex-wrap items-center gap-3">
        <h1 className="text-xl font-semibold">Documents</h1>
        <Button nativeButton={false} size="sm" className="ml-auto" render={<Link href="/app/documents/upload" data-testid="go-upload" />}>
          Upload PDF
        </Button>
      </div>
      <div className="flex flex-wrap gap-2" role="search">
        <Input value={q} onChange={(e) => setQ(e.target.value)} placeholder="Search titles" aria-label="Search documents" className="w-64" />
        <Select value={scope} onValueChange={(v) => setScope(v ?? ANY)}>
          <SelectTrigger className="w-52" aria-label="Filter by scope"><SelectValue /></SelectTrigger>
          <SelectContent>
            <SelectItem value={ANY}>All scopes</SelectItem>
            <SelectItem value="AUTHORITATIVE_ORG">Verified regulations</SelectItem>
            <SelectItem value="WORKSPACE">Workspace documents</SelectItem>
            <SelectItem value="PRIVATE_USER">My private documents</SelectItem>
          </SelectContent>
        </Select>
        <Select value={status} onValueChange={(v) => setStatus(v ?? ANY)}>
          <SelectTrigger className="w-44" aria-label="Filter by status"><SelectValue /></SelectTrigger>
          <SelectContent>
            <SelectItem value={ANY}>All statuses</SelectItem>
            {["READY", "UPLOADED", "PARSING", "FAILED", "QUARANTINED"].map((s) => (
              <SelectItem key={s} value={s}>{s.toLowerCase()}</SelectItem>
            ))}
          </SelectContent>
        </Select>
      </div>
      {docs.isPending || me.isPending ? (
        <LoadingState rows={5} label="Loading documents" />
      ) : docs.isError || me.isError ? (
        <ErrorState message={errorMessage(docs.error ?? me.error)} onRetry={() => docs.refetch()} />
      ) : docs.data.items.length === 0 ? (
        <EmptyState title="No documents match" hint="Upload a PDF or change the filters." action={<Button nativeButton={false} size="sm" variant="outline" render={<Link href="/app/documents/upload" />}>Upload PDF</Button>} />
      ) : (
        <DocumentTable items={docs.data.items} me={me.data} />
      )}
    </div>
  );
}

export default function DocumentsPage() {
  return (
    <Suspense fallback={<LoadingState rows={5} />}>
      <Library />
    </Suspense>
  );
}
