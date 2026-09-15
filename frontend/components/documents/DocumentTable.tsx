"use client";
import Link from "next/link";

import { DocumentStatus, ScopeBadge } from "@/components/common/StatusBadge";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import type { DocumentSummary, Me } from "@/lib/types";

export function ownerLabel(d: DocumentSummary, me: Me): string {
  if (d.scope === "WORKSPACE") return me.workspaces.find((w) => w.id === d.workspace_id)?.name ?? "Workspace";
  if (d.scope === "PRIVATE_USER") return d.owner_user_id === me.user.id ? "You" : "Another user";
  return "Organization";
}

export function DocumentTable({ items, me }: { items: DocumentSummary[]; me: Me }) {
  return (
    <div className="overflow-x-auto rounded-xl border bg-card">
      <Table data-testid="document-table">
        <TableHeader>
          <TableRow>
            <TableHead>Document</TableHead>
            <TableHead>Scope</TableHead>
            <TableHead>Version</TableHead>
            <TableHead>Status</TableHead>
            <TableHead>Updated</TableHead>
            <TableHead>Owner / workspace</TableHead>
          </TableRow>
        </TableHeader>
        <TableBody>
          {items.map((d) => (
            <TableRow key={d.id} data-testid="document-row" data-status={d.status}>
              <TableCell className="max-w-[28rem]">
                <Link href={`/app/documents/${d.id}`} className="font-medium text-primary hover:underline">
                  {d.title}
                </Link>
                <div className="truncate text-xs text-muted-foreground">
                  {d.document_key} · {d.document_type.toLowerCase().replace("_", " ")}
                </div>
              </TableCell>
              <TableCell>
                <ScopeBadge scope={d.scope} kind={d.document_type} />
              </TableCell>
              <TableCell className="text-sm">{d.version?.label ?? "—"}</TableCell>
              <TableCell>
                <DocumentStatus status={d.status} />
              </TableCell>
              <TableCell className="text-sm text-text-secondary">
                {new Date(d.latest_job?.completed_at ?? d.created_at).toLocaleDateString()}
              </TableCell>
              <TableCell className="text-sm text-text-secondary">{ownerLabel(d, me)}</TableCell>
            </TableRow>
          ))}
        </TableBody>
      </Table>
    </div>
  );
}
