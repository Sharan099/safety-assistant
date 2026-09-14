"use client";
import { Badge } from "@/components/ui/badge";
import { EmptyState, ErrorState, LoadingState } from "@/components/common/States";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { useRegulations } from "@/features/queries";
import { errorMessage } from "@/lib/errors";
import type { RegulationSummary } from "@/lib/types";

// Groups in the order an engineer expects them; the key prefix decides membership.
const GROUPS: { id: string; label: string; match: (r: RegulationSummary) => boolean }[] = [
  { id: "unece", label: "UNECE regulations (1958 Agreement)", match: (r) => r.regulation_key.startsWith("UN-R") },
  { id: "us", label: "US federal standards (FMVSS)", match: (r) => r.jurisdiction === "US-FMVSS" },
  { id: "ncap", label: "Euro NCAP test protocols", match: (r) => r.regulation_key.startsWith("EURONCAP-") },
  { id: "cae", label: "CAE solver manuals", match: (r) => r.kind === "MANUAL" },
  { id: "ref", label: "Standards and reference handbooks", match: () => true },
];

// UN R94 before UN R129 (numeric), the consolidated text before its amendment sheets, then title order.
function sortKey(r: RegulationSummary): [number, number, string] {
  const m = /^UN-R(\d+)(-.*)?$/.exec(r.regulation_key);
  return m ? [Number(m[1]), m[2] ? 1 : 0, r.regulation_key] : [Number.MAX_SAFE_INTEGER, 0, r.title];
}

function compare(a: RegulationSummary, b: RegulationSummary): number {
  const [an, as, at] = sortKey(a);
  const [bn, bs, bt] = sortKey(b);
  return an - bn || as - bs || at.localeCompare(bt);
}

function shortName(r: RegulationSummary): string {
  const m = /^UN-R(\d+)(?:-(.+))?$/.exec(r.regulation_key);
  if (!m) return r.regulation_key.replace(/-/g, " ");
  if (!m[2]) return `UN R${m[1]}`;
  const suppl = /^SUPPL(\d+)(?:-(\d+))?$/.exec(m[2]);
  const amend = /^AMEND-(\d+)$/.exec(m[2]);
  const note = suppl ? `Suppl. ${suppl[1]}${suppl[2] ? ` to ${suppl[2]} series` : ""}` : amend ? `${amend[1]} series amendment` : m[2];
  return `UN R${m[1]} · ${note}`;
}

function current(r: RegulationSummary) {
  return r.versions.find((v) => v.status === "ACTIVE") ?? r.versions[r.versions.length - 1];
}

export default function SourcesPage() {
  const regs = useRegulations();
  if (regs.isPending) return <LoadingState rows={8} label="Loading sources" />;
  if (regs.isError) return <ErrorState message={errorMessage(regs.error)} onRetry={() => regs.refetch()} />;
  const corpus = regs.data.filter((r) => r.kind !== "PROJECT_DOCUMENT"); // uploads live under Documents
  const remaining = [...corpus];
  const groups = GROUPS.map((g) => {
    const items = remaining.filter(g.match).sort(compare);
    items.forEach((i) => remaining.splice(remaining.indexOf(i), 1));
    return { ...g, items };
  }).filter((g) => g.items.length);

  return (
    <div className="mx-auto max-w-6xl space-y-6 px-4 py-6">
      <div>
        <h1 className="text-xl font-semibold">Sources</h1>
        <p className="text-sm text-text-secondary">
          The verified corpus every answer can cite: {corpus.length} documents. Only the version in force is
          retrievable; amendment sheets newer than a consolidated text are listed under it.
        </p>
      </div>
      {groups.length === 0 ? (
        <EmptyState title="No sources yet" hint="Ingest the registry with `safety-assistant ingest`." />
      ) : (
        groups.map((g) => (
          <section key={g.id} aria-labelledby={`sources-${g.id}`} className="space-y-2">
            <h2 id={`sources-${g.id}`} className="text-sm font-semibold uppercase tracking-wide text-text-secondary">
              {g.label} <span className="font-normal">({g.items.length})</span>
            </h2>
            <div className="overflow-x-auto rounded-xl border bg-card">
              <Table data-testid={`sources-${g.id}`}>
                <TableHeader>
                  <TableRow>
                    <TableHead className="w-36">Source</TableHead>
                    <TableHead>Title</TableHead>
                    <TableHead className="w-40">Version</TableHead>
                    <TableHead className="w-28">In force from</TableHead>
                    <TableHead className="w-24">Status</TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {g.items.map((r) => {
                    const v = current(r);
                    return (
                      <TableRow key={r.regulation_key} data-testid="source-row" data-key={r.regulation_key}>
                        <TableCell className="whitespace-normal font-medium">{shortName(r)}</TableCell>
                        <TableCell className="whitespace-normal text-sm">
                          {r.title}
                          <div className="text-xs text-muted-foreground">
                            {r.kind.toLowerCase().replace("_", " ")} · {r.authority_level.toLowerCase().replace(/_/g, " ")}
                          </div>
                        </TableCell>
                        <TableCell className="whitespace-normal text-sm">{v?.label ?? "—"}</TableCell>
                        <TableCell className="text-sm text-text-secondary">{v?.valid_from ?? "—"}</TableCell>
                        <TableCell>
                          <Badge variant={v?.status === "ACTIVE" ? "default" : "secondary"}>{(v?.status ?? "—").toLowerCase()}</Badge>
                        </TableCell>
                      </TableRow>
                    );
                  })}
                </TableBody>
              </Table>
            </div>
          </section>
        ))
      )}
    </div>
  );
}
