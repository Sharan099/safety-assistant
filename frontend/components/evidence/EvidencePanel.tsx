"use client";
import { Copy, ExternalLink, X } from "lucide-react";
import { useEffect, useRef, useSyncExternalStore } from "react";
import { toast } from "sonner";

import { ScopeBadge } from "@/components/common/StatusBadge";
import { Button } from "@/components/ui/button";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Sheet, SheetContent, SheetHeader, SheetTitle } from "@/components/ui/sheet";
import { api } from "@/lib/api";
import { cn } from "@/lib/utils";

import { type EvidenceItem, useEvidence } from "./EvidenceContext";

function validity(item: EvidenceItem): string {
  if (item.valid_from && item.valid_to) return `valid ${item.valid_from} → ${item.valid_to}`;
  if (item.valid_from) return `in force from ${item.valid_from}`;
  if (item.version_status) return item.version_status.toLowerCase();
  return "validity unknown";
}

export function EvidenceCard({ item, active }: { item: EvidenceItem; active: boolean }) {
  const ref = useRef<HTMLElement>(null);
  useEffect(() => {
    if (active) ref.current?.scrollIntoView({ block: "nearest" });
  }, [active]);
  const citation = `${item.label}${item.source_sha256 ? ` (sha256 ${item.source_sha256.slice(0, 12)})` : ""}`;
  return (
    <article
      ref={ref}
      id={`evidence-${item.id}`}
      tabIndex={-1}
      data-testid="evidence-card"
      data-active={active}
      className={cn("rounded-xl border bg-card p-3 text-sm outline-none", active && "border-primary ring-2 ring-primary/30")}
      aria-label={`Evidence ${item.order}: ${item.label}`}
    >
      <div className="flex items-start justify-between gap-2">
        <div className="min-w-0">
          <div className="font-medium">
            <span className="mr-1 rounded bg-secondary px-1.5 py-0.5 font-mono text-xs">[{item.order}]</span>
            {item.regulation_key.replace("UN-", "UN ")}
            {item.cited === false && <span className="ml-2 text-xs font-normal text-muted-foreground">retrieved, not cited</span>}
          </div>
          <div className="mt-0.5 text-xs text-text-secondary">
            {item.version_label} · §{item.section_path}
            {item.page_start ? ` · p. ${item.page_start}${item.page_end && item.page_end !== item.page_start ? `–${item.page_end}` : ""}` : ""}
          </div>
          <div className="text-xs text-text-secondary">{validity(item)}</div>
        </div>
        {item.scope && <ScopeBadge scope={item.scope} />}
      </div>
      {item.excerpt ? (
        <p className="mt-2 max-h-48 overflow-y-auto whitespace-pre-wrap rounded-md bg-secondary/60 p-2 font-mono text-xs leading-relaxed text-foreground">
          {item.excerpt}
        </p>
      ) : (
        <p className="mt-2 text-xs italic text-muted-foreground">
          {item.available ? "Excerpt not stored." : "The cited version is no longer active; the excerpt is unavailable."}
        </p>
      )}
      <div className="mt-2 flex flex-wrap gap-2">
        {item.chunk_id && item.available && (
          <Button nativeButton={false}
            size="sm"
            variant="outline"
            render={<a href={api.evidenceUrl(item.chunk_id)} target="_blank" rel="noopener noreferrer" />}
          >
            <ExternalLink className="size-3.5" aria-hidden /> Open source
          </Button>
        )}
        <Button
          size="sm"
          variant="ghost"
          onClick={() => navigator.clipboard.writeText(citation).then(() => toast.success("Citation copied"))}
        >
          <Copy className="size-3.5" aria-hidden /> Copy citation
        </Button>
      </div>
    </article>
  );
}

function PanelBody() {
  const { items, activeId } = useEvidence();
  if (items.length === 0) {
    return (
      <p className="p-4 text-sm text-text-secondary">
        Evidence for the selected answer appears here. Select a citation marker like <span className="font-mono">[1]</span> in
        an answer to focus it.
      </p>
    );
  }
  return (
    <ScrollArea className="h-full">
      <ol className="space-y-3 p-3" aria-label="Evidence list">
        {items.map((it) => (
          <li key={it.id}>
            <EvidenceCard item={it} active={it.id === activeId} />
          </li>
        ))}
      </ol>
    </ScrollArea>
  );
}

const DESKTOP = "(min-width: 1280px)";
function useIsDesktop(): boolean {
  return useSyncExternalStore(
    (cb) => {
      const mq = window.matchMedia(DESKTOP);
      mq.addEventListener("change", cb);
      return () => mq.removeEventListener("change", cb);
    },
    () => window.matchMedia(DESKTOP).matches,
    () => true,
  );
}

/** Desktop column (≥1280px) + sheet for narrower widths (03_UI_UX "Responsive behavior"). */
export function EvidencePanel() {
  const { open, setOpen, items } = useEvidence();
  const desktop = useIsDesktop();
  return (
    <>
      <aside
        className="hidden w-[400px] shrink-0 flex-col border-l bg-background xl:flex"
        aria-label="Evidence"
        data-testid="evidence-panel"
      >
        <div className="flex h-12 items-center justify-between border-b px-4">
          <h2 className="text-sm font-semibold">Evidence {items.length ? `(${items.length})` : ""}</h2>
        </div>
        <div className="min-h-0 flex-1">
          <PanelBody />
        </div>
      </aside>
      <Sheet open={open && !desktop} onOpenChange={setOpen}>
        <SheetContent side="right" className="w-full p-0 sm:max-w-md" data-testid="evidence-sheet">
          <SheetHeader className="flex h-12 flex-row items-center justify-between border-b px-4 py-0">
            <SheetTitle className="text-sm">Evidence {items.length ? `(${items.length})` : ""}</SheetTitle>
            <Button size="icon" variant="ghost" aria-label="Close evidence" onClick={() => setOpen(false)}>
              <X className="size-4" />
            </Button>
          </SheetHeader>
          <div className="h-[calc(100vh-3rem)]">
            <PanelBody />
          </div>
        </SheetContent>
      </Sheet>
    </>
  );
}
