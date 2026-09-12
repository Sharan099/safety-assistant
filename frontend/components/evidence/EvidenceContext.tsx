"use client";
// Evidence panel state shared between the message list (citation markers) and the shell's panel.
import { createContext, useCallback, useContext, useMemo, useState } from "react";

import type { Citation, Evidence, MessageCitation, SourceScopeName } from "@/lib/types";

/** One row the panel can render, from either a live answer (Evidence) or persisted history. */
export interface EvidenceItem {
  id: string; // evidence_id ("E1") for live answers, citation label for history
  order: number;
  label: string;
  regulation_key: string;
  regulation_title?: string;
  version_label: string;
  version_status?: string;
  section_path: string;
  page_start: number | null;
  page_end: number | null;
  valid_from: string | null;
  valid_to: string | null;
  scope: SourceScopeName | null;
  authority_level?: string;
  excerpt: string | null;
  chunk_id: string | null;
  source_sha256: string | null;
  available: boolean;
  cited?: boolean;
}

/** Cited evidence first, numbered in citation order (matching the answer's markers), then the rest. */
export function fromEvidence(list: Evidence[], citations: Citation[] = []): EvidenceItem[] {
  const citedIds = citations.map((c) => c.evidence_id);
  const ordered = [
    ...citedIds.map((id) => list.find((e) => e.evidence_id === id)).filter((e): e is Evidence => !!e),
    ...list.filter((e) => !citedIds.includes(e.evidence_id)),
  ];
  return ordered.map((e, i) => ({
    id: e.evidence_id,
    order: i + 1,
    cited: i < citedIds.length,
    label: e.citation_label,
    regulation_key: e.regulation_key,
    regulation_title: e.regulation_title,
    version_label: e.version_label,
    version_status: e.version_status,
    section_path: e.section_path,
    page_start: e.page_start,
    page_end: e.page_end,
    valid_from: e.valid_from,
    valid_to: e.valid_to,
    scope: scopeFromAuthority(e.authority_level, e.regulation_key),
    authority_level: e.authority_level,
    excerpt: e.content,
    chunk_id: e.chunk_id,
    source_sha256: e.source_sha256,
    available: true,
  }));
}

export function fromCitations(list: MessageCitation[]): EvidenceItem[] {
  return list.map((c) => ({
    id: c.label,
    order: c.order + 1,
    label: c.label,
    regulation_key: c.regulation_key,
    version_label: c.version_label,
    section_path: c.section_path,
    page_start: c.page_start,
    page_end: c.page_end,
    valid_from: null,
    valid_to: null,
    scope: scopeFromAuthority(undefined, c.regulation_key),
    excerpt: c.quote_excerpt,
    chunk_id: c.chunk_id,
    source_sha256: c.source_sha256,
    available: c.evidence_available,
  }));
}

// Uploaded documents carry DOC-… keys and REFERENCE/INTERNAL_APPROVED authority; registry sources are
// AUTHORITATIVE. The API does not (yet) echo the scope on evidence, so this is the display rule.
function scopeFromAuthority(authority: string | undefined, key: string): SourceScopeName | null {
  if (key.startsWith("DOC-")) return authority === "INTERNAL_APPROVED" ? "AUTHORITATIVE_ORG" : "PRIVATE_USER";
  return "AUTHORITATIVE_ORG";
}

interface Ctx {
  items: EvidenceItem[];
  activeId: string | null;
  open: boolean;
  show: (items: EvidenceItem[], focus?: string) => void;
  focus: (id: string) => void;
  setOpen: (open: boolean) => void;
}

const EvidenceCtx = createContext<Ctx | null>(null);

export function EvidenceProvider({ children }: { children: React.ReactNode }) {
  const [items, setItems] = useState<EvidenceItem[]>([]);
  const [activeId, setActiveId] = useState<string | null>(null);
  const [open, setOpen] = useState(false);
  const show = useCallback((next: EvidenceItem[], focusId?: string) => {
    setItems(next);
    setActiveId(focusId ?? next[0]?.id ?? null);
    if (focusId) setOpen(true);
  }, []);
  const focus = useCallback((id: string) => {
    setActiveId(id);
    setOpen(true);
  }, []);
  const value = useMemo(() => ({ items, activeId, open, show, focus, setOpen }), [items, activeId, open, show, focus]);
  return <EvidenceCtx.Provider value={value}>{children}</EvidenceCtx.Provider>;
}

export function useEvidence(): Ctx {
  const ctx = useContext(EvidenceCtx);
  if (!ctx) throw new Error("useEvidence outside EvidenceProvider");
  return ctx;
}
