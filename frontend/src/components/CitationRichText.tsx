"use client";

import type { Citation } from "@/lib/types";

const CHIP_RE =
  /\[([^\]]+?)\s*§([^,\]]+)\s*,\s*p\.(\d+|\?)\]/g;

function matchCitation(label: string, citations: Citation[]): Citation | undefined {
  const norm = label.replace(/\s+/g, " ").trim();
  return (
    citations.find((c) => (c.label || c.citation || "").replace(/\s+/g, " ").trim() === norm) ||
    citations.find((c) => {
      const short = (c.regulation_id || "").replace("UN-ECE-", "");
      const needle = `${short} §${c.section_number}`;
      return norm.includes(needle) || norm.includes(c.section_number || "");
    })
  );
}

export function CitationRichText({
  text,
  citations,
  onCite,
}: {
  text: string;
  citations: Citation[];
  onCite: (c: Citation) => void;
}) {
  const nodes: React.ReactNode[] = [];
  let last = 0;
  let m: RegExpExecArray | null;
  const re = new RegExp(CHIP_RE.source, "g");
  while ((m = re.exec(text)) !== null) {
    if (m.index > last) {
      nodes.push(<span key={`t-${last}`}>{text.slice(last, m.index)}</span>);
    }
    const full = m[0];
    const cite = matchCitation(full, citations);
    nodes.push(
      <button
        key={`c-${m.index}`}
        type="button"
        className="cite-chip"
        title={cite?.section_title || cite?.text?.slice(0, 120) || full}
        onClick={() => cite && onCite(cite)}
        disabled={!cite}
      >
        {full}
      </button>
    );
    last = m.index + full.length;
  }
  if (last < text.length) {
    nodes.push(<span key={`t-end`}>{text.slice(last)}</span>);
  }
  return <div className="rich-answer">{nodes}</div>;
}
