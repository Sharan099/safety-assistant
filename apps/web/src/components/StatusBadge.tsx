// UI_UX_DESIGN_BRIEF.md Section 5: "Never use color alone." Every status
// carries a symbol + word, color is reinforcement only.

const STYLES: Record<string, { symbol: string; className: string }> = {
  PASS: { symbol: "✓", className: "bg-emerald-950 text-emerald-300 border-emerald-800" },
  COMPARABLE: { symbol: "✓", className: "bg-emerald-950 text-emerald-300 border-emerald-800" },
  SAME: { symbol: "✓", className: "bg-emerald-950 text-emerald-300 border-emerald-800" },

  WARNING: { symbol: "⚠", className: "bg-amber-950 text-amber-300 border-amber-800" },
  CONDITIONAL: { symbol: "⚠", className: "bg-amber-950 text-amber-300 border-amber-800" },
  PARTIALLY_SUPPORTED: { symbol: "⚠", className: "bg-amber-950 text-amber-300 border-amber-800" },

  FAIL: { symbol: "✕", className: "bg-red-950 text-red-300 border-red-800" },
  NOT_COMPARABLE: { symbol: "✕", className: "bg-red-950 text-red-300 border-red-800" },
  REJECTED: { symbol: "✕", className: "bg-red-950 text-red-300 border-red-800" },
  CONTRADICTED: { symbol: "✕", className: "bg-red-950 text-red-300 border-red-800" },
  BLOCKED: { symbol: "✕", className: "bg-red-950 text-red-300 border-red-800" },

  CHANGED: { symbol: "±", className: "bg-sky-950 text-sky-300 border-sky-800" },

  UNKNOWN: { symbol: "?", className: "bg-neutral-800 text-neutral-300 border-neutral-600" },
  NOT_ESTABLISHED: { symbol: "?", className: "bg-neutral-800 text-neutral-300 border-neutral-600" },
  INCONCLUSIVE: { symbol: "?", className: "bg-neutral-800 text-neutral-300 border-neutral-600" },
};

const DEFAULT_STYLE = { symbol: "•", className: "bg-neutral-800 text-neutral-300 border-neutral-600" };

export function StatusBadge({ status }: { status: string }) {
  const style = STYLES[status] ?? DEFAULT_STYLE;
  return (
    <span
      className={`inline-flex items-center gap-1 rounded border px-2 py-0.5 text-xs font-medium tabular-nums ${style.className}`}
    >
      <span aria-hidden>{style.symbol}</span>
      {status}
    </span>
  );
}
