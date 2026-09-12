import { ShieldCheck } from "lucide-react";

export function ProductMark({ compact = false }: { compact?: boolean }) {
  return (
    <div className="flex items-center gap-2">
      <span className="grid size-8 place-items-center rounded-md bg-primary text-primary-foreground" aria-hidden>
        <ShieldCheck className="size-5" />
      </span>
      {!compact && <span className="text-base font-semibold tracking-tight">Safety Assistant</span>}
    </div>
  );
}
