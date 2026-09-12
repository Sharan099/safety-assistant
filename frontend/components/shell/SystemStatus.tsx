"use client";
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip";
import { useReady } from "@/features/queries";
import { cn } from "@/lib/utils";

export function SystemStatus() {
  const { data, isError } = useReady();
  const ok = !isError && data?.status === "ready";
  const label = isError ? "API unreachable" : data ? (ok ? "System ready" : "System not ready") : "Checking…";
  const detail = data
    ? Object.entries(data.deps)
        .map(([k, v]) => `${k}: ${v.ok ? "ok" : "down"}`)
        .join(" · ")
    : "";
  return (
    <Tooltip>
      <TooltipTrigger
        render={<span className="flex items-center gap-1.5 text-xs text-text-secondary" data-testid="system-status" aria-live="polite" />}
      >
        <span className={cn("size-2 rounded-full", ok ? "bg-success" : isError ? "bg-destructive" : "bg-warning")} aria-hidden />
        <span className="hidden md:inline">{label}</span>
      </TooltipTrigger>
      <TooltipContent>{detail || label}</TooltipContent>
    </Tooltip>
  );
}
