"use client";
import { useHealth } from "@/hooks/useHealth";

export function SystemStatus() {
  const { health, error } = useHealth();
  const ok = health?.status === "ready";
  const db = health?.deps?.database as { active_versions?: number; migration?: string } | undefined;
  const emb = health?.deps?.embeddings as { model?: string } | undefined;
  return (
    <div data-testid="system-status" className="flex flex-wrap items-center gap-3 text-xs text-zinc-400">
      <span className={`inline-block h-2 w-2 rounded-full ${error ? "bg-rose-500" : ok ? "bg-emerald-500" : "bg-amber-500"}`} aria-hidden />
      <span>{error ? "API unreachable" : ok ? "ready" : health ? "not ready" : "checking…"}</span>
      {db?.active_versions !== undefined && <span>{db.active_versions} active regulation versions</span>}
      {emb?.model && <span>embeddings: {emb.model}</span>}
      {health?.app_env && <span>env: {health.app_env}</span>}
    </div>
  );
}
