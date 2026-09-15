"use client";
import { ChevronDown } from "lucide-react";

import { Button } from "@/components/ui/button";
import { Checkbox } from "@/components/ui/checkbox";
import { DropdownMenu, DropdownMenuContent, DropdownMenuGroup, DropdownMenuLabel, DropdownMenuSeparator, DropdownMenuTrigger } from "@/components/ui/dropdown-menu";
import type { Me, SourceScope, SourceScopeName } from "@/lib/types";

// FR-CHAT-04: Verified regulations | Workspace documents | My private documents | All authorized sources.
const OPTIONS: { value: SourceScopeName; label: string }[] = [
  { value: "AUTHORITATIVE_ORG", label: "Verified regulations" },
  { value: "WORKSPACE", label: "Workspace documents" },
  { value: "PRIVATE_USER", label: "My private documents" },
];

export function scopeSummary(scope: SourceScope): string {
  if (scope.document_ids.length > 0) {
    return `${scope.document_ids.length === 1 ? "1 selected document" : `${scope.document_ids.length} selected documents`} + any regulation you name`;
  }
  if (scope.scopes.length === 3) return "All authorized sources";
  return OPTIONS.filter((o) => scope.scopes.includes(o.value))
    .map((o) => o.label)
    .join(" + ");
}

export function SourceScopeSelector({
  value,
  onChange,
  me,
  disabled,
}: {
  value: SourceScope;
  onChange: (next: SourceScope) => void;
  me: Me;
  disabled?: boolean;
}) {
  function toggle(name: SourceScopeName) {
    const has = value.scopes.includes(name);
    const scopes = has ? value.scopes.filter((s) => s !== name) : [...value.scopes, name];
    if (scopes.length === 0) return; // at least one source
    onChange({ ...value, scopes });
  }
  function toggleWorkspace(id: string) {
    const has = value.workspace_ids.includes(id);
    onChange({ ...value, workspace_ids: has ? value.workspace_ids.filter((w) => w !== id) : [...value.workspace_ids, id] });
  }
  return (
    <DropdownMenu>
      <DropdownMenuTrigger
        nativeButton
        render={<Button variant="outline" size="sm" disabled={disabled} data-testid="source-scope" aria-label="Source scope" />}
      >
        {scopeSummary(value)} <ChevronDown className="size-3.5" aria-hidden />
      </DropdownMenuTrigger>
      <DropdownMenuContent align="start" className="w-72 p-2">
        <DropdownMenuGroup>
          <DropdownMenuLabel>Search in</DropdownMenuLabel>
        </DropdownMenuGroup>
        {OPTIONS.map((o) => {
          const unavailable = o.value === "WORKSPACE" && me.workspaces.length === 0;
          return (
            <label key={o.value} className="flex cursor-pointer items-center gap-2 rounded-md px-2 py-1.5 text-sm hover:bg-secondary">
              <Checkbox
                checked={value.scopes.includes(o.value)}
                onCheckedChange={() => toggle(o.value)}
                disabled={unavailable}
                aria-label={o.label}
              />
              {o.label}
              {unavailable && <span className="ml-auto text-xs text-muted-foreground">no workspaces</span>}
            </label>
          );
        })}
        {value.scopes.includes("WORKSPACE") && me.workspaces.length > 1 && (
          <>
            <DropdownMenuSeparator />
            <DropdownMenuGroup>
              <DropdownMenuLabel>Workspaces (all when none selected)</DropdownMenuLabel>
            </DropdownMenuGroup>
            {me.workspaces.map((w) => (
              <label key={w.id} className="flex cursor-pointer items-center gap-2 rounded-md px-2 py-1.5 text-sm hover:bg-secondary">
                <Checkbox checked={value.workspace_ids.includes(w.id)} onCheckedChange={() => toggleWorkspace(w.id)} aria-label={w.name} />
                {w.name}
              </label>
            ))}
          </>
        )}
        <DropdownMenuSeparator />
        <Button
          size="sm"
          variant="ghost"
          className="w-full justify-start"
          onClick={() => onChange({ scopes: OPTIONS.map((o) => o.value), workspace_ids: [], document_ids: [] })}
        >
          All authorized sources
        </Button>
      </DropdownMenuContent>
    </DropdownMenu>
  );
}
