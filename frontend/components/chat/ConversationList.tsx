"use client";
import { Archive, Search } from "lucide-react";
import Link from "next/link";
import { useState } from "react";

import { EmptyState, ErrorState, LoadingState } from "@/components/common/States";
import { Input } from "@/components/ui/input";
import { useConversations } from "@/features/queries";
import { errorMessage } from "@/lib/errors";
import { cn } from "@/lib/utils";

export function ConversationList({ activeId }: { activeId: string | null }) {
  const [q, setQ] = useState("");
  const [archived, setArchived] = useState(false);
  const list = useConversations(q || undefined, archived);
  return (
    <div className="flex h-full flex-col">
      <div className="space-y-2 border-b p-2">
        <div className="relative">
          <Search className="pointer-events-none absolute left-2 top-1/2 size-4 -translate-y-1/2 text-muted-foreground" aria-hidden />
          <Input value={q} onChange={(e) => setQ(e.target.value)} placeholder="Search investigations" aria-label="Search investigations" className="pl-8" />
        </div>
        <button
          type="button"
          className={cn("flex items-center gap-1.5 text-xs text-text-secondary hover:text-foreground", archived && "text-primary")}
          onClick={() => setArchived((a) => !a)}
          aria-pressed={archived}
        >
          <Archive className="size-3.5" aria-hidden /> {archived ? "Showing archived" : "Show archived"}
        </button>
      </div>
      <div className="min-h-0 flex-1 overflow-y-auto p-2">
        {list.isPending ? (
          <LoadingState rows={4} />
        ) : list.isError ? (
          <ErrorState message={errorMessage(list.error)} onRetry={() => list.refetch()} />
        ) : list.data.items.length === 0 ? (
          <EmptyState title={archived ? "No archived investigations" : "No investigations yet"} hint="Start one with “New investigation”." />
        ) : (
          <ul className="space-y-0.5" data-testid="conversation-list">
            {list.data.items.map((c) => (
              <li key={c.id}>
                <Link
                  href={`/app/chat/${c.id}`}
                  aria-current={c.id === activeId ? "page" : undefined}
                  className={cn(
                    "block truncate rounded-md px-2.5 py-2 text-sm hover:bg-secondary",
                    c.id === activeId && "bg-primary-soft font-medium text-primary hover:bg-primary-soft",
                  )}
                  title={c.title}
                >
                  {c.title}
                  <span className="block text-xs font-normal text-muted-foreground">{new Date(c.updated_at).toLocaleString()}</span>
                </Link>
              </li>
            ))}
          </ul>
        )}
      </div>
    </div>
  );
}
