"use client";
import { Archive, Pencil } from "lucide-react";
import { useParams } from "next/navigation";
import { useEffect, useRef, useState } from "react";
import { toast } from "sonner";

import { Composer } from "@/components/chat/Composer";
import { type FailedTurn, MessageList } from "@/components/chat/MessageList";
import { SourceScopeSelector, scopeSummary } from "@/components/chat/SourceScopeSelector";
import { EmptyState, ErrorState, LoadingState } from "@/components/common/States";
import { type EvidenceItem, fromEvidence, useEvidence } from "@/components/evidence/EvidenceContext";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { useConversation, useMe, usePatchConversation, useSendMessage } from "@/features/queries";
import { errorMessage } from "@/lib/errors";

export default function ConversationPage() {
  const { id } = useParams<{ id: string }>();
  const me = useMe();
  const conv = useConversation(id);
  const send = useSendMessage(id);
  const patchConv = usePatchConversation(id);
  const { show } = useEvidence();
  const [editing, setEditing] = useState(false);
  const [title, setTitle] = useState("");
  // Live evidence (full excerpts) for answers produced in this session; history falls back to stored citations.
  const [live, setLive] = useState<Record<string, EvidenceItem[]>>({});
  const started = useRef(false);
  const pending = send.isPending ? (send.variables?.content ?? null) : null;
  // A failed turn stays visible with a retry; the question is never silently dropped.
  const [failed, setFailed] = useState<FailedTurn | null>(null);

  function ask(content: string, asOf: string | null) {
    setFailed(null);
    send.mutate(
      { content, as_of: asOf },
      {
        onSuccess: (ex) => {
          const items = fromEvidence(ex.answer.evidence, ex.answer.citations);
          setLive((prev) => ({ ...prev, [ex.assistant_message.id]: items }));
          show(items);
        },
        onError: (e) => setFailed({ content, message: errorMessage(e), retry: () => ask(content, asOf) }),
      },
    );
  }

  // First question carried over from /app/chat (new investigation).
  useEffect(() => {
    if (started.current || !conv.data) return;
    started.current = true;
    try {
      const raw = sessionStorage.getItem(`sa.pending.${id}`);
      if (raw) {
        sessionStorage.removeItem(`sa.pending.${id}`);
        const { content, asOf } = JSON.parse(raw) as { content: string; asOf: string | null };
        ask(content, asOf);
      }
    } catch {
      /* ignore */
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [conv.data, id]);

  if (conv.isPending || me.isPending) return <div className="p-4"><LoadingState rows={4} label="Loading conversation" /></div>;
  if (conv.isError || me.isError) {
    return (
      <div className="p-4">
        <ErrorState message={errorMessage(conv.error ?? me.error)} onRetry={() => conv.refetch()} />
      </div>
    );
  }
  const c = conv.data;
  const archived = !!c.archived_at;

  return (
    <>
      <header className="flex h-12 items-center gap-2 border-b bg-card px-4" data-testid="conversation-header">
        {editing ? (
          <form
            className="flex items-center gap-2"
            onSubmit={(e) => {
              e.preventDefault();
              patchConv.mutate({ title: title.trim() || c.title }, { onSuccess: () => setEditing(false) });
            }}
          >
            <Input value={title} onChange={(e) => setTitle(e.target.value)} aria-label="Conversation title" className="h-8 w-72" autoFocus />
            <Button size="sm" type="submit">Save</Button>
            <Button size="sm" variant="ghost" type="button" onClick={() => setEditing(false)}>Cancel</Button>
          </form>
        ) : (
          <>
            <h1 className="truncate text-sm font-semibold" data-testid="conversation-title">{c.title}</h1>
            <Button size="icon-sm" variant="ghost" aria-label="Rename" onClick={() => { setTitle(c.title); setEditing(true); }}>
              <Pencil className="size-3.5" />
            </Button>
          </>
        )}
        <div className="ml-2">
          <SourceScopeSelector value={c.source_scope} onChange={(s) => patchConv.mutate({ source_scope: s })} me={me.data} disabled={archived} />
        </div>
        <div className="ml-auto">
          <Button
            size="sm"
            variant="ghost"
            onClick={() => patchConv.mutate({ archived: !archived }, { onSuccess: () => toast.success(archived ? "Restored" : "Archived") })}
            data-testid="archive-conversation"
          >
            <Archive className="size-4" aria-hidden /> {archived ? "Restore" : "Archive"}
          </Button>
        </div>
      </header>
      <div className="min-h-0 flex-1 overflow-y-auto">
        {c.messages.length === 0 && !pending && !failed ? (
          <div className="p-6">
            <EmptyState title="No messages yet" hint="Ask the first question below." />
          </div>
        ) : (
          <MessageList messages={c.messages} liveEvidence={live} pending={pending} failed={failed} />
        )}
      </div>
      {archived ? (
        <p className="border-t bg-card px-4 py-3 text-sm text-text-secondary">This investigation is archived. Restore it to continue.</p>
      ) : (
        <Composer onSend={ask} busy={send.isPending} scopeSummary={scopeSummary(c.source_scope)} />
      )}
    </>
  );
}
