"use client";
import { useRouter, useSearchParams } from "next/navigation";
import { Suspense, useState } from "react";

import { Composer } from "@/components/chat/Composer";
import { SourceScopeSelector, scopeSummary } from "@/components/chat/SourceScopeSelector";
import { ErrorState, LoadingState } from "@/components/common/States";
import { useCreateConversation, useMe } from "@/features/queries";
import { errorMessage } from "@/lib/errors";
import type { SourceScope } from "@/lib/types";

const DEFAULT_SCOPE: SourceScope = { scopes: ["AUTHORITATIVE_ORG"], workspace_ids: [], document_ids: [] };

// New investigation: pick a source scope, ask the first question → the conversation is created and
// the question is carried to /app/chat/[id] via sessionStorage (never the URL).
function NewInvestigation() {
  const router = useRouter();
  const search = useSearchParams();
  const me = useMe();
  const create = useCreateConversation();
  const [scope, setScope] = useState<SourceScope>(() => {
    const doc = search.get("document");
    return doc ? { scopes: ["AUTHORITATIVE_ORG", "WORKSPACE", "PRIVATE_USER"], workspace_ids: [], document_ids: [doc] } : DEFAULT_SCOPE;
  });

  if (me.isPending) return <LoadingState rows={2} />;
  if (me.isError) return <ErrorState message={errorMessage(me.error)} />;

  function start(content: string, asOf: string | null) {
    create.mutate(
      { source_scope: scope },
      {
        onSuccess: (conv) => {
          try {
            sessionStorage.setItem(`sa.pending.${conv.id}`, JSON.stringify({ content, asOf }));
          } catch {
            /* storage unavailable: the user re-types the question */
          }
          router.push(`/app/chat/${conv.id}`);
        },
      },
    );
  }

  return (
    <>
      <div className="flex h-12 items-center gap-3 border-b bg-card px-4">
        <h1 className="text-sm font-semibold">New investigation</h1>
        <SourceScopeSelector value={scope} onChange={setScope} me={me.data} />
        {scope.document_ids.length > 0 && <span className="text-xs text-text-secondary">Limited to 1 selected document</span>}
      </div>
      <div className="flex flex-1 flex-col justify-end">
        <div className="mx-auto max-w-3xl px-4 py-8 text-sm text-text-secondary">
          <p className="text-base font-medium text-foreground">Ask a regulatory question.</p>
          <p className="mt-1">
            Answers are grounded in the selected sources only; every claim cites clause, version and page. When the evidence is
            insufficient, the assistant says so instead of guessing.
          </p>
          {create.isError && (
            <div className="mt-4">
              <ErrorState message={errorMessage(create.error)} />
            </div>
          )}
        </div>
        <Composer onSend={start} busy={create.isPending} scopeSummary={scopeSummary(scope)} autoFocus />
      </div>
    </>
  );
}

export default function ChatIndexPage() {
  return (
    <Suspense fallback={<LoadingState rows={2} />}>
      <NewInvestigation />
    </Suspense>
  );
}
