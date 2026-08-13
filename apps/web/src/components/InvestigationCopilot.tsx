"use client";

import { useEffect, useRef, useState } from "react";
import { useQuery, useQueryClient } from "@tanstack/react-query";
import { api } from "@/lib/api";
import type { CopilotMessageSummary, CopilotStepEvent } from "@/lib/types";

// PRD_COPILOT_UPDATE.md — contextual chat panel embedded in the investigation
// workspace (explicitly NOT a standalone /chat page). The live step feed
// below is what gives the engineer real-time visibility into the LangGraph
// agent's own workflow as it runs — one line per node, straight off the SSE
// stream in apps/api/routers/copilot.py.
export function InvestigationCopilot({ investigationId }: { investigationId: string }) {
  const qc = useQueryClient();
  const [collapsed, setCollapsed] = useState(false);
  const [input, setInput] = useState("");
  const [isStreaming, setIsStreaming] = useState(false);
  const [liveSteps, setLiveSteps] = useState<CopilotStepEvent[]>([]);
  const [pendingUserMessage, setPendingUserMessage] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const scrollRef = useRef<HTMLDivElement>(null);

  const history = useQuery({
    queryKey: ["copilot-messages", investigationId],
    queryFn: () => api.listCopilotMessages(investigationId),
    enabled: !!investigationId && !collapsed,
  });

  const messages = history.data ?? [];
  const lastAssistant = [...messages].reverse().find((m) => m.role === "assistant");

  useEffect(() => {
    scrollRef.current?.scrollTo({ top: scrollRef.current.scrollHeight });
  }, [messages.length, liveSteps.length, pendingUserMessage]);

  const send = async (text: string) => {
    const trimmed = text.trim();
    if (!trimmed || isStreaming) return;
    setError(null);
    setInput("");
    setPendingUserMessage(trimmed);
    setLiveSteps([]);
    setIsStreaming(true);

    await api.streamCopilotMessage(investigationId, trimmed, {
      onStep: (event) => setLiveSteps((prev) => [...prev, event]),
      onFinal: () => {
        setIsStreaming(false);
        setPendingUserMessage(null);
        setLiveSteps([]);
        qc.invalidateQueries({ queryKey: ["copilot-messages", investigationId] });
      },
      onError: (detail) => {
        setIsStreaming(false);
        setPendingUserMessage(null);
        setLiveSteps([]);
        setError(detail);
      },
    });
  };

  return (
    <section className="flex h-full flex-col rounded border border-neutral-800 bg-neutral-950">
      <button
        onClick={() => setCollapsed((c) => !c)}
        className="flex w-full shrink-0 items-center justify-between px-4 py-3 text-left"
      >
        <span className="flex items-center gap-2 text-sm font-semibold text-neutral-200">
          Investigation Copilot
          {isStreaming && <span className="text-xs font-normal text-sky-400">working…</span>}
        </span>
        <span className="text-neutral-500">{collapsed ? "Expand" : "Collapse"}</span>
      </button>

      {!collapsed && (
        <div className="flex min-h-0 flex-1 flex-col border-t border-neutral-800 px-4 py-3">
          <div ref={scrollRef} className="min-h-[16rem] flex-1 space-y-3 overflow-y-auto pr-1">
            {history.isLoading && <p className="text-sm text-neutral-500">Loading conversation…</p>}
            {!history.isLoading && messages.length === 0 && !pendingUserMessage && (
              <p className="text-sm text-neutral-500">
                Ask a question about this investigation — e.g. &quot;Why is the restraint configuration a leading
                contributor?&quot; The Copilot answers only from this investigation&apos;s evidence, retrieved
                documents, and analysis tools.
              </p>
            )}

            {messages.map((m) => (
              <CopilotMessageBubble key={m.id} message={m} />
            ))}

            {pendingUserMessage && (
              <div className="ml-8 rounded bg-sky-950/40 px-3 py-2 text-sm text-sky-100">{pendingUserMessage}</div>
            )}

            {isStreaming && (
              <div className="space-y-1 rounded border border-neutral-800 bg-neutral-900/40 px-3 py-2">
                {liveSteps.length === 0 && <StepLine label="Starting…" active />}
                {liveSteps.map((s, i) => (
                  <StepLine key={i} label={s.detail} active={i === liveSteps.length - 1} />
                ))}
              </div>
            )}
          </div>

          {error && <p className="mt-2 text-sm text-red-400">{error}</p>}

          {lastAssistant && lastAssistant.suggested_actions.length > 0 && !isStreaming && (
            <div className="mt-3 flex flex-wrap gap-2">
              {lastAssistant.suggested_actions.map((action) => (
                <button
                  key={action}
                  onClick={() => send(action)}
                  className="rounded-full border border-neutral-700 px-3 py-1 text-xs text-neutral-300 hover:bg-neutral-900"
                >
                  {action}
                </button>
              ))}
            </div>
          )}

          <form
            onSubmit={(e) => {
              e.preventDefault();
              send(input);
            }}
            className="mt-3 flex items-center gap-2"
          >
            <input
              className="w-full rounded border border-neutral-700 bg-neutral-900 px-3 py-1.5 text-sm text-neutral-100 disabled:opacity-50"
              value={input}
              onChange={(e) => setInput(e.target.value)}
              placeholder="Ask the Copilot about this investigation…"
              disabled={isStreaming}
            />
            <button
              type="submit"
              disabled={isStreaming || !input.trim()}
              className="rounded bg-sky-600 px-4 py-2 text-sm font-medium text-white hover:bg-sky-500 disabled:opacity-50"
            >
              {isStreaming ? "…" : "Send"}
            </button>
          </form>
        </div>
      )}
    </section>
  );
}

function StepLine({ label, active }: { label: string; active: boolean }) {
  return (
    <div className={`flex items-center gap-2 text-xs ${active ? "text-sky-300" : "text-neutral-500"}`}>
      <span className={`h-1.5 w-1.5 rounded-full ${active ? "bg-sky-400" : "bg-neutral-600"}`} />
      {label}
    </div>
  );
}

function CopilotMessageBubble({ message }: { message: CopilotMessageSummary }) {
  const isUser = message.role === "user";
  return (
    <div className={isUser ? "ml-8" : "mr-8"}>
      <div
        className={`rounded px-3 py-2 text-sm ${
          isUser ? "bg-sky-950/40 text-sky-100" : "border border-neutral-800 bg-neutral-900/60 text-neutral-200"
        }`}
      >
        {message.content}
      </div>

      {!isUser && message.llm_degraded && (
        <p className="mt-1 text-xs text-amber-400">
          Note: natural-language drafting was unavailable — this is the deterministic, evidence-grounded answer.
        </p>
      )}

      {!isUser && message.tool_activity.length > 0 && (
        <div className="mt-1 flex flex-wrap gap-1.5">
          {message.tool_activity.map((t, i) => (
            <span
              key={i}
              title={t.result_summary}
              className="rounded border border-neutral-700 px-1.5 py-0.5 font-mono text-[10px] text-neutral-400"
            >
              {t.tool_name} · {t.status}
            </span>
          ))}
        </div>
      )}

      {!isUser && message.evidence_refs.length > 0 && (
        <div className="mt-1 flex flex-wrap gap-1.5">
          {message.evidence_refs.map((ref) => (
            <span
              key={ref}
              className="rounded border border-emerald-800 px-1.5 py-0.5 font-mono text-[10px] text-emerald-300"
            >
              evidence: {ref.slice(0, 8)}
            </span>
          ))}
        </div>
      )}

      {!isUser && message.unknowns.length > 0 && (
        <ul className="mt-1 list-inside list-disc text-xs text-amber-400">
          {message.unknowns.map((u, i) => (
            <li key={i}>{u}</li>
          ))}
        </ul>
      )}
    </div>
  );
}
