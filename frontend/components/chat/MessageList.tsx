"use client";
import { AlertTriangle } from "lucide-react";
import { useEffect, useRef } from "react";

import { AnswerModeBadge } from "@/components/common/StatusBadge";
import { type EvidenceItem, fromCitations, useEvidence } from "@/components/evidence/EvidenceContext";
import { Button } from "@/components/ui/button";
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip";
import type { Message } from "@/lib/types";
import { cn } from "@/lib/utils";

function pages(item: EvidenceItem): string {
  if (!item.page_start) return "";
  return item.page_end && item.page_end !== item.page_start ? `pp. ${item.page_start}–${item.page_end}` : `p. ${item.page_start}`;
}

/** Hover / focus preview of the cited lines: what the answer rests on, before opening the panel. */
export function CitationPreview({ item, children }: { item: EvidenceItem; children: React.ReactElement }) {
  return (
    <Tooltip>
      <TooltipTrigger render={children} />
      <TooltipContent side="top" className="max-w-md flex-col items-stretch p-0 text-left" data-testid="citation-preview">
        <div className="border-b px-3 py-2 text-xs">
          <span className="font-medium">{item.label}</span>
          {pages(item) && <span className="text-text-secondary"> · {pages(item)}</span>}
          <span className="text-text-secondary"> · {item.version_label}</span>
        </div>
        <pre className="max-h-48 overflow-y-auto whitespace-pre-wrap px-3 py-2 font-sans text-xs leading-relaxed">
          {item.excerpt ?? "Excerpt not stored for this citation — open the evidence panel."}
        </pre>
      </TooltipContent>
    </Tooltip>
  );
}

/** Renders "[E1]" / "[1]" markers in answer text as buttons that focus the evidence panel. */
export function CitationMarkers({ text, items, onFocus }: { text: string; items: EvidenceItem[]; onFocus: (id: string) => void }) {
  const parts = text.split(/(\[E?\d+\])/g);
  return (
    <>
      {parts.map((part, i) => {
        const m = /^\[E?(\d+)\]$/.exec(part);
        if (!m) return <span key={i}>{part}</span>;
        const n = Number(m[1]);
        // "[E7]" (live answers) refers to an evidence id; "[2]" (rendered chips / history) to citation order.
        const item = m[0].startsWith("[E") ? items.find((it) => it.id === `E${n}`) : items.find((it) => it.order === n);
        if (!item) return <span key={i}>{part}</span>;
        return (
          <CitationPreview key={i} item={item}>
            <button
              type="button"
              data-testid="citation-marker"
              className="mx-0.5 inline-flex rounded bg-evidence-soft px-1.5 py-0 font-mono text-xs font-medium text-evidence hover:ring-2 hover:ring-evidence/40 focus-visible:ring-2 focus-visible:ring-evidence"
              aria-label={`Show evidence ${n}: ${item.label}`}
              onClick={() => onFocus(item.id)}
            >
              [{item.order}]
            </button>
          </CitationPreview>
        );
      })}
    </>
  );
}

export function AssistantMessage({ message, live }: { message: Message; live?: EvidenceItem[] }) {
  const { show, focus } = useEvidence();
  const items = live ?? fromCitations(message.citations);
  const mode = message.answer_mode ?? "EVIDENCE_ONLY";
  return (
    <article
      className="rounded-xl border bg-card p-4"
      data-testid="answer"
      data-mode={mode}
      aria-label="Assistant answer"
    >
      <div className="mb-2 flex flex-wrap items-center gap-2">
        {message.abstain_reason === "small_talk" ? (
          <span className="text-xs font-medium text-text-secondary" data-testid="answer-mode" data-mode="ASSISTANT">Assistant</span>
        ) : (
          <AnswerModeBadge mode={mode} />
        )}
        {message.citations.length > 0 && (
          <Button size="sm" variant="ghost" onClick={() => show(items, items[0]?.id)} data-testid="view-evidence">
            View evidence ({message.citations.length})
          </Button>
        )}
      </div>
      {mode === "ABSTAINED" ? (
        <p className="text-sm text-text-secondary" data-testid="abstain">
          {message.content || "The authorized sources do not contain enough evidence to answer. Try naming the regulation, broadening the source scope, or uploading the relevant document."}
        </p>
      ) : (
        <div className="whitespace-pre-wrap text-sm leading-relaxed">
          <CitationMarkers text={message.content} items={items} onFocus={(id) => { show(items, id); focus(id); }} />
        </div>
      )}
      {message.warnings.length > 0 && (
        <ul className="mt-3 space-y-1 rounded-md bg-warning-soft p-2 text-xs text-warning" data-testid="warnings" aria-label="Warnings">
          {message.warnings.map((w, i) => (
            <li key={i} className="flex items-start gap-1.5">
              <AlertTriangle className="mt-0.5 size-3.5 shrink-0" aria-hidden /> {w}
            </li>
          ))}
        </ul>
      )}
      {message.citations.length > 0 && (
        <ol className="mt-3 flex flex-wrap gap-1.5" aria-label="Citations">
          {message.citations.map((c) => {
            const item = items.find((it) => it.label === c.label) ?? items[c.order];
            const chip = (
              <button
                type="button"
                data-testid="citation"
                className={cn(
                  "rounded-md border px-2 py-1 text-left text-xs hover:bg-secondary focus-visible:ring-2 focus-visible:ring-ring",
                  !c.evidence_available && "opacity-60",
                )}
                onClick={() => {
                  const id = item?.id ?? c.label;
                  show(items, id);
                  focus(id);
                }}
              >
                <span className="font-mono">[{c.order + 1}]</span> {c.label}
              </button>
            );
            return <li key={c.order}>{item ? <CitationPreview item={item}>{chip}</CitationPreview> : chip}</li>;
          })}
        </ol>
      )}
    </article>
  );
}

export interface FailedTurn {
  content: string;
  message: string;
  retry: () => void;
}

export function MessageList({
  messages,
  liveEvidence,
  pending,
  failed,
}: {
  messages: Message[];
  liveEvidence?: Record<string, EvidenceItem[]>;
  pending?: string | null;
  failed?: FailedTurn | null;
}) {
  const end = useRef<HTMLDivElement>(null);
  useEffect(() => {
    end.current?.scrollIntoView({ block: "end" });
  }, [messages.length, pending, failed]);
  return (
    <div className="mx-auto flex max-w-3xl flex-col gap-4 px-4 py-4" aria-live="polite">
      {messages.map((m) =>
        m.role === "user" ? (
          <div key={m.id} className="self-end rounded-xl bg-primary-soft px-4 py-2.5 text-sm" data-testid="user-message">
            {m.content}
          </div>
        ) : (
          <AssistantMessage key={m.id} message={m} live={liveEvidence?.[m.id]} />
        ),
      )}
      {pending && (
        <>
          <div className="self-end rounded-xl bg-primary-soft px-4 py-2.5 text-sm">{pending}</div>
          <div className="rounded-xl border bg-card p-4 text-sm text-text-secondary" role="status" data-testid="thinking">
            Retrieving evidence and drafting a grounded answer…
          </div>
        </>
      )}
      {failed && !pending && (
        <>
          <div className="self-end rounded-xl bg-primary-soft px-4 py-2.5 text-sm">{failed.content}</div>
          <div className="rounded-xl border border-destructive/40 bg-card p-4 text-sm" role="alert" data-testid="send-failed">
            <p className="font-medium">The answer could not be produced.</p>
            <p className="text-text-secondary">{failed.message} Your question was not saved; retry to send it again.</p>
            <Button size="sm" variant="outline" className="mt-2" onClick={failed.retry} data-testid="send-retry">
              Retry
            </Button>
          </div>
        </>
      )}
      <div ref={end} />
    </div>
  );
}
