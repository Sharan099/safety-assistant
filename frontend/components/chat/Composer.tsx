"use client";
import { Paperclip, SendHorizontal } from "lucide-react";
import Link from "next/link";
import { useRef, useState } from "react";

import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Textarea } from "@/components/ui/textarea";

export function Composer({
  onSend,
  busy,
  scopeSummary,
  autoFocus,
}: {
  onSend: (content: string, asOf: string | null) => void;
  busy: boolean;
  scopeSummary: string;
  autoFocus?: boolean;
}) {
  const [text, setText] = useState("");
  const [asOf, setAsOf] = useState("");
  const area = useRef<HTMLTextAreaElement>(null);

  function submit() {
    const content = text.trim();
    if (!content || busy) return;
    onSend(content, asOf || null);
    setText("");
    area.current?.focus();
  }

  return (
    <form
      className="border-t bg-card px-4 py-3"
      onSubmit={(e) => {
        e.preventDefault();
        submit();
      }}
      aria-label="Ask a question"
    >
      <div className="mx-auto max-w-3xl">
        <Textarea
          ref={area}
          value={text}
          onChange={(e) => setText(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter" && !e.shiftKey) {
              e.preventDefault();
              submit();
            }
          }}
          rows={3}
          autoFocus={autoFocus}
          placeholder="Ask about a requirement, limit, test procedure or definition…"
          aria-label="Question"
          data-testid="query"
          disabled={busy}
          className="resize-y"
        />
        <div className="mt-2 flex flex-wrap items-center gap-3">
          <span className="text-xs text-text-secondary" data-testid="composer-scope">
            Searching: {scopeSummary}
          </span>
          <div className="flex items-center gap-1.5">
            <Label htmlFor="as-of" className="text-xs text-text-secondary">
              As of
            </Label>
            <Input id="as-of" type="date" value={asOf} onChange={(e) => setAsOf(e.target.value)} className="h-7 w-36 text-xs" data-testid="as-of" />
          </div>
          <div className="ml-auto flex items-center gap-2">
            <Button nativeButton={false} size="sm" variant="ghost" render={<Link href="/app/documents/upload" />} aria-label="Upload a document">
              <Paperclip className="size-4" aria-hidden /> Upload
            </Button>
            <span className="hidden text-xs text-muted-foreground sm:inline">Enter to send · Shift+Enter for newline</span>
            <Button type="submit" size="sm" disabled={busy || !text.trim()} data-testid="ask">
              <SendHorizontal className="size-4" aria-hidden /> {busy ? "Working…" : "Ask"}
            </Button>
          </div>
        </div>
      </div>
    </form>
  );
}
