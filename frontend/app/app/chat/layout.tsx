"use client";
import { useParams } from "next/navigation";

import { ConversationList } from "@/components/chat/ConversationList";

export default function ChatLayout({ children }: { children: React.ReactNode }) {
  const params = useParams<{ id?: string }>();
  return (
    <div className="flex h-full">
      <aside className="hidden w-64 shrink-0 border-r bg-card md:block" aria-label="Investigations">
        <ConversationList activeId={params?.id ?? null} />
      </aside>
      <div className="flex min-w-0 flex-1 flex-col">{children}</div>
    </div>
  );
}
