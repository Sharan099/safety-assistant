import { AuthPanel } from "@/components/AuthPanel";
import { ChatPanel } from "@/components/ChatPanel";
import { SystemStatus } from "@/components/SystemStatus";

export default function Home() {
  return (
    <main className="mx-auto max-w-4xl space-y-6 px-4 py-8">
      <header className="space-y-2">
        <div className="flex flex-wrap items-center justify-between gap-3">
          <h1 className="text-xl font-semibold">Safety Assistant</h1>
          <AuthPanel />
        </div>
        <p className="text-sm text-zinc-400">
          Answers come only from ingested, versioned regulation text. Every claim cites an evidence id; every citation opens
          the exact clause, version, page and source hash. When the corpus cannot support an answer, the system says so.
        </p>
        <SystemStatus />
      </header>
      <ChatPanel />
    </main>
  );
}
