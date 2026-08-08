"use client";

import { useMemo, useState } from "react";
import { AddRegulationModal } from "@/components/AddRegulationModal";
import { ChatPane } from "@/components/ChatPane";
import { MetricsPanel } from "@/components/MetricsPanel";
import { PdfPane } from "@/components/PdfPane";
import { pdfUrl } from "@/lib/api";
import type { Citation, QueryMetrics } from "@/lib/types";

const TOPICS = [
  { id: "t1", title: "Frontal injury limits", subtitle: "R94 HIC / ThCC / TCFC" },
  { id: "t2", title: "Side impact", subtitle: "R95 TTI / VC" },
  { id: "t3", title: "Safety-belts", subtitle: "R16 ELR / SBR" },
  { id: "t4", title: "Child restraints", subtitle: "R129 i-Size" },
];

export default function HomePage() {
  const [active, setActive] = useState<Citation | null>(null);
  const [topicId, setTopicId] = useState("t1");
  const [showMetrics, setShowMetrics] = useState(false);
  const [lastTraceId, setLastTraceId] = useState<string | null>(null);
  const [uploadOpen, setUploadOpen] = useState(false);
  const [regsRefreshKey, setRegsRefreshKey] = useState(0);
  const [toast, setToast] = useState<string | null>(null);

  const activePdf = useMemo(() => {
    if (!active?.regulation_id) return null;
    return pdfUrl(active.regulation_id);
  }, [active]);

  return (
    <div
      className={`shell ${active ? "shell--split" : ""} ${showMetrics ? "shell--metrics" : ""}`}
    >
      <nav className="icon-rail" aria-label="Primary">
        <div className="rail-logo" title="Passive Safety RAG">
          PS
        </div>
        <button
          type="button"
          className={`rail-btn ${!showMetrics ? "is-active" : ""}`}
          title="Chat"
          onClick={() => setShowMetrics(false)}
        >
          ⌘
        </button>
        <button type="button" className="rail-btn" title="Sources" disabled>
          ▦
        </button>
        <button
          type="button"
          className={`rail-btn ${showMetrics ? "is-active" : ""}`}
          title="Metrics"
          onClick={() => setShowMetrics(true)}
        >
          ◇
        </button>
      </nav>

      <aside className="topics">
        <div className="topics__head">
          <h2>Conversations</h2>
          <span className="pill-mini">Local</span>
        </div>
        <button
          type="button"
          className="add-conversation"
          onClick={() => setUploadOpen(true)}
        >
          <span className="add-conversation__plus" aria-hidden>
            +
          </span>
          Add regulation
        </button>
        <ul>
          {TOPICS.map((t) => (
            <li key={t.id}>
              <button
                type="button"
                className={`topic-card ${topicId === t.id ? "is-active" : ""}`}
                onClick={() => setTopicId(t.id)}
              >
                <strong>{t.title}</strong>
                <span>{t.subtitle}</span>
              </button>
            </li>
          ))}
        </ul>
      </aside>

      <main className="main">
        {showMetrics ? (
          <MetricsPanel lastTraceId={lastTraceId} onClose={() => setShowMetrics(false)} />
        ) : (
          <ChatPane
            regsRefreshKey={regsRefreshKey}
            onOpenCitation={setActive}
            onTrace={(id: string, _m?: QueryMetrics) => {
              setLastTraceId(id);
            }}
          />
        )}
      </main>

      {active ? (
        <PdfPane
          citation={active}
          pdfUrl={activePdf}
          onClose={() => setActive(null)}
        />
      ) : null}

      <AddRegulationModal
        open={uploadOpen}
        onClose={() => setUploadOpen(false)}
        onComplete={(job) => {
          setRegsRefreshKey((k) => k + 1);
          setToast(
            `${job.regulation_id} (${job.revision}) indexed — ${job.chunk_count ?? "?"} chunks`
          );
          window.setTimeout(() => setToast(null), 4500);
          setUploadOpen(false);
        }}
      />

      {toast ? (
        <div className="toast" role="status">
          {toast}
        </div>
      ) : null}
    </div>
  );
}
