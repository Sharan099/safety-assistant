"use client";

import { useEffect, useRef, useState } from "react";
import type { Citation } from "@/lib/types";

type Props = {
  citation: Citation | null;
  pdfUrl: string | null;
  onClose: () => void;
};

/**
 * Docling PDF provenance bboxes are typically BOTTOMLEFT (l, t, r, b).
 * PDF.js canvas uses TOPLEFT. We convert when drawing the highlight.
 */
function bboxToCanvasRect(
  bbox: number[],
  pageWidth: number,
  pageHeight: number,
  viewportWidth: number,
  viewportHeight: number,
  origin: string
) {
  const [l, t, r, b] = bbox;
  const sx = viewportWidth / pageWidth;
  const sy = viewportHeight / pageHeight;
  if ((origin || "BOTTOMLEFT").toUpperCase() === "TOPLEFT") {
    return {
      left: l * sx,
      top: t * sy,
      width: (r - l) * sx,
      height: (b - t) * sy,
    };
  }
  // BOTTOMLEFT: y up — flip top edge into canvas space
  return {
    left: l * sx,
    top: (pageHeight - t) * sy,
    width: (r - l) * sx,
    height: (t - b) * sy,
  };
}

export function PdfPane({ citation, pdfUrl, onClose }: Props) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const wrapRef = useRef<HTMLDivElement>(null);
  const [status, setStatus] = useState("Idle");
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!citation || !pdfUrl) return;
    let cancelled = false;

    async function render() {
      setError(null);
      setStatus("Loading PDF…");
      try {
        const pdfjs = await import("pdfjs-dist/legacy/build/pdf.mjs");
        pdfjs.GlobalWorkerOptions.workerSrc =
          `https://cdnjs.cloudflare.com/ajax/libs/pdf.js/${pdfjs.version}/pdf.worker.min.mjs`;

        const doc = await pdfjs.getDocument({ url: pdfUrl!, withCredentials: false }).promise;
        if (cancelled) return;
        const pageNum = Math.max(1, Math.min(citation!.page_number || 1, doc.numPages));
        setStatus(`Rendering p.${pageNum}…`);
        const page = await doc.getPage(pageNum);
        const base = page.getViewport({ scale: 1 });
        const wrap = wrapRef.current;
        const targetWidth = Math.min(wrap?.clientWidth || 520, 720);
        const scale = targetWidth / base.width;
        const viewport = page.getViewport({ scale });

        const canvas = canvasRef.current;
        if (!canvas) return;
        const ctx = canvas.getContext("2d");
        if (!ctx) return;
        canvas.width = viewport.width;
        canvas.height = viewport.height;
        await page.render({ canvasContext: ctx, viewport }).promise;
        if (cancelled) return;

        const bbox = citation!.bounding_box || [];
        if (bbox.length >= 4) {
          const rect = bboxToCanvasRect(
            bbox,
            base.width,
            base.height,
            viewport.width,
            viewport.height,
            citation!.coord_origin || "BOTTOMLEFT"
          );
          ctx.save();
          ctx.fillStyle = "rgba(196, 92, 38, 0.28)";
          ctx.strokeStyle = "rgba(196, 92, 38, 0.95)";
          ctx.lineWidth = 2;
          ctx.fillRect(rect.left, rect.top, Math.max(rect.width, 2), Math.max(rect.height, 2));
          ctx.strokeRect(rect.left, rect.top, Math.max(rect.width, 2), Math.max(rect.height, 2));
          ctx.restore();
          setStatus(`Highlighted §${citation!.section_number || "?"} on p.${pageNum}`);
        } else {
          setStatus(`Opened p.${pageNum} (no bbox on chunk)`);
        }
      } catch (err) {
        console.error(err);
        setError(err instanceof Error ? err.message : "PDF render failed");
        setStatus("Error");
      }
    }

    render();
    return () => {
      cancelled = true;
    };
  }, [citation, pdfUrl]);

  if (!citation || !pdfUrl) {
    return (
      <aside className="pdf-pane pdf-pane--empty">
        <div className="pdf-empty">
          <h3>Source viewer</h3>
          <p>Click a citation chip in the answer to open the regulation PDF and highlight the grounded span.</p>
        </div>
      </aside>
    );
  }

  const title =
    (citation.regulation_id || "").replace("UN-ECE-", "") +
    (citation.section_number ? ` §${citation.section_number}` : "");

  return (
    <aside className="pdf-pane">
      <header className="pdf-pane__header">
        <div>
          <div className="pdf-pane__eyebrow">Source</div>
          <h3>{title}</h3>
          <p className="pdf-pane__meta">
            {citation.section_title || "—"} · p.{citation.page_number ?? "?"}
          </p>
        </div>
        <button type="button" className="icon-btn" onClick={onClose} aria-label="Close PDF">
          ✕
        </button>
      </header>
      <div className="pdf-pane__status">{error || status}</div>
      <div className="pdf-pane__scroll" ref={wrapRef}>
        <canvas ref={canvasRef} className="pdf-canvas" />
      </div>
      {citation.text ? (
        <footer className="pdf-pane__excerpt">
          <strong>Chunk text</strong>
          <p>{citation.text.slice(0, 420)}{citation.text.length > 420 ? "…" : ""}</p>
        </footer>
      ) : null}
    </aside>
  );
}
