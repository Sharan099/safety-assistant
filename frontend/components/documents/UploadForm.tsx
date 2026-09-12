"use client";
import { FileUp } from "lucide-react";
import { useRef, useState } from "react";
import { z } from "zod";

import { ErrorState } from "@/components/common/States";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Textarea } from "@/components/ui/textarea";
import { useUpload } from "@/features/queries";
import { errorMessage } from "@/lib/errors";
import type { Me, UploadResult } from "@/lib/types";
import { cn } from "@/lib/utils";

const MAX_MB = 200;
const TYPES = ["PROJECT_DOCUMENT", "REGULATION", "STANDARD", "TECHNICAL_REPORT", "MANUAL"] as const;

const Meta = z.object({
  title: z.string().trim().min(1, "Title is required").max(300),
  document_type: z.enum(TYPES),
  scope: z.enum(["PRIVATE_USER", "WORKSPACE"]),
  workspace_id: z.string().optional(),
  version_label: z.string().trim().max(100).default("v1"),
  effective_from: z.string().optional(),
  notes: z.string().max(2000).optional(),
});

export function UploadForm({ me, onUploaded }: { me: Me; onUploaded: (r: UploadResult) => void }) {
  const upload = useUpload();
  const [file, setFile] = useState<File | null>(null);
  const [drag, setDrag] = useState(false);
  const [fieldError, setFieldError] = useState<string | null>(null);
  const [meta, setMeta] = useState({
    title: "",
    document_type: "PROJECT_DOCUMENT" as (typeof TYPES)[number],
    scope: "PRIVATE_USER" as "PRIVATE_USER" | "WORKSPACE",
    workspace_id: me.workspaces[0]?.id ?? "",
    version_label: "v1",
    effective_from: "",
    notes: "",
  });
  const input = useRef<HTMLInputElement>(null);

  function pick(f: File | null) {
    setFieldError(null);
    if (!f) return;
    if (!/\.pdf$/i.test(f.name) && f.type !== "application/pdf") return setFieldError("Only PDF files are accepted.");
    if (f.size > MAX_MB * 1024 * 1024) return setFieldError(`File exceeds ${MAX_MB} MB.`);
    setFile(f);
    if (!meta.title) setMeta((m) => ({ ...m, title: f.name.replace(/\.pdf$/i, "") }));
  }

  function submit(e: React.FormEvent) {
    e.preventDefault();
    setFieldError(null);
    if (!file) return setFieldError("Choose a PDF first.");
    const parsed = Meta.safeParse(meta);
    if (!parsed.success) return setFieldError(parsed.error.issues[0]?.message ?? "Check the form.");
    if (parsed.data.scope === "WORKSPACE" && !parsed.data.workspace_id) return setFieldError("Select a workspace.");
    const form = new FormData();
    form.append("file", file, file.name);
    form.append("title", parsed.data.title);
    form.append("document_type", parsed.data.document_type);
    form.append("scope", parsed.data.scope);
    if (parsed.data.scope === "WORKSPACE") form.append("workspace_id", parsed.data.workspace_id ?? "");
    form.append("version_label", parsed.data.version_label || "v1");
    if (parsed.data.effective_from) form.append("effective_from", parsed.data.effective_from);
    if (parsed.data.notes) form.append("notes", parsed.data.notes);
    upload.mutate(form, { onSuccess: onUploaded });
  }

  return (
    <form onSubmit={submit} className="space-y-5" aria-label="Upload a document">
      {/* Step 1 — file */}
      <section className="space-y-2">
        <h2 className="text-sm font-semibold">1. Choose a PDF</h2>
        <div
          role="button"
          tabIndex={0}
          aria-label="Drop a PDF here or press Enter to browse"
          data-testid="dropzone"
          onClick={() => input.current?.click()}
          onKeyDown={(e) => (e.key === "Enter" || e.key === " ") && input.current?.click()}
          onDragOver={(e) => { e.preventDefault(); setDrag(true); }}
          onDragLeave={() => setDrag(false)}
          onDrop={(e) => { e.preventDefault(); setDrag(false); pick(e.dataTransfer.files[0] ?? null); }}
          className={cn(
            "flex cursor-pointer flex-col items-center gap-2 rounded-xl border-2 border-dashed bg-card px-6 py-8 text-center text-sm focus-visible:outline-2 focus-visible:outline-ring",
            drag && "border-primary bg-primary-soft",
          )}
        >
          <FileUp className="size-6 text-muted-foreground" aria-hidden />
          {file ? (
            <span className="font-medium">{file.name} · {(file.size / 1024 / 1024).toFixed(1)} MB</span>
          ) : (
            <span>Drag a PDF here or click to browse</span>
          )}
          <span className="text-xs text-text-secondary">PDF only · max {MAX_MB} MB · processed asynchronously · private by default</span>
        </div>
        <input ref={input} type="file" accept="application/pdf,.pdf" className="sr-only" data-testid="file-input" onChange={(e) => pick(e.target.files?.[0] ?? null)} />
      </section>

      {/* Step 2 — metadata */}
      <section className="grid gap-4 sm:grid-cols-2">
        <h2 className="text-sm font-semibold sm:col-span-2">2. Describe it</h2>
        <div className="space-y-1.5 sm:col-span-2">
          <Label htmlFor="title">Display title</Label>
          <Input id="title" value={meta.title} onChange={(e) => setMeta({ ...meta, title: e.target.value })} required data-testid="upload-title" />
        </div>
        <div className="space-y-1.5">
          <Label htmlFor="doctype">Document type</Label>
          <Select value={meta.document_type} onValueChange={(v) => setMeta({ ...meta, document_type: v as (typeof TYPES)[number] })}>
            <SelectTrigger id="doctype" className="w-full"><SelectValue /></SelectTrigger>
            <SelectContent>
              {TYPES.map((t) => (
                <SelectItem key={t} value={t}>{t.toLowerCase().replace("_", " ")}</SelectItem>
              ))}
            </SelectContent>
          </Select>
        </div>
        <div className="space-y-1.5">
          <Label htmlFor="scope">Scope</Label>
          <Select value={meta.scope} onValueChange={(v) => setMeta({ ...meta, scope: v as "PRIVATE_USER" | "WORKSPACE" })}>
            <SelectTrigger id="scope" className="w-full" data-testid="upload-scope"><SelectValue /></SelectTrigger>
            <SelectContent>
              <SelectItem value="PRIVATE_USER">Private — only you</SelectItem>
              <SelectItem value="WORKSPACE" disabled={me.workspaces.length === 0}>Workspace — members of a workspace</SelectItem>
            </SelectContent>
          </Select>
        </div>
        {meta.scope === "WORKSPACE" && (
          <div className="space-y-1.5">
            <Label htmlFor="workspace">Workspace</Label>
            <Select value={meta.workspace_id} onValueChange={(v) => setMeta({ ...meta, workspace_id: v ?? "" })}>
              <SelectTrigger id="workspace" className="w-full"><SelectValue /></SelectTrigger>
              <SelectContent>
                {me.workspaces.map((w) => (
                  <SelectItem key={w.id} value={w.id}>{w.name}</SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>
        )}
        <div className="space-y-1.5">
          <Label htmlFor="version">Version label</Label>
          <Input id="version" value={meta.version_label} onChange={(e) => setMeta({ ...meta, version_label: e.target.value })} />
        </div>
        <div className="space-y-1.5">
          <Label htmlFor="effective">Effective from (optional)</Label>
          <Input id="effective" type="date" value={meta.effective_from} onChange={(e) => setMeta({ ...meta, effective_from: e.target.value })} />
        </div>
        <div className="space-y-1.5 sm:col-span-2">
          <Label htmlFor="notes">Notes (optional)</Label>
          <Textarea id="notes" rows={2} value={meta.notes} onChange={(e) => setMeta({ ...meta, notes: e.target.value })} />
        </div>
      </section>

      <Alert>
        <AlertDescription>
          Uploads are validated, parsed, chunked, embedded and verified before they become searchable. They never become
          organization-wide verified sources unless a knowledge admin promotes them.
        </AlertDescription>
      </Alert>
      {fieldError && <p className="text-sm text-destructive" role="alert" data-testid="upload-field-error">{fieldError}</p>}
      {upload.isError && <ErrorState message={errorMessage(upload.error)} />}
      <Button type="submit" disabled={upload.isPending} data-testid="upload-submit">
        {upload.isPending ? "Uploading…" : "Upload and process"}
      </Button>
    </form>
  );
}
