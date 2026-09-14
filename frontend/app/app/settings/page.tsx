"use client";
import { toast } from "sonner";

import { ErrorState, LoadingState } from "@/components/common/States";
import { Label } from "@/components/ui/label";
import { Textarea } from "@/components/ui/textarea";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { useMe, usePatchPreferences } from "@/features/queries";
import { errorMessage } from "@/lib/errors";
import type { Preferences } from "@/lib/types";

const NONE = "__none";

// Explicit preferences only (06_SECURITY "User preferences"): never regulatory memory.
export default function SettingsPage() {
  const me = useMe();
  const patch = usePatchPreferences();
  if (me.isPending) return <div className="p-6"><LoadingState rows={3} /></div>;
  if (me.isError) return <div className="p-6"><ErrorState message={errorMessage(me.error)} /></div>;
  const p = me.data.preferences;
  const save = (next: Partial<Preferences>) =>
    patch.mutate(next, { onSuccess: () => toast.success("Saved"), onError: (e) => toast.error(errorMessage(e)) });

  return (
    <div className="mx-auto max-w-2xl space-y-6 px-4 py-6">
      <h1 className="text-xl font-semibold">Settings</h1>
      <section className="rounded-xl border bg-card p-4">
        <h2 className="text-sm font-semibold">Profile</h2>
        <dl className="mt-2 space-y-1 text-sm">
          <div className="flex gap-2"><dt className="w-32 text-text-secondary">Name</dt><dd>{me.data.user.display_name}</dd></div>
          <div className="flex gap-2"><dt className="w-32 text-text-secondary">Email</dt><dd>{me.data.user.email}</dd></div>
          <div className="flex gap-2"><dt className="w-32 text-text-secondary">Roles</dt><dd>{me.data.user.roles.join(", ")}</dd></div>
          <div className="flex gap-2"><dt className="w-32 text-text-secondary">Workspaces</dt><dd>{me.data.workspaces.map((w) => w.name).join(", ") || "none"}</dd></div>
        </dl>
      </section>
      <section className="grid gap-4 rounded-xl border bg-card p-4 sm:grid-cols-2">
        <h2 className="text-sm font-semibold sm:col-span-2">Preferences</h2>
        <div className="space-y-1.5">
          <Label htmlFor="ws">Default workspace</Label>
          <Select value={p.default_workspace_id ?? NONE} onValueChange={(v) => save({ default_workspace_id: v === NONE || v == null ? null : v })}>
            <SelectTrigger id="ws" className="w-full"><SelectValue /></SelectTrigger>
            <SelectContent>
              <SelectItem value={NONE}>None</SelectItem>
              {me.data.workspaces.map((w) => (
                <SelectItem key={w.id} value={w.id}>{w.name}</SelectItem>
              ))}
            </SelectContent>
          </Select>
        </div>
        <div className="space-y-1.5">
          <Label htmlFor="density">Answer density</Label>
          <Select value={p.answer_density} onValueChange={(v) => v && save({ answer_density: v as Preferences["answer_density"] })}>
            <SelectTrigger id="density" className="w-full" data-testid="pref-density"><SelectValue /></SelectTrigger>
            <SelectContent>
              <SelectItem value="concise">Concise</SelectItem>
              <SelectItem value="standard">Standard</SelectItem>
              <SelectItem value="detailed">Detailed</SelectItem>
            </SelectContent>
          </Select>
        </div>
        <div className="space-y-1.5">
          <Label htmlFor="lang">Preferred language</Label>
          <Select value={p.preferred_language} onValueChange={(v) => v && save({ preferred_language: v })}>
            <SelectTrigger id="lang" className="w-full"><SelectValue /></SelectTrigger>
            <SelectContent>
              <SelectItem value="en">English</SelectItem>
              <SelectItem value="de">Deutsch</SelectItem>
              <SelectItem value="fr">Français</SelectItem>
            </SelectContent>
          </Select>
        </div>
        <div className="space-y-1.5">
          <Label htmlFor="theme">Theme</Label>
          <Select value={p.ui_theme} onValueChange={(v) => v && save({ ui_theme: v as Preferences["ui_theme"] })}>
            <SelectTrigger id="theme" className="w-full"><SelectValue /></SelectTrigger>
            <SelectContent>
              <SelectItem value="light">Light</SelectItem>
              <SelectItem value="system">System</SelectItem>
              <SelectItem value="dark">Dark (coming later)</SelectItem>
            </SelectContent>
          </Select>
        </div>
        <div className="space-y-1.5 sm:col-span-2">
          <Label htmlFor="project">Current project</Label>
          <Textarea
            id="project"
            data-testid="pref-project"
            defaultValue={p.project_context ?? ""}
            maxLength={800}
            rows={3}
            placeholder="e.g. M1 passenger car, 1,850 kg, EU + UK markets, SOP 2027, Euro NCAP 5-star target"
            onBlur={(e) => {
              const v = e.target.value.trim();
              if (v !== (p.project_context ?? "")) save({ project_context: v || null });
            }}
          />
          <p className="text-xs text-text-secondary">
            Lets you ask “does my vehicle need …” in short form. Shown to the model as context, never as evidence.
          </p>
        </div>
        <p className="text-xs text-text-secondary sm:col-span-2">
          Preferences shape wording and defaults only. Regulatory facts always come from the current authorized corpus.
        </p>
      </section>
    </div>
  );
}
