"use client";
import { useState } from "react";
import { useAuth } from "@/hooks/useAuth";

export function AuthPanel() {
  const { token, setToken, signedIn } = useAuth();
  const [draft, setDraft] = useState("");
  const [open, setOpen] = useState(false);
  return (
    <div className="text-xs text-zinc-400">
      <button onClick={() => setOpen((o) => !o)} className="underline" data-testid="auth-toggle">
        {signedIn ? `token set (${token.slice(0, 4)}…)` : "no token (anonymous viewer)"}
      </button>
      {open && (
        <form
          className="mt-2 flex gap-2"
          onSubmit={(e) => {
            e.preventDefault();
            setToken(draft);
            setDraft("");
            setOpen(false);
          }}
        >
          <input
            type="password"
            value={draft}
            onChange={(e) => setDraft(e.target.value)}
            placeholder="API token or OIDC access token"
            className="rounded border border-zinc-700 bg-zinc-950 p-1"
            data-testid="token-input"
          />
          <button type="submit" className="rounded border border-zinc-600 px-2">save</button>
          <button type="button" className="rounded border border-zinc-600 px-2" onClick={() => setToken("")}>clear</button>
        </form>
      )}
    </div>
  );
}
