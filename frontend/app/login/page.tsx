"use client";
import { useRouter } from "next/navigation";
import { useState } from "react";

import { ProductMark } from "@/components/common/ProductMark";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { api } from "@/lib/api";
import { errorMessage } from "@/lib/errors";

// Minimal login (03_UI_UX "Login"). Dev login only exists when the API enables it; in production the
// API returns 404 for it and only the organization sign-in (OIDC) remains.
export default function LoginPage() {
  const router = useRouter();
  const [email, setEmail] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  async function submit(e: React.FormEvent) {
    e.preventDefault();
    setBusy(true);
    setError(null);
    try {
      await api.devLogin(email.trim());
      router.replace("/app/home");
    } catch (err) {
      setError(errorMessage(err));
    } finally {
      setBusy(false);
    }
  }

  return (
    <main className="flex min-h-screen items-center justify-center px-4 py-8">
      <div className="w-full max-w-sm rounded-xl border bg-card p-6 shadow-none">
        <ProductMark />
        <p className="mt-2 text-sm text-text-secondary">
          Cited, versioned answers across approved passive-safety regulations and your authorized documents.
        </p>
        <form onSubmit={submit} className="mt-6 space-y-4" aria-label="Sign in">
          <div className="space-y-1.5">
            <Label htmlFor="email">Work email</Label>
            <Input
              id="email"
              type="email"
              autoComplete="username"
              required
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              data-testid="login-email"
            />
          </div>
          {error && (
            <Alert variant="destructive" role="alert">
              <AlertDescription>{error}</AlertDescription>
            </Alert>
          )}
          <Button type="submit" className="w-full" disabled={busy} data-testid="login-submit">
            {busy ? "Signing in…" : "Sign in"}
          </Button>
          <p className="text-xs text-muted-foreground">
            Organization sign-in (OIDC) is configured by your administrator. No membership? Contact an admin.
          </p>
        </form>
      </div>
    </main>
  );
}
