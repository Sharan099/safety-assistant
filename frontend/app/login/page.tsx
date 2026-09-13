"use client";
import { useQuery } from "@tanstack/react-query";
import { useRouter } from "next/navigation";
import { useState } from "react";

import { ProductMark } from "@/components/common/ProductMark";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { api } from "@/lib/api";
import { errorMessage } from "@/lib/errors";

// Minimal login. The API says which methods exist: organization sign-in (OIDC authorization code +
// PKCE, handled entirely by the API) and, outside production, a password-less dev login.
export default function LoginPage() {
  const router = useRouter();
  const [email, setEmail] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const methods = useQuery({
    queryKey: ["auth-methods"],
    queryFn: api.authMethods,
    staleTime: Infinity,
    retry: 1,
  });

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
          Cited, versioned answers across approved passive-safety regulations
          and your authorized documents.
        </p>
        {methods.data?.oidc && (
          <Button
            className="mt-6 w-full"
            render={<a href="/api/v1/auth/oidc/login" />}
            nativeButton={false}
            data-testid="oidc-login"
          >
            Sign in with your organization
          </Button>
        )}
        {methods.data && !methods.data.oidc && !methods.data.dev_login && (
          <p className="mt-6 text-sm text-text-secondary" role="status">
            No sign-in method is configured. Ask an administrator to configure
            organization sign-in.
          </p>
        )}
        {methods.data?.dev_login && (
          <form
            onSubmit={submit}
            className="mt-6 space-y-4"
            aria-label="Sign in"
          >
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
            <Button
              type="submit"
              className="w-full"
              disabled={busy}
              data-testid="login-submit"
            >
              {busy ? "Signing in…" : "Sign in"}
            </Button>
            <p className="text-xs text-muted-foreground">
              Development sign-in for seeded users. No membership? Contact an
              admin.
            </p>
          </form>
        )}
      </div>
    </main>
  );
}
