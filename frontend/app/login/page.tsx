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
import { authErrorMessage } from "@/lib/errors";

// The API says which methods exist: organization sign-in (OIDC authorization code + PKCE, handled
// entirely by the API), email/password sign-in and sign-up for a passive-safety engineer, and,
// outside production, a password-less dev login for seeded users.
export default function LoginPage() {
  const router = useRouter();
  const [mode, setMode] = useState<"signin" | "signup">("signin");
  const [email, setEmail] = useState("");
  const [displayName, setDisplayName] = useState("");
  const [password, setPassword] = useState("");
  const [devEmail, setDevEmail] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const methods = useQuery({
    queryKey: ["auth-methods"],
    queryFn: api.authMethods,
    staleTime: Infinity,
    retry: 1,
  });

  async function submitPassword(e: React.FormEvent) {
    e.preventDefault();
    setBusy(true);
    setError(null);
    try {
      if (mode === "signup") {
        await api.signup({ email: email.trim(), display_name: displayName.trim(), password });
      } else {
        await api.login({ email: email.trim(), password });
      }
      router.replace("/app/home");
    } catch (err) {
      setError(authErrorMessage(err));
    } finally {
      setBusy(false);
    }
  }

  async function submitDev(e: React.FormEvent) {
    e.preventDefault();
    setBusy(true);
    setError(null);
    try {
      await api.devLogin(devEmail.trim());
      router.replace("/app/home");
    } catch (err) {
      setError(authErrorMessage(err));
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
        {methods.data && !methods.data.oidc && !methods.data.password && !methods.data.dev_login && (
          <p className="mt-6 text-sm text-text-secondary" role="status">
            No sign-in method is configured. Ask an administrator to configure
            organization sign-in.
          </p>
        )}
        {methods.data?.password && (
          <>
            {methods.data.oidc && (
              <div className="my-4 flex items-center gap-2 text-xs text-text-secondary">
                <div className="h-px flex-1 bg-border" />
                or
                <div className="h-px flex-1 bg-border" />
              </div>
            )}
            <div className="mb-4 flex gap-1 rounded-lg bg-muted p-1 text-sm" role="tablist">
              <button
                type="button"
                role="tab"
                aria-selected={mode === "signin"}
                onClick={() => { setMode("signin"); setError(null); }}
                className={`flex-1 rounded-md py-1.5 font-medium transition-colors ${mode === "signin" ? "bg-card shadow-none" : "text-text-secondary"}`}
                data-testid="mode-signin"
              >
                Sign in
              </button>
              <button
                type="button"
                role="tab"
                aria-selected={mode === "signup"}
                onClick={() => { setMode("signup"); setError(null); }}
                className={`flex-1 rounded-md py-1.5 font-medium transition-colors ${mode === "signup" ? "bg-card shadow-none" : "text-text-secondary"}`}
                data-testid="mode-signup"
              >
                Create account
              </button>
            </div>
            <form onSubmit={submitPassword} className="space-y-4" aria-label={mode === "signup" ? "Create account" : "Sign in"}>
              {mode === "signup" && (
                <div className="space-y-1.5">
                  <Label htmlFor="display-name">Full name</Label>
                  <Input
                    id="display-name"
                    autoComplete="name"
                    required
                    value={displayName}
                    onChange={(e) => setDisplayName(e.target.value)}
                    data-testid="signup-name"
                  />
                </div>
              )}
              <div className="space-y-1.5">
                <Label htmlFor="email">Work email</Label>
                <Input
                  id="email"
                  type="email"
                  autoComplete="username"
                  required
                  value={email}
                  onChange={(e) => setEmail(e.target.value)}
                  data-testid="password-email"
                />
              </div>
              <div className="space-y-1.5">
                <Label htmlFor="password">Password</Label>
                <Input
                  id="password"
                  type="password"
                  autoComplete={mode === "signup" ? "new-password" : "current-password"}
                  required
                  minLength={mode === "signup" ? 10 : undefined}
                  value={password}
                  onChange={(e) => setPassword(e.target.value)}
                  data-testid="password-input"
                />
                {mode === "signup" && (
                  <p className="text-xs text-muted-foreground">At least 10 characters. Not your email address.</p>
                )}
              </div>
              {error && (
                <Alert variant="destructive" role="alert" data-testid="password-error">
                  <AlertDescription>{error}</AlertDescription>
                </Alert>
              )}
              <Button type="submit" className="w-full" disabled={busy} data-testid="password-submit">
                {busy ? (mode === "signup" ? "Creating account…" : "Signing in…") : mode === "signup" ? "Create account" : "Sign in"}
              </Button>
            </form>
          </>
        )}
        {methods.data?.dev_login && (
          <form onSubmit={submitDev} className="mt-6 space-y-3 border-t pt-4" aria-label="Developer sign-in">
            <p className="text-xs font-medium text-muted-foreground">Developer sign-in (no password)</p>
            <Input
              type="email"
              autoComplete="off"
              placeholder="seeded-user@example.test"
              value={devEmail}
              onChange={(e) => setDevEmail(e.target.value)}
              data-testid="login-email"
            />
            <Button type="submit" variant="outline" className="w-full" disabled={busy} data-testid="login-submit">
              Sign in as seeded user
            </Button>
          </form>
        )}
      </div>
    </main>
  );
}
