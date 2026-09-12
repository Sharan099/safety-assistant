"use client";
import { useRouter } from "next/navigation";
import { useEffect } from "react";

import { ErrorState, LoadingState } from "@/components/common/States";
import { EvidenceProvider } from "@/components/evidence/EvidenceContext";
import { AppShell } from "@/components/shell/AppShell";
import { useMe } from "@/features/queries";
import { ApiError, errorMessage } from "@/lib/errors";

// Client-side guard only for UX; every route is authorized by the backend (04_APP_FLOWS cross-user flow).
export default function AppLayout({ children }: { children: React.ReactNode }) {
  const router = useRouter();
  const me = useMe();
  const unauthenticated = me.error instanceof ApiError && (me.error.status === 401 || me.error.status === 403);
  useEffect(() => {
    if (unauthenticated) router.replace("/login");
  }, [unauthenticated, router]);

  if (me.isPending || unauthenticated) {
    return (
      <div className="p-6">
        <LoadingState rows={4} label="Signing in" />
      </div>
    );
  }
  if (me.isError) {
    return (
      <div className="p-6">
        <ErrorState message={errorMessage(me.error)} onRetry={() => me.refetch()} />
      </div>
    );
  }
  return (
    <EvidenceProvider>
      <AppShell me={me.data}>{children}</AppShell>
    </EvidenceProvider>
  );
}
