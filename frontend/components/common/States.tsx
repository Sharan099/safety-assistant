import { AlertTriangle, Inbox, Lock, RefreshCw } from "lucide-react";

import { Button } from "@/components/ui/button";
import { Skeleton } from "@/components/ui/skeleton";

// The four non-success states every screen must define (03_UI_UX "State design").

export function LoadingState({ rows = 3, label = "Loading" }: { rows?: number; label?: string }) {
  return (
    <div className="space-y-3" role="status" aria-live="polite" aria-label={label}>
      {Array.from({ length: rows }).map((_, i) => (
        <Skeleton key={i} className="h-10 w-full" />
      ))}
    </div>
  );
}

export function EmptyState({ title, hint, action }: { title: string; hint?: string; action?: React.ReactNode }) {
  return (
    <div className="flex flex-col items-center gap-2 rounded-xl border border-dashed px-6 py-10 text-center">
      <Inbox className="size-6 text-muted-foreground" aria-hidden />
      <p className="text-sm font-medium">{title}</p>
      {hint && <p className="max-w-md text-sm text-text-secondary">{hint}</p>}
      {action}
    </div>
  );
}

export function ErrorState({ message, onRetry }: { message: string; onRetry?: () => void }) {
  return (
    <div role="alert" className="flex flex-col items-start gap-3 rounded-xl border border-destructive/40 bg-destructive-soft p-4">
      <div className="flex items-center gap-2 text-sm font-medium text-destructive">
        <AlertTriangle className="size-4" aria-hidden />
        Something went wrong
      </div>
      <p className="text-sm text-foreground">{message}</p>
      {onRetry && (
        <Button size="sm" variant="outline" onClick={onRetry}>
          <RefreshCw className="size-4" aria-hidden /> Retry
        </Button>
      )}
    </div>
  );
}

export function PermissionDenied({ what = "this page" }: { what?: string }) {
  return (
    <div role="alert" className="flex flex-col items-start gap-2 rounded-xl border bg-card p-4">
      <div className="flex items-center gap-2 text-sm font-medium">
        <Lock className="size-4" aria-hidden /> Permission required
      </div>
      <p className="text-sm text-text-secondary">Your role does not allow access to {what}. Ask an administrator.</p>
    </div>
  );
}
