"use client";
// Server-state hooks (TanStack Query). Keys are namespaced so mutations can invalidate precisely.
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";

import { api } from "@/lib/api";
import { ApiError } from "@/lib/errors";
import type { DisplayStatus, Preferences, SourceScope } from "@/lib/types";

export const keys = {
  me: ["me"] as const,
  ready: ["ready"] as const,
  regulations: ["regulations"] as const,
  conversations: (q?: string, archived?: boolean) => ["conversations", q ?? "", archived ?? false] as const,
  conversation: (id: string) => ["conversation", id] as const,
  documents: (params: Record<string, string | undefined>) => ["documents", params] as const,
  document: (id: string) => ["document", id] as const,
  job: (id: string) => ["job", id] as const,
  runs: ["admin", "runs"] as const,
};

const TERMINAL: DisplayStatus[] = ["READY", "FAILED", "QUARANTINED", "ARCHIVED"];
export const isTerminal = (s: DisplayStatus | undefined) => !!s && TERMINAL.includes(s);

export function useMe() {
  return useQuery({
    queryKey: keys.me,
    queryFn: api.me,
    retry: (count, err) => !(err instanceof ApiError && (err.status === 401 || err.status === 403)) && count < 2,
    staleTime: 60_000,
  });
}

export function useReady() {
  return useQuery({ queryKey: keys.ready, queryFn: api.ready, refetchInterval: 30_000, retry: 1 });
}

export function useRegulations() {
  return useQuery({ queryKey: keys.regulations, queryFn: api.regulations, staleTime: 5 * 60_000 });
}

export function useConversations(q?: string, archived = false) {
  return useQuery({ queryKey: keys.conversations(q, archived), queryFn: () => api.conversations(q, archived) });
}

export function useConversation(id: string | null) {
  return useQuery({ queryKey: keys.conversation(id ?? ""), queryFn: () => api.conversation(id as string), enabled: !!id });
}

export function useCreateConversation() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: api.createConversation,
    onSuccess: () => qc.invalidateQueries({ queryKey: ["conversations"] }),
  });
}

export function usePatchConversation(id: string) {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (body: { title?: string; archived?: boolean; source_scope?: SourceScope }) => api.patchConversation(id, body),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["conversations"] });
      qc.invalidateQueries({ queryKey: keys.conversation(id) });
    },
  });
}

export function useSendMessage(id: string) {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (body: { content: string; as_of?: string | null; k?: number }) => api.sendMessage(id, body),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: keys.conversation(id) });
      qc.invalidateQueries({ queryKey: ["conversations"] });
    },
  });
}

export function useDocuments(params: Record<string, string | undefined> = {}) {
  return useQuery({
    queryKey: keys.documents(params),
    queryFn: () => api.documents(params),
    // keep the library live while anything is processing
    refetchInterval: (q) => (q.state.data?.items.some((d) => !isTerminal(d.status)) ? 4_000 : false),
  });
}

export function useDocument(id: string | null) {
  return useQuery({
    queryKey: keys.document(id ?? ""),
    queryFn: () => api.document(id as string),
    enabled: !!id,
    refetchInterval: (q) => (q.state.data && !isTerminal(q.state.data.status) ? 3_000 : false),
  });
}

export function useJob(id: string | null) {
  return useQuery({
    queryKey: keys.job(id ?? ""),
    queryFn: () => api.job(id as string),
    enabled: !!id,
    refetchInterval: (q) => {
      const s = q.state.data?.status;
      return s && (s === "QUEUED" || s === "RUNNING") ? 2_500 : false;
    },
  });
}

export function useUpload() {
  const qc = useQueryClient();
  return useMutation({ mutationFn: api.upload, onSuccess: () => qc.invalidateQueries({ queryKey: ["documents"] }) });
}

export function useDocumentAction(id: string) {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (action: "archive" | "promote") => (action === "archive" ? api.archiveDocument(id) : api.promoteDocument(id)),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: keys.document(id) });
      qc.invalidateQueries({ queryKey: ["documents"] });
    },
  });
}

export function useRetryJob() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: api.retryJob,
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["documents"] });
      qc.invalidateQueries({ queryKey: ["document"] });
      qc.invalidateQueries({ queryKey: ["job"] });
    },
  });
}

export function usePatchPreferences() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (p: Partial<Preferences>) => api.patchPreferences(p),
    onSuccess: (me) => qc.setQueryData(keys.me, me),
  });
}

export function useLogout() {
  const qc = useQueryClient();
  return useMutation({ mutationFn: api.logout, onSuccess: () => qc.clear() });
}

export function useIngestionRuns(enabled: boolean) {
  return useQuery({ queryKey: keys.runs, queryFn: api.ingestionRuns, enabled, refetchInterval: 10_000 });
}
