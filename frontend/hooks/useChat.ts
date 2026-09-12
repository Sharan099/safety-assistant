"use client";
import { useCallback, useState } from "react";
import { api } from "@/lib/apiClient";
import { ApiError } from "@/lib/errors";
import type { AnswerResponse, AskRequest } from "@/lib/types";

export interface Turn {
  request: AskRequest;
  response?: AnswerResponse;
  error?: string;
  pending: boolean;
}

export function useChat() {
  const [turns, setTurns] = useState<Turn[]>([]);
  const ask = useCallback(async (request: AskRequest) => {
    const index = turns.length;
    setTurns((t) => [...t, { request, pending: true }]);
    try {
      const response = await api.ask(request);
      setTurns((t) => t.map((x, i) => (i === index ? { request, response, pending: false } : x)));
    } catch (e) {
      const message = e instanceof ApiError ? e.userMessage : String(e);
      setTurns((t) => t.map((x, i) => (i === index ? { request, error: message, pending: false } : x)));
    }
  }, [turns.length]);
  return { turns, ask, clear: () => setTurns([]) };
}
