"use client";
import { useEffect, useState } from "react";
import { api } from "@/lib/apiClient";
import type { Readiness } from "@/lib/types";

export function useHealth(intervalMs = 30000) {
  const [health, setHealth] = useState<Readiness | null>(null);
  const [error, setError] = useState<string | null>(null);
  useEffect(() => {
    let alive = true;
    const tick = () =>
      api
        .ready()
        .then((h) => alive && (setHealth(h), setError(null)))
        .catch((e) => alive && setError(String(e?.message ?? e)));
    tick();
    const id = setInterval(tick, intervalMs);
    return () => {
      alive = false;
      clearInterval(id);
    };
  }, [intervalMs]);
  return { health, error };
}
