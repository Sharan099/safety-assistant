"use client";
import { useCallback, useState } from "react";
import { getToken, setToken } from "@/lib/apiClient";

export function useAuth() {
  const [token, setLocal] = useState(() => getToken());
  const save = useCallback((t: string) => {
    setToken(t);
    setLocal(t);
  }, []);
  return { token, setToken: save, signedIn: token.length > 0 };
}
