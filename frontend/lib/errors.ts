export class ApiError extends Error {
  status: number;
  detail: unknown;
  requestId: string | null;

  constructor(status: number, detail: unknown, requestId: string | null) {
    super(typeof detail === "string" ? detail : JSON.stringify(detail));
    this.status = status;
    this.detail = detail;
    this.requestId = requestId;
  }

  get userMessage(): string {
    if (this.status === 401) return "Your session has expired. Sign in again.";
    if (this.status === 403) return typeof this.detail === "string" ? this.detail : "Your role does not allow this action.";
    if (this.status === 404) return "Not found, or you do not have access to it.";
    if (this.status === 409) return typeof this.detail === "string" ? this.detail : "This action conflicts with the current state.";
    if (this.status === 429) return "Too many requests; wait a moment and retry.";
    if (this.status >= 500) return `Server error${this.requestId ? ` (request ${this.requestId})` : ""}.`;
    return typeof this.detail === "string" ? this.detail : this.message;
  }
}

export function errorMessage(e: unknown): string {
  if (e instanceof ApiError) return e.userMessage;
  if (e instanceof TypeError) return "The API is unreachable.";
  return e instanceof Error ? e.message : "Unexpected error.";
}

/** Sign-up/sign-in specific: a 401 here means "wrong credentials", not "your session expired" —
 * there was no session yet. Every other status still reads through the shared mapping. */
export function authErrorMessage(e: unknown): string {
  if (e instanceof ApiError && e.status === 401) {
    return typeof e.detail === "string" ? e.detail : "Invalid email or password.";
  }
  return errorMessage(e);
}
