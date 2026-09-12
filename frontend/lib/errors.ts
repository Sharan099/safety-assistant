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
    if (this.status === 401) return "Sign in required: add an API token.";
    if (this.status === 403) return "Your role does not allow this action.";
    if (this.status === 429) return "Too many requests; wait a moment and retry.";
    if (this.status >= 500) return `Server error${this.requestId ? ` (request ${this.requestId})` : ""}.`;
    return this.message;
  }
}
