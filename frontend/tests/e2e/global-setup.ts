import { execFileSync } from "node:child_process";
import path from "node:path";

// Seeds the two engineers and an admin the flows use (idempotent: "already exists" is fine).
export default function globalSetup() {
  const root = path.resolve(__dirname, "../..");
  const users = [
    ["e2e.alice@example.test", "Alice E2E", "engineer", "e2e-team"],
    ["e2e.bob@example.test", "Bob E2E", "engineer", ""],
    ["e2e.admin@example.test", "Admin E2E", "knowledge_admin", ""],
  ];
  for (const [email, name, role, ws] of users) {
    const args = ["run", "safety-assistant", "users", "add", "--email", email, "--name", name, "--role", role];
    if (ws) args.push("--workspace", ws);
    try {
      execFileSync("uv", args, { cwd: root, stdio: "pipe" });
    } catch (e) {
      const msg = String((e as { stderr?: Buffer }).stderr ?? e);
      if (!/already exists/i.test(msg)) throw e;
    }
  }
}
