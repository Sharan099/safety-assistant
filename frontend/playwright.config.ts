import { defineConfig } from "@playwright/test";

// Runs against a live API (DEV_LOGIN_ENABLED=true) and a running ingestion worker
// (`uv run safety-assistant worker`) with the Next dev server. See README "End-to-end tests".
export default defineConfig({
  testDir: "./tests/e2e",
  timeout: 180_000,
  expect: { timeout: 60_000 },
  workers: 1,
  globalSetup: "./tests/e2e/global-setup.ts",
  use: { baseURL: "http://localhost:3010", trace: "retain-on-failure", viewport: { width: 1440, height: 900 } },
  webServer: [
    { command: "npm run dev", url: "http://localhost:3010", reuseExistingServer: true, timeout: 120_000 },
  ],
});
