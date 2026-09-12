import { defineConfig } from "@playwright/test";

// Runs against a live API (default http://localhost:8010) with the Next dev server.
export default defineConfig({
  testDir: "./tests",
  timeout: 90_000,
  expect: { timeout: 30_000 },
  use: { baseURL: "http://localhost:3010", trace: "retain-on-failure" },
  webServer: {
    command: "npm run dev",
    url: "http://localhost:3010",
    reuseExistingServer: true,
    timeout: 120_000,
  },
});
