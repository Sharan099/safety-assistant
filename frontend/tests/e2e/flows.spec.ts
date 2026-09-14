import { expect, test } from "@playwright/test";
import path from "node:path";

import { ALICE, BOB, FIXTURES, login, logout } from "./helpers";

// The five required E2E flows (07_TESTING "Frontend E2E"). Real API, real worker, real corpus.
// Without the licensed corpus (CI) point the flows at the synthetic regulation seeded by
// scripts/maintenance/seed_synthetic_corpus.py via E2E_REGULATION / E2E_QUESTION / E2E_TERM.
const REG = process.env.E2E_REGULATION ?? "UN R94";
const QUESTION = process.env.E2E_QUESTION ?? "What is the tibia index limit in UN R94?";
const TERM = new RegExp(process.env.E2E_TERM ?? "tibia", "i");
const QUESTION_2 = process.env.E2E_QUESTION_2 ?? "What is the head performance criterion limit in UN R94?";
const QUESTION_2_KEY = QUESTION_2.split(" ").slice(3, 6).join(" ");

test("1. login → new chat → answer → open evidence", async ({ page }) => {
  await login(page, ALICE);
  await page.getByTestId("home-new").click();
  await expect(page).toHaveURL(/\/app\/chat$/);
  await expect(page.getByTestId("composer-scope")).toContainText("Verified regulations");
  await page.getByTestId("query").fill(QUESTION);
  await page.getByTestId("ask").click();
  await expect(page).toHaveURL(/\/app\/chat\/[0-9a-f-]+/);
  const answer = page.getByTestId("answer");
  await expect(answer).toBeVisible();
  await expect(answer).toHaveAttribute("data-mode", /GENERATED|EVIDENCE_ONLY|ABSTAINED/);
  const citation = page.getByTestId("citation").first();
  await expect(citation).toContainText(REG);
  await citation.hover(); // the cited lines and page are previewed before the panel opens
  const preview = page.getByTestId("citation-preview");
  await expect(preview).toBeVisible();
  await expect(preview).toContainText(/p\. \d+|pp\. \d+/);
  await citation.click();
  const card = page.getByTestId("evidence-panel").getByTestId("evidence-card").first();
  await expect(card).toBeVisible();
  await expect(card).toContainText(REG);
  await expect(card).toContainText(/§/);
  await expect(card).toContainText(/in force|valid/);
  await expect(card).toContainText(/Verified regulation/);
  await expect(card).toContainText(TERM);
});

test("2. login → upload PDF → processing → READY → ask uploaded document", async ({ page }) => {
  await login(page, ALICE);
  const title = `E2E sled note ${Date.now()}`;
  await page.goto("/app/documents/upload");
  await page.getByTestId("file-input").setInputFiles(path.join(FIXTURES, "project-note.pdf"));
  await page.getByTestId("upload-title").fill(title);
  await page.getByTestId("upload-submit").click();
  const progress = page.getByTestId("upload-progress");
  await expect(progress).toBeVisible();
  await expect(progress.getByTestId("ingestion-timeline")).toHaveAttribute("data-status", "READY", { timeout: 150_000 });
  await progress.getByTestId("ask-document").click();
  await expect(page).toHaveURL(/\/app\/chat\?document=/);
  await expect(page.getByText("Limited to 1 selected document")).toBeVisible();
  await page.getByTestId("query").fill("What is the sled pulse peak limit for the front row?");
  await page.getByTestId("ask").click();
  const answer = page.getByTestId("answer");
  await expect(answer).toBeVisible();
  await expect(answer).toHaveAttribute("data-mode", /GENERATED|EVIDENCE_ONLY/);
  await page.getByTestId("citation").first().click();
  const card = page.getByTestId("evidence-panel").getByTestId("evidence-card").first();
  await expect(card).toContainText(/Private document/);
  await expect(card).toContainText(/37 g/);
});

test("3. logout → login → conversation restored with messages, citations and scope", async ({ page }) => {
  await login(page, ALICE);
  await page.goto("/app/chat");
  await page.getByTestId("query").fill(QUESTION_2);
  await page.getByTestId("ask").click();
  await expect(page).toHaveURL(/\/app\/chat\/[0-9a-f-]+/);
  const url = page.url();
  await expect(page.getByTestId("answer")).toBeVisible();
  const citationCount = await page.getByTestId("citation").count();
  await logout(page);
  await page.goto(url);
  await expect(page).toHaveURL(/\/login/); // guarded
  await login(page, ALICE);
  await page.goto(url);
  await expect(page.getByTestId("user-message")).toContainText(QUESTION_2_KEY);
  await expect(page.getByTestId("answer")).toBeVisible();
  expect(await page.getByTestId("citation").count()).toBe(citationCount);
  await expect(page.getByTestId("source-scope")).toContainText("Verified regulations");
  await expect(page.getByTestId("conversation-list")).toContainText(QUESTION_2_KEY);
});

test("4. user A private upload → user B denied", async ({ page }) => {
  await login(page, ALICE);
  const title = `E2E private ${Date.now()}`;
  await page.goto("/app/documents/upload");
  await page.getByTestId("file-input").setInputFiles(path.join(FIXTURES, "project-note.pdf"));
  await page.getByTestId("upload-title").fill(title);
  await page.getByTestId("upload-submit").click();
  const progress = page.getByTestId("upload-progress");
  await expect(progress).toBeVisible();
  const detail = await progress.getByTestId("view-details").first().getAttribute("href", { timeout: 150_000 });
  expect(detail).toBeTruthy();
  await logout(page);
  await login(page, BOB);
  await page.goto("/app/documents?scope=PRIVATE_USER");
  await expect(page.getByTestId("document-table").or(page.getByText("No documents match"))).toBeVisible();
  await expect(page.getByText(title)).toHaveCount(0);
  await page.goto(detail as string);
  await expect(page.getByTestId("document-error")).toContainText(/does not exist or you do not have access/);
});

test("5. failed upload → actionable error UI", async ({ page }) => {
  await login(page, ALICE);
  await page.goto("/app/documents/upload");
  // a file that only pretends to be a PDF is rejected at the API boundary (magic bytes) with a clear message
  await page.getByTestId("file-input").setInputFiles(path.join(FIXTURES, "not-a-pdf.pdf"));
  await page.getByTestId("upload-title").fill(`E2E not a pdf ${Date.now()}`);
  await page.getByTestId("upload-submit").click();
  await expect(page.getByRole("alert").filter({ hasText: /not a PDF/ })).toBeVisible();
  await expect(page.getByTestId("upload-progress")).toHaveCount(0);
  // a PDF that fails validation in the pipeline ends QUARANTINED with a safe message and a diagnostic reference
  await page.getByTestId("file-input").setInputFiles(path.join(FIXTURES, "poison.pdf"));
  await page.getByTestId("upload-title").fill(`E2E poison ${Date.now()}`);
  await page.getByTestId("upload-submit").click();
  const progress = page.getByTestId("upload-progress");
  await expect(progress.getByTestId("ingestion-timeline")).toHaveAttribute("data-status", "QUARANTINED", { timeout: 120_000 });
  const err = progress.getByTestId("ingestion-error");
  await expect(err).toContainText(/Document quarantined/);
  await expect(err).toContainText(/Diagnostic reference/);
  await expect(err).not.toContainText(/garbage|Traceback/);
  await expect(progress.getByTestId("upload-replace")).toBeVisible();
});

test("6. sources page lists the verified corpus grouped and in regulation order", async ({ page }) => {
  await login(page, ALICE);
  await page.getByRole("link", { name: "Sources" }).first().click();
  await expect(page).toHaveURL(/\/app\/sources$/);
  const rows = page.getByTestId("sources-unece").getByTestId("source-row");
  await expect(rows.first()).toBeVisible();
  const keys = await rows.evaluateAll((els) => els.map((e) => e.getAttribute("data-key") ?? ""));
  const numbers = keys.map((k) => Number(/^UN-R(\d+)/.exec(k)?.[1] ?? 0));
  expect(numbers).toEqual([...numbers].sort((a, b) => a - b)); // numeric, not lexical (R94 before R129)
  await expect(rows.first()).toContainText(/UN R\d+/);
});
