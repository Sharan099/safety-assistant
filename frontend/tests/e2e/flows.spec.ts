import { expect, test } from "@playwright/test";
import path from "node:path";

import { ALICE, BOB, FIXTURES, login, logout } from "./helpers";

// The five required E2E flows (07_TESTING "Frontend E2E"). Real API, real worker, real corpus.

test("1. login → new chat → answer → open evidence", async ({ page }) => {
  await login(page, ALICE);
  await page.getByTestId("home-new").click();
  await expect(page).toHaveURL(/\/app\/chat$/);
  await expect(page.getByTestId("composer-scope")).toContainText("Verified regulations");
  await page.getByTestId("query").fill("What is the tibia index limit in UN R94?");
  await page.getByTestId("ask").click();
  await expect(page).toHaveURL(/\/app\/chat\/[0-9a-f-]+/);
  const answer = page.getByTestId("answer");
  await expect(answer).toBeVisible();
  await expect(answer).toHaveAttribute("data-mode", /GENERATED|EVIDENCE_ONLY|ABSTAINED/);
  const citation = page.getByTestId("citation").first();
  await expect(citation).toContainText(/UN R94/);
  await citation.click();
  const card = page.getByTestId("evidence-panel").getByTestId("evidence-card").first();
  await expect(card).toBeVisible();
  await expect(card).toContainText(/UN R94/);
  await expect(card).toContainText(/§5\.2\.1\.8|§/);
  await expect(card).toContainText(/in force|valid/);
  await expect(card).toContainText(/Verified regulation/);
  await expect(card).toContainText(/tibia/i);
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
  await page.getByTestId("query").fill("What is the head performance criterion limit in UN R94?");
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
  await expect(page.getByTestId("user-message")).toContainText("head performance criterion");
  await expect(page.getByTestId("answer")).toBeVisible();
  expect(await page.getByTestId("citation").count()).toBe(citationCount);
  await expect(page.getByTestId("source-scope")).toContainText("Verified regulations");
  await expect(page.getByTestId("conversation-list")).toContainText("head performance criterion");
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
