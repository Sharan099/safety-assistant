import { expect, type Page } from "@playwright/test";
import path from "node:path";

export const ALICE = "e2e.alice@example.test";
export const BOB = "e2e.bob@example.test";
export const FIXTURES = path.resolve(__dirname, "fixtures");

export async function login(page: Page, email: string) {
  await page.context().clearCookies();
  await page.goto("/login");
  await page.getByTestId("login-email").fill(email);
  await page.getByTestId("login-submit").click();
  await expect(page).toHaveURL(/\/app\/home/);
}

export async function logout(page: Page) {
  await page.getByTestId("user-menu").click();
  await page.getByTestId("logout").click();
  await expect(page).toHaveURL(/\/login/);
}

/** Upload a fixture with a unique title; returns the document id once processing is shown. */
export async function upload(page: Page, file: string, title: string, scope: "PRIVATE_USER" | "WORKSPACE" = "PRIVATE_USER") {
  await page.goto("/app/documents/upload");
  await page.getByTestId("file-input").setInputFiles(path.join(FIXTURES, file));
  await page.getByTestId("upload-title").fill(title);
  if (scope === "WORKSPACE") {
    await page.getByTestId("upload-scope").click();
    await page.getByRole("option", { name: /Workspace/ }).click();
  }
  await page.getByTestId("upload-submit").click();
  await expect(page.getByTestId("upload-progress")).toBeVisible();
  const url = await page.getByTestId("upload-progress").getByRole("link", { name: "View details" }).getAttribute("href").catch(() => null);
  return url?.split("/").pop() ?? null;
}
