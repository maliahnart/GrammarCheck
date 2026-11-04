import {test, expect, chromium} from "@playwright/test";
import path from "node:path";
import {fileURLToPath} from "node:url";

test("extension connects to real ML API, reviews edits, teaches, and exports datasets", async () => {
  const extension = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
  const context = await chromium.launchPersistentContext("", {
    channel: "chromium", headless: true,
    args: [`--disable-extensions-except=${extension}`, `--load-extension=${extension}`],
  });
  try {
    const worker = context.serviceWorkers()[0] || await context.waitForEvent("serviceworker");
    const id = worker.url().split("/")[2];
    await worker.evaluate(() => chrome.storage.local.set({endpoint: "http://127.0.0.1:8765"}));
    const page = await context.newPage();
    const errors = []; page.on("pageerror", error => errors.push(error.message));
    await page.goto(`chrome-extension://${id}/index.html`);
    await expect(page.locator("#endpoint")).toHaveValue("http://127.0.0.1:8765");
    await page.locator("#source").fill("Tuy trời mưa nhưng nên tôi vẫn đi học.\n\nKinh tế dang phát triển nhanh.");
    await page.locator("#check").click();
    await expect(page.locator("#results")).toBeVisible();
    await expect(page.locator("#output")).toHaveValue(/Kinh tế đang phát triển nhanh/);
    await expect(page.locator("#output")).not.toHaveValue(/nhưng nên/);
    for (const checkbox of await page.locator("#edits input").all()) await checkbox.uncheck();
    await expect(page.locator("#output")).toHaveValue("Tuy trời mưa nhưng nên tôi vẫn đi học.\n\nKinh tế dang phát triển nhanh.");
    await page.locator('[data-mode="learn"]').click();
    await page.locator("#load-lessons").click();
    await expect(page.locator(".lesson")).toHaveCount(6);
    await page.locator("#lesson-vocabulary").getByRole("button", {name: "giành", exact: true}).click();
    await expect(page.locator("#lesson-vocabulary [role=status]")).toContainText("Đúng.");
    await page.screenshot({path: "test-results/learning.png", fullPage: true});
    await page.locator('[data-mode="dataset"]').click();
    await page.locator("#data-file").setInputFiles({name: "sample.jsonl", mimeType: "application/x-ndjson", buffer: Buffer.from('{"text":"Kinh tế dang phát triển nhanh.","label":"neutral"}\n')});
    const downloadPromise = page.waitForEvent("download");
    await page.locator("#clean").click();
    const download = await downloadPromise;
    expect(download.suggestedFilename()).toBe("normalized.jsonl");
    await expect(page.locator("#status")).toContainText("Đã xử lý 1 bản ghi");
    expect(errors).toEqual([]);
  } finally { await context.close(); }
});
