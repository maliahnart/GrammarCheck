import {defineConfig} from "@playwright/test";

export default defineConfig({
  testDir: "./e2e", workers: 1, timeout: 60000,
  reporter: "list",
  webServer: {
    command: "../../.venv/bin/python ../../scripts/run_api.py --port 8765",
    url: "http://127.0.0.1:8765/health", reuseExistingServer: false,
    env: {GRAMMARCHECK_API_KEY: ""}, timeout: 30000,
  },
});
