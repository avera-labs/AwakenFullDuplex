import { defineConfig } from "@playwright/test";

const port = Number(process.env.E2E_PORT || 18765);
export default defineConfig({
  testDir: "./tests/e2e",
  fullyParallel: false,
  workers: 1, // The real inference protocol supports one conversation at a time.
  timeout: 20000,
  expect: { timeout: 8000 },
  reporter: "list",
  use: {
    baseURL: `http://127.0.0.1:${port}`,
    browserName: "chromium",
    viewport: { width: 1440, height: 1100 },
    permissions: ["microphone"],
    launchOptions: {
      args: [
        "--use-fake-device-for-media-stream",
        "--use-fake-ui-for-media-stream",
      ],
    },
    screenshot: "only-on-failure",
    trace: "retain-on-failure",
  },
  webServer: {
    command: "uv run python tests/serve_e2e.py",
    url: `http://127.0.0.1:${port}/api/config`,
    reuseExistingServer: false, // Never accidentally test against a real GPU server.
    timeout: 30000,
  },
});
