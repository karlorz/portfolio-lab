import { defineConfig, devices } from '@playwright/test';

const PORT = Number(process.env.PORTFOLIO_LAB_E2E_PORT ?? 4173);
const baseURL = `http://127.0.0.1:${PORT}`;

export default defineConfig({
  testDir: './tests/browser',
  outputDir: '.playwright/output',
  timeout: 45_000,
  expect: {
    timeout: 10_000,
  },
  fullyParallel: false,
  workers: 1,
  use: {
    ...devices['Desktop Chrome'],
    baseURL,
    trace: 'retain-on-failure',
  },
  webServer: {
    command: `bunx --bun vite --host 127.0.0.1 --port ${PORT}`,
    url: baseURL,
    // Local-only suite (playwright is absent from .github/workflows/ci.yml):
    // always launch a fresh server. Reusing a leftover PORT listener caused
    // a zero-output 120s timeout flake; a stale listener now fails fast with
    // EADDRINUSE (kill the stray process and re-run). 300s is the fail-fast
    // ceiling — a cold first-in-session start can exceed it under load
    // (prewarm per the Makefile test-browser comment).
    reuseExistingServer: false,
    timeout: 300_000,
  },
});
