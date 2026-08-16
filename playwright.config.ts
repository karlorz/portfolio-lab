import { defineConfig, devices } from '@playwright/test';

const PORT = Number(process.env.PORTFOLIO_LAB_E2E_PORT ?? 4173);
const baseURL = `http://127.0.0.1:${PORT}`;

export default defineConfig({
  testDir: './tests/browser',
  outputDir: '.playwright/output',
  timeout: 45_000,
  expect: {
    timeout: 25_000,
  },
  fullyParallel: false,
  workers: 1,
  use: {
    ...devices['Desktop Chrome'],
    baseURL,
    trace: 'retain-on-failure',
  },
  webServer: {
    command: `node node_modules/vite/bin/vite.js --host 127.0.0.1 --port ${PORT}`,
    url: baseURL,
    // Local-only suite (playwright is absent from .github/workflows/ci.yml):
    // always launch a fresh server. Running vite via plain `node` (not
    // `bunx --bun`, which stalled ~67% of spawned cold starts — Items
    // 14/15) binds in <1s. A stale PORT listener fails fast with EADDRINUSE
    // (kill the stray process and re-run).
    reuseExistingServer: false,
    timeout: 300_000,
  },
});
