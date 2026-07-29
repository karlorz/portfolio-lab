import { expect, test, type Page } from '@playwright/test';

const TAB_LABELS = [
  'Overview',
  'Health',
  'Risk',
  'History',
  'Performance',
  'Rebalance',
  'Analytics',
  'Options',
  'Auction',
  'Labs',
  'Decisions',
  'Tasks',
  'Chat',
] as const;

const VIEWPORTS = [
  { width: 1200, height: 900 },
  { width: 1024, height: 900 },
  { width: 768, height: 900 },
  { width: 390, height: 900 },
] as const;

type DashboardTabLabel = (typeof TAB_LABELS)[number];

const LOADED_TAB_SELECTORS: Record<DashboardTabLabel, readonly string[]> = {
  'Overview': ['.overview-panel .metrics-grid'],
  'Health': ['.health-panel-container'],
  'Risk': ['.risk-primary-grid'],
  'History': ['.history-panel'],
  'Performance': ['.performance-summary'],
  'Rebalance': ['.rebalance-health-panel'],
  'Analytics': ['.analytics-summary', '.analytics-empty', '.explainability-panel'],
  'Options': ['.zero-dte-panel'],
  'Auction': ['.closing-auction-panel'],
  'Labs': ['.labs-panel .positions-table', '.labs-panel .analytics-empty:not([role="status"])'],
  'Decisions': ['.decision-replay-panel', '.decision-replay-panel .analytics-empty'],
  'Tasks': ['.tasks-panel-container'],
  'Chat': ['.chat-panel-container'],
} as const;

async function openDashboard(page: Page, viewport = { width: 1200, height: 900 }) {
  await page.setViewportSize(viewport);
  await page.goto('/');
  await expect(page.getByRole('heading', { name: 'Live Paper Trading' })).toBeVisible();
  await expect(page.getByRole('navigation', { name: 'Portfolio Lab workspaces' })).toBeVisible();
}

async function expectNoDocumentOverflow(page: Page) {
  const overflow = await page.evaluate(() => ({
    documentWidth: document.documentElement.scrollWidth,
    viewportWidth: window.innerWidth,
    bodyWidth: document.body.scrollWidth,
  }));

  expect(overflow.documentWidth).toBeLessThanOrEqual(overflow.viewportWidth + 1);
  expect(overflow.bodyWidth).toBeLessThanOrEqual(overflow.viewportWidth + 1);
}

async function expectGridColumns(page: Page, selector: string, minColumns: number) {
  const grid = page.locator(selector).first();
  await expect(grid).toHaveCSS('display', 'grid');
  const columnCount = await grid.evaluate((element) =>
    getComputedStyle(element).gridTemplateColumns.split(' ').filter(Boolean).length
  );
  expect(columnCount).toBeGreaterThanOrEqual(minColumns);
}

async function expectCardSurface(page: Page, selector: string) {
  const card = page.locator(selector).first();
  await expect(card).toBeVisible();
  await expect(card).toHaveCSS('background-color', 'rgb(30, 41, 59)');
  await expect(card).toHaveCSS('border-radius', '8px');
  const paddingTop = await card.evaluate((element) => parseFloat(getComputedStyle(element).paddingTop));
  expect(paddingTop).toBeGreaterThan(0);
}

async function expectStyledSurface(page: Page, selector: string) {
  const card = page.locator(selector).first();
  await expect(card).toBeVisible();
  const style = await card.evaluate((element) => {
    const computed = getComputedStyle(element);
    return {
      backgroundColor: computed.backgroundColor,
      borderRadius: parseFloat(computed.borderTopLeftRadius),
      paddingTop: parseFloat(computed.paddingTop),
    };
  });

  expect(style.backgroundColor).not.toBe('rgba(0, 0, 0, 0)');
  expect(style.borderRadius).toBeGreaterThan(0);
  expect(style.paddingTop).toBeGreaterThan(0);
}

async function expectExplainabilityPanelStyles(page: Page, expectedProvenanceColumns: number) {
  await expectStyledSurface(page, '.explainability-panel');
  await expectStyledSurface(page, '.ex-decision-card');
  await expectGridColumns(page, '.ex-provenance-grid', expectedProvenanceColumns);

  const signalRow = page.locator('.ex-signal-row').first();
  await expect(signalRow).toBeVisible();
  await expect(signalRow).toHaveCSS('display', 'grid');

  const barMetrics = await page.locator('.ex-signal-bar-container').first().evaluate((container) => {
    const bar = container.querySelector('.ex-signal-bar');
    const containerBox = container.getBoundingClientRect();
    const barBox = bar?.getBoundingClientRect();
    const computed = getComputedStyle(container);
    return {
      overflow: computed.overflow,
      containerWidth: containerBox.width,
      barWidth: barBox?.width ?? 0,
    };
  });

  expect(barMetrics.overflow).toBe('hidden');
  expect(barMetrics.barWidth).toBeLessThanOrEqual(barMetrics.containerWidth + 1);
}

function dashboardTab(page: Page, label: string) {
  return page
    .getByRole('navigation', { name: 'Portfolio Lab workspaces' })
    .getByRole('link', { name: label, exact: true });
}

async function waitForLoadedDashboardTab(page: Page, label: DashboardTabLabel) {
  const selectors = LOADED_TAB_SELECTORS[label];
  await expect(page.locator(selectors.join(', ')).first()).toBeVisible();
}

function collectPresentationConsoleFailures(page: Page): string[] {
  const consoleMessages: string[] = [];
  page.on('console', (message) => {
    if (message.type() !== 'error' && message.type() !== 'warning') return;
    if (message.type() === 'warning' && message.text().startsWith('[graduation] Validation failed:')) {
      return;
    }
    consoleMessages.push(`${message.type()}: ${message.text()}`);
  });
  return consoleMessages;
}

test.describe('dashboard browser presentation smoke', () => {
  test('contains the live missing-title kill alert without losing authority or the React root', async ({ page }) => {
    const consoleMessages = collectPresentationConsoleFailures(page);
    const signalsFixture = {
      timestamp: '2026-07-28T03:40:23Z',
      generated_at: '2026-07-28T03:40:23Z',
      regime: { regime: 'normal', vix: 18.67, detected: null },
      latest_prices: { SPY: 635, GLD: 305, TLT: 88 },
      current_positions: [],
      target_allocations: { SPY: 0.46, GLD: 0.38, TLT: 0.16 },
      cash: 0,
      total_value: 100000,
      recent_orders: [],
      ml_signals: {
        available: false,
        timestamp: null,
        predictions: {},
        features: {},
        grid_search: {
          available: false,
          timestamp: null,
          top_allocation: null,
          sharpe: null,
          volatility: null,
        },
      },
      marl_status: {
        schema_version: 'marl-runtime-status/v1',
        available: false,
        timestamp: '2026-07-28T03:40:23Z',
        runtime: {
          version: 'unknown',
          device: 'unknown',
          agents_loaded: [],
          signal_integrator_connected: false,
          checkpoint_loaded: false,
          inference_count: 0,
          current_allocation: {},
          graph_metrics: {},
        },
        execution_role: {
          role: 'research_shadow_non_routed',
          routed: false,
          routed_by: null,
          live_authoritative: false,
          description: 'Research shadow only.',
        },
      },
    };

    await page.route('**/data/signals.json', (route) => route.fulfill({ json: signalsFixture }));
    await page.route('**/data/alerts.json', (route) => route.fulfill({
      json: {
        generated_at: '2026-07-28T03:40:23Z',
        alerts: [{
          type: 'kill_switch',
          level: 'warning',
          reason: 'unresolved_incident:signal_staleness',
          incident_id: 'bb796837-2920-47af-a1b0-b67d9c08d356',
          enabled: true,
          message: '1/23 signals unavailable: regime_transition',
        }],
      },
    }));

    await openDashboard(page);
    await expect(page.locator('#root')).not.toBeEmpty();
    await expect(page.locator('.allocation-spine')).toContainText('SPY');
    await expect(page.locator('.allocation-spine')).toContainText('46%');
    await expect(page.locator('.allocation-spine')).toContainText('Research shadow · non-routed');
    const operatorContext = page.getByRole('complementary', { name: 'Operator context' });
    await expect(operatorContext.getByText('Kill Switch', { exact: true })).toBeVisible();
    await expect(operatorContext.getByText('Review kill-switch state before placing new orders.')).toBeVisible();
    expect(consoleMessages).toEqual([]);
  });

  for (const viewport of VIEWPORTS) {
    test(`keeps workspace navigation usable without document overflow at ${viewport.width}px`, async ({ page }) => {
      const consoleMessages = collectPresentationConsoleFailures(page);

      await openDashboard(page, viewport);
      await expectNoDocumentOverflow(page);

      for (const label of TAB_LABELS) {
        const tab = dashboardTab(page, label);
        await expect(tab).toBeVisible();
        await tab.click();
        await expect(tab).toHaveAttribute('aria-current', 'page');
        await waitForLoadedDashboardTab(page, label);
        await expectNoDocumentOverflow(page);
      }

      expect(consoleMessages).toEqual([]);
    });
  }

  test('computes real card and grid styles for dashboard presentation panels', async ({ page }) => {
    const consoleMessages = collectPresentationConsoleFailures(page);

    await openDashboard(page, { width: 1200, height: 900 });

    await dashboardTab(page, 'Performance').click();
    await expect(page.locator('.performance-summary')).toBeVisible();
    await expectGridColumns(page, '.performance-summary', 3);
    await expectStyledSurface(page, '.perf-card');
    await expectNoDocumentOverflow(page);

    await dashboardTab(page, 'Risk').click();
    await expectGridColumns(page, '.risk-primary-grid', 2);
    await expectCardSurface(page, '.risk-card');

    await dashboardTab(page, 'Analytics').click();
    await expect(page.locator('.analytics-summary')).toBeVisible();
    await expectStyledSurface(page, '.underwater-chart');
    await expectStyledSurface(page, '.rolling-metrics-chart');
    await expectStyledSurface(page, '.crisis-overlay');
    await expectStyledSurface(page, '.analytics-card');
    await expectExplainabilityPanelStyles(page, 4);
    await expectGridColumns(page, '.dashboard-grid.dashboard-grid-two.analytics-panel-group', 2);
    await expectNoDocumentOverflow(page);

    await dashboardTab(page, 'Rebalance').click();
    await expectCardSurface(page, '.rebalance-health-panel');
    await expectGridColumns(page, '.rh-state-grid', 2);

    await dashboardTab(page, 'Options').click();
    await expectCardSurface(page, '.zero-dte-panel');
    await expectCardSurface(page, '.panel');

    await dashboardTab(page, 'Auction').click();
    await expectCardSurface(page, '.closing-auction-panel');

    expect(consoleMessages).toEqual([]);
  });

  test('keeps loaded Analytics child panels within mobile viewport', async ({ page }) => {
    const consoleMessages = collectPresentationConsoleFailures(page);

    await openDashboard(page, { width: 390, height: 900 });
    await dashboardTab(page, 'Analytics').click();
    await expect(page.locator('.analytics-summary')).toBeVisible();
    await expect(page.locator('.explainability-panel')).toBeVisible();
    await expectExplainabilityPanelStyles(page, 1);
    await expect(page.getByText('VIXY Hedge Sizing').first()).toBeVisible();
    await expect(page.getByText('Hedge Selector').first()).toBeVisible();
    await expect(page.getByText('Black-Litterman Mapper').first()).toBeVisible();
    await expect(page.getByText('Regime Gate (v5.00)').first()).toBeVisible();
    await expect(page.getByText('TSMOM Overlay').first()).toBeVisible();
    await expectNoDocumentOverflow(page);

    expect(consoleMessages).toEqual([]);
  });

  test('keeps loaded Labs experiment table within mobile viewport', async ({ page }) => {
    const consoleMessages = collectPresentationConsoleFailures(page);

    await openDashboard(page, { width: 390, height: 900 });
    await dashboardTab(page, 'Labs').click();
    await expect(page.locator('.labs-panel')).toBeVisible();
    const table = page.locator('.labs-panel .positions-table');
    const empty = page.locator('.labs-panel .analytics-empty:not([role="status"])');
    await expect(table.or(empty).first()).toBeVisible();

    if (await table.isVisible()) {
      const tableMetrics = await table.evaluate((element) => {
        const panel = element.closest('.labs-panel');
        return {
          tableWidth: element.scrollWidth,
          panelWidth: panel?.clientWidth ?? 0,
        };
      });
      expect(tableMetrics.tableWidth).toBeGreaterThan(tableMetrics.panelWidth);
    }
    await expectNoDocumentOverflow(page);

    expect(consoleMessages).toEqual([]);
  });
});
