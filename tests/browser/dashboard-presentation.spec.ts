import { expect, test, type Page } from '@playwright/test';

const TAB_LABELS = [
  'Overview',
  'Health',
  'Risk',
  'History',
  'Performance',
  'Rebalance',
  'Analytics',
  '0DTE',
  'Auction',
  'Labs',
  'Tasks',
  'Chat',
] as const;

const VIEWPORTS = [
  { width: 1200, height: 900 },
  { width: 1024, height: 900 },
  { width: 768, height: 900 },
  { width: 390, height: 900 },
] as const;

async function openDashboard(page: Page, viewport = { width: 1200, height: 900 }) {
  await page.setViewportSize(viewport);
  await page.goto('/');
  await expect(page.getByRole('heading', { name: 'Live Paper Trading' })).toBeVisible();
  await expect(page.locator('.dashboard-tabs')).toBeVisible();
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

function dashboardTab(page: Page, label: string) {
  return page
    .locator('.dashboard-tabs')
    .getByRole('button', { name: new RegExp(`^${label}(?:\\s+\\d+)?$`) });
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
  for (const viewport of VIEWPORTS) {
    test(`keeps dashboard tabs clickable without document overflow at ${viewport.width}px`, async ({ page }) => {
      const consoleMessages = collectPresentationConsoleFailures(page);

      await openDashboard(page, viewport);
      await expectNoDocumentOverflow(page);

      for (const label of TAB_LABELS) {
        const tab = dashboardTab(page, label);
        await expect(tab).toBeVisible();
        await tab.click();
        await expect(tab).toHaveClass(/active/);
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
    await expectGridColumns(page, '.dashboard-grid.dashboard-grid-two.analytics-panel-group', 2);
    await expectNoDocumentOverflow(page);

    await dashboardTab(page, 'Rebalance').click();
    await expectCardSurface(page, '.rebalance-health-panel');
    await expectGridColumns(page, '.rh-state-grid', 2);

    await dashboardTab(page, '0DTE').click();
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
    await expect(page.locator('.labs-panel .positions-table')).toBeVisible();

    const tableMetrics = await page.locator('.labs-panel .positions-table').evaluate((table) => {
      const panel = table.closest('.labs-panel');
      return {
        tableWidth: table.scrollWidth,
        panelWidth: panel?.clientWidth ?? 0,
      };
    });
    expect(tableMetrics.tableWidth).toBeGreaterThan(tableMetrics.panelWidth);
    await expectNoDocumentOverflow(page);

    expect(consoleMessages).toEqual([]);
  });
});
