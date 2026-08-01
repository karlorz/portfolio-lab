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
  { width: 1440, height: 900 },
  { width: 1200, height: 900 },
  { width: 1024, height: 900 },
  { width: 768, height: 900 },
  { width: 390, height: 900 },
  { width: 320, height: 900 },
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
  const navigation = page.getByRole('navigation', { name: 'Portfolio Lab workspaces' });
  if (viewport.width <= 720) {
    await expect(page.getByRole('button', { name: 'Menu' })).toBeVisible();
    await expect(navigation).not.toBeVisible();
    await page.getByRole('button', { name: 'Menu' }).click();
  }
  await expect(navigation).toBeVisible();
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

function makeSignalsFixture(factorRotation: Record<string, unknown>) {
  return {
    timestamp: '2026-07-30T20:06:46Z',
    generated_at: '2026-07-30T20:06:46Z',
    regime: { regime: 'normal', vix: 18.67, detected: null },
    latest_prices: { SPY: 635, GLD: 305, TLT: 88 },
    current_positions: [],
    target_allocations: { SPY: 0.46, GLD: 0.38, TLT: 0.16 },
    regime_authority: {
      schema_version: 'regime-authority/v1',
      live_controller: 'signals.json.target_allocations',
      live_controller_module: 'src.broker.order_router',
      live_regime: 'normal',
      allocation_regime: 'normal',
      routed_surface: 'target_allocations',
      target_allocations: { SPY: 0.46, GLD: 0.38, TLT: 0.16 },
      regime_controller: 'classify_vix_regime',
      regime_controller_module: 'src.utils.classify_vix_regime',
      regime_routed: false,
      advanced_regime_signals: {
        two_stage_regime: { role: 'advisory_shadow', routed: false },
        bocd_regime: { role: 'advisory_shadow', routed: false },
        regime_transition: { role: 'advisory_shadow', routed: false },
      },
    },
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
      timestamp: '2026-07-30T20:06:46Z',
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
    factor_rotation: factorRotation,
  };
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
    await expect(page.locator('.metric-card.primary')).not.toContainText('Cash:');
    const operatorContext = page.getByRole('complementary', { name: 'Operator context' });
    await expect(operatorContext.getByText('Kill Switch', { exact: true })).toBeVisible();
    await expect(operatorContext.getByText('Review kill-switch state before placing new orders.')).toBeVisible();
    expect(consoleMessages).toEqual([]);
  });

  test('renders the bounded IC evidence brief in the first-view cockpit', async ({ page }) => {
    const consoleMessages = collectPresentationConsoleFailures(page);
    const signalsFixture = {
      ...makeSignalsFixture({
        selected_factors: ['VLUE'],
        allocation: { VLUE: 1 },
        signal_strength: 0.5,
        recommendation: 'Hold value sleeve',
      }),
      ic_decay: {
        status: 'critical',
        signals: {
          ensemble_duration: {
            ic_rolling: 0.0045,
            ic_trend: 'unknown',
            observations: 20,
            status: 'critical',
          },
          ensemble_consensus: {
            ic_rolling: 0.0391,
            ic_trend: 'unknown',
            observations: 20,
            status: 'critical',
          },
          ensemble_equity: {
            ic_rolling: 0.0692,
            ic_trend: 'unknown',
            observations: 20,
            status: 'warning',
          },
        },
      },
    };
    const healthFixture = {
      cron_jobs: [],
      data_freshness: {},
      system_status: 'critical',
      generated_at: '2026-08-01T09:40:23Z',
      kill_switch: {
        status: 'critical',
        enabled: true,
        level: 'halt',
        reason: 'unresolved_incident:ic_decay',
        mode: 'paper',
      },
      ic_decay_summary: {
        status: 'critical',
        critical_signals: ['ensemble_consensus', 'ensemble_duration'],
        warning_signals: ['ensemble_equity'],
        insufficient_data_signals: [],
        resolved_signal_count: 4,
        min_observations: 20,
        staged_pending_predictions: 7,
        staged_pending_signal_names: ['ensemble_consensus', 'ensemble_duration'],
        staged_date: '2026-08-01',
        staged_pending_scope: 'ic_staged_date_window',
        historical_unlabeled_rows: 1663,
        historical_unlabeled_dates: 2,
        historical_unlabeled_oldest_date: '2026-07-31',
        historical_unlabeled_scope: 'historical_db_unlabeled_rows',
        evidence_generated_at: '2026-08-01T09:40:23Z',
        evidence_freshness: 'captured_runtime_snapshot',
        routing_authority: 'advisory_only',
        routing_control: 'routing_blocked',
        control_effect: 'paper_warning',
        kill_switch_level: 'halt',
      },
    };

    await page.route('**/data/signals.json', (route) => route.fulfill({ json: signalsFixture }));
    await page.route('**/data/health.json', (route) => route.fulfill({ json: healthFixture }));
    await page.route('**/data/alerts.json', (route) => route.fulfill({
      json: { generated_at: '2026-08-01T09:40:23Z', alerts: [] },
    }));
    await page.route('**/data/incidents.json', (route) => route.fulfill({
      json: {
        generated_at: '2026-08-01T09:40:23Z',
        open_count: 0,
        incidents: [],
        metrics: {
          incident_frequency: 0,
          open_count: 0,
          resolved_count: 0,
          mean_mttr_seconds: null,
        },
      },
    }));

    await openDashboard(page, { width: 1024, height: 900 });
    const brief = page.locator('.operator-brief');
    await expect(brief).toContainText('Market regime');
    await expect(brief).toContainText('Normal');
    await expect(brief).toContainText('Signal quality: Critical');
    await expect(brief).toContainText('ensemble_duration IC 0.0045 (20/20)');
    await expect(brief).toContainText('Staged pending labels: 7');
    await expect(brief).toContainText('Historical unlabeled rows: 1663');
    await expect(brief).toContainText('Routing blocked · kill halt');
    await expect(brief.getByRole('button', { name: 'Review IC evidence' })).toBeVisible();
    expect(consoleMessages).toEqual([]);
  });

  test('keeps mobile navigation collapsed until requested and restores focusable content', async ({ page }) => {
    const consoleMessages = collectPresentationConsoleFailures(page);
    await page.setViewportSize({ width: 390, height: 900 });
    await page.goto('/');

    const menu = page.getByRole('button', { name: 'Menu' });
    const navigation = page.getByRole('navigation', { name: 'Portfolio Lab workspaces' });
    await expect(menu).toHaveAttribute('aria-expanded', 'false');
    await expect(navigation).not.toBeVisible();

    await menu.click();
    await expect(menu).toHaveAttribute('aria-expanded', 'true');
    await expect(navigation).toBeVisible();
    await menu.click();
    await expect(menu).toHaveAttribute('aria-expanded', 'false');
    await expect(navigation).not.toBeVisible();
    await expect(page.getByRole('heading', { name: 'Live Paper Trading' })).toBeVisible();
    expect(consoleMessages).toEqual([]);
  });

  test('keeps keyboard focus usable at a 400% zoom approximation', async ({ page }) => {
    const consoleMessages = collectPresentationConsoleFailures(page);
    await page.setViewportSize({ width: 390, height: 900 });
    await page.goto('/');

    const menu = page.getByRole('button', { name: 'Menu' });
    await menu.focus();
    await expect(menu).toBeFocused();
    await page.keyboard.press('Enter');
    await expect(page.getByRole('navigation', { name: 'Portfolio Lab workspaces' })).toBeVisible();
    await page.keyboard.press('Tab');
    await expect(page.locator(':focus')).toBeVisible();

    // Playwright does not expose browser UI zoom, so use CSS zoom to exercise
    // the same dense/reflow surfaces without changing application data.
    await page.evaluate(() => {
      document.documentElement.style.zoom = '4';
    });
    await expect(menu).toBeVisible();
    await expect(page.locator('#root')).not.toBeEmpty();
    expect(await page.locator(':focus').count()).toBe(1);
    expect(consoleMessages).toEqual([]);
  });

  test('loads every primary operator data request successfully', async ({ page }) => {
    const consoleMessages = collectPresentationConsoleFailures(page);
    const primaryPaths = [
      '/data/signals.json',
      '/data/dashboard.json',
      '/data/alerts.json',
      '/data/stats.json',
      '/data/health.json',
      '/data/incidents.json',
      '/data/health_ops.json',
    ];
    const responses = primaryPaths.map((path) => page.waitForResponse((response) => (
      new URL(response.url()).pathname === path && response.status() === 200
    )));

    await openDashboard(page, { width: 1024, height: 900 });
    await Promise.all(responses);
    expect(consoleMessages).toEqual([]);
  });

  for (const scenario of [
    {
      name: 'production',
      payload: {
        selected_factors: ['VLUE', 'VBR'],
        allocation: { VLUE: 0.27, VBR: 0.73 },
        signal_strength: 0.53,
        recommendation: 'Rotate to Value',
      },
      expected: 'Rotate to Value',
    },
    {
      name: 'partial',
      payload: {
        selected_factors: ['QUAL'],
        recommendation: 'Hold quality sleeve',
      },
      expected: 'Unavailable',
    },
    {
      name: 'malformed numeric',
      payload: {
        selected_factors: ['VLUE'],
        allocation: { VLUE: 'not-a-number' },
        signal_strength: 'bad',
        recommendation: 'Advisory data degraded',
      },
      expected: 'Advisory data degraded',
    },
  ] as const) {
    test(`renders Factor Rotation ${scenario.name} payload without console failures`, async ({ page }) => {
      const consoleMessages = collectPresentationConsoleFailures(page);
      await page.route('**/data/signals.json', (route) => route.fulfill({
        json: makeSignalsFixture(scenario.payload),
      }));

      await openDashboard(page, { width: 1200, height: 900 });
      await dashboardTab(page, 'Analytics').click();

      const panel = page.locator('.factor-rotation-card').first();
      await expect(panel).toBeVisible();
      await expect(panel).toContainText('Factor Rotation');
      await expect(panel).toContainText('Advisory');
      await expect(panel).toContainText(scenario.expected);
      await expectNoDocumentOverflow(page);
      expect(consoleMessages).toEqual([]);
    });
  }

  for (const viewport of VIEWPORTS) {
    test(`keeps workspace navigation usable without document overflow at ${viewport.width}px`, async ({ page }) => {
      const consoleMessages = collectPresentationConsoleFailures(page);

      await openDashboard(page, viewport);
      await expectNoDocumentOverflow(page);

      for (const label of TAB_LABELS) {
        if (viewport.width <= 720) {
          const navigation = page.getByRole('navigation', { name: 'Portfolio Lab workspaces' });
          if (!(await navigation.isVisible())) {
            await page.getByRole('button', { name: 'Menu' }).click();
            await expect(navigation).toBeVisible();
          }
        }
        const tab = dashboardTab(page, label);
        await expect(tab).toBeVisible();
        await tab.click();
        await expect(page.locator('.navigation-rail a').filter({ hasText: label })).toHaveAttribute('aria-current', 'page');
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
    await waitForLoadedDashboardTab(page, 'Analytics');
    await expect(page.locator('.analytics-summary')).toBeVisible({ timeout: 20_000 });
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

  test('keeps action and advisory badges legible in the context rail', async ({ page }) => {
    const consoleMessages = collectPresentationConsoleFailures(page);

    await openDashboard(page, { width: 1200, height: 900 });
    const badges = page.locator('.action-center .control-status');
    // The number of live action/advisory statuses is data-dependent.  The
    // presentation contract is that every rendered status remains legible,
    // not that a particular day's health payload produces a fixed count.
    await expect(badges.first()).toBeVisible();
    expect(await badges.count()).toBeGreaterThan(0);

    const badgeMetrics = await badges.evaluateAll((elements) => elements.map((element) => {
      const computed = getComputedStyle(element);
      const rect = element.getBoundingClientRect();
      return {
        whiteSpace: computed.whiteSpace,
        width: rect.width,
        height: rect.height,
        lineHeight: parseFloat(computed.lineHeight),
      };
    }));

    for (const badge of badgeMetrics) {
      expect(badge.whiteSpace).toBe('nowrap');
      expect(badge.width).toBeGreaterThan(30);
      expect(badge.height).toBeLessThanOrEqual(badge.lineHeight * 2.1);
    }
    expect(consoleMessages).toEqual([]);
  });

  test('keeps action totals consistent with the named condition collection', async ({ page }) => {
    const consoleMessages = collectPresentationConsoleFailures(page);
    await openDashboard(page, { width: 1024, height: 900 });

    const actionCenter = page.locator('.action-center');
    const overflow = actionCenter.locator('.action-list-overflow');
    await expect.poll(async () => actionCenter.evaluate((element) => {
      const countLabel = element.querySelector('.control-count')?.getAttribute('aria-label') ?? '';
      const overflowText = element.querySelector('.action-list-overflow')?.textContent ?? '';
      const count = countLabel.match(/^(\d+) actions required$/)?.[1];
      const overflowCount = overflowText.match(/;\s*(\d+) actions required/)?.[1];
      if (!overflowText) return count ? 'consistent' : countLabel;
      return count && overflowCount && count === overflowCount ? 'consistent' : `${countLabel} / ${overflowText}`;
    })).toBe('consistent');
    if (await overflow.count() > 0) await expect(overflow).toContainText('conditions');
    await expect(actionCenter).toContainText('Action Center');
    expect(consoleMessages).toEqual([]);
  });

  test('keeps live authority provenance readable at 320px', async ({ page }) => {
    const consoleMessages = collectPresentationConsoleFailures(page);

    await openDashboard(page, { width: 320, height: 900 });
    const authoritySource = page.locator('.authority-badge code').first();
    await expect(authoritySource).toContainText('signals.json.target_allocations');

    const sourceMetrics = await authoritySource.evaluate((element) => {
      const rect = element.getBoundingClientRect();
      return {
        width: rect.width,
        height: rect.height,
        scrollWidth: element.scrollWidth,
      };
    });

    expect(sourceMetrics.width).toBeGreaterThan(120);
    expect(sourceMetrics.height).toBeLessThan(48);
    expect(sourceMetrics.scrollWidth).toBeLessThanOrEqual(sourceMetrics.width + 1);
    await expectNoDocumentOverflow(page);
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
