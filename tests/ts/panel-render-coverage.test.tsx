import { describe, expect, it } from 'bun:test';
import React from 'react';
import { renderToStaticMarkup } from 'react-dom/server';
import { ComparisonTable } from '../../src/components/ComparisonTable';
import { CorrelationMatrix } from '../../src/components/CorrelationMatrix';
import { CrisisAnalysis } from '../../src/components/CrisisAnalysis';
import { DrawdownChart } from '../../src/components/DrawdownChart';
import { EquityCurve } from '../../src/components/EquityCurve';
import { IncidentSummary } from '../../src/components/IncidentSummary';
import { MetricsCards } from '../../src/components/MetricsCards';
import { PortfolioSelector } from '../../src/components/PortfolioSelector';
import { RiskReturnChart } from '../../src/components/RiskReturnChart';
import { RollingWindow } from '../../src/components/RollingWindow';
import type {
  BacktestResult,
  PerformanceMetrics,
  PortfolioConfig,
  PriceData,
} from '../../src/backtest/engine';
import type { DashboardIncident } from '../../src/components/dashboardIncidents';

// Render-smoke coverage for the 10 live dashboard panels that had zero
// imports in tests/ts (I15; see architecture/05-tech-debt.md). Hermetic
// fixtures only — no fetch, no live data, CI fresh-clone safe (Item 1
// s4/s5 lessons). renderToStaticMarkup does not run useEffect-dependent
// rendering (recharts ResponsiveContainer bodies render client-side); these
// cases are a crash/import/type regression net, per the panel-smoke
// precedent.

function metrics(overrides: Partial<PerformanceMetrics> = {}): PerformanceMetrics {
  return {
    cagr: 0.082,
    volatility: 0.12,
    sharpeRatio: 0.68,
    maxDrawdown: -0.19,
    calmarRatio: 0.43,
    sortinoRatio: 0.95,
    positiveMonths: 180,
    totalReturn: 3.1,
    ...overrides,
  };
}

const RESULT_ROWS = [
  { name: 'SPY/GLD/TLT', metrics: metrics(), color: '#3b82f6' },
  { name: '60/40 (S&P 500)', metrics: metrics({ cagr: 0.061, sharpeRatio: 0.55 }), color: '#10b981' },
];

// 30 consecutive daily points per symbol: enough return history for the
// correlation matrix (needs >= 20 overlapping returns).
const DAILY_30D: PriceData[] = ['SPY', 'GLD', 'TLT'].flatMap((symbol, i) =>
  Array.from({ length: 30 }, (_, k) => ({
    date: `2026-06-${String(k + 1).padStart(2, '0')}`,
    symbol,
    price: 100 + i * 10 + k * 0.5,
  })),
);

function backtestResult(dates: string[], startValue = 10000, drift = 1.005): BacktestResult {
  const portfolioValues = dates.map((_, i) => startValue * Math.pow(drift, i));
  return {
    dates,
    portfolioValues,
    returns: dates.map((_, i) => (i > 0 ? drift - 1 : 0)),
    drawdowns: dates.map((_, i) => (i < 10 ? -0.02 * i : -0.2 - 0.01 * (i - 10))),
    holdings: dates.map(() => ({ SPY: 46, GLD: 38, TLT: 16 })),
    trades: [],
  };
}

// Monthly grid spanning the fixed CRISIS_PERIODS (2008 / 2020 / 2022).
function monthlyDates(start: [number, number], months: number): string[] {
  return Array.from({ length: months }, (_, m) => {
    const d = new Date(Date.UTC(start[0], start[1] - 1 + m, 15));
    return d.toISOString().slice(0, 10);
  });
}

const CRISIS_DATES = monthlyDates([2007, 1], 204); // 2007-01 .. 2023-12
const CRISIS_RESULTS = [
  { name: 'SPY/GLD/TLT', result: backtestResult(CRISIS_DATES), color: '#3b82f6' },
  { name: '60/40 (S&P 500)', result: backtestResult(CRISIS_DATES, 10000, 1.004), color: '#10b981' },
];

const SHORT_DATES = monthlyDates([2026, 1], 5);
const SHORT_RESULTS = [
  { name: 'SPY/GLD/TLT', result: backtestResult(SHORT_DATES), metrics: metrics(), color: '#3b82f6' },
  { name: '60/40 (S&P 500)', result: backtestResult(SHORT_DATES, 10000, 1.003), metrics: metrics({ cagr: 0.061 }), color: '#10b981' },
];

const INCIDENTS: DashboardIncident[] = [
  {
    id: 'inc-1',
    tab: 'risk',
    severity: 'critical',
    attention: 'action',
    title: 'IC decay critical',
    source: 'ensemble_gold',
    currentValue: '-0.31',
    threshold: '-0.2',
    message: 'Rolling IC below floor for 14 days.',
    nextAction: 'Review regime exposure',
    timestamp: '2026-08-16T00:00:00Z',
  },
  {
    id: 'inc-2',
    tab: 'health',
    severity: 'info',
    attention: 'advisory',
    title: 'Kill switch disarmed',
    source: 'operator',
    message: 'Operator resolution, level clear.',
  },
];

const PORTFOLIOS: PortfolioConfig[] = [
  { name: 'SPY/GLD/TLT', allocation: { SPY: 0.46, GLD: 0.38, TLT: 0.16 }, rebalanceFrequency: 'monthly' },
  { name: '60/40 (S&P 500)', allocation: { SPY: 0.6, TLT: 0.4 }, rebalanceFrequency: 'quarterly' },
];

const COLORS = ['#3b82f6', '#10b981'];

describe('panel render coverage (10 live components, I15)', () => {
  it('ComparisonTable renders rows with formatted metrics', () => {
    const html = renderToStaticMarkup(<ComparisonTable results={RESULT_ROWS} />);
    expect(html).toContain('comparison-table');
    expect(html).toContain('Detailed Comparison');
    expect(html).toContain('SPY/GLD/TLT');
    expect(html).toContain('8.2%'); // cagr 0.082 formatted
    expect(html).toContain('0.68'); // sharpeRatio formatted
  });

  it('CorrelationMatrix renders the daily-return matrix for key symbols', () => {
    const html = renderToStaticMarkup(<CorrelationMatrix priceData={DAILY_30D} />);
    expect(html).toContain('Asset Correlation Matrix (Daily Returns)');
    expect(html).toContain('SPY');
    expect(html).toContain('GLD');
    expect(html).toContain('TLT');
    expect(html).toContain('1.00'); // diagonal cell
  });

  it('CrisisAnalysis renders both crisis-period charts', () => {
    const html = renderToStaticMarkup(<CrisisAnalysis results={CRISIS_RESULTS} />);
    expect(html).toContain('chart-container');
    expect(html).toContain('Crisis Period Performance (% Return)');
    expect(html).toContain('Max Drawdown During Crisis Periods');
    // Period rows live in recharts' client-rendered data, not SSR markup
    expect((html.match(/<h3/g) ?? []).length).toBe(2);
  });

  it('DrawdownChart renders heading and chart container', () => {
    const html = renderToStaticMarkup(<DrawdownChart results={SHORT_RESULTS} />);
    expect(html).toContain('chart-container');
    expect(html).toContain('Drawdown History (%)');
  });

  it('EquityCurve renders heading and chart container', () => {
    const html = renderToStaticMarkup(<EquityCurve results={SHORT_RESULTS} />);
    expect(html).toContain('chart-container');
    expect(html).toContain('Equity Curve (Normalized to 1.0)');
  });

  it('IncidentSummary renders incidents with severity classes', () => {
    const html = renderToStaticMarkup(
      <IncidentSummary title="Active incidents" incidents={INCIDENTS} showTab />,
    );
    expect(html).toContain('incident-summary');
    expect(html).toContain('aria-label="Active incidents"');
    expect(html).toContain('incident-row-critical');
    expect(html).toContain('IC decay critical');
    expect(html).toContain('2 active');
  });

  it('IncidentSummary renders selectable rows as buttons', () => {
    const html = renderToStaticMarkup(
      <IncidentSummary
        title="Active incidents"
        incidents={INCIDENTS}
        onIncidentSelect={() => undefined}
      />,
    );
    expect(html).toContain('<button');
    expect(html).toContain('incident-row-critical');
    expect(html).toContain('Next: Review regime exposure');
  });

  it('MetricsCards renders all five metric cards', () => {
    const html = renderToStaticMarkup(<MetricsCards results={RESULT_ROWS} />);
    expect(html).toContain('metrics-grid');
    expect(html).toContain('CAGR');
    expect(html).toContain('Sharpe Ratio');
    expect(html).toContain('Max Drawdown');
    expect((html.match(/metric-card/g) ?? []).length).toBe(5);
  });

  it('PortfolioSelector renders categories with checked selection', () => {
    const html = renderToStaticMarkup(
      <PortfolioSelector
        portfolios={PORTFOLIOS}
        selected={['SPY/GLD/TLT']}
        onToggle={() => undefined}
        colors={COLORS}
      />,
    );
    expect(html).toContain('portfolio-selector');
    expect(html).toContain('Select Portfolios to Compare');
    expect(html).toContain('★ Winners');
    expect(html).toContain('Traditional');
    expect(html).toContain('SPY 46%, GLD 38%, TLT 16%');
    expect((html.match(/checked=""/g) ?? []).length).toBe(1);
  });

  it('RiskReturnChart renders scatter heading and container', () => {
    const html = renderToStaticMarkup(<RiskReturnChart results={RESULT_ROWS} />);
    expect(html).toContain('chart-container');
    expect(html).toContain('Risk-Return Scatter (Bubble = Max Drawdown)');
  });

  it('RollingWindow renders window table over synthetic price history', () => {
    const priceData: PriceData[] = ['SPY', 'GLD', 'TLT'].flatMap((symbol, i) =>
      monthlyDates([2005, 1], 264).map((date) => ({
        date,
        symbol,
        price: 100 + i * 20 + (parseInt(date.slice(0, 4), 10) - 2005) * 8,
      })),
    );
    const html = renderToStaticMarkup(
      <RollingWindow portfolios={PORTFOLIOS} priceData={priceData} colors={COLORS} />,
    );
    expect(html).toContain('Rolling-Window Sharpe Ratios');
    expect(html).toContain('Full (2005-2026)');
    expect(html).toContain('GFC (2007-2009)');
    expect(html).toContain('SPY/GLD/TLT');
  });
});
