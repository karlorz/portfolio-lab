/**
 * Rolling-Window Analysis — tests strategy robustness across sub-periods
 * Validates that grid-search winners aren't just artifacts of the full-period backtest.
 */
import { BacktestEngine } from './engine';
import type { PortfolioConfig, PriceData } from './engine';

function toBacktestData(prices: Record<string, Array<{ d: string; p: number }>>): PriceData[] {
  const result: PriceData[] = [];
  for (const [symbol, entries] of Object.entries(prices)) {
    for (const { d, p } of entries) {
      result.push({ date: d, symbol, price: p });
    }
  }
  return result.sort((a, b) => a.date.localeCompare(b.date));
}

const CHAMPIONS: PortfolioConfig[] = [
  { name: 'SPY/GLD/TLT 46/38/16', allocation: { SPY: 0.46, GLD: 0.38, TLT: 0.16 }, rebalanceFrequency: 'annual' },
  { name: 'SPY/GLD/TLT 50/35/15', allocation: { SPY: 0.50, GLD: 0.35, TLT: 0.15 }, rebalanceFrequency: 'annual' },
  { name: 'SPY/GLD 55/45', allocation: { SPY: 0.55, GLD: 0.45 }, rebalanceFrequency: 'annual' },
  { name: 'SPY (S&P 500)', allocation: { SPY: 1 }, rebalanceFrequency: 'none' },
  { name: 'Golden Butterfly', allocation: { VTI: 0.20, VBR: 0.20, TLT: 0.20, SHY: 0.20, GLD: 0.20 }, rebalanceFrequency: 'annual' },
  { name: 'All Weather', allocation: { VTI: 0.30, TLT: 0.40, IEF: 0.15, GLD: 0.075, DBC: 0.075 }, rebalanceFrequency: 'annual' },
];

// International variants of champion
const INTL_VARIANTS: PortfolioConfig[] = [
  { name: 'SPY/EFA/GLD/TLT 36/10/38/16', allocation: { SPY: 0.36, EFA: 0.10, GLD: 0.38, TLT: 0.16 }, rebalanceFrequency: 'annual' },
  { name: 'VTI/VXUS/GLD/TLT 36/10/38/16', allocation: { VTI: 0.36, VXUS: 0.10, GLD: 0.38, TLT: 0.16 }, rebalanceFrequency: 'annual' },
];

const WINDOWS = [
  { name: 'Full (2005-2026)', start: '2005-01-01', end: '2026-12-31' },
  { name: 'Pre-GFC (2005-2007)', start: '2005-01-01', end: '2007-12-31' },
  { name: 'GFC (2007-2009)', start: '2007-10-01', end: '2009-06-30' },
  { name: 'Recovery (2009-2013)', start: '2009-03-01', end: '2013-12-31' },
  { name: 'Mid-cycle (2013-2019)', start: '2013-01-01', end: '2019-12-31' },
  { name: '2020 COVID', start: '2020-01-01', end: '2020-12-31' },
  { name: 'Post-COVID (2020-2026)', start: '2020-03-01', end: '2026-12-31' },
  { name: 'Rate hikes (2022-2023)', start: '2022-01-01', end: '2023-12-31' },
  { name: '2025-2026 YTD', start: '2025-01-01', end: '2026-12-31' },
];

async function main() {
  console.log('Loading extended price data (2005-2026)...');
  const response = await fetch('file://' + new URL('../../public/data/prices.json', import.meta.url).pathname);
  const priceJson = await response.json() as Record<string, Array<{ d: string; p: number }>>;
  const priceData = toBacktestData(priceJson);

  const engine = new BacktestEngine();
  engine.loadData(priceData);

  const allConfigs = [...CHAMPIONS, ...INTL_VARIANTS];

  // Header
  console.log('\n=== ROLLING-WINDOW ANALYSIS (2005-2026) ===\n');

  for (const window of WINDOWS) {
    console.log(`\n--- ${window.name} (${window.start} to ${window.end}) ---`);
    console.log('| Portfolio | CAGR | Vol | Sharpe | Max DD |');
    console.log('|-----------|------|-----|--------|--------|');

    for (const config of allConfigs) {
      const result = engine.runBacktest(config, window.start, window.end, 10000);
      const metrics = engine.calculateMetrics(result);
      if (metrics.cagr === 0 && metrics.volatility === 0) continue; // Skip if no data
      const cagr = (metrics.cagr * 100).toFixed(1);
      const vol = (metrics.volatility * 100).toFixed(1);
      const sharpe = metrics.sharpeRatio.toFixed(2);
      const maxDD = (metrics.maxDrawdown * 100).toFixed(1);
      console.log(`| ${config.name} | ${cagr}% | ${vol}% | ${sharpe} | ${maxDD}% |`);
    }
  }

  // Summary: how often does each portfolio beat SPY on Sharpe?
  console.log('\n=== SHARPE DOMINANCE: times beating SPY across windows ===');
  const spySharpes: number[] = [];
  const portfolioName = 'SPY (S&P 500)';
  for (const window of WINDOWS) {
    const result = engine.runBacktest(CHAMPIONS.find(c => c.name === portfolioName)!, window.start, window.end, 10000);
    const metrics = engine.calculateMetrics(result);
    spySharpes.push(metrics.sharpeRatio);
  }

  for (const config of allConfigs) {
    let beats = 0;
    for (let i = 0; i < WINDOWS.length; i++) {
      const result = engine.runBacktest(config, WINDOWS[i].start, WINDOWS[i].end, 10000);
      const metrics = engine.calculateMetrics(result);
      if (metrics.sharpeRatio > spySharpes[i]) beats++;
    }
    console.log(`  ${config.name}: beats SPY in ${beats}/${WINDOWS.length} windows`);
  }
}

main().catch(console.error);
