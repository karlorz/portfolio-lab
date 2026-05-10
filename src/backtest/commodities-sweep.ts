/**
 * Commodities Sweep — test DBC as partial GLD replacement
 * DBC has data from 2006, so we test on 2006-2026 overlapping period
 */
import { BacktestEngine } from './engine';
import type { PortfolioConfig, PriceData } from './engine';
import * as fs from 'fs';

const DBC_PORTFOLIOS: PortfolioConfig[] = [
  // Base champion (no DBC)
  {
    name: 'SPY/GLD/TLT 46/38/16 (base)',
    allocation: { SPY: 0.46, GLD: 0.38, TLT: 0.16 },
    rebalanceFrequency: 'annual',
  },
  // Replace some GLD with DBC
  {
    name: 'SPY/GLD/DBC/TLT 46/30/8/16',
    allocation: { SPY: 0.46, GLD: 0.30, DBC: 0.08, TLT: 0.16 },
    rebalanceFrequency: 'annual',
  },
  {
    name: 'SPY/GLD/DBC/TLT 46/28/10/16',
    allocation: { SPY: 0.46, GLD: 0.28, DBC: 0.10, TLT: 0.16 },
    rebalanceFrequency: 'annual',
  },
  {
    name: 'SPY/GLD/DBC/TLT 46/26/12/16',
    allocation: { SPY: 0.46, GLD: 0.26, DBC: 0.12, TLT: 0.16 },
    rebalanceFrequency: 'annual',
  },
  {
    name: 'SPY/GLD/DBC/TLT 46/24/14/16',
    allocation: { SPY: 0.46, GLD: 0.24, DBC: 0.14, TLT: 0.16 },
    rebalanceFrequency: 'annual',
  },
  {
    name: 'SPY/GLD/DBC/TLT 46/22/16/16',
    allocation: { SPY: 0.46, GLD: 0.22, DBC: 0.16, TLT: 0.16 },
    rebalanceFrequency: 'annual',
  },
  {
    name: 'SPY/GLD/DBC/TLT 46/20/18/16',
    allocation: { SPY: 0.46, GLD: 0.20, DBC: 0.18, TLT: 0.16 },
    rebalanceFrequency: 'annual',
  },
  {
    name: 'SPY/GLD/DBC/TLT 46/18/20/16',
    allocation: { SPY: 0.46, GLD: 0.18, DBC: 0.20, TLT: 0.16 },
    rebalanceFrequency: 'annual',
  },
  // Also test All Weather-style DBC allocation
  {
    name: 'SPY/GLD/DBC/TLT 30/38/16/16',
    allocation: { SPY: 0.30, GLD: 0.38, DBC: 0.16, TLT: 0.16 },
    rebalanceFrequency: 'annual',
  },
  {
    name: 'SPY/GLD/DBC/TLT/IEF 30/30/8/16/16',
    allocation: { SPY: 0.30, GLD: 0.30, DBC: 0.08, TLT: 0.16, IEF: 0.16 },
    rebalanceFrequency: 'annual',
  },
  {
    name: 'SPY (S&P 500)',
    allocation: { SPY: 1 },
    rebalanceFrequency: 'none',
  },
];

function loadData(): PriceData[] {
  const raw = fs.readFileSync(new URL('../../public/data/prices.json', import.meta.url).pathname, 'utf-8');
  const prices: Record<string, Array<{ d: string; p: number }>> = JSON.parse(raw);
  const result: PriceData[] = [];
  for (const [symbol, entries] of Object.entries(prices)) {
    for (const { d, p } of entries) {
      result.push({ date: d, symbol, price: p });
    }
  }
  return result.sort((a, b) => a.date.localeCompare(b.date));
}

function runAnalysis() {
  const priceData = loadData();
  const engine = new BacktestEngine();
  engine.loadData(priceData);

  // DBC starts 2006-02-06
  const startDate = '2006-02-06';
  const dates = priceData.map(p => p.date).sort();
  const endDate = dates[dates.length - 1];

  console.log(`\n=== COMMODITIES SWEEP (DBC as partial GLD replacement) ===`);
  console.log(`Period: ${startDate} to ${endDate}`);
  console.log(`DBC data available from 2006-02-06\n`);

  console.log('| Portfolio | CAGR | Vol | Sharpe | Max DD | 2008 | 2020 | 2022 |');
  console.log('|-----------|------|-----|--------|--------|------|------|------|');

  for (const pf of DBC_PORTFOLIOS) {
    const result = engine.runBacktest(pf, startDate, endDate, 10000);
    const m = engine.calculateMetrics(result);
    const c2008 = getCrisisReturn(result, '2008-01-02', '2008-12-31');
    const c2020 = getCrisisReturn(result, '2020-02-19', '2020-03-23');
    const c2022 = getCrisisReturn(result, '2022-01-03', '2022-12-30');
    console.log(`| ${pf.name} | ${(m.cagr * 100).toFixed(1)}% | ${(m.volatility * 100).toFixed(1)}% | ${m.sharpeRatio.toFixed(2)} | ${(m.maxDrawdown * 100).toFixed(1)}% | ${c2008} | ${c2020} | ${c2022} |`);
  }
}

function getCrisisReturn(result: import('./engine').BacktestResult, start: string, end: string): string {
  const dates = result.dates;
  const values = result.portfolioValues;
  let startVal = 0;
  let endVal = 0;
  for (let i = 0; i < dates.length; i++) {
    if (dates[i]! >= start && startVal === 0) startVal = values[i]!;
    if (dates[i]! <= end) endVal = values[i]!;
  }
  if (startVal === 0) return 'N/A';
  const ret = ((endVal - startVal) / startVal) * 100;
  return `${ret.toFixed(1)}%`;
}

runAnalysis();
