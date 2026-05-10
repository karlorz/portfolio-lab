/**
 * Factor Tilt Analysis — test momentum, value, and min-vol tilts
 * against the champion SPY/GLD/TLT 46/38/16
 *
 * Factor ETFs have limited history (MTUM/VLUE from 2013, USMV from 2011),
 * so we test on the overlapping 2013-2026 period for fair comparison.
 */
import { BacktestEngine } from './engine';
import type { PortfolioConfig, PriceData } from './engine';
import * as fs from 'fs';

const FACTOR_PORTFOLIOS: PortfolioConfig[] = [
  // Base champion (no factor)
  {
    name: 'SPY/GLD/TLT 46/38/16 (base)',
    allocation: { SPY: 0.46, GLD: 0.38, TLT: 0.16 },
    rebalanceFrequency: 'annual',
  },
  // Momentum tilt: replace ~20% of SPY with MTUM
  {
    name: 'Momentum Tilt SPY/MTUM/GLD/TLT 30/20/34/16',
    allocation: { SPY: 0.30, MTUM: 0.20, GLD: 0.34, TLT: 0.16 },
    rebalanceFrequency: 'annual',
  },
  // Value tilt: replace ~20% of SPY with VLUE
  {
    name: 'Value Tilt SPY/VLUE/GLD/TLT 30/20/34/16',
    allocation: { SPY: 0.30, VLUE: 0.20, GLD: 0.34, TLT: 0.16 },
    rebalanceFrequency: 'annual',
  },
  // Minimum volatility tilt: replace ~20% of SPY with USMV
  {
    name: 'Min Vol Tilt SPY/USMV/GLD/TLT 30/20/34/16',
    allocation: { SPY: 0.30, USMV: 0.20, GLD: 0.34, TLT: 0.16 },
    rebalanceFrequency: 'annual',
  },
  // Multi-factor blend
  {
    name: 'Factor Blend SPY/MTUM/VLUE/USMV/GLD/TLT 20/10/10/10/34/16',
    allocation: { SPY: 0.20, MTUM: 0.10, VLUE: 0.10, USMV: 0.10, GLD: 0.34, TLT: 0.16 },
    rebalanceFrequency: 'annual',
  },
  // Just SPY for benchmark
  {
    name: 'SPY (S&P 500)',
    allocation: { SPY: 1 },
    rebalanceFrequency: 'none',
  },
];

// Wide factor sweep: test different factor weight levels
const FACTOR_SWEEP: PortfolioConfig[] = [];
for (const factor of ['MTUM', 'VLUE', 'USMV'] as const) {
  for (const factorWt of [0.10, 0.15, 0.20, 0.25, 0.30]) {
    const spyWt = 0.46 - factorWt;
    if (spyWt < 0.10) continue;
    const gldWt = 0.38;
    const tltWt = 0.16;
    FACTOR_SWEEP.push({
      name: `SPY/${factor}/GLD/TLT ${(spyWt * 100).toFixed(0)}/${(factorWt * 100).toFixed(0)}/${gldWt * 100}/${tltWt * 100}`,
      allocation: { SPY: spyWt, [factor]: factorWt, GLD: gldWt, TLT: tltWt },
      rebalanceFrequency: 'annual',
    });
  }
}

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

  // Find the earliest date where MTUM/VLUE have data (2013-04-18)
  const factorStart = '2013-04-18';
  const dates = priceData.map(p => p.date).sort();
  const endDate = dates[dates.length - 1];

  console.log(`\n=== FACTOR TILT ANALYSIS ===`);
  console.log(`Period: ${factorStart} to ${endDate}`);
  console.log(`Factor ETFs: MTUM (momentum), VLUE (value), USMV (min vol)\n`);

  // First: primary comparison
  console.log('=== PRIMARY COMPARISON ===\n');
  console.log('| Portfolio | CAGR | Vol | Sharpe | Max DD | 2020 | 2022 |');
  console.log('|-----------|------|-----|--------|--------|------|------|');

  for (const pf of FACTOR_PORTFOLIOS) {
    const result = engine.runBacktest(pf, factorStart, endDate, 10000);
    const m = engine.calculateMetrics(result);
    const crisis2020 = getCrisisReturn(result, pf, '2020-02-19', '2020-03-23');
    const crisis2022 = getCrisisReturn(result, pf, '2022-01-03', '2022-12-30');
    console.log(`| ${pf.name} | ${(m.cagr * 100).toFixed(1)}% | ${(m.volatility * 100).toFixed(1)}% | ${m.sharpeRatio.toFixed(2)} | ${(m.maxDrawdown * 100).toFixed(1)}% | ${crisis2020} | ${crisis2022} |`);
  }

  // Factor sweep
  console.log(`\n=== FACTOR WEIGHT SWEEP ===\n`);
  console.log('| Portfolio | CAGR | Vol | Sharpe | Max DD |');
  console.log('|-----------|------|-----|--------|--------|');

  for (const pf of FACTOR_SWEEP) {
    const result = engine.runBacktest(pf, factorStart, endDate, 10000);
    const m = engine.calculateMetrics(result);
    console.log(`| ${pf.name} | ${(m.cagr * 100).toFixed(1)}% | ${(m.volatility * 100).toFixed(1)}% | ${m.sharpeRatio.toFixed(2)} | ${(m.maxDrawdown * 100).toFixed(1)}% |`);
  }

  // Summary
  console.log('\n=== SUMMARY ===\n');
  const baseResult = engine.runBacktest(FACTOR_PORTFOLIOS[0], factorStart, endDate, 10000);
  const baseMetrics = engine.calculateMetrics(baseResult);
  console.log(`Base (SPY/GLD/TLT 46/38/16): CAGR ${(baseMetrics.cagr * 100).toFixed(1)}%, Sharpe ${baseMetrics.sharpeRatio.toFixed(2)}`);

  for (const pf of FACTOR_PORTFOLIOS.slice(1, -1)) {
    const result = engine.runBacktest(pf, factorStart, endDate, 10000);
    const m = engine.calculateMetrics(result);
    const cagrDiff = (m.cagr - baseMetrics.cagr) * 100;
    const sharpeDiff = m.sharpeRatio - baseMetrics.sharpeRatio;
    console.log(`${pf.name}: CAGR ${(m.cagr * 100).toFixed(1)}% (${cagrDiff >= 0 ? '+' : ''}${cagrDiff.toFixed(1)}pp), Sharpe ${m.sharpeRatio.toFixed(2)} (${sharpeDiff >= 0 ? '+' : ''}${sharpeDiff.toFixed(2)})`);
  }
}

function getCrisisReturn(
  result: import('./engine').BacktestResult,
  _pf: PortfolioConfig,
  start: string,
  end: string
): string {
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
