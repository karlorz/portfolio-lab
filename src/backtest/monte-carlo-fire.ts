/**
 * Monte Carlo FIRE Simulation — bootstrap resampling of daily returns
 * Answers: what's the probability of portfolio survival at each withdrawal rate?
 * Unlike our historical sweep (single path), Monte Carlo gives confidence intervals.
 * Uses block bootstrap (20-day blocks) to preserve autocorrelation structure.
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

interface SimulationResult {
  portfolio: string;
  withdrawalRate: number;
  simulations: number;
  successRate: number; // % where portfolio survives 30 years
  p10EndValue: number; // 10th percentile terminal value
  p25EndValue: number;
  p50EndValue: number; // median
  p75EndValue: number;
  p90EndValue: number;
  meanEndValue: number;
  worstCase: number; // minimum terminal value across all sims
  brokeRate: number; // % where portfolio goes to zero
  medianYearsToBroke: number | null; // for those that do go broke
}

const PORTFOLIOS: PortfolioConfig[] = [
  { name: 'SPY (S&P 500)', allocation: { SPY: 1 }, rebalanceFrequency: 'none' },
  { name: 'SPY/GLD 55/45', allocation: { SPY: 0.55, GLD: 0.45 }, rebalanceFrequency: 'annual' },
  { name: 'SPY/GLD/TLT 46/38/16', allocation: { SPY: 0.46, GLD: 0.38, TLT: 0.16 }, rebalanceFrequency: 'annual' },
  { name: 'SPY/GLD/TLT 50/35/15', allocation: { SPY: 0.50, GLD: 0.35, TLT: 0.15 }, rebalanceFrequency: 'annual' },
  { name: 'All Weather', allocation: { VTI: 0.30, TLT: 0.40, IEF: 0.15, GLD: 0.075, DBC: 0.075 }, rebalanceFrequency: 'annual' },
  { name: '60/40', allocation: { SPY: 0.6, AGG: 0.4 }, rebalanceFrequency: 'annual' },
];

const WITHDRAWAL_RATES = [0.03, 0.035, 0.04, 0.045, 0.05, 0.055, 0.06];
const NUM_SIMULATIONS = 1000;
const RETIREMENT_YEARS = 30;
const BLOCK_SIZE = 20; // ~1 month blocks to preserve autocorrelation
const INITIAL_AMOUNT = 1000000;

/**
 * Block bootstrap: resample daily returns in blocks to preserve autocorrelation.
 * Returns an array of simulated daily portfolio returns for RETIREMENT_YEARS.
 */
function blockBootstrapReturns(
  dailyReturns: number[],
  numDays: number,
  rng: () => number = Math.random,
): number[] {
  const result: number[] = [];
  const n = dailyReturns.length;
  while (result.length < numDays) {
    // Pick a random starting point
    const start = Math.floor(rng() * (n - BLOCK_SIZE));
    for (let i = 0; i < BLOCK_SIZE && result.length < numDays; i++) {
      result.push(dailyReturns[start + i]);
    }
  }
  return result.slice(0, numDays);
}

/**
 * Simulate a single retirement scenario with withdrawals.
 * Returns terminal value (0 if broke) and whether portfolio survived.
 */
function simulateRetirement(
  bootstrapReturns: number[],
  initialAmount: number,
  annualWithdrawalRate: number,
): { endValue: number; survived: boolean; brokeYear: number | null } {
  const monthlyWithdrawal = (initialAmount * annualWithdrawalRate) / 12;
  const tradingDaysPerMonth = 21;
  const totalDays = bootstrapReturns.length;

  let portfolio = initialAmount;
  let broke = false;
  let brokeYear: number | null = null;

  for (let i = 0; i < totalDays; i++) {
    // Apply daily return
    portfolio *= (1 + bootstrapReturns[i]);

    // Monthly withdrawal
    if (i > 0 && i % tradingDaysPerMonth === 0) {
      portfolio -= monthlyWithdrawal;
    }

    // Check if broke
    if (portfolio <= 0 && !broke) {
      broke = true;
      brokeYear = Math.floor(i / 252) + 1;
      break;
    }
  }

  return {
    endValue: broke ? 0 : portfolio,
    survived: !broke,
    brokeYear,
  };
}

async function main() {
  console.log('Loading price data...\n');
  const response = await fetch('file://' + new URL('../../public/data/prices.json', import.meta.url).pathname);
  const priceJson = await response.json() as Record<string, Array<{ d: string; p: number }>>;
  const priceData = toBacktestData(priceJson);

  const engine = new BacktestEngine();
  engine.loadData(priceData);

  const dates = priceData.map(p => p.date).sort();
  const startDate = dates[0];
  const endDate = dates[dates.length - 1];

  console.log('=== MONTE CARLO FIRE SIMULATION ===\n');
  console.log(`${NUM_SIMULATIONS} simulations × ${RETIREMENT_YEARS}-year retirement`);
  console.log(`Block bootstrap with ${BLOCK_SIZE}-day blocks (preserves autocorrelation)`);
  console.log(`Starting portfolio: $${(INITIAL_AMOUNT / 1000).toFixed(0)}k\n`);

  const allResults: SimulationResult[] = [];
  const totalDays = RETIREMENT_YEARS * 252;

  for (const config of PORTFOLIOS) {
    // Get historical daily returns for this portfolio
    const result = engine.runBacktest(config, startDate, endDate, 10000);
    const dailyReturns = result.returns.slice(1); // Skip first 0

    console.log(`\n${'='.repeat(70)}`);
    console.log(`${config.name} (${dailyReturns.length} historical daily returns)`);
    console.log('='.repeat(70));

    for (const rate of WITHDRAWAL_RATES) {
      const endValues: number[] = [];
      const brokeYears: number[] = [];
      let survived = 0;

      for (let sim = 0; sim < NUM_SIMULATIONS; sim++) {
        const bootstrapReturns = blockBootstrapReturns(dailyReturns, totalDays);
        const { endValue, survived: didSurvive, brokeYear } = simulateRetirement(
          bootstrapReturns,
          INITIAL_AMOUNT,
          rate,
        );

        if (didSurvive) {
          survived++;
          endValues.push(endValue);
        } else {
          endValues.push(0);
          if (brokeYear !== null) brokeYears.push(brokeYear);
        }
      }

      // Sort for percentiles
      endValues.sort((a, b) => a - b);

      const simResult: SimulationResult = {
        portfolio: config.name,
        withdrawalRate: rate,
        simulations: NUM_SIMULATIONS,
        successRate: survived / NUM_SIMULATIONS,
        p10EndValue: endValues[Math.floor(NUM_SIMULATIONS * 0.10)] ?? 0,
        p25EndValue: endValues[Math.floor(NUM_SIMULATIONS * 0.25)] ?? 0,
        p50EndValue: endValues[Math.floor(NUM_SIMULATIONS * 0.50)] ?? 0,
        p75EndValue: endValues[Math.floor(NUM_SIMULATIONS * 0.75)] ?? 0,
        p90EndValue: endValues[Math.floor(NUM_SIMULATIONS * 0.90)] ?? 0,
        meanEndValue: endValues.reduce((a, b) => a + b, 0) / NUM_SIMULATIONS,
        worstCase: endValues[0] ?? 0,
        brokeRate: (NUM_SIMULATIONS - survived) / NUM_SIMULATIONS,
        medianYearsToBroke: brokeYears.length > 0
          ? brokeYears.sort((a, b) => a - b)[Math.floor(brokeYears.length / 2)]
          : null,
      };
      allResults.push(simResult);

      const successPct = (simResult.successRate * 100).toFixed(0);
      const p50 = `$${(simResult.p50EndValue / 1000).toFixed(0)}k`;
      const p10 = `$${(simResult.p10EndValue / 1000).toFixed(0)}k`;
      const p90 = `$${(simResult.p90EndValue / 1000).toFixed(0)}k`;

      console.log(`  ${(rate * 100).toFixed(1)}%: success ${successPct}% | median ${p50} | 10th pct ${p10} | 90th pct ${p90}${brokeYears.length > 0 ? ` | broke median yr ${simResult.medianYearsToBroke}` : ''}`);
    }
  }

  // Summary table: success rates
  console.log('\n\n' + '='.repeat(80));
  console.log('SUCCESS RATE SUMMARY (% of simulations where portfolio survives 30 years)');
  console.log('='.repeat(80));

  // Header
  let header = '| Portfolio |';
  for (const rate of WITHDRAWAL_RATES) {
    header += ` ${(rate * 100).toFixed(1)}% |`;
  }
  console.log(header);
  console.log('-'.repeat(header.length));

  for (const config of PORTFOLIOS) {
    let row = `| ${config.name} |`;
    for (const rate of WITHDRAWAL_RATES) {
      const r = allResults.find(x => x.portfolio === config.name && x.withdrawalRate === rate);
      const pct = r ? (r.successRate * 100).toFixed(0) : '—';
      row += ` ${pct.padStart(3)}% |`;
    }
    console.log(row);
  }

  // Terminal value distribution at 4% and 5%
  for (const rate of [0.04, 0.05]) {
    console.log(`\n\n${(rate * 100).toFixed(1)}% WITHDRAWAL — TERMINAL VALUE DISTRIBUTION (30 years, $1M start)`);
    console.log('| Portfolio | P10 | P25 | P50 (median) | P75 | P90 | Mean | Broke% |');
    console.log('|-----------|-----|-----|--------------|-----|-----|------|--------|');

    for (const config of PORTFOLIOS) {
      const r = allResults.find(x => x.portfolio === config.name && x.withdrawalRate === rate);
      if (!r) continue;
      const fmt = (v: number) => `$${(v / 1000).toFixed(0)}k`;
      console.log(`| ${r.portfolio} | ${fmt(r.p10EndValue)} | ${fmt(r.p25EndValue)} | ${fmt(r.p50EndValue)} | ${fmt(r.p75EndValue)} | ${fmt(r.p90EndValue)} | ${fmt(r.meanEndValue)} | ${(r.brokeRate * 100).toFixed(0)}% |`);
    }
  }

  // Safe withdrawal rate at different confidence levels
  console.log('\n\n' + '='.repeat(80));
  console.log('SAFE WITHDRAWAL RATE BY CONFIDENCE LEVEL');
  console.log('(Maximum rate where ≥X% of simulations survive 30 years)');
  console.log('='.repeat(80));

  for (const confidence of [0.95, 0.90, 0.80, 0.70]) {
    console.log(`\n${(confidence * 100).toFixed(0)}% confidence:`);
    for (const config of PORTFOLIOS) {
      const configResults = allResults.filter(r => r.portfolio === config.name);
      // Find the highest rate where successRate >= confidence
      let safeRate = 0;
      for (const r of configResults) {
        if (r.successRate >= confidence) {
          safeRate = Math.max(safeRate, r.withdrawalRate);
        }
      }
      console.log(`  ${config.name}: ${(safeRate * 100).toFixed(1)}%`);
    }
  }

  // Comparison: historical vs Monte Carlo
  console.log('\n\n' + '='.repeat(80));
  console.log('HISTORICAL vs MONTE CARLO COMPARISON (4% withdrawal, 30 years)');
  console.log('='.repeat(80));
  console.log('| Portfolio | Historical End | MC Median | MC P10 | MC P90 | MC Success% |');
  console.log('|-----------|----------------|-----------|--------|--------|-------------|');

  // Compute historical 30-year result
  const histStart = '2005-01-03';
  const histEnd30yr = '2035-01-01'; // We only have 20 years of data, so use full period
  // Actually use the longest available: 2005-2026 = 21 years
  for (const config of PORTFOLIOS) {
    const result = engine.runBacktest(config, startDate, endDate, INITIAL_AMOUNT);
    const values = result.portfolioValues;

    // Simulate 4% withdrawal
    let portfolio = INITIAL_AMOUNT;
    for (let i = 0; i < values.length; i++) {
      if (i > 0) portfolio *= (values[i] / values[i - 1]);
      if (i > 0 && i % 21 === 0) portfolio -= (INITIAL_AMOUNT * 0.04) / 12;
      if (portfolio <= 0) break;
    }

    const mcResult = allResults.find(r => r.portfolio === config.name && r.withdrawalRate === 0.04);
    if (!mcResult) continue;

    const fmt = (v: number) => `$${(v / 1000).toFixed(0)}k`;
    console.log(`| ${config.name} | ${fmt(portfolio)} | ${fmt(mcResult.p50EndValue)} | ${fmt(mcResult.p10EndValue)} | ${fmt(mcResult.p90EndValue)} | ${(mcResult.successRate * 100).toFixed(0)}% |`);
  }
}

main().catch(console.error);
