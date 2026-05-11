/**
 * Grid Search — sweeps allocation combinations near winning regions
 * Finds optimal SPY/GLD/TLT mixes that meet the success criterion:
 * CAGR ≥ 90% of SPY, Volatility ≤ 70% of SPY
 * Prioritizes Sharpe ratio and crisis resilience.
 */

import { BacktestEngine } from './engine';
import type { PortfolioConfig, PriceData, PerformanceMetrics } from './engine';
import { batchCalculateDSR, flagOverfitConfigs, estimateIndependentTrials } from '../utils/dsr-calculator';

interface GridResult {
  name: string;
  allocation: Record<string, number>;
  rebalanceFrequency: PortfolioConfig['rebalanceFrequency'];
  trendFollowing?: PortfolioConfig['trendFollowing'];
  volatilityTarget?: PortfolioConfig['volatilityTarget'];
  metrics: PerformanceMetrics;
  crisis2008: number;
  crisis2020: number;
  crisis2022: number;
  meetsTarget: boolean;
  sharpeRank: number;
  returns: number[];
  permutationPValue?: number;
  dsr?: number;
  isSignificant?: boolean;
}

function toBacktestData(prices: Record<string, Array<{ d: string; p: number }>>): PriceData[] {
  const result: PriceData[] = [];
  for (const [symbol, entries] of Object.entries(prices)) {
    for (const { d, p } of entries) {
      result.push({ date: d, symbol, price: p });
    }
  }
  return result.sort((a, b) => a.date.localeCompare(b.date));
}

function computeCrisisReturn(engine: BacktestEngine, config: PortfolioConfig, startDate: string, endDate: string): number {
  const result = engine.runBacktest(config, startDate, endDate, 10000);
  if (result.portfolioValues.length < 2) return 0;
  return (result.portfolioValues[result.portfolioValues.length - 1] / result.portfolioValues[0]) - 1;
}

// Grid search configurations
function generateConfigs(): PortfolioConfig[] {
  const configs: PortfolioConfig[] = [];

  // Region 1: SPY/GLD sweep (40/60 to 70/30 in 5% steps)
  for (let spy = 40; spy <= 70; spy += 5) {
    const gld = 100 - spy;
    configs.push({
      name: `SPY/GLD ${spy}/${gld}`,
      allocation: { SPY: spy / 100, GLD: gld / 100 },
      rebalanceFrequency: 'annual',
    });
  }

  // Region 2: SPY/GLD/TLT sweep — TLT 5-20%, SPY 50-65%, GLD remainder
  for (let tlt = 5; tlt <= 20; tlt += 5) {
    for (let spy = 50; spy <= 65; spy += 5) {
      const gld = 100 - spy - tlt;
      if (gld < 10 || gld > 60) continue;
      configs.push({
        name: `SPY/GLD/TLT ${spy}/${gld}/${tlt}`,
        allocation: { SPY: spy / 100, GLD: gld / 100, TLT: tlt / 100 },
        rebalanceFrequency: 'annual',
      });
    }
  }

  // Region 3: SPY/GLD/IEF sweep — shorter-duration bond hedge
  for (let ief = 5; ief <= 15; ief += 5) {
    for (let spy = 50; spy <= 65; spy += 5) {
      const gld = 100 - spy - ief;
      if (gld < 10 || gld > 60) continue;
      configs.push({
        name: `SPY/GLD/IEF ${spy}/${gld}/${ief}`,
        allocation: { SPY: spy / 100, GLD: gld / 100, IEF: ief / 100 },
        rebalanceFrequency: 'annual',
      });
    }
  }

  // Region 4: Trend-following on top SPY/GLD candidates
  for (const [spy, gld] of [[50, 50], [55, 45], [60, 40]]) {
    configs.push({
      name: `SPY/GLD ${spy}/${gld} +Trend`,
      allocation: { SPY: spy / 100, GLD: gld / 100 },
      rebalanceFrequency: 'monthly',
      trendFollowing: { enabled: true, lookbackMonths: 12, movingAverageMonths: 10 },
    });
  }

  // Region 5: SPY/GLD/TLT +Trend on promising triples
  for (const [spy, gld, tlt] of [[55, 35, 10], [58, 32, 10], [60, 30, 10]]) {
    configs.push({
      name: `SPY/GLD/TLT ${spy}/${gld}/${tlt} +Trend`,
      allocation: { SPY: spy / 100, GLD: gld / 100, TLT: tlt / 100 },
      rebalanceFrequency: 'monthly',
      trendFollowing: { enabled: true, lookbackMonths: 12, movingAverageMonths: 10 },
    });
  }

  // Region 6: Quarterly rebalancing variants
  for (const [spy, gld, tlt] of [[55, 45, 0], [58, 32, 10]]) {
    const alloc: Record<string, number> = { SPY: spy / 100, GLD: gld / 100 };
    if (tlt > 0) alloc.TLT = tlt / 100;
    configs.push({
      name: `SPY/GLD${tlt > 0 ? '/TLT' : ''} ${spy}/${gld}${tlt > 0 ? `/${tlt}` : ''} Q`,
      allocation: alloc,
      rebalanceFrequency: 'quarterly',
    });
  }

  // Region 7: VTI instead of SPY + small-cap value tilt (Golden Butterfly inspired)
  for (let vti = 25; vti <= 40; vti += 5) {
    for (let vbr = 10; vbr <= 20; vbr += 5) {
      const gld = 100 - vti - vbr - 20; // 20% TLT
      if (gld < 15 || gld > 45) continue;
      configs.push({
        name: `VTI/VBR/TLT/GLD ${vti}/${vbr}/20/${gld}`,
        allocation: { VTI: vti / 100, VBR: vbr / 100, TLT: 0.20, GLD: gld / 100 },
        rebalanceFrequency: 'annual',
      });
    }
  }

  // Region 8: Fine sweep around top 3 winners (2% steps)
  for (let spy = 46; spy <= 54; spy += 2) {
    for (let tlt = 10; tlt <= 20; tlt += 2) {
      const gld = 100 - spy - tlt;
      if (gld < 25 || gld > 45) continue;
      configs.push({
        name: `SPY/GLD/TLT ${spy}/${gld}/${tlt}`,
        allocation: { SPY: spy / 100, GLD: gld / 100, TLT: tlt / 100 },
        rebalanceFrequency: 'annual',
      });
    }
  }

  // Region 9: Volatility targeting on top winners
  for (const base of [
    { spy: 50, gld: 35, tlt: 15 },
    { spy: 50, gld: 40, tlt: 10 },
    { spy: 55, gld: 45, tlt: 0 },
  ]) {
    const alloc: Record<string, number> = { SPY: base.spy / 100, GLD: base.gld / 100 };
    if (base.tlt > 0) alloc.TLT = base.tlt / 100;
    for (const targetVol of [0.10, 0.12, 0.14]) {
      configs.push({
        name: `SPY/GLD${base.tlt > 0 ? '/TLT' : ''} ${base.spy}/${base.gld}${base.tlt > 0 ? `/${base.tlt}` : ''} +Vol${(targetVol * 100).toFixed(0)}`,
        allocation: alloc,
        rebalanceFrequency: 'quarterly',
        volatilityTarget: { enabled: true, targetVol },
      });
    }
  }

  return configs;
}

async function main() {
  // Load price data
  console.log('Loading price data...');
  const response = await fetch('file://' + new URL('../../public/data/prices.json', import.meta.url).pathname);
  const priceJson = await response.json() as Record<string, Array<{ d: string; p: number }>>;
  const priceData = toBacktestData(priceJson);

  const engine = new BacktestEngine();
  engine.loadData(priceData);

  const dates = priceData.map(p => p.date).sort();
  const startDate = dates[0];
  const endDate = dates[dates.length - 1];

  // Get SPY benchmark metrics
  const spyResult = engine.runBacktest(
    { name: 'SPY', allocation: { SPY: 1 }, rebalanceFrequency: 'none' },
    startDate, endDate, 10000,
  );
  const spyMetrics = engine.calculateMetrics(spyResult);
  const spyCagr = spyMetrics.cagr;
  const spyVol = spyMetrics.volatility;
  console.log(`SPY benchmark: CAGR ${(spyCagr * 100).toFixed(1)}%, Vol ${(spyVol * 100).toFixed(1)}%`);
  console.log(`Target: CAGR ≥ ${(spyCagr * 0.9 * 100).toFixed(1)}%, Vol ≤ ${(spyVol * 0.7 * 100).toFixed(1)}%\n`);

  // Generate and run all configs
  const configs = generateConfigs();
  console.log(`Running ${configs.length} portfolio configurations...\n`);

  const results: GridResult[] = [];
  const crisisPeriods = [
    { name: '2008', start: '2007-10-01', end: '2009-03-31' },
    { name: '2020', start: '2020-02-01', end: '2020-03-31' },
    { name: '2022', start: '2022-01-01', end: '2022-12-31' },
  ];

  for (const config of configs) {
    const result = engine.runBacktest(config, startDate, endDate, 10000);
    const metrics = engine.calculateMetrics(result);

    const crisis2008 = computeCrisisReturn(engine, config, crisisPeriods[0].start, crisisPeriods[0].end);
    const crisis2020 = computeCrisisReturn(engine, config, crisisPeriods[1].start, crisisPeriods[1].end);
    const crisis2022 = computeCrisisReturn(engine, config, crisisPeriods[2].start, crisisPeriods[2].end);

    const meetsTarget = metrics.cagr >= spyCagr * 0.9 && metrics.volatility <= spyVol * 0.7;

    results.push({
      name: config.name,
      allocation: config.allocation,
      rebalanceFrequency: config.rebalanceFrequency,
      trendFollowing: config.trendFollowing,
      volatilityTarget: config.volatilityTarget,
      metrics,
      crisis2008,
      crisis2020,
      crisis2022,
      meetsTarget,
      sharpeRank: 0,
      returns: result.returns,
    });
  }

  // Rank by Sharpe
  const sorted = [...results].sort((a, b) => b.metrics.sharpeRatio - a.metrics.sharpeRatio);
  sorted.forEach((r, i) => r.sharpeRank = i + 1);

  // Calculate DSR for top 10 configurations
  console.log('\n=== STATISTICAL VALIDATION (Deflated Sharpe Ratio) ===\n');
  const top10 = sorted.slice(0, 10);
  const totalTrials = results.length;
  const effectiveTrials = estimateIndependentTrials(totalTrials, 0.7);

  const dsrResults = batchCalculateDSR(
    top10.map(r => ({
      name: r.name,
      sharpe: r.metrics.sharpeRatio,
      returns: r.returns,
    })),
    effectiveTrials
  );

  console.log('| Portfolio | Sharpe | DSR | p-value | Significant |');
  console.log('|-----------|--------|-----|---------|-------------|');
  for (const r of dsrResults) {
    const sig = r.isSignificant ? '✅' : '❌';
    console.log(`| ${r.name} | ${r.sharpe.toFixed(2)} | ${r.dsr.toFixed(2)} | ${r.pValue.toFixed(3)} | ${sig} |`);
  }

  // Flag overfit configurations
  const overfitAnalysis = flagOverfitConfigs(dsrResults, 0);
  console.log(`\nOverfit Analysis: ${(overfitAnalysis.overfitRatio * 100).toFixed(0)}% of top 10 configs flagged as potentially overfit`);
  if (overfitAnalysis.likelyOverfit.length > 0) {
    console.log('Flagged:', overfitAnalysis.likelyOverfit.join(', '));
  }

  // Print all results
  console.log('=== ALL CONFIGURATIONS ===\n');
  console.log('| # | Portfolio | CAGR | Vol | Sharpe | MaxDD | 2008 | 2020 | 2022 | Target |');
  console.log('|---|-----------|------|-----|--------|-------|------|------|------|--------|');
  for (const r of sorted) {
    const cagr = (r.metrics.cagr * 100).toFixed(1);
    const vol = (r.metrics.volatility * 100).toFixed(1);
    const sharpe = r.metrics.sharpeRatio.toFixed(2);
    const maxDD = (r.metrics.maxDrawdown * 100).toFixed(1);
    const c08 = (r.crisis2008 * 100).toFixed(1);
    const c20 = (r.crisis2020 * 100).toFixed(1);
    const c22 = (r.crisis2022 * 100).toFixed(1);
    const target = r.meetsTarget ? '✅' : '❌';
    console.log(`| ${r.sharpeRank} | ${r.name} | ${cagr}% | ${vol}% | ${sharpe} | ${maxDD}% | ${c08}% | ${c20}% | ${c22}% | ${target} |`);
  }

  // Print winners only
  const winners = sorted.filter(r => r.meetsTarget);
  console.log(`\n=== WINNERS (meet target: ${winners.length} of ${results.length}) ===\n`);
  if (winners.length > 0) {
    console.log('| # | Portfolio | CAGR | Vol | Sharpe | MaxDD | Calmar | Sortino | 2008 | 2020 | 2022 |');
    console.log('|---|-----------|------|-----|--------|-------|--------|---------|------|------|------|');
    for (const r of winners) {
      const cagr = (r.metrics.cagr * 100).toFixed(1);
      const vol = (r.metrics.volatility * 100).toFixed(1);
      const sharpe = r.metrics.sharpeRatio.toFixed(2);
      const maxDD = (r.metrics.maxDrawdown * 100).toFixed(1);
      const calmar = r.metrics.calmarRatio.toFixed(2);
      const sortino = r.metrics.sortinoRatio.toFixed(2);
      const c08 = (r.crisis2008 * 100).toFixed(1);
      const c20 = (r.crisis2020 * 100).toFixed(1);
      const c22 = (r.crisis2022 * 100).toFixed(1);
      console.log(`| ${r.sharpeRank} | ${r.name} | ${cagr}% | ${vol}% | ${sharpe} | ${maxDD}% | ${calmar} | ${sortino} | ${c08}% | ${c20}% | ${c22}% |`);
    }
  }

  // Output winners as JSON for programmatic use
  const winnersJson = winners.map(r => ({
    name: r.name,
    allocation: r.allocation,
    rebalanceFrequency: r.rebalanceFrequency,
    trendFollowing: r.trendFollowing,
    volatilityTarget: r.volatilityTarget,
    cagr: r.metrics.cagr,
    volatility: r.metrics.volatility,
    sharpeRatio: r.metrics.sharpeRatio,
    maxDrawdown: r.metrics.maxDrawdown,
    calmarRatio: r.metrics.calmarRatio,
    sortinoRatio: r.metrics.sortinoRatio,
    crisis2008: r.crisis2008,
    crisis2020: r.crisis2020,
    crisis2022: r.crisis2022,
  }));
  console.log('\n=== WINNERS JSON ===');
  console.log(JSON.stringify(winnersJson, null, 2));
}

main().catch(console.error);
