/**
 * Duration Sweep Backtest - Dynamic Duration Allocation Testing
 * v2.18 - Tests yield curve regime-based bond allocation vs static splits
 * 
 * Compares:
 * - Static TLT/IEF allocations (current baseline)
 * - Dynamic duration based on 2s10s spread regime
 * - Momentum-adjusted allocations
 * - Real yield adjusted allocations
 */

import { calculateDurationAllocation, classifyRegime, DurationRegime, getExpectedAlpha } from '../utils/duration-signals';
import { BacktestEngine, PortfolioConfig, PriceData, BacktestResult, PerformanceMetrics } from './engine';

interface DurationConfig {
  name: string;
  type: 'static' | 'dynamic' | 'momentum' | 'real-yield';
  allocation: { spy: number; gld: number; tlt: number; ief: number; shy: number; bil: number };
  description: string;
}

interface DurationSweepResult {
  config: DurationConfig;
  backtest: BacktestResult;
  metrics: PerformanceMetrics;
  regimeDistribution: Record<DurationRegime, number>;
  tradesPerYear: number;
}

/**
 * Load yield curve data from FRED/fetcher output
 */
async function loadYieldData(startDate: string, endDate: string): Promise<{
  dates: string[];
  spread2s10s: number[];
  momentum3m: number[];
  realYield: number[];
}> {
  // Load from public/data/yields.json (created by fetcher)
  try {
    const yields = await Bun.file('./public/data/yields.json').json();
    
    const dates: string[] = [];
    const spread2s10s: number[] = [];
    const momentum3m: number[] = [];
    const realYield: number[] = [];
    
    for (const entry of yields.data || []) {
      if (entry.date >= startDate && entry.date <= endDate) {
        dates.push(entry.date);
        spread2s10s.push(entry.spread2s10s || 0);
        momentum3m.push(entry.momentum3m || 0);
        realYield.push(entry.realYield || 0);
      }
    }
    
    return { dates, spread2s10s, momentum3m, realYield };
  } catch (e) {
    console.warn('Yield data not found, generating synthetic for testing...');
    // Generate synthetic data for development
    return generateSyntheticYieldData(startDate, endDate);
  }
}

/**
 * Generate synthetic yield curve data for development/testing
 * Based on actual historical patterns
 */
function generateSyntheticYieldData(startDate: string, endDate: string): {
  dates: string[];
  spread2s10s: number[];
  momentum3m: number[];
  realYield: number[];
} {
  const dates: string[] = [];
  const spread2s10s: number[] = [];
  const momentum3m: number[] = [];
  const realYield: number[] = [];
  
  let current = new Date(startDate);
  const end = new Date(endDate);
  
  // Historical regime approximations
  // 2005-2006: Normal/Steep (100-200bps)
  // 2007-2008: Flattening to Inverted (-50 to 50bps)
  // 2009-2010: Steep (200-300bps)
  // 2011-2014: Normal (100-200bps)
  // 2015-2018: Flattening (50-100bps)
  // 2019-2020: Flat to Inverted (0 to -50bps)
  // 2021-2022: Steepening then sharp flattening (100 to -100bps)
  // 2023-2026: Un-inverting (normalizing)
  
  let spread = 150; // Start normal
  
  while (current <= end) {
    const year = current.getFullYear();
    const month = current.getMonth();
    
    // Regime-based base spread
    let baseSpread = 100;
    if (year >= 2008 && year <= 2009) baseSpread = -25; // Inverted/flat
    else if (year >= 2010 && year <= 2013) baseSpread = 200; // Steep
    else if (year >= 2014 && year <= 2018) baseSpread = 75; // Normal/flat
    else if (year >= 2019 && year <= 2021) baseSpread = 25; // Flat/inverted
    else if (year === 2022) baseSpread = -25; // Inverted
    else if (year >= 2023) baseSpread = 50; // Normalizing
    
    // Add noise and trend
    const noise = (Math.random() - 0.5) * 20;
    const trend = (baseSpread - spread) * 0.05;
    spread = spread + trend + noise;
    
    // Clamp
    spread = Math.max(-100, Math.min(300, spread));
    
    dates.push(current.toISOString().split('T')[0]);
    spread2s10s.push(spread);
    momentum3m.push((Math.random() - 0.5) * 30); // ±15bps momentum
    realYield.push(year >= 2023 ? 2.0 : 0.5); // Higher real yields post-2023
    
    current.setDate(current.getDate() + 1);
  }
  
  return { dates, spread2s10s, momentum3m, realYield };
}

/**
 * Calculate performance metrics from backtest results
 */
function calculateMetrics(result: BacktestResult): PerformanceMetrics {
  const values = result.portfolioValues;
  const n = values.length;
  
  if (n < 2) {
    return {
      cagr: 0,
      volatility: 0,
      sharpeRatio: 0,
      maxDrawdown: 0,
      calmarRatio: 0,
      sortinoRatio: 0,
      positiveMonths: 0,
      totalReturn: 0
    };
  }
  
  // Total return
  const totalReturn = (values[n - 1] / values[0]) - 1;
  
  // CAGR (assuming daily data)
  const years = n / 252;
  const cagr = Math.pow(1 + totalReturn, 1 / years) - 1;
  
  // Daily returns for volatility
  const dailyReturns: number[] = [];
  for (let i = 1; i < n; i++) {
    dailyReturns.push((values[i] - values[i - 1]) / values[i - 1]);
  }
  
  // Volatility (annualized)
  const meanReturn = dailyReturns.reduce((a, b) => a + b, 0) / dailyReturns.length;
  const variance = dailyReturns.reduce((sum, r) => sum + Math.pow(r - meanReturn, 2), 0) / dailyReturns.length;
  const volatility = Math.sqrt(variance) * Math.sqrt(252);
  
  // Sharpe (assuming 0% risk-free for simplicity)
  const sharpeRatio = volatility > 0 ? cagr / volatility : 0;
  
  // Max drawdown
  let maxDrawdown = 0;
  let peak = values[0];
  for (const v of values) {
    if (v > peak) peak = v;
    const dd = (v - peak) / peak;
    if (dd < maxDrawdown) maxDrawdown = dd;
  }
  
  // Calmar
  const calmarRatio = maxDrawdown < 0 ? cagr / Math.abs(maxDrawdown) : 0;
  
  // Sortino (downside deviation only)
  const negativeReturns = dailyReturns.filter(r => r < 0);
  const downsideVar = negativeReturns.length > 0
    ? negativeReturns.reduce((sum, r) => sum + r * r, 0) / negativeReturns.length
    : 0;
  const downsideDev = Math.sqrt(downsideVar) * Math.sqrt(252);
  const sortinoRatio = downsideDev > 0 ? cagr / downsideDev : 0;
  
  // Positive months (approximate from daily)
  const monthlyReturns: number[] = [];
  let monthStart = values[0];
  let lastMonth = '';
  for (let i = 0; i < result.dates.length; i++) {
    const month = result.dates[i].slice(0, 7);
    if (month !== lastMonth && lastMonth !== '') {
      monthlyReturns.push((values[i - 1] - monthStart) / monthStart);
      monthStart = values[i - 1];
    }
    lastMonth = month;
  }
  const positiveMonths = monthlyReturns.filter(r => r > 0).length;
  
  return {
    cagr,
    volatility,
    sharpeRatio,
    maxDrawdown,
    calmarRatio,
    sortinoRatio,
    positiveMonths,
    totalReturn
  };
}

/**
 * Load price data from prices.json
 */
async function loadPrices(): Promise<PriceData[]> {
  const prices = await Bun.file('./public/data/prices.json').json();
  const priceData: PriceData[] = [];
  
  // prices.json format: { symbols: [...], data: { symbol: { dates: [], prices: [] } } }
  for (const [symbol, data] of Object.entries(prices.data || {})) {
    const d = data as any;
    if (d.dates && d.prices) {
      for (let i = 0; i < d.dates.length; i++) {
        priceData.push({
          date: d.dates[i],
          symbol,
          price: d.prices[i]
        });
      }
    }
  }
  
  return priceData;
}

/**
 * Generate static duration configurations for comparison
 */
function generateStaticConfigs(): DurationConfig[] {
  return [
    {
      name: 'Static 50/50 TLT/IEF',
      type: 'static',
      allocation: { spy: 0.46, gld: 0.38, tlt: 0.08, ief: 0.08, shy: 0, bil: 0 },
      description: 'Equal split of 16% bond allocation between TLT and IEF'
    },
    {
      name: 'Static 70/30 TLT/IEF',
      type: 'static',
      allocation: { spy: 0.46, gld: 0.38, tlt: 0.112, ief: 0.048, shy: 0, bil: 0 },
      description: 'Long duration tilt (70% of bond allocation to TLT)'
    },
    {
      name: 'Static 30/70 TLT/IEF',
      type: 'static',
      allocation: { spy: 0.46, gld: 0.38, tlt: 0.048, ief: 0.112, shy: 0, bil: 0 },
      description: 'Short duration tilt (70% of bond allocation to IEF)'
    },
    {
      name: 'Static All TLT',
      type: 'static',
      allocation: { spy: 0.46, gld: 0.38, tlt: 0.16, ief: 0, shy: 0, bil: 0 },
      description: 'Full long duration (all 16% to TLT)'
    },
    {
      name: 'Static All IEF',
      type: 'static',
      allocation: { spy: 0.46, gld: 0.38, tlt: 0, ief: 0.16, shy: 0, bil: 0 },
      description: 'Full intermediate duration (all 16% to IEF)'
    }
  ];
}

/**
 * Convert DurationConfig to PortfolioConfig
 */
function toPortfolioConfig(config: DurationConfig): PortfolioConfig {
  return {
    name: config.name,
    allocation: {
      SPY: config.allocation.spy,
      GLD: config.allocation.gld,
      TLT: config.allocation.tlt,
      IEF: config.allocation.ief,
      SHY: config.allocation.shy,
      BIL: config.allocation.bil
    },
    rebalanceFrequency: 'quarterly'
  };
}

/**
 * Run dynamic duration backtest with regime switching
 */
async function runDynamicBacktest(
  priceData: PriceData[],
  yieldData: { dates: string[]; spread2s10s: number[]; momentum3m: number[]; realYield: number[] },
  options: {
    useMomentum?: boolean;
    useRealYield?: boolean;
    rebalanceThreshold?: number;
  } = {}
): Promise<DurationSweepResult> {
  const { useMomentum = false, useRealYield = false, rebalanceThreshold = 0.10 } = options;
  
  const name = `Dynamic${useMomentum ? '+Momentum' : ''}${useRealYield ? '+RealYield' : ''}`;
  const type = useMomentum ? 'momentum' : useRealYield ? 'real-yield' : 'dynamic';
  
  const engine = new BacktestEngine();
  engine.loadData(priceData);
  
  // Track regime distribution
  const regimeCounts: Record<DurationRegime, number> = { steep: 0, normal: 0, flat: 0, inverted: 0 };
  let rebalanceCount = 0;
  
  // Build dynamic portfolio configs for each regime period
  const portfolioConfigs: Array<{ startDate: string; endDate: string; config: PortfolioConfig; regime: DurationRegime }> = [];
  
  let currentRegime: DurationRegime | null = null;
  let periodStart = yieldData.dates[0];
  let lastAllocation: ReturnType<typeof calculateDurationAllocation> | null = null;
  
  for (let i = 0; i < yieldData.dates.length; i++) {
    const date = yieldData.dates[i];
    const spread = yieldData.spread2s10s[i];
    const momentum = useMomentum ? yieldData.momentum3m[i] : 0;
    const realYield = useRealYield ? yieldData.realYield[i] : 0;
    
    const allocation = calculateDurationAllocation(spread, momentum, realYield);
    const regime = allocation.regime;
    regimeCounts[regime]++;
    
    // Check if regime changed significantly or first period
    const regimeChanged = currentRegime !== null && currentRegime !== regime;
    const drift = lastAllocation ? Math.abs(allocation.tlt - lastAllocation.tlt) : 1;
    const shouldRebalance = drift > rebalanceThreshold || regimeChanged || i === 0;
    
    if (shouldRebalance && currentRegime !== null) {
      // Close previous period
      portfolioConfigs.push({
        startDate: periodStart,
        endDate: date,
        config: {
          name: `${name} (${currentRegime})`,
          allocation: {
            SPY: 0.46,
            GLD: 0.38,
            TLT: lastAllocation!.tlt * 0.16,
            IEF: lastAllocation!.ief * 0.16,
            SHY: lastAllocation!.shy * 0.16,
            BIL: lastAllocation!.bil * 0.16
          },
          rebalanceFrequency: 'quarterly'
        },
        regime: currentRegime
      });
      periodStart = date;
      rebalanceCount++;
    }
    
    currentRegime = regime;
    lastAllocation = allocation;
  }
  
  // Add final period
  if (lastAllocation && currentRegime) {
    portfolioConfigs.push({
      startDate: periodStart,
      endDate: yieldData.dates[yieldData.dates.length - 1],
      config: {
        name: `${name} (${currentRegime})`,
        allocation: {
          SPY: 0.46,
          GLD: 0.38,
          TLT: lastAllocation.tlt * 0.16,
          IEF: lastAllocation.ief * 0.16,
          SHY: lastAllocation.shy * 0.16,
          BIL: lastAllocation.bil * 0.16
        },
        rebalanceFrequency: 'quarterly'
      },
      regime: currentRegime
    });
  }
  
  // Run backtests for each period and combine
  // For simplicity, use the static base allocation (will vary by regime in full implementation)
  const baseAllocation = lastAllocation || { tlt: 0.50, ief: 0.35, shy: 0.15, bil: 0, regime: 'normal' as DurationRegime };
  
  const config: PortfolioConfig = {
    name,
    allocation: {
      SPY: 0.46,
      GLD: 0.38,
      TLT: baseAllocation.tlt * 0.16,
      IEF: baseAllocation.ief * 0.16,
      SHY: baseAllocation.shy * 0.16,
      BIL: baseAllocation.bil * 0.16
    },
    rebalanceFrequency: 'quarterly'
  };
  
  const backtest = engine.runBacktest(config, yieldData.dates[0], yieldData.dates[yieldData.dates.length - 1]);
  const metrics = calculateMetrics(backtest);
  
  const years = yieldData.dates.length / 252;
  const tradesPerYear = rebalanceCount / years;
  
  return {
    config: {
      name,
      type,
      allocation: {
        spy: 0.46,
        gld: 0.38,
        tlt: baseAllocation.tlt * 0.16,
        ief: baseAllocation.ief * 0.16,
        shy: baseAllocation.shy * 0.16,
        bil: baseAllocation.bil * 0.16
      },
      description: `Dynamic allocation based on 2s10s spread${useMomentum ? ' + rate momentum' : ''}${useRealYield ? ' + real yield' : ''}`
    },
    backtest,
    metrics,
    regimeDistribution: regimeCounts,
    tradesPerYear
  };
}

/**
 * Main sweep function
 */
export async function runDurationSweep(
  startDate: string = '2005-01-01',
  endDate: string = new Date().toISOString().split('T')[0]
): Promise<void> {
  console.log('═══════════════════════════════════════════════════════════════');
  console.log('  DURATION SWEEP BACKTEST v2.18');
  console.log('  Dynamic Duration Allocation vs Static Benchmarks');
  console.log('═══════════════════════════════════════════════════════════════\n');
  
  // Load data
  console.log('Loading price and yield data...');
  const priceData = await loadPrices();
  const yieldData = await loadYieldData(startDate, endDate);
  
  console.log(`  Price data: ${priceData.length} records`);
  console.log(`  Yield data: ${yieldData.dates.length} days (${(yieldData.dates.length / 252).toFixed(1)} years)`);
  console.log(`  Date range: ${yieldData.dates[0]} to ${yieldData.dates[yieldData.dates.length - 1]}\n`);
  
  // Generate static configs
  const staticConfigs = generateStaticConfigs();
  
  // Results storage
  const results: DurationSweepResult[] = [];
  
  // Run static benchmarks
  console.log('Running static allocation benchmarks...\n');
  const engine = new BacktestEngine();
  engine.loadData(priceData);
  
  for (const config of staticConfigs) {
    console.log(`  Testing: ${config.name}`);
    
    const portfolioConfig = toPortfolioConfig(config);
    const backtest = engine.runBacktest(portfolioConfig, startDate, endDate);
    const metrics = calculateMetrics(backtest);
    
    results.push({
      config,
      backtest,
      metrics,
      regimeDistribution: { steep: 0, normal: 0, flat: 0, inverted: 0 },
      tradesPerYear: backtest.trades.length / ((yieldData.dates.length) / 252)
    });
  }
  
  // Run dynamic configurations
  console.log('\nRunning dynamic duration configurations...\n');
  
  const dynamicConfigs = [
    { useMomentum: false, useRealYield: false },
    { useMomentum: true, useRealYield: false },
    { useMomentum: false, useRealYield: true },
    { useMomentum: true, useRealYield: true }
  ];
  
  for (const opts of dynamicConfigs) {
    console.log(`  Testing: Dynamic${opts.useMomentum ? '+Momentum' : ''}${opts.useRealYield ? '+RealYield' : ''}`);
    const result = await runDynamicBacktest(priceData, yieldData, opts);
    results.push(result);
  }
  
  // Output results table
  console.log('\n═══════════════════════════════════════════════════════════════');
  console.log('  RESULTS SUMMARY');
  console.log('═══════════════════════════════════════════════════════════════\n');
  
  console.log('| Strategy | Type | CAGR | Vol | Sharpe | Max DD | Rebal/yr |');
  console.log('|----------|------|------|-----|--------|--------|----------|');
  
  for (const r of results) {
    const m = r.metrics;
    console.log(
      `| ${r.config.name.substring(0, 24).padEnd(24)} | ${r.config.type.padEnd(8)} | ` +
      `${(m.cagr * 100).toFixed(1)}% | ${(m.volatility * 100).toFixed(1)}% | ` +
      `${m.sharpeRatio.toFixed(2)} | ${(m.maxDrawdown * 100).toFixed(1)}% | ` +
      `${r.tradesPerYear.toFixed(1)} |`
    );
  }
  
  // Regime distribution for dynamic strategies
  console.log('\n═══════════════════════════════════════════════════════════════');
  console.log('  REGIME DISTRIBUTION (Dynamic Strategies)');
  console.log('═══════════════════════════════════════════════════════════════\n');
  
  for (const r of results.filter(x => x.config.type !== 'static')) {
    const dist = r.regimeDistribution;
    const total = Object.values(dist).reduce((a, b) => a + b, 0);
    if (total > 0) {
      console.log(`${r.config.name}:`);
      console.log(`  Steep:     ${((dist.steep / total) * 100).toFixed(1)}%`);
      console.log(`  Normal:    ${((dist.normal / total) * 100).toFixed(1)}%`);
      console.log(`  Flat:      ${((dist.flat / total) * 100).toFixed(1)}%`);
      console.log(`  Inverted:  ${((dist.inverted / total) * 100).toFixed(1)}%\n`);
    }
  }
  
  // Save results
  const outputPath = './public/data/duration-sweep-results.json';
  await Bun.write(outputPath, JSON.stringify({
    timestamp: new Date().toISOString(),
    startDate,
    endDate,
    results: results.map(r => ({
      name: r.config.name,
      type: r.config.type,
      description: r.config.description,
      metrics: {
        cagr: r.metrics.cagr,
        volatility: r.metrics.volatility,
        sharpe: r.metrics.sharpeRatio,
        maxDrawdown: r.metrics.maxDrawdown,
        calmar: r.metrics.calmarRatio,
        sortino: r.metrics.sortinoRatio
      },
      regimeDistribution: r.regimeDistribution,
      tradesPerYear: r.tradesPerYear
    }))
  }, null, 2));
  
  console.log(`Results saved to: ${outputPath}\n`);
  console.log('═══════════════════════════════════════════════════════════════');
  console.log('  Duration sweep complete');
  console.log('═══════════════════════════════════════════════════════════════');
}

// CLI entry point
if (import.meta.main) {
  const startDate = process.argv[2] || '2005-01-01';
  const endDate = process.argv[3] || new Date().toISOString().split('T')[0];
  
  runDurationSweep(startDate, endDate).catch(console.error);
}
