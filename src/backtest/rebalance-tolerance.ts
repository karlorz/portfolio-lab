/**
 * Rebalancing Tolerance — how much can allocations drift before performance degrades?
 * Answers: do I need exact 46/38/16, or is ±5-10% fine? What's the Sharpe degradation curve?
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

// Base allocations to test tolerance around
const BASE_CONFIGS = [
  { name: 'SPY/GLD/TLT 46/38/16', base: { SPY: 46, GLD: 38, TLT: 16 } },
  { name: 'SPY/GLD 55/45', base: { SPY: 55, GLD: 45 } },
  { name: 'SPY/GLD/TLT 50/35/15', base: { SPY: 50, GLD: 35, TLT: 15 } },
];

interface ToleranceResult {
  name: string;
  spyPct: number;
  gldPct: number;
  tltPct: number;
  sharpe: number;
  cagr: number;
  volatility: number;
  maxDrawdown: number;
  calmarRatio: number;
  meetsTarget: boolean; // CAGR ≥90% SPY, vol ≤70% SPY
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

  // Get SPY baseline for target comparison
  const spyConfig: PortfolioConfig = { name: 'SPY', allocation: { SPY: 1 }, rebalanceFrequency: 'none' };
  const spyResult = engine.runBacktest(spyConfig, startDate, endDate, 10000);
  const spyMetrics = engine.calculateMetrics(spyResult);
  const spyCagr = spyMetrics.cagr;
  const spyVol = spyMetrics.volatility;

  console.log(`SPY baseline: CAGR ${(spyCagr * 100).toFixed(1)}%, Vol ${(spyVol * 100).toFixed(1)}%`);
  console.log(`Target: CAGR ≥ ${(spyCagr * 0.9 * 100).toFixed(1)}%, Vol ≤ ${(spyVol * 0.7 * 100).toFixed(1)}%\n`);

  console.log('=== REBALANCING TOLERANCE ANALYSIS ===\n');

  for (const baseConfig of BASE_CONFIGS) {
    console.log(`\n${'='.repeat(80)}`);
    console.log(`BASE: ${baseConfig.name}`);
    console.log('='.repeat(80));

    const symbols = Object.keys(baseConfig.base);
    const results: ToleranceResult[] = [];

    // Generate all combinations within ±10% of base allocation (2% steps)
    const offsets: number[] = [];
    for (let d = -10; d <= 10; d += 2) offsets.push(d);

    if (symbols.length === 2) {
      // 2-asset: just vary SPY/GLD split
      for (const spyOff of offsets) {
        const spyPct = baseConfig.base['SPY']! + spyOff;
        const gldPct = 100 - spyPct;
        if (spyPct < 10 || gldPct < 10) continue; // sanity bounds

        const config: PortfolioConfig = {
          name: `SPY/GLD ${spyPct}/${gldPct}`,
          allocation: { SPY: spyPct / 100, GLD: gldPct / 100 },
          rebalanceFrequency: 'annual',
        };

        const result = engine.runBacktest(config, startDate, endDate, 10000);
        const metrics = engine.calculateMetrics(result);
        results.push({
          name: config.name,
          spyPct,
          gldPct,
          tltPct: 0,
          sharpe: metrics.sharpeRatio,
          cagr: metrics.cagr,
          volatility: metrics.volatility,
          maxDrawdown: metrics.maxDrawdown,
          calmarRatio: metrics.calmarRatio,
          meetsTarget: metrics.cagr >= spyCagr * 0.9 && metrics.volatility <= spyVol * 0.7,
        });
      }
    } else {
      // 3-asset: vary SPY and GLD independently, TLT = 100 - SPY - GLD
      for (const spyOff of offsets) {
        for (const gldOff of offsets) {
          const spyPct = baseConfig.base['SPY']! + spyOff;
          const gldPct = baseConfig.base['GLD']! + gldOff;
          const tltPct = 100 - spyPct - gldPct;
          if (spyPct < 10 || gldPct < 5 || tltPct < 5) continue; // sanity bounds

          const config: PortfolioConfig = {
            name: `SPY/GLD/TLT ${spyPct}/${gldPct}/${tltPct}`,
            allocation: { SPY: spyPct / 100, GLD: gldPct / 100, TLT: tltPct / 100 },
            rebalanceFrequency: 'annual',
          };

          const result = engine.runBacktest(config, startDate, endDate, 10000);
          const metrics = engine.calculateMetrics(result);
          results.push({
            name: config.name,
            spyPct,
            gldPct,
            tltPct,
            sharpe: metrics.sharpeRatio,
            cagr: metrics.cagr,
            volatility: metrics.volatility,
            maxDrawdown: metrics.maxDrawdown,
            calmarRatio: metrics.calmarRatio,
            meetsTarget: metrics.cagr >= spyCagr * 0.9 && metrics.volatility <= spyVol * 0.7,
          });
        }
      }
    }

    // Sort by Sharpe descending
    results.sort((a, b) => b.sharpe - a.sharpe);

    // Print top 20
    console.log(`\nTop 20 by Sharpe (${results.length} configurations tested):`);
    console.log('| Allocation | Sharpe | CAGR | Vol | Max DD | Calmar | Meets Target? |');
    console.log('|------------|--------|------|-----|--------|--------|---------------|');
    for (const r of results.slice(0, 20)) {
      const marker = r.name === baseConfig.name ? ' ← BASE' : '';
      const alloc = r.tltPct > 0
        ? `${r.spyPct}/${r.gldPct}/${r.tltPct}`
        : `${r.spyPct}/${r.gldPct}`;
      console.log(`| ${alloc} | ${r.sharpe.toFixed(3)} | ${(r.cagr * 100).toFixed(1)}% | ${(r.volatility * 100).toFixed(1)}% | ${(r.maxDrawdown * 100).toFixed(1)}% | ${r.calmarRatio.toFixed(2)} | ${r.meetsTarget ? '✓' : '✗'} |${marker}`);
    }

    // Tolerance zones
    const baseResult = results.find(r => r.name === baseConfig.name);
    if (!baseResult) {
      console.log('Base config not found in results');
      continue;
    }
    const baseSharpe = baseResult.sharpe;

    // Count configs within X% of base Sharpe
    console.log(`\nTolerance zones (Sharpe within X% of base ${baseSharpe.toFixed(3)}):`);
    for (const threshold of [0.02, 0.05, 0.10]) {
      const withinThreshold = results.filter(r => r.sharpe >= baseSharpe - threshold);
      const meetsTarget = withinThreshold.filter(r => r.meetsTarget);
      console.log(`  Within ${threshold.toFixed(2)} Sharpe: ${withinThreshold.length} configs, ${meetsTarget.length} meet target`);
      // Show allocation range
      const spyRange = [Math.min(...withinThreshold.map(r => r.spyPct)), Math.max(...withinThreshold.map(r => r.spyPct))];
      const gldRange = [Math.min(...withinThreshold.map(r => r.gldPct)), Math.max(...withinThreshold.map(r => r.gldPct))];
      const tltRange = withinThreshold[0]?.tltPct > 0
        ? [Math.min(...withinThreshold.map(x => x.tltPct)), Math.max(...withinThreshold.map(x => x.tltPct))]
        : [0, 0];
      console.log(`    SPY: ${spyRange[0]}-${spyRange[1]}%, GLD: ${gldRange[0]}-${gldRange[1]}%${tltRange[0] > 0 ? `, TLT: ${tltRange[0]}-${tltRange[1]}%` : ''}`);
    }

    // Sharpe heatmap (3-asset only)
    if (symbols.length === 3) {
      console.log('\nSharpe Heatmap (rows=GLD%, cols=SPY%, TLT=100-SPY-GLD):');
      const spyValues = [...new Set(results.map(r => r.spyPct))].sort((a, b) => a - b);
      const gldValues = [...new Set(results.map(r => r.gldPct))].sort((a, b) => a - b);

      // Header
      let header = 'GLD\\SPY |';
      for (const s of spyValues) header += ` ${String(s).padStart(5)} |`;
      console.log(header);
      console.log('-'.repeat(header.length));

      for (const g of gldValues) {
        let row = `  ${String(g).padStart(3)}   |`;
        for (const s of spyValues) {
          const r = results.find(x => x.spyPct === s && x.gldPct === g);
          if (r) {
            const sharpeStr = r.sharpe.toFixed(2);
            const isBase = s === baseConfig.base['SPY'] && g === baseConfig.base['GLD'];
            row += ` ${isBase ? '[' : ' '}${sharpeStr}${isBase ? ']' : ' '} |`;
          } else {
            row += '   -   |';
          }
        }
        console.log(row);
      }
    }

    // Rebalancing frequency comparison for champion
    console.log('\nRebalancing frequency comparison:');
    const bestConfig = results[0];
    for (const freq of ['monthly', 'quarterly', 'annual'] as const) {
      const tltPct = bestConfig.tltPct;
      const config: PortfolioConfig = tltPct > 0
        ? { name: `${bestConfig.spyPct}/${bestConfig.gldPct}/${tltPct} ${freq}`, allocation: { SPY: bestConfig.spyPct / 100, GLD: bestConfig.gldPct / 100, TLT: tltPct / 100 }, rebalanceFrequency: freq }
        : { name: `${bestConfig.spyPct}/${bestConfig.gldPct} ${freq}`, allocation: { SPY: bestConfig.spyPct / 100, GLD: bestConfig.gldPct / 100 }, rebalanceFrequency: freq };

      const result = engine.runBacktest(config, startDate, endDate, 10000);
      const metrics = engine.calculateMetrics(result);
      console.log(`  ${freq}: Sharpe ${metrics.sharpeRatio.toFixed(3)}, CAGR ${(metrics.cagr * 100).toFixed(1)}%, Vol ${(metrics.volatility * 100).toFixed(1)}%, MaxDD ${(metrics.maxDrawdown * 100).toFixed(1)}%`);
    }
  }

  // Drift tolerance: no rebalance for N months
  console.log('\n\n' + '='.repeat(80));
  console.log('REBALANCING DELAY TOLERANCE');
  console.log('(SPY/GLD/TLT 46/38/16: what happens if you skip rebalancing?)');
  console.log('='.repeat(80));

  const champConfig: PortfolioConfig = {
    name: 'SPY/GLD/TLT 46/38/16',
    allocation: { SPY: 0.46, GLD: 0.38, TLT: 0.16 },
    rebalanceFrequency: 'annual',
  };

  for (const freq of ['monthly', 'quarterly', 'annual', 'none'] as const) {
    const config = { ...champConfig, rebalanceFrequency: freq };
    const result = engine.runBacktest(config, startDate, endDate, 10000);
    const metrics = engine.calculateMetrics(result);

    // Also compute max allocation drift
    const holdings = result.holdings;
    const values = result.portfolioValues;
    let maxSpyDrift = 0, maxGldDrift = 0, maxTltDrift = 0;
    for (let i = 0; i < holdings.length; i++) {
      const v = values[i];
      if (v === 0) continue;
      const spyWt = (holdings[i]['SPY'] || 0) * (engine as any).getPrice('SPY', result.dates[i]) / v;
      const gldWt = (holdings[i]['GLD'] || 0) * (engine as any).getPrice('GLD', result.dates[i]) / v;
      const tltWt = (holdings[i]['TLT'] || 0) * (engine as any).getPrice('TLT', result.dates[i]) / v;
      maxSpyDrift = Math.max(maxSpyDrift, Math.abs(spyWt - 0.46));
      maxGldDrift = Math.max(maxGldDrift, Math.abs(gldWt - 0.38));
      maxTltDrift = Math.max(maxTltDrift, Math.abs(tltWt - 0.16));
    }

    console.log(`  ${freq.padEnd(10)}: Sharpe ${metrics.sharpeRatio.toFixed(3)}, CAGR ${(metrics.cagr * 100).toFixed(1)}%, Max drift: SPY ±${(maxSpyDrift * 100).toFixed(0)}%, GLD ±${(maxGldDrift * 100).toFixed(0)}%, TLT ±${(maxTltDrift * 100).toFixed(0)}%`);
  }
}

main().catch(console.error);
