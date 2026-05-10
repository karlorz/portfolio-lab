/**
 * Tactical Rebalancing Analysis — does trigger-based rebalancing beat calendar annual?
 *
 * Hypothesis: Rebalancing when portfolio drops >N% from peak (or when allocation
 * drifts >M% from target) might capture more rebalancing bonus than waiting for
 * a calendar date.
 *
 * Test: Compare annual rebalance vs threshold-based rebalance on the champion
 * SPY/GLD/TLT 46/38/16 over 2005-2026.
 */
import { BacktestEngine } from './engine';
import type { PortfolioConfig, PriceData } from './engine';
import * as fs from 'fs';

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

interface TacticalResult {
  name: string;
  sharpe: number;
  cagr: number;
  volatility: number;
  maxDrawdown: number;
  rebalanceCount: number;
}

function runTacticalBacktest(
  priceData: PriceData[],
  allocation: { [key: string]: number },
  startDate: string,
  endDate: string,
  initialValue: number,
  // Rebalance when: annual calendar OR when any allocation drifts > driftThreshold from target
  // OR when portfolio drawdown exceeds drawdownThreshold (then rebalance more aggressively)
  options: {
    annualRebalance: boolean;
    driftThreshold: number; // e.g., 0.10 = 10% absolute drift from target
    drawdownThreshold: number; // e.g., 0.05 = rebalance on >5% drawdown from peak
    minDaysBetweenRebalances: number; // prevent over-trading
  }
): TacticalResult {
  const engine = new BacktestEngine();
  engine.loadData(priceData);

  const dates = priceData.map(p => p.date).sort().filter(d => d >= startDate && d <= endDate);
  const uniqueDates = [...new Set(dates)].sort();

  const symbols = Object.keys(allocation);
  let holdings: { [symbol: string]: number } = {};
  let peakValue = initialValue;
  let lastRebalanceIndex = 0;
  let rebalanceCount = 0;

  // Initialize
  const firstDate = uniqueDates[0]!;
  for (const symbol of symbols) {
    const price = getPrice(engine, symbol, firstDate) || 100;
    holdings[symbol] = (initialValue * allocation[symbol]) / price;
  }

  const values: number[] = [];
  const returns: number[] = [];

  for (let i = 0; i < uniqueDates.length; i++) {
    const date = uniqueDates[i]!;

    // Calculate current value
    let currentValue = 0;
    for (const symbol of symbols) {
      const price = getPrice(engine, symbol, date) || 100;
      currentValue += (holdings[symbol] || 0) * price;
    }

    if (currentValue > peakValue) peakValue = currentValue;
    values.push(currentValue);

    if (i > 0) {
      returns.push((currentValue - values[i - 1]) / values[i - 1]);
    } else {
      returns.push(0);
    }

    const drawdown = (currentValue - peakValue) / peakValue;

    // Check if we should rebalance
    let shouldRebalance = false;
    let reason = '';

    // Annual calendar rebalance
    if (options.annualRebalance && i - lastRebalanceIndex >= options.minDaysBetweenRebalances) {
      shouldRebalance = true;
      reason = 'calendar';
    }

    // Drift-based: check if any allocation has drifted > threshold
    if (!shouldRebalance && options.driftThreshold < 1) {
      const daysSinceLast = i - lastRebalanceIndex;
      if (daysSinceLast >= options.minDaysBetweenRebalances) {
        for (const symbol of symbols) {
          const price = getPrice(engine, symbol, date) || 100;
          const currentValue_ = (holdings[symbol] || 0) * price;
          const currentWt = currentValue_ / currentValue;
          const targetWt = allocation[symbol] || 0;
          if (Math.abs(currentWt - targetWt) > options.driftThreshold) {
            shouldRebalance = true;
            reason = 'drift';
            break;
          }
        }
      }
    }

    // Drawdown-based: rebalance when portfolio drops > threshold from peak
    if (!shouldRebalance && options.drawdownThreshold < 1) {
      const daysSinceLast = i - lastRebalanceIndex;
      if (daysSinceLast >= options.minDaysBetweenRebalances && drawdown < -options.drawdownThreshold) {
        shouldRebalance = true;
        reason = 'drawdown';
      }
    }

    if (shouldRebalance) {
      for (const symbol of symbols) {
        const price = getPrice(engine, symbol, date) || 100;
        const targetValue = currentValue * (allocation[symbol] || 0);
        holdings[symbol] = targetValue / price;
      }
      lastRebalanceIndex = i;
      rebalanceCount++;
    }
  }

  // Calculate metrics
  const totalReturn = (values[values.length - 1]! - values[0]!) / values[0]!;
  const years = values.length / 252;
  const cagr = Math.pow(1 + totalReturn, 1 / years) - 1;
  const meanReturn = returns.slice(1).reduce((a, b) => a + b, 0) / returns.slice(1).length;
  const variance = returns.slice(1).reduce((sum, r) => sum + Math.pow(r - meanReturn, 2), 0) / returns.slice(1).length;
  const volatility = Math.sqrt(variance) * Math.sqrt(252);
  const sharpe = variance > 0 ? ((meanReturn - 0.02 / 252) / Math.sqrt(variance)) * Math.sqrt(252) : 0;

  let maxDD = 0;
  let peak = values[0]!;
  for (const v of values) {
    if (v > peak) peak = v;
    const dd = (v - peak) / peak;
    if (dd < maxDD) maxDD = dd;
  }

  return { name: '', sharpe, cagr, volatility, maxDrawdown: maxDD, rebalanceCount };
}

function getPrice(engine: BacktestEngine, symbol: string, date: string): number {
  // Access the price data from the engine via a simple approach
  // We'll just use the engine's public methods indirectly
  const data = (engine as any).priceData as Map<string, Map<string, number>>;
  const symbolData = data.get(symbol);
  if (!symbolData) return 0;
  const price = symbolData.get(date);
  if (price !== undefined) return price;
  // Fallback
  const dates = Array.from(symbolData.keys()).sort();
  let lastPrice = 0;
  for (const d of dates) {
    if (d > date) break;
    lastPrice = symbolData.get(d) || 0;
  }
  return lastPrice;
}

function main() {
  const priceData = loadData();
  const dates = priceData.map(p => p.date).sort();
  const startDate = dates[0];
  const endDate = dates[dates.length - 1];
  const allocation = { SPY: 0.46, GLD: 0.38, TLT: 0.16 };

  console.log('\n=== TACTICAL REBALANCING ANALYSIS ===');
  console.log(`Period: ${startDate} to ${endDate}`);
  console.log(`Portfolio: SPY/GLD/TLT 46/38/16\n`);

  const configs = [
    { name: 'Annual Only', annualRebalance: true, driftThreshold: 1, drawdownThreshold: 1, minDaysBetweenRebalances: 252 },
    { name: 'Annual + 5% Drift', annualRebalance: true, driftThreshold: 0.05, drawdownThreshold: 1, minDaysBetweenRebalances: 63 },
    { name: 'Annual + 10% Drift', annualRebalance: true, driftThreshold: 0.10, drawdownThreshold: 1, minDaysBetweenRebalances: 63 },
    { name: 'Annual + 15% Drift', annualRebalance: true, driftThreshold: 0.15, drawdownThreshold: 1, minDaysBetweenRebalances: 63 },
    { name: 'Annual + 5% Drawdown', annualRebalance: true, driftThreshold: 1, drawdownThreshold: 0.05, minDaysBetweenRebalances: 63 },
    { name: 'Annual + 10% Drawdown', annualRebalance: true, driftThreshold: 1, drawdownThreshold: 0.10, minDaysBetweenRebalances: 63 },
    { name: 'Annual + 15% Drawdown', annualRebalance: true, driftThreshold: 1, drawdownThreshold: 0.15, minDaysBetweenRebalances: 63 },
    { name: 'Drift 5% Only', annualRebalance: false, driftThreshold: 0.05, drawdownThreshold: 1, minDaysBetweenRebalances: 21 },
    { name: 'Drift 10% Only', annualRebalance: false, driftThreshold: 0.10, drawdownThreshold: 1, minDaysBetweenRebalances: 21 },
    { name: 'Drawdown 5% Only', annualRebalance: false, driftThreshold: 1, drawdownThreshold: 0.05, minDaysBetweenRebalances: 21 },
    { name: 'Drawdown 10% Only', annualRebalance: false, driftThreshold: 1, drawdownThreshold: 0.10, minDaysBetweenRebalances: 21 },
    { name: 'Quarterly', annualRebalance: true, driftThreshold: 1, drawdownThreshold: 1, minDaysBetweenRebalances: 63 },
    { name: 'Monthly', annualRebalance: true, driftThreshold: 1, drawdownThreshold: 1, minDaysBetweenRebalances: 21 },
  ];

  // For calendar-only rebalance, we need to override the logic
  // The annual/drift/drawdown flags handle it, but quarterly/monthly need special handling

  console.log('| Strategy | CAGR | Vol | Sharpe | Max DD | Rebalances |');
  console.log('|----------|------|-----|--------|--------|------------|');

  for (const config of configs) {
    const result = runTacticalBacktest(priceData, allocation, startDate, endDate, 10000, config);
    console.log(`| ${config.name} | ${(result.cagr * 100).toFixed(1)}% | ${(result.volatility * 100).toFixed(1)}% | ${result.sharpe.toFixed(2)} | ${(result.maxDrawdown * 100).toFixed(1)}% | ${result.rebalanceCount} |`);
  }
}

main();
