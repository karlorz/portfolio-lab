/**
 * Factor Timing Backtest Engine (v2.43 Phase 2)
 * 
 * Simulates factor rotation strategy that adjusts exposure to style factors
 * (MTUM, VLUE, USMV, QUAL, IJR) based on valuation, momentum, and regime signals.
 * 
 * Compares performance vs static SPY/GLD/TLT 46/38/16 baseline.
 */

import * as fs from 'fs';
import * as path from 'path';
import { fileURLToPath } from 'url';
import { BacktestEngine, PortfolioConfig, BacktestResult, PerformanceMetrics } from './engine.js';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

interface FactorSignal {
  symbol: string;
  valuationZScore: number;
  momentum6mo: number;
  momentum12mo: number;
  momentumScore: number; // 6mo - 12mo
  regime: string;
  historicalAlpha: number; // avg excess return in current regime
}

interface FactorTimingConfig {
  name: string;
  // Base allocation: how much of portfolio goes to factor sleeve (vs bonds/gold)
  factorSleeveWeight: number; // e.g., 0.46 for 46% equity
  bondWeight: number;          // e.g., 0.16 for TLT
  goldWeight: number;          // e.g., 0.38 for GLD
  // Timing parameters
  valuationThreshold: number;  // z-score threshold for timing (default: 1.5)
  momentumThreshold: number; // momentum score threshold (default: 0.05)
  minFactorWeight: number;     // minimum weight per factor (default: 0.05)
  maxFactorWeight: number;     // maximum weight per factor (default: 0.20)
  rebalanceFrequency: 'monthly' | 'quarterly';
  transactionCost: number;     // per trade (default: 0.001 = 0.1%)
  strategy: 'static' | 'valuation' | 'valuation_momentum' | 'full_ensemble';
}

interface FactorBacktestResult {
  config: FactorTimingConfig;
  baselineResult: {
    metrics: PerformanceMetrics;
    trades: number;
  };
  strategyResult: {
    metrics: PerformanceMetrics;
    trades: number;
  };
  improvement: {
    sharpeDelta: number;
    cagrDelta: number;
    maxDdDelta: number;
    turnoverEstimate: number;
  };
  monthlyAllocations: Array<{
    date: string;
    regime: string;
    factorWeights: Record<string, number>;
    signals: Record<string, { valuation: number; momentum: number; composite: number }>;
  }>;
}

// Factor ETF metadata
const FACTOR_ETFS = ['MTUM', 'VLUE', 'USMV', 'QUAL', 'IJR'];
const BASE_DATE = '2013-07-18'; // QUAL inception

// Historical factor return data for regime analysis (simplified)
// In production, this would come from data/factor_correlation_analysis.json
const REGIME_PERFORMANCE: Record<string, Record<string, number>> = {
  bull: {
    MTUM: -0.346,
    VLUE: -0.123,
    USMV: -0.089,
    QUAL: -0.056,
    IJR: 0.011,
  },
  bear: {
    MTUM: -0.12,
    VLUE: 0.08,
    USMV: 0.15,
    QUAL: 0.06,
    IJR: -0.08,
  },
  neutral: {
    MTUM: 0.02,
    VLUE: 0.01,
    USMV: 0.03,
    QUAL: 0.02,
    IJR: 0.01,
  },
  strong_bull: {
    MTUM: 0.05,
    VLUE: -0.05,
    USMV: -0.03,
    QUAL: -0.02,
    IJR: 0.08,
  },
};

function loadPrices(): Record<string, Array<{ d: string; p: number }>> {
  const dataPath = path.resolve(__dirname, '../../public/data/prices.json');
  const raw = fs.readFileSync(dataPath, 'utf-8');
  return JSON.parse(raw);
}

function getPriceOnDate(prices: Array<{ d: string; p: number }>, date: string): number | null {
  const entry = prices.find(p => p.d === date);
  if (entry) return entry.p;
  const earlier = prices.filter(p => p.d < date);
  if (earlier.length === 0) return null;
  return earlier[earlier.length - 1].p;
}

function calculateHistoricalZScore(
  factorPrices: Array<{ d: string; p: number }>,
  spyPrices: Array<{ d: string; p: number }>,
  currentDate: string
): { currentSpread: number; zScore: number; mean: number; std: number } | null {
  const spreads: number[] = [];
  
  for (let i = 0; i < Math.min(factorPrices.length, spyPrices.length); i++) {
    if (factorPrices[i]?.p && spyPrices[i]?.p) {
      spreads.push(factorPrices[i].p / spyPrices[i].p);
    }
  }
  
  if (spreads.length < 252) return null;
  
  const currentFactor = getPriceOnDate(factorPrices, currentDate);
  const currentSpy = getPriceOnDate(spyPrices, currentDate);
  if (!currentFactor || !currentSpy) return null;
  
  const currentSpread = currentFactor / currentSpy;
  const mean = spreads.reduce((a, b) => a + b, 0) / spreads.length;
  const variance = spreads.reduce((sum, s) => sum + Math.pow(s - mean, 2), 0) / spreads.length;
  const std = Math.sqrt(variance);
  
  return { currentSpread, zScore: (currentSpread - mean) / std, mean, std };
}

function calculateMomentum(
  prices: Array<{ d: string; p: number }>,
  currentDate: string
): { return6mo: number; return12mo: number; score: number } | null {
  const currentIdx = prices.findIndex(p => p.d === currentDate);
  if (currentIdx === -1) return null;
  
  const currentPrice = prices[currentIdx]?.p;
  const price6mo = prices[currentIdx - 126]?.p;
  const price12mo = prices[currentIdx - 252]?.p;
  
  if (!currentPrice || !price6mo || !price12mo) return null;
  
  const return6mo = (currentPrice - price6mo) / price6mo;
  const return12mo = (currentPrice - price12mo) / price12mo;
  return { return6mo, return12mo, score: return6mo - return12mo };
}

function identifyRegime(spyPrices: Array<{ d: string; p: number }>, currentDate: string): string {
  const currentIdx = spyPrices.findIndex(p => p.d === currentDate);
  if (currentIdx === -1 || currentIdx < 252) return 'neutral';
  
  const currentPrice = spyPrices[currentIdx].p;
  const price6mo = spyPrices[currentIdx - 126]?.p;
  const price12mo = spyPrices[currentIdx - 252]?.p;
  
  if (!price6mo || !price12mo) return 'neutral';
  
  const ret6mo = (currentPrice - price6mo) / price6mo;
  const ret12mo = (currentPrice - price12mo) / price12mo;
  
  if (ret6mo > 0.15 && ret12mo > 0.2) return 'strong_bull';
  if (ret6mo > 0.05 && ret12mo > 0.1) return 'bull';
  if (ret6mo < -0.1 && ret12mo < -0.15) return 'bear';
  if (ret6mo < -0.05) return 'correction';
  if (ret6mo > 0 && ret12mo < 0) return 'recovery';
  if (ret6mo < 0 && ret12mo > 0) return 'consolidation';
  return 'neutral';
}

function calculateSignals(
  prices: Record<string, Array<{ d: string; p: number }>>,
  date: string
): { signals: Record<string, FactorSignal>; regime: string } {
  const spyPrices = prices['SPY'];
  const regime = identifyRegime(spyPrices, date);
  const signals: Record<string, FactorSignal> = {};
  
  for (const symbol of FACTOR_ETFS) {
    const factorPrices = prices[symbol];
    if (!factorPrices) continue;
    
    const zData = calculateHistoricalZScore(factorPrices, spyPrices, date);
    const momData = calculateMomentum(factorPrices, date);
    
    if (zData && momData) {
      signals[symbol] = {
        symbol,
        valuationZScore: zData.zScore,
        momentum6mo: momData.return6mo,
        momentum12mo: momData.return12mo,
        momentumScore: momData.score,
        regime,
        historicalAlpha: REGIME_PERFORMANCE[regime]?.[symbol] || 0,
      };
    }
  }
  
  return { signals, regime };
}

function calculateFactorWeights(
  signals: Record<string, FactorSignal>,
  config: FactorTimingConfig,
  prevWeights?: Record<string, number>
): { weights: Record<string, number>; turnover: number } {
  const baseWeight = config.factorSleeveWeight / FACTOR_ETFS.length; // equal base
  const weights: Record<string, number> = {};
  
  // Calculate raw scores based on strategy
  const rawScores: Record<string, number> = {};
  
  for (const [symbol, signal] of Object.entries(signals)) {
    let score = 1; // neutral
    
    switch (config.strategy) {
      case 'static':
        score = 1;
        break;
      
      case 'valuation':
        // Cheap (negative z-score) = overweight
        if (signal.valuationZScore < -config.valuationThreshold) {
          score = 1.5;
        } else if (signal.valuationZScore > config.valuationThreshold) {
          score = 0.5;
        }
        break;
      
      case 'valuation_momentum':
        // Valuation + momentum overlay
        let valScore = 1;
        if (signal.valuationZScore < -config.valuationThreshold) valScore = 1.5;
        else if (signal.valuationZScore > config.valuationThreshold) valScore = 0.5;
        
        // Momentum penalty (avoid declining momentum)
        if (signal.momentumScore < -config.momentumThreshold) {
          valScore *= 0.8;
        } else if (signal.momentumScore > config.momentumThreshold) {
          valScore *= 1.2;
        }
        score = valScore;
        break;
      
      case 'full_ensemble':
        // Valuation + momentum + regime
        let ensembleScore = 1;
        
        // Valuation component (40%)
        if (signal.valuationZScore < -config.valuationThreshold) {
          ensembleScore += 0.4 * Math.min(1, Math.abs(signal.valuationZScore) / 3);
        } else if (signal.valuationZScore > config.valuationThreshold) {
          ensembleScore -= 0.4 * Math.min(1, Math.abs(signal.valuationZScore) / 3);
        }
        
        // Momentum component (30%)
        if (signal.momentumScore < -config.momentumThreshold) {
          ensembleScore -= 0.3;
        } else if (signal.momentumScore > config.momentumThreshold) {
          ensembleScore += 0.3;
        }
        
        // Regime component (30%)
        if (signal.historicalAlpha > 0.02) {
          ensembleScore += 0.3;
        } else if (signal.historicalAlpha < -0.02) {
          ensembleScore -= 0.3;
        }
        
        score = Math.max(0.2, ensembleScore);
        break;
    }
    
    rawScores[symbol] = score;
  }
  
  // Normalize to sum to factorSleeveWeight
  const totalScore = Object.values(rawScores).reduce((a, b) => a + b, 0);
  for (const symbol of Object.keys(rawScores)) {
    const normalized = (rawScores[symbol] / totalScore) * config.factorSleeveWeight;
    // Clamp to min/max constraints
    weights[symbol] = Math.max(config.minFactorWeight, Math.min(config.maxFactorWeight, normalized));
  }
  
  // Re-normalize after clamping
  const clampedTotal = Object.values(weights).reduce((a, b) => a + b, 0);
  for (const symbol of Object.keys(weights)) {
    weights[symbol] = (weights[symbol] / clampedTotal) * config.factorSleeveWeight;
  }
  
  // Calculate turnover vs previous weights
  let turnover = 0;
  if (prevWeights) {
    for (const symbol of FACTOR_ETFS) {
      turnover += Math.abs((weights[symbol] || 0) - (prevWeights[symbol] || 0));
    }
    turnover /= 2; // Only count one side of trade
  }
  
  return { weights, turnover };
}

function calculatePerformanceMetrics(result: BacktestResult): PerformanceMetrics {
  const values = result.portfolioValues;
  const returns = result.returns.slice(1); // Skip first 0 return
  
  if (values.length < 2 || returns.length === 0) {
    return { cagr: 0, volatility: 0, sharpeRatio: 0, maxDrawdown: 0, calmarRatio: 0, sortinoRatio: 0, positiveMonths: 0, totalReturn: 0 };
  }
  
  const totalReturn = (values[values.length - 1] - values[0]) / values[0];
  const years = values.length / 252;
  const cagr = Math.pow(1 + totalReturn, 1 / years) - 1;
  
  const meanReturn = returns.reduce((a, b) => a + b, 0) / returns.length;
  const variance = returns.reduce((sum, r) => sum + Math.pow(r - meanReturn, 2), 0) / returns.length;
  const dailyVol = Math.sqrt(variance);
  const annualVol = dailyVol * Math.sqrt(252);
  
  const riskFreeRate = 0.02;
  const sharpeRatio = annualVol > 0 ? (cagr - riskFreeRate) / annualVol : 0;
  
  // Max drawdown
  let peak = values[0];
  let maxDD = 0;
  for (const v of values) {
    if (v > peak) peak = v;
    const dd = (peak - v) / peak;
    if (dd > maxDD) maxDD = dd;
  }
  
  // Calmar
  const calmarRatio = maxDD > 0 ? cagr / maxDD : 0;
  
  // Sortino (downside deviation)
  const downsideReturns = returns.filter(r => r < 0);
  const downsideVar = downsideReturns.length > 0 
    ? downsideReturns.reduce((sum, r) => sum + r * r, 0) / downsideReturns.length 
    : 0;
  const downsideDev = Math.sqrt(downsideVar) * Math.sqrt(252);
  const sortinoRatio = downsideDev > 0 ? (cagr - riskFreeRate) / downsideDev : 0;
  
  return {
    cagr,
    volatility: annualVol,
    sharpeRatio,
    maxDrawdown: -maxDD,
    calmarRatio,
    sortinoRatio,
    positiveMonths: 0, // Would need monthly aggregation
    totalReturn,
  };
}

export async function runFactorTimingBacktest(
  config: FactorTimingConfig
): Promise<FactorBacktestResult> {
  const prices = loadPrices();
  const spyPrices = prices['SPY'];
  
  // Get date range from 2013-07-18 to latest available
  const startIdx = spyPrices.findIndex(p => p.d >= BASE_DATE);
  const dates = spyPrices.slice(startIdx).map(p => p.d);
  
  // Run baseline backtest (static SPY/GLD/TLT 46/38/16)
  const engine = new BacktestEngine();
  const allPrices: any[] = [];
  for (const [symbol, priceList] of Object.entries(prices)) {
    for (const { d, p } of priceList) {
      allPrices.push({ date: d, symbol, price: p });
    }
  }
  engine.loadData(allPrices);
  
  const baselineConfig: PortfolioConfig = {
    name: 'Baseline SPY/GLD/TLT',
    allocation: { SPY: 0.46, GLD: 0.38, TLT: 0.16 },
    rebalanceFrequency: config.rebalanceFrequency,
  };
  
  const baselineResult = engine.runBacktest(baselineConfig, dates[0], dates[dates.length - 1], 10000);
  const baselineMetrics = calculatePerformanceMetrics(baselineResult);
  
  // Simulate factor timing strategy
  let portfolioValue = 10000;
  const portfolioValues: number[] = [];
  let prevWeights: Record<string, number> | undefined;
  let totalTurnover = 0;
  let tradeCount = 0;
  const monthlyAllocations: FactorBacktestResult['monthlyAllocations'] = [];
  
  // Monthly rebalancing check
  let lastRebalanceMonth = -1;
  
  for (let i = 0; i < dates.length; i++) {
    const date = dates[i];
    const currentMonth = parseInt(date.split('-')[1]);
    
    // Calculate factor signals
    const { signals, regime } = calculateSignals(prices, date);
    
    // Check if we should rebalance
    const shouldRebalance = config.rebalanceFrequency === 'monthly' 
      ? currentMonth !== lastRebalanceMonth
      : (currentMonth !== lastRebalanceMonth && [1, 4, 7, 10].includes(currentMonth));
    
    let currentWeights: Record<string, number>;
    
    if (shouldRebalance || !prevWeights) {
      const { weights, turnover } = calculateFactorWeights(signals, config, prevWeights);
      currentWeights = weights;
      totalTurnover += turnover;
      tradeCount++;
      lastRebalanceMonth = currentMonth;
      prevWeights = weights;
    } else {
      currentWeights = prevWeights;
    }
    
    // Calculate daily portfolio return
    let dailyReturn = 0;
    
    // Factor sleeve return
    for (const [symbol, weight] of Object.entries(currentWeights)) {
      const priceList = prices[symbol];
      if (!priceList || i === 0) continue;
      
      const todayPrice = priceList[startIdx + i]?.p;
      const yesterdayPrice = priceList[startIdx + i - 1]?.p;
      
      if (todayPrice && yesterdayPrice) {
        const symbolReturn = (todayPrice - yesterdayPrice) / yesterdayPrice;
        dailyReturn += weight * symbolReturn;
      }
    }
    
    // Bond (TLT) return
    const tltPrices = prices['TLT'];
    if (tltPrices && i > 0) {
      const tltToday = tltPrices[startIdx + i]?.p;
      const tltYesterday = tltPrices[startIdx + i - 1]?.p;
      if (tltToday && tltYesterday) {
        dailyReturn += config.bondWeight * ((tltToday - tltYesterday) / tltYesterday);
      }
    }
    
    // Gold (GLD) return
    const gldPrices = prices['GLD'];
    if (gldPrices && i > 0) {
      const gldToday = gldPrices[startIdx + i]?.p;
      const gldYesterday = gldPrices[startIdx + i - 1]?.p;
      if (gldToday && gldYesterday) {
        dailyReturn += config.goldWeight * ((gldToday - gldYesterday) / gldYesterday);
      }
    }
    
    // Apply transaction costs on rebalance days
    if (shouldRebalance && i > 0) {
      const rebalanceCost = config.transactionCost * (totalTurnover / Math.max(1, tradeCount));
      dailyReturn -= rebalanceCost;
    }
    
    portfolioValue *= (1 + dailyReturn);
    portfolioValues.push(portfolioValue);
    
    // Record monthly allocation (monthly snapshot)
    if (shouldRebalance || i === 0) {
      const signalData: Record<string, { valuation: number; momentum: number; composite: number }> = {};
      for (const [symbol, signal] of Object.entries(signals)) {
        signalData[symbol] = {
          valuation: signal.valuationZScore,
          momentum: signal.momentumScore,
          composite: signal.valuationZScore + signal.momentumScore * 10 + signal.historicalAlpha * 100,
        };
      }
      
      monthlyAllocations.push({
        date,
        regime,
        factorWeights: { ...currentWeights },
        signals: signalData,
      });
    }
  }
  
  // Calculate strategy metrics
  const strategyReturns = portfolioValues.map((v, i) => 
    i === 0 ? 0 : (v - portfolioValues[i - 1]) / portfolioValues[i - 1]
  ).slice(1);
  
  const totalReturn = (portfolioValues[portfolioValues.length - 1] - 10000) / 10000;
  const years = dates.length / 252;
  const cagr = Math.pow(1 + totalReturn, 1 / years) - 1;
  
  const meanReturn = strategyReturns.reduce((a, b) => a + b, 0) / strategyReturns.length;
  const variance = strategyReturns.reduce((sum, r) => sum + Math.pow(r - meanReturn, 2), 0) / strategyReturns.length;
  const annualVol = Math.sqrt(variance) * Math.sqrt(252);
  
  const riskFreeRate = 0.02;
  const sharpeRatio = annualVol > 0 ? (cagr - riskFreeRate) / annualVol : 0;
  
  // Max drawdown
  let peak = portfolioValues[0];
  let maxDD = 0;
  for (const v of portfolioValues) {
    if (v > peak) peak = v;
    const dd = (peak - v) / peak;
    if (dd > maxDD) maxDD = dd;
  }
  
  const strategyMetrics: PerformanceMetrics = {
    cagr,
    volatility: annualVol,
    sharpeRatio,
    maxDrawdown: -maxDD,
    calmarRatio: maxDD > 0 ? cagr / maxDD : 0,
    sortinoRatio: 0, // Calculate if needed
    positiveMonths: 0,
    totalReturn,
  };
  
  // Annualized turnover estimate
  const yearsSimulated = dates.length / 252;
  const annualTurnover = totalTurnover / yearsSimulated;
  
  return {
    config,
    baselineResult: {
      metrics: baselineMetrics,
      trades: baselineResult.trades.length,
    },
    strategyResult: {
      metrics: strategyMetrics,
      trades: tradeCount,
    },
    improvement: {
      sharpeDelta: sharpeRatio - baselineMetrics.sharpeRatio,
      cagrDelta: (cagr - baselineMetrics.cagr) * 100, // in percentage points
      maxDdDelta: (strategyMetrics.maxDrawdown - baselineMetrics.maxDrawdown) * 100,
      turnoverEstimate: annualTurnover,
    },
    monthlyAllocations,
  };
}

// Run all strategy configurations and compare
export async function runFullFactorBacktestSuite(): Promise<void> {
  console.log('\n=== FACTOR TIMING BACKTEST SUITE (v2.43 Phase 2) ===\n');
  
  const configs: FactorTimingConfig[] = [
    {
      name: 'Baseline (Static Equal Weight)',
      strategy: 'static',
      factorSleeveWeight: 0.46,
      bondWeight: 0.16,
      goldWeight: 0.38,
      valuationThreshold: 1.5,
      momentumThreshold: 0.05,
      minFactorWeight: 0.05,
      maxFactorWeight: 0.20,
      rebalanceFrequency: 'quarterly',
      transactionCost: 0.001,
    },
    {
      name: 'Valuation Timing',
      strategy: 'valuation',
      factorSleeveWeight: 0.46,
      bondWeight: 0.16,
      goldWeight: 0.38,
      valuationThreshold: 1.5,
      momentumThreshold: 0.05,
      minFactorWeight: 0.05,
      maxFactorWeight: 0.20,
      rebalanceFrequency: 'quarterly',
      transactionCost: 0.001,
    },
    {
      name: 'Valuation + Momentum',
      strategy: 'valuation_momentum',
      factorSleeveWeight: 0.46,
      bondWeight: 0.16,
      goldWeight: 0.38,
      valuationThreshold: 1.5,
      momentumThreshold: 0.05,
      minFactorWeight: 0.05,
      maxFactorWeight: 0.20,
      rebalanceFrequency: 'quarterly',
      transactionCost: 0.001,
    },
    {
      name: 'Full Ensemble (Regime + Valuation + Momentum)',
      strategy: 'full_ensemble',
      factorSleeveWeight: 0.46,
      bondWeight: 0.16,
      goldWeight: 0.38,
      valuationThreshold: 1.5,
      momentumThreshold: 0.05,
      minFactorWeight: 0.05,
      maxFactorWeight: 0.20,
      rebalanceFrequency: 'quarterly',
      transactionCost: 0.001,
    },
  ];
  
  const results: FactorBacktestResult[] = [];
  
  for (const config of configs) {
    console.log(`Running: ${config.name}...`);
    const result = await runFactorTimingBacktest(config);
    results.push(result);
    
    console.log(`  Baseline  - CAGR: ${(result.baselineResult.metrics.cagr * 100).toFixed(2)}%, Sharpe: ${result.baselineResult.metrics.sharpeRatio.toFixed(3)}, MaxDD: ${(result.baselineResult.metrics.maxDrawdown * 100).toFixed(1)}%`);
    console.log(`  Strategy  - CAGR: ${(result.strategyResult.metrics.cagr * 100).toFixed(2)}%, Sharpe: ${result.strategyResult.metrics.sharpeRatio.toFixed(3)}, MaxDD: ${(result.strategyResult.metrics.maxDrawdown * 100).toFixed(1)}%`);
    console.log(`  Improvement - Sharpe: ${result.improvement.sharpeDelta > 0 ? '+' : ''}${result.improvement.sharpeDelta.toFixed(3)}, CAGR: ${result.improvement.cagrDelta > 0 ? '+' : ''}${result.improvement.cagrDelta.toFixed(2)}pp, Turnover: ${result.improvement.turnoverEstimate.toFixed(2)}x\n`);
  }
  
  // Summary table
  console.log('\n=== SUMMARY TABLE ===');
  console.log('| Strategy | CAGR | Sharpe | Max DD | Sharpe Δ |');
  console.log('|----------|------|--------|--------|----------|');
  
  for (const r of results) {
    const sig = r.strategyResult.metrics;
    const imp = r.improvement;
    console.log(`| ${r.config.name.slice(0, 30).padEnd(30)} | ${(sig.cagr * 100).toFixed(1)}% | ${sig.sharpeRatio.toFixed(3)} | ${(sig.maxDrawdown * 100).toFixed(1)}% | ${imp.sharpeDelta > 0 ? '+' : ''}${imp.sharpeDelta.toFixed(3)} |`);
  }
  
  // Save results
  const outputPath = path.resolve(__dirname, '../../data/factor_timing_backtest_results.json');
  fs.writeFileSync(outputPath, JSON.stringify({
    timestamp: new Date().toISOString(),
    results: results.map(r => ({
      config: r.config,
      baseline: r.baselineResult.metrics,
      strategy: r.strategyResult.metrics,
      improvement: r.improvement,
    })),
  }, null, 2));
  
  console.log(`\nResults saved to: ${outputPath}`);
  
  // Find best strategy
  const best = results.reduce((best, current) => 
    current.improvement.sharpeDelta > best.improvement.sharpeDelta ? current : best
  );
  
  console.log(`\n=== BEST STRATEGY ===`);
  console.log(`${best.config.name}`);
  console.log(`Sharpe Improvement: ${best.improvement.sharpeDelta > 0 ? '+' : ''}${best.improvement.sharpeDelta.toFixed(3)}`);
  console.log(`CAGR Improvement: ${best.improvement.cagrDelta > 0 ? '+' : ''}${best.improvement.cagrDelta.toFixed(2)}pp`);
}

// Main execution
if (import.meta.url === `file://${process.argv[1]}`) {
  runFullFactorBacktestSuite().catch(console.error);
}
