/**
 * Factor Valuation & Momentum Signal Analyzer (v2.43 Phase 1.3)
 * 
 * Calculates:
 * 1. Factor valuation spreads vs SPY (historical z-scores)
 * 2. Factor momentum signals (6mo vs 12mo lookback)
 * 3. Regime-sensitivity analysis by market phase
 * 
 * Research goal: Identify when factor timing adds value
 */

import * as fs from 'fs';
import * as path from 'path';
import { fileURLToPath } from 'url';

// ESM-compatible __dirname
const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

interface PricePoint {
  d: string;
  p: number;
}

interface FactorMetrics {
  symbol: string;
  currentPrice: number;
  spyPrice: number;
  // Valuation metrics
  valuationSpread: number;  // vs SPY (ratio)
  valuationZScore: number; // Historical z-score of spread
  // Momentum metrics
  return6mo: number;
  return12mo: number;
  momentumSignal: 'positive' | 'negative' | 'neutral';
  momentumScore: number; // 6mo - 12mo (reversal signal)
  // Regime analysis
  regimePerformance: Record<string, number>;
}

interface SignalOutput {
  timestamp: string;
  analysisDate: string;
  spyPrice: number;
  factors: FactorMetrics[];
  recommendations: FactorRecommendation[];
  regime: string;
}

interface FactorRecommendation {
  factor: string;
  currentWeight: number;
  recommendedWeight: number;
  signal: 'overweight' | 'underweight' | 'neutral';
  confidence: number;
  rationale: string;
}

const FACTOR_ETFS = ['MTUM', 'VLUE', 'USMV', 'QUAL', 'IJR'];
const BASE_DATE = '2013-07-18'; // QUAL inception date (most recent factor ETF)

function loadData(): Record<string, PricePoint[]> {
  const dataPath = path.resolve(__dirname, '../../public/data/prices.json');
  const raw = fs.readFileSync(dataPath, 'utf-8');
  return JSON.parse(raw);
}

function getPriceOnDate(prices: PricePoint[], date: string): number | null {
  // Find exact match or nearest earlier date
  const entry = prices.find(p => p.d === date);
  if (entry) return entry.p;
  
  // Find nearest earlier date
  const earlier = prices.filter(p => p.d < date);
  if (earlier.length === 0) return null;
  return earlier[earlier.length - 1].p;
}

function getPriceNDaysAgo(prices: PricePoint[], currentDate: string, days: number): number | null {
  const currentIdx = prices.findIndex(p => p.d === currentDate);
  if (currentIdx === -1) return null;
  const targetIdx = currentIdx - days;
  if (targetIdx < 0) return null;
  return prices[targetIdx].p;
}

function calculateReturn(prices: PricePoint[], startDate: string, endDate: string): number | null {
  const startPrice = getPriceOnDate(prices, startDate);
  const endPrice = getPriceOnDate(prices, endDate);
  if (!startPrice || !endPrice) return null;
  return (endPrice - startPrice) / startPrice;
}

function calculateMomentumScore(prices: PricePoint[], currentDate: string): { return6mo: number; return12mo: number; score: number } | null {
  // Approximate: 6 months ≈ 126 trading days, 12 months ≈ 252 trading days
  const currentPrice = getPriceOnDate(prices, currentDate);
  if (!currentPrice) return null;
  
  const currentIdx = prices.findIndex(p => p.d === currentDate);
  if (currentIdx === -1) return null;
  
  const price6moAgo = prices[currentIdx - 126]?.p;
  const price12moAgo = prices[currentIdx - 252]?.p;
  
  if (!price6moAgo || !price12moAgo) return null;
  
  const return6mo = (currentPrice - price6moAgo) / price6moAgo;
  const return12mo = (currentPrice - price12moAgo) / price12moAgo;
  const score = return6mo - return12mo; // Positive = accelerating momentum
  
  return { return6mo, return12mo, score };
}

function calculateHistoricalSpreadStats(
  spyPrices: PricePoint[],
  factorPrices: PricePoint[],
  currentDate: string
): { currentSpread: number; zScore: number; mean: number; std: number } | null {
  // Calculate daily spreads over full history
  const spreads: number[] = [];
  
  for (let i = 0; i < Math.min(spyPrices.length, factorPrices.length); i++) {
    const spyPrice = spyPrices[i]?.p;
    const factorPrice = factorPrices[i]?.p;
    if (spyPrice && factorPrice) {
      // Spread = factor price / SPY price (relative valuation)
      spreads.push(factorPrice / spyPrice);
    }
  }
  
  if (spreads.length < 252) return null; // Need at least 1 year
  
  const currentSpy = getPriceOnDate(spyPrices, currentDate);
  const currentFactor = getPriceOnDate(factorPrices, currentDate);
  if (!currentSpy || !currentFactor) return null;
  
  const currentSpread = currentFactor / currentSpy;
  
  const mean = spreads.reduce((a, b) => a + b, 0) / spreads.length;
  const variance = spreads.reduce((sum, s) => sum + Math.pow(s - mean, 2), 0) / spreads.length;
  const std = Math.sqrt(variance);
  
  const zScore = (currentSpread - mean) / std;
  
  return { currentSpread, zScore, mean, std };
}

function identifyRegime(spyPrices: PricePoint[], currentDate: string): string {
  // Simple regime detection based on SPY 6mo and 12mo returns
  // Convert to array if needed (JSON structure may vary)
  const prices: PricePoint[] = Array.isArray(spyPrices) ? spyPrices : Object.values(spyPrices);
  const currentIdx = prices.findIndex(p => p.d === currentDate);
  if (currentIdx === -1 || currentIdx < 252) return 'unknown';
  
  const currentPrice = spyPrices[currentIdx].p;
  const price6mo = spyPrices[currentIdx - 126]?.p;
  const price12mo = spyPrices[currentIdx - 252]?.p;
  
  if (!price6mo || !price12mo) return 'unknown';
  
  const ret6mo = (currentPrice - price6mo) / price6mo;
  const ret12mo = (currentPrice - price12mo) / price12mo;
  
  // Regime classification
  if (ret6mo > 0.15 && ret12mo > 0.2) return 'strong_bull';
  if (ret6mo > 0.05 && ret12mo > 0.1) return 'bull';
  if (ret6mo < -0.1 && ret12mo < -0.15) return 'bear';
  if (ret6mo < -0.05) return 'correction';
  if (ret6mo > 0 && ret12mo < 0) return 'recovery';
  if (ret6mo < 0 && ret12mo > 0) return 'consolidation';
  return 'neutral';
}

function calculateRegimePerformance(
  spyPrices: PricePoint[],
  factorPrices: PricePoint[],
  currentDate: string
): Record<string, number> {
  // Calculate factor performance in different historical regimes
  const regimes: Record<string, number[]> = {
    strong_bull: [],
    bull: [],
    neutral: [],
    correction: [],
    bear: [],
  };
  
  // Sample every 20 days to capture different market conditions
  for (let i = 252; i < spyPrices.length; i += 20) {
    const date = spyPrices[i].d;
    const regime = identifyRegime(spyPrices, date);
    
    const spyRet = calculateReturn(spyPrices, spyPrices[i - 63].d, date);
    const factorRet = calculateReturn(factorPrices, factorPrices[i - 63]?.d || spyPrices[i - 63].d, date);
    
    if (spyRet && factorRet && regimes[regime]) {
      const excess = factorRet - spyRet;
      regimes[regime].push(excess);
    }
  }
  
  // Calculate average excess return per regime
  const result: Record<string, number> = {};
  for (const [regime, returns] of Object.entries(regimes)) {
    if (returns.length > 0) {
      result[regime] = returns.reduce((a, b) => a + b, 0) / returns.length;
    } else {
      result[regime] = 0;
    }
  }
  
  return result;
}

function generateRecommendations(factors: FactorMetrics[]): FactorRecommendation[] {
  const recommendations: FactorRecommendation[] = [];
  
  for (const factor of factors) {
    const baseWeight = 0.092; // 46% SPY replacement split 5 ways = ~9.2% each
    let signal: 'overweight' | 'underweight' | 'neutral' = 'neutral';
    let confidence = 0.5;
    let recommendedWeight = baseWeight;
    let rationale = '';
    
    // Valuation signal: negative z-score = cheap = overweight
    if (factor.valuationZScore < -1.5) {
      signal = 'overweight';
      confidence = Math.min(0.9, 0.6 + Math.abs(factor.valuationZScore) * 0.1);
      recommendedWeight = baseWeight * 1.5;
      rationale = `Cheap valuation (z=${factor.valuationZScore.toFixed(2)})`;
    } else if (factor.valuationZScore > 1.5) {
      signal = 'underweight';
      confidence = Math.min(0.9, 0.6 + Math.abs(factor.valuationZScore) * 0.1);
      recommendedWeight = baseWeight * 0.5;
      rationale = `Expensive valuation (z=${factor.valuationZScore.toFixed(2)})`;
    }
    
    // Momentum overlay: avoid negative momentum
    if (factor.momentumSignal === 'negative' && factor.momentumScore < -0.05) {
      if (signal !== 'underweight') {
        signal = confidence > 0.6 ? 'neutral' : 'underweight';
        confidence += 0.1;
        rationale += `; Declining momentum (6mo vs 12mo)`;
        recommendedWeight *= 0.8;
      }
    } else if (factor.momentumSignal === 'positive' && factor.momentumScore > 0.05) {
      if (signal !== 'overweight') {
        signal = confidence > 0.6 ? 'neutral' : 'overweight';
        confidence += 0.1;
        rationale += `; Accelerating momentum`;
        recommendedWeight *= 1.2;
      }
    }
    
    // Regime overlay
    const currentRegime = identifyRegime(factors[0]?.regimePerformance as unknown as PricePoint[], new Date().toISOString().split('T')[0]);
    const regimeAlpha = factor.regimePerformance[currentRegime] || 0;
    if (regimeAlpha > 0.02) {
      confidence += 0.1;
      rationale += `; Historically strong in ${currentRegime} regime`;
    } else if (regimeAlpha < -0.02) {
      confidence -= 0.1;
      rationale += `; Historically weak in ${currentRegime} regime`;
    }
    
    // Clean up rationale
    if (rationale.startsWith('; ')) rationale = rationale.slice(2);
    if (!rationale) rationale = 'Neutral signals';
    
    recommendations.push({
      factor: factor.symbol,
      currentWeight: baseWeight,
      recommendedWeight: Math.min(0.2, Math.max(0.02, recommendedWeight)),
      signal,
      confidence: Math.min(0.95, Math.max(0.3, confidence)),
      rationale,
    });
  }
  
  return recommendations;
}

function runFactorAnalysis() {
  const data = loadData();
  const today = new Date().toISOString().split('T')[0];
  
  console.log(`\n=== FACTOR VALUATION & MOMENTUM ANALYSIS (v2.43 Phase 1.3) ===`);
  console.log(`Analysis Date: ${today}\n`);
  
  const spyPrices = data['SPY'];
  const currentSpyPrice = getPriceOnDate(spyPrices, today);
  
  if (!currentSpyPrice) {
    // Use latest available date
    const latestDate = spyPrices[spyPrices.length - 1]?.d;
    console.log(`Using latest available date: ${latestDate}`);
  }
  
  const analysisDate = spyPrices[spyPrices.length - 1]?.d;
  const spyPrice = spyPrices[spyPrices.length - 1]?.p;
  
  console.log(`SPY Price: $${spyPrice?.toFixed(2)} (${analysisDate})\n`);
  
  const factors: FactorMetrics[] = [];
  
  console.log('=== FACTOR METRICS ===\n');
  console.log('Factor | Price | vs SPY | Val Z-Score | 6mo Ret | 12mo Ret | Mom Score | Signal');
  console.log('-------|-------|--------|-------------|---------|----------|-----------|--------');
  
  for (const symbol of FACTOR_ETFS) {
    const prices = data[symbol];
    if (!prices || prices.length === 0) {
      console.log(`${symbol}: No data available`);
      continue;
    }
    
    const currentPrice = prices[prices.length - 1].p;
    const currentDate = prices[prices.length - 1].d;
    
    // Valuation spread
    const spreadStats = calculateHistoricalSpreadStats(spyPrices, prices, currentDate);
    const valuationSpread = spreadStats?.currentSpread || 0;
    const zScore = spreadStats?.zScore || 0;
    
    // Momentum
    const momentum = calculateMomentumScore(prices, currentDate);
    const return6mo = momentum?.return6mo || 0;
    const return12mo = momentum?.return12mo || 0;
    const momScore = momentum?.score || 0;
    
    // Momentum signal
    let momentumSignal: 'positive' | 'negative' | 'neutral' = 'neutral';
    if (momScore > 0.03) momentumSignal = 'positive';
    else if (momScore < -0.03) momentumSignal = 'negative';
    
    // Regime performance
    const regimePerf = calculateRegimePerformance(spyPrices, prices, currentDate);
    
    const factor: FactorMetrics = {
      symbol,
      currentPrice,
      spyPrice: spyPrice || 0,
      valuationSpread,
      valuationZScore: zScore,
      return6mo,
      return12mo,
      momentumSignal,
      momentumScore: momScore,
      regimePerformance: regimePerf,
    };
    
    factors.push(factor);
    
    console.log(
      `${symbol.padEnd(6)} | $${currentPrice.toFixed(2).padEnd(5)} | ` +
      `${(valuationSpread * 100).toFixed(1)}% | ` +
      `${zScore.toFixed(2).padStart(6)} | ` +
      `${(return6mo * 100).toFixed(1)}% | ` +
      `${(return12mo * 100).toFixed(1)}% | ` +
      `${momScore.toFixed(3).padStart(7)} | ` +
      `${momentumSignal}`
    );
  }
  
  // Regime analysis
  console.log(`\n=== REGIME-SENSITIVITY ANALYSIS ===\n`);
  const currentRegime = identifyRegime(spyPrices, analysisDate);
  console.log(`Current Market Regime: ${currentRegime}\n`);
  
  console.log('Factor | Strong Bull | Bull | Neutral | Correction | Bear');
  console.log('-------|-------------|------|---------|------------|-----');
  for (const factor of factors) {
    const perf = factor.regimePerformance;
    console.log(
      `${factor.symbol.padEnd(6)} | ` +
      `${((perf.strong_bull || 0) * 100).toFixed(1)}% | ` +
      `${((perf.bull || 0) * 100).toFixed(1)}% | ` +
      `${((perf.neutral || 0) * 100).toFixed(1)}% | ` +
      `${((perf.correction || 0) * 100).toFixed(1)}% | ` +
      `${((perf.bear || 0) * 100).toFixed(1)}%`
    );
  }
  
  // Generate recommendations
  const recommendations = generateRecommendations(factors);
  
  console.log(`\n=== DYNAMIC FACTOR ALLOCATION RECOMMENDATIONS ===\n`);
  console.log('Factor | Base Wt | Rec Wt | Signal | Conf | Rationale');
  console.log('-------|---------|--------|--------|------|----------');
  
  for (const rec of recommendations) {
    console.log(
      `${rec.factor.padEnd(6)} | ` +
      `${(rec.currentWeight * 100).toFixed(1)}% | ` +
      `${(rec.recommendedWeight * 100).toFixed(1)}% | ` +
      `${rec.signal.padEnd(11)} | ` +
      `${(rec.confidence * 100).toFixed(0)}% | ` +
      `${rec.rationale.slice(0, 40)}`
    );
  }
  
  // Save output
  const output: SignalOutput = {
    timestamp: new Date().toISOString(),
    analysisDate,
    spyPrice: spyPrice || 0,
    factors,
    recommendations,
    regime: currentRegime,
  };
  
  const outputPath = './data/factor_valuation_momentum_signals.json';
  fs.writeFileSync(outputPath, JSON.stringify(output, null, 2));
  console.log(`\n✓ Results saved to ${outputPath}\n`);
  
  // Summary insights
  console.log('=== KEY INSIGHTS ===\n');
  
  const cheapFactors = factors.filter(f => f.valuationZScore < -1);
  const expensiveFactors = factors.filter(f => f.valuationZScore > 1);
  const positiveMomentum = factors.filter(f => f.momentumSignal === 'positive');
  const negativeMomentum = factors.filter(f => f.momentumSignal === 'negative');
  
  if (cheapFactors.length > 0) {
    console.log(`📉 Cheap factors (z < -1): ${cheapFactors.map(f => f.symbol).join(', ')}`);
  }
  if (expensiveFactors.length > 0) {
    console.log(`📈 Expensive factors (z > 1): ${expensiveFactors.map(f => f.symbol).join(', ')}`);
  }
  if (positiveMomentum.length > 0) {
    console.log(`🚀 Positive momentum: ${positiveMomentum.map(f => f.symbol).join(', ')}`);
  }
  if (negativeMomentum.length > 0) {
    console.log(`📉 Negative momentum: ${negativeMomentum.map(f => f.symbol).join(', ')}`);
  }
  
  console.log(`\n🎯 Regime context: ${currentRegime}`);
  const bestInRegime = factors
    .map(f => ({ symbol: f.symbol, alpha: f.regimePerformance[currentRegime] || 0 }))
    .sort((a, b) => b.alpha - a.alpha)[0];
  if (bestInRegime && bestInRegime.alpha > 0) {
    console.log(`🏆 Best historical performance in ${currentRegime}: ${bestInRegime.symbol} (+${(bestInRegime.alpha * 100).toFixed(1)}% vs SPY)`);
  }
  
  // Phase 1 completion status
  console.log(`\n=== v2.43 PHASE 1 STATUS ===\n`);
  console.log('✅ Phase 1.1: Data fetch for 5 factor ETFs - COMPLETE');
  console.log('✅ Phase 1.2: Factor correlation analysis - COMPLETE');
  console.log('✅ Phase 1.3: Factor valuation & momentum signals - COMPLETE');
  console.log('⏭️  Next: Phase 2 - Backtest engine for dynamic factor timing');
  
  return output;
}

// Run analysis
runFactorAnalysis();
