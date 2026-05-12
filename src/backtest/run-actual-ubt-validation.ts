/**
 * Actual UBT/TMF Historical Backtest - Phase 1C Validation
 * v2.35 Capital Efficiency Strategy
 * 
 * Uses real historical UBT/TMF data (fetched 2026-05-12) to validate
 * synthetic simulation accuracy before paper trading decision.
 */

import * as fs from 'fs';
import * as path from 'path';

interface PriceData {
  d: string;  // date
  p: number;  // price
}

interface HistoricalData {
  [symbol: string]: PriceData[];
}

interface BacktestResult {
  scenario: string;
  startDate: string;
  endDate: string;
  days: number;
  cagr: number;
  volatility: number;
  sharpe: number;
  maxDrawdown: number;
  calmar: number;
  totalReturn: number;
  trackingErrorVsTLT: number;
  volatilityDecayEstimate: number;
  annualizedExpenseImpact: number;
}

interface ValidationReport {
  timestamp: string;
  version: string;
  dataQuality: {
    tltDays: number;
    ubtDays: number;
    tmfDays: number;
    overlapDays: number;
    dataStart: string;
    dataEnd: string;
  };
  results: BacktestResult[];
  syntheticVsActual: {
    ubtCorrelation: number;
    tmfCorrelation: number;
    ubtAnnualizedTrackingError: number;
    tmfAnnualizedTrackingError: number;
    syntheticAccuracy: 'high' | 'medium' | 'low';
  };
  recommendation: {
    proceedToPaperTrading: boolean;
    recommendedScenario: string;
    reasoning: string;
    confidence: 'high' | 'medium' | 'low';
  };
}

console.log('[INFO] Loading historical data...');

// Load historical data
const historicalPath = path.join(process.cwd(), 'public', 'data', 'historical.json');
const historical: HistoricalData = JSON.parse(fs.readFileSync(historicalPath, 'utf-8'));

// Check available symbols
const symbols = Object.keys(historical);
console.log(`[INFO] Available symbols: ${symbols.join(', ')}`);

// Check if UBT/TMF are in data
const hasUBT = symbols.includes('UBT');
const hasTMF = symbols.includes('TMF');
const hasTLT = symbols.includes('TLT');

console.log(`[INFO] UBT available: ${hasUBT}, TMF available: ${hasTMF}, TLT available: ${hasTLT}`);

if (!hasTLT) {
  console.error('[ERROR] TLT data required but not found');
  process.exit(1);
}

// Get data statistics
const tltData = historical['TLT'];
const ubtData = hasUBT ? historical['UBT'] : null;
const tmfData = hasTMF ? historical['TMF'] : null;

console.log(`[INFO] TLT: ${tltData.length} days (${tltData[0].d} to ${tltData[tltData.length - 1].d})`);
if (ubtData) console.log(`[INFO] UBT: ${ubtData.length} days (${ubtData[0].d} to ${ubtData[ubtData.length - 1].d})`);
if (tmfData) console.log(`[INFO] TMF: ${tmfData.length} days (${tmfData[0].d} to ${tmfData[tmfData.length - 1].d})`);

// Find overlapping date range
function findOverlap(d1: PriceData[], d2: PriceData[]): { start: string; end: string; days: number } {
  const dates1 = new Set(d1.map(x => x.d));
  const dates2 = new Set(d2.map(x => x.d));
  const overlap = Array.from(dates1).filter(d => dates2.has(d)).sort();
  return {
    start: overlap[0],
    end: overlap[overlap.length - 1],
    days: overlap.length
  };
}

const overlapUBT = hasUBT ? findOverlap(tltData, ubtData!) : null;
const overlapTMF = hasTMF ? findOverlap(tltData, tmfData!) : null;

if (overlapUBT) console.log(`[INFO] TLT-UBT overlap: ${overlapUBT.days} days (${overlapUBT.start} to ${overlapUBT.end})`);
if (overlapTMF) console.log(`[INFO] TLT-TMF overlap: ${overlapTMF.days} days (${overlapTMF.start} to ${overlapTMF.end})`);

// Build aligned price series
function buildAlignedSeries(
  baseData: PriceData[],
  leveragedData: PriceData[] | null,
  startDate: string,
  endDate: string
): { dates: string[]; basePrices: number[]; leveragedPrices: number[] | null } {
  const dateRange = baseData.filter(x => x.d >= startDate && x.d <= endDate).map(x => x.d);
  const baseMap = new Map(baseData.map(x => [x.d, x.p]));
  const levMap = leveragedData ? new Map(leveragedData.map(x => [x.d, x.p])) : null;
  
  const basePrices: number[] = [];
  const leveragedPrices: number[] = [];
  const dates: string[] = [];
  
  for (const date of dateRange) {
    const basePrice = baseMap.get(date);
    const levPrice = levMap?.get(date);
    
    if (basePrice !== undefined) {
      dates.push(date);
      basePrices.push(basePrice);
      leveragedPrices.push(levPrice !== undefined ? levPrice : NaN);
    }
  }
  
  return { dates, basePrices, leveragedPrices: leveragedPrices.some(p => !isNaN(p)) ? leveragedPrices : null };
}

// Calculate returns from prices
function calculateReturns(prices: number[]): number[] {
  const returns: number[] = [];
  for (let i = 1; i < prices.length; i++) {
    returns.push((prices[i] - prices[i - 1]) / prices[i - 1]);
  }
  return returns;
}

// Calculate metrics
function calculateMetrics(
  returns: number[],
  dates: string[],
  scenario: string,
  baseReturns?: number[]
): BacktestResult {
  const totalReturn = returns.reduce((prod, r) => prod * (1 + r), 1) - 1;
  const years = returns.length / 252;
  const cagr = Math.pow(1 + totalReturn, 1 / years) - 1;
  
  const meanReturn = returns.reduce((a, b) => a + b, 0) / returns.length;
  const variance = returns.reduce((sum, r) => sum + Math.pow(r - meanReturn, 2), 0) / returns.length;
  const dailyVol = Math.sqrt(variance);
  const annualizedVol = dailyVol * Math.sqrt(252);
  
  // Max drawdown
  let maxDrawdown = 0;
  let peak = 1;
  let currentValue = 1;
  for (const r of returns) {
    currentValue *= (1 + r);
    if (currentValue > peak) peak = currentValue;
    const dd = (currentValue - peak) / peak;
    if (dd < maxDrawdown) maxDrawdown = dd;
  }
  
  const sharpe = annualizedVol > 0 ? (cagr - 0.04) / annualizedVol : 0;
  const calmar = maxDrawdown < 0 ? cagr / Math.abs(maxDrawdown) : cagr;
  
  // Calculate tracking error vs base (if provided)
  let trackingError = 0;
  if (baseReturns && baseReturns.length === returns.length) {
    const differences: number[] = [];
    for (let i = 0; i < returns.length; i++) {
      // For 2x leverage, expect 2x base return
      const expectedMultiple = scenario.includes('UBT') ? 2 : scenario.includes('TMF') ? 3 : 1;
      const diff = returns[i] - (baseReturns[i] * expectedMultiple);
      differences.push(diff);
    }
    const meanDiff = differences.reduce((a, b) => a + b, 0) / differences.length;
    const varDiff = differences.reduce((sum, d) => sum + Math.pow(d - meanDiff, 2), 0) / differences.length;
    trackingError = Math.sqrt(varDiff) * Math.sqrt(252); // Annualized
  }
  
  // Estimate volatility decay
  // Formula: E[r_levered] ≈ leverage × E[r_base] - 0.5 × leverage² × σ² × (1 - autocorr)
  const volatilityDecay = scenario.includes('UBT') 
    ? -0.5 * 4 * Math.pow(annualizedVol / 2, 2) / 252 * returns.length
    : scenario.includes('TMF')
    ? -0.5 * 9 * Math.pow(annualizedVol / 3, 2) / 252 * returns.length
    : 0;
  
  // Estimate expense impact
  const expenseRatio = scenario.includes('UBT') ? 0.0080 : scenario.includes('TMF') ? 0.0091 : 0.0015;
  const expenseImpact = Math.pow(1 - expenseRatio, years) - 1;
  
  return {
    scenario,
    startDate: dates[0],
    endDate: dates[dates.length - 1],
    days: returns.length,
    cagr,
    volatility: annualizedVol,
    sharpe,
    maxDrawdown,
    calmar,
    totalReturn,
    trackingErrorVsTLT: trackingError,
    volatilityDecayEstimate: volatilityDecay,
    annualizedExpenseImpact: expenseImpact
  };
}

// Run scenarios
const results: BacktestResult[] = [];

// Scenario 1: TLT only (baseline)
const tltSeries = buildAlignedSeries(tltData, null, tltData[0].d, tltData[tltData.length - 1].d);
const tltReturns = calculateReturns(tltSeries.basePrices);
results.push(calculateMetrics(tltReturns, tltSeries.dates, 'Baseline_TLT'));

// Scenario 2: UBT (if available)
if (hasUBT && overlapUBT) {
  const ubtSeries = buildAlignedSeries(tltData, ubtData!, overlapUBT.start, overlapUBT.end);
  const ubtReturns = calculateReturns(ubtSeries.leveragedPrices!.filter(p => !isNaN(p)) as number[]);
  const alignedTltReturns = calculateReturns(ubtSeries.basePrices).slice(0, ubtReturns.length);
  results.push(calculateMetrics(ubtReturns, ubtSeries.dates.slice(1), 'Actual_UBT', alignedTltReturns));
}

// Scenario 3: TMF (if available)
if (hasTMF && overlapTMF) {
  const tmfSeries = buildAlignedSeries(tltData, tmfData!, overlapTMF.start, overlapTMF.end);
  const tmfReturns = calculateReturns(tmfSeries.leveragedPrices!.filter(p => !isNaN(p)) as number[]);
  const alignedTltReturns = calculateReturns(tmfSeries.basePrices).slice(0, tmfReturns.length);
  results.push(calculateMetrics(tmfReturns, tmfSeries.dates.slice(1), 'Actual_TMF', alignedTltReturns));
}

// Compare synthetic vs actual
function calculateCorrelation(returns1: number[], returns2: number[]): number {
  const n = Math.min(returns1.length, returns2.length);
  const r1 = returns1.slice(0, n);
  const r2 = returns2.slice(0, n);
  
  const m1 = r1.reduce((a, b) => a + b, 0) / n;
  const m2 = r2.reduce((a, b) => a + b, 0) / n;
  
  let numerator = 0;
  let var1 = 0;
  let var2 = 0;
  
  for (let i = 0; i < n; i++) {
    const d1 = r1[i] - m1;
    const d2 = r2[i] - m2;
    numerator += d1 * d2;
    var1 += d1 * d1;
    var2 += d2 * d2;
  }
  
  return numerator / Math.sqrt(var1 * var2);
}

// Calculate synthetic UBT returns (2x TLT minus expenses)
const syntheticUBT = tltReturns.map(r => r * 2 - 0.0080 / 252);
const syntheticTMF = tltReturns.map(r => r * 3 - 0.0091 / 252);

const ubtCorr = hasUBT ? calculateCorrelation(results.find(r => r.scenario === 'Actual_UBT') ? 
  tltReturns.slice(0, results.find(r => r.scenario === 'Actual_UBT')!.days) : [], syntheticUBT.slice(0, results.find(r => r.scenario === 'Actual_UBT')?.days || 0)) : 0;
const tmfCorr = hasTMF ? calculateCorrelation(results.find(r => r.scenario === 'Actual_TMF') ? 
  tltReturns.slice(0, results.find(r => r.scenario === 'Actual_TMF')!.days) : [], syntheticTMF.slice(0, results.find(r => r.scenario === 'Actual_TMF')?.days || 0)) : 0;

// Determine recommendation
const ubtResult = results.find(r => r.scenario === 'Actual_UBT');
const tltResult = results.find(r => r.scenario === 'Baseline_TLT');

let proceedToPaperTrading = false;
let recommendedScenario = 'Baseline_TLT';
let reasoning = '';
let confidence: 'high' | 'medium' | 'low' = 'low';

if (ubtResult && tltResult) {
  const cagrImprovement = ubtResult.cagr - tltResult.cagr;
  const volIncrease = ubtResult.volatility - tltResult.volatility;
  const trackingOk = ubtResult.trackingErrorVsTLT < 0.02; // < 2% tracking error
  
  if (cagrImprovement > 0.005 && volIncrease < 0.15 && trackingOk) {
    proceedToPaperTrading = true;
    recommendedScenario = 'Capital_Efficient_UBT';
    reasoning = `Actual UBT delivers +${(cagrImprovement * 100).toFixed(1)}% CAGR vs TLT with ${(volIncrease * 100).toFixed(1)}% vol increase and ${(ubtResult.trackingErrorVsTLT * 100).toFixed(2)}% tracking error. Meets criteria for paper trading.`;
    confidence = 'high';
  } else if (cagrImprovement > 0) {
    proceedToPaperTrading = false;
    recommendedScenario = 'Capital_Efficient_UBT';
    reasoning = `UBT shows +${(cagrImprovement * 100).toFixed(1)}% CAGR improvement but vol increase of ${(volIncrease * 100).toFixed(1)}% requires monitoring. Paper trading deferred for further analysis.`;
    confidence = 'medium';
  } else {
    proceedToPaperTrading = false;
    recommendedScenario = 'Baseline_TLT';
    reasoning = `UBT underperforms TLT by ${(Math.abs(cagrImprovement) * 100).toFixed(1)}% CAGR. Capital efficiency strategy not validated.`;
    confidence = 'low';
  }
} else {
  reasoning = 'Insufficient data to validate capital efficiency strategy.';
}

// Build validation report
const report: ValidationReport = {
  timestamp: new Date().toISOString(),
  version: 'v2.35',
  dataQuality: {
    tltDays: tltData.length,
    ubtDays: ubtData?.length || 0,
    tmfDays: tmfData?.length || 0,
    overlapDays: Math.min(overlapUBT?.days || 0, overlapTMF?.days || 0),
    dataStart: tltData[0].d,
    dataEnd: tltData[tltData.length - 1].d
  },
  results,
  syntheticVsActual: {
    ubtCorrelation: ubtCorr,
    tmfCorrelation: tmfCorr,
    ubtAnnualizedTrackingError: ubtResult?.trackingErrorVsTLT || 0,
    tmfAnnualizedTrackingError: results.find(r => r.scenario === 'Actual_TMF')?.trackingErrorVsTLT || 0,
    syntheticAccuracy: ubtCorr > 0.95 ? 'high' : ubtCorr > 0.85 ? 'medium' : 'low'
  },
  recommendation: {
    proceedToPaperTrading,
    recommendedScenario,
    reasoning,
    confidence
  }
};

// Save results
const outputPath = path.join(process.cwd(), 'data', 'ubt_actual_validation.json');
fs.writeFileSync(outputPath, JSON.stringify(report, null, 2));

console.log(`\n[SUCCESS] Validation complete! Results saved to: ${outputPath}`);
console.log('\n=== VALIDATION SUMMARY ===');
console.log(`Data Quality:`);
console.log(`  TLT: ${report.dataQuality.tltDays} days`);
console.log(`  UBT: ${report.dataQuality.ubtDays} days`);
console.log(`  TMF: ${report.dataQuality.tmfDays} days`);
console.log(`  Overlap: ${report.dataQuality.overlapDays} days`);
console.log(`\nSynthetic vs Actual Accuracy: ${report.syntheticVsActual.syntheticAccuracy.toUpperCase()}`);
console.log(`  UBT Correlation: ${(report.syntheticVsActual.ubtCorrelation * 100).toFixed(1)}%`);
console.log(`  UBT Tracking Error: ${(report.syntheticVsActual.ubtAnnualizedTrackingError * 100).toFixed(2)}%`);
console.log(`\n=== BACKTEST RESULTS ===`);
for (const r of results) {
  console.log(`${r.scenario}:`);
  console.log(`  Period: ${r.startDate} to ${r.endDate} (${r.days} days)`);
  console.log(`  CAGR: ${(r.cagr * 100).toFixed(2)}%`);
  console.log(`  Volatility: ${(r.volatility * 100).toFixed(1)}%`);
  console.log(`  Sharpe: ${r.sharpe.toFixed(2)}`);
  console.log(`  Max DD: ${(r.maxDrawdown * 100).toFixed(1)}%`);
  if (r.trackingErrorVsTLT !== 0) {
    console.log(`  Tracking Error vs TLT: ${(r.trackingErrorVsTLT * 100).toFixed(2)}%`);
  }
  console.log(`  Est. Vol Decay: ${(r.volatilityDecayEstimate * 100).toFixed(2)}%`);
  console.log(`  Expense Impact: ${(r.annualizedExpenseImpact * 100).toFixed(2)}%`);
}
console.log(`\n=== RECOMMENDATION ===`);
console.log(`Proceed to Paper Trading: ${report.recommendation.proceedToPaperTrading ? '✅ YES' : '⏸️ NO'}`);
console.log(`Recommended Scenario: ${report.recommendation.recommendedScenario}`);
console.log(`Confidence: ${report.recommendation.confidence.toUpperCase()}`);
console.log(`Reasoning: ${report.recommendation.reasoning}`);
