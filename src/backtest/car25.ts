/**
 * CAR25 Performance Metric - Bandy's risk-normalized objective function
 * 
 * CAR25 = Compound Annual Rate of Return at the 25th percentile
 * after position-sizing via safe-f (max drawdown-constrained).
 * 
 * Two-stage process:
 *   1. Safe-f: Binary search for position size where 95th %ile max DD = tolerance
 *   2. CAR25: Monte Carlo at safe-f, extract 25th percentile CAGR
 * 
 * Companion metric: Correlation to SPY benchmark
 */

import type { PriceData, PortfolioConfig } from './engine';

// Default simulation parameters
const DEFAULT_SIMULATIONS = 1000;
const DEFAULT_HORIZON_YEARS = 2;
const DEFAULT_RISK_TOLERANCE = 0.20; // 20% max DD
const DEFAULT_CONFIDENCE = 0.95;
const DEFAULT_BLOCK_SIZE = 20; // ~1 month blocks for autocorrelation
const TRADING_DAYS_PER_YEAR = 252;

interface CAR25Config {
  simulations?: number;
  horizonYears?: number;
  riskTolerance?: number; // e.g., 0.20 = 20% max drawdown
  confidenceLevel?: number; // e.g., 0.95 = 95th percentile
  blockSize?: number;
  seed?: number;
}

interface SafeFResult {
  safeF: number; // Position size fraction (0.01 to 4.0)
  drawdown95: number; // Actual 95th %ile max DD achieved
  iterations: number;
  converged: boolean;
  toleranceUsed: number;
}

interface CAR25Result {
  car25: number; // 25th percentile annualized return
  car50: number; // Median annualized return
  car75: number; // 75th percentile (optimistic)
  twr25: number; // 25th percentile terminal wealth ratio
  twr50: number;
  twr75: number;
  safeF: number;
  finalEquity25: number; // 25th percentile final portfolio value
  finalEquity50: number;
  finalEquity75: number;
}

interface MarketCorrelationResult {
  correlation: number; // Pearson's ρ (-1 to 1)
  classification: 'low' | 'moderate' | 'high';
  commonDays: number;
}

interface CAR25FullResult {
  portfolio: string;
  safeF: SafeFResult;
  car25: CAR25Result;
  correlation: MarketCorrelationResult;
  config: Required<CAR25Config>;
  inputDays: number;
}

// Seeded RNG for reproducibility
function makeSeededRng(seed: number): () => number {
  let s = seed;
  return () => {
    s = Math.sin(s * 12.9898 + 78.233) * 43758.5453;
    return s - Math.floor(s);
  };
}

/**
 * Calculate daily returns from price series
 */
function calculateDailyReturns(prices: number[]): number[] {
  const returns: number[] = [];
  for (let i = 1; i < prices.length; i++) {
    returns.push((prices[i] - prices[i - 1]) / prices[i - 1]);
  }
  return returns;
}

/**
 * Block bootstrap: resample daily returns in blocks to preserve autocorrelation
 */
function blockBootstrapReturns(
  dailyReturns: number[],
  numDays: number,
  blockSize: number,
  rng: () => number,
): number[] {
  const result: number[] = [];
  const n = dailyReturns.length;
  while (result.length < numDays) {
    const start = Math.floor(rng() * (n - blockSize));
    for (let i = 0; i < blockSize && result.length < numDays; i++) {
      result.push(dailyReturns[start + i]);
    }
  }
  return result.slice(0, numDays);
}

/**
 * Calculate max drawdown from equity curve
 */
function calculateMaxDrawdown(equityCurve: number[]): number {
  let peak = equityCurve[0];
  let maxDD = 0;
  for (const value of equityCurve) {
    if (value > peak) peak = value;
    const dd = (peak - value) / peak;
    if (dd > maxDD) maxDD = dd;
  }
  return maxDD;
}

/**
 * Simulate equity curve at given position size fraction f
 */
function simulateAtFraction(
  bootstrapReturns: number[],
  initialEquity: number,
  f: number,
): number[] {
  const equity: number[] = [initialEquity];
  for (const ret of bootstrapReturns) {
    const leveragedReturn = ret * f;
    equity.push(equity[equity.length - 1] * (1 + leveragedReturn));
  }
  return equity;
}

/**
 * Stage 1: Calculate safe-f via binary search
 * Find position size fraction where 95th %ile max DD = risk tolerance
 */
export function calculateSafeF(
  dailyReturns: number[],
  config: CAR25Config = {},
): SafeFResult {
  const {
    simulations = DEFAULT_SIMULATIONS,
    horizonYears = DEFAULT_HORIZON_YEARS,
    riskTolerance = DEFAULT_RISK_TOLERANCE,
    confidenceLevel = DEFAULT_CONFIDENCE,
    blockSize = DEFAULT_BLOCK_SIZE,
    seed = 42,
  } = config;

  const rng = makeSeededRng(seed);
  const numDays = Math.floor(horizonYears * TRADING_DAYS_PER_YEAR);
  const targetQuantile = Math.floor(simulations * confidenceLevel);
  const initialEquity = 100000;

  // Binary search bounds: 0.01 to 4.0 (1% to 400% position size)
  let lowF = 0.01;
  let highF = 4.0;
  let iterations = 0;
  const maxIterations = 20;

  let bestF = 0.01;
  let bestDD = 0;
  let converged = false;

  while (iterations < maxIterations) {
    const midF = (lowF + highF) / 2;
    const drawdowns: number[] = [];

    // Run Monte Carlo simulations at this f
    for (let sim = 0; sim < simulations; sim++) {
      const bootstrapReturns = blockBootstrapReturns(dailyReturns, numDays, blockSize, rng);
      const equity = simulateAtFraction(bootstrapReturns, initialEquity, midF);
      const maxDD = calculateMaxDrawdown(equity);
      drawdowns.push(maxDD);
    }

    // Sort to get 95th percentile drawdown
    drawdowns.sort((a, b) => a - b);
    const dd95 = drawdowns[targetQuantile] ?? drawdowns[Math.floor(drawdowns.length * confidenceLevel)];

    bestF = midF;
    bestDD = dd95;

    // Check convergence (within 0.5% of target)
    if (Math.abs(dd95 - riskTolerance) < 0.005) {
      converged = true;
      break;
    }

    // Adjust search bounds
    if (dd95 > riskTolerance) {
      // Drawdown too high, reduce position size
      highF = midF;
    } else {
      // Drawdown acceptable, can increase position size
      lowF = midF;
    }

    iterations++;
  }

  return {
    safeF: bestF,
    drawdown95: bestDD,
    iterations,
    converged,
    toleranceUsed: riskTolerance,
  };
}

/**
 * Stage 2: Calculate CAR25 at safe-f position size
 * Monte Carlo to get distribution of terminal wealth, extract 25th %ile CAGR
 */
export function calculateCAR25(
  dailyReturns: number[],
  safeF: number,
  config: CAR25Config = {},
): CAR25Result {
  const {
    simulations = DEFAULT_SIMULATIONS,
    horizonYears = DEFAULT_HORIZON_YEARS,
    blockSize = DEFAULT_BLOCK_SIZE,
    seed = 42,
  } = config;

  const rng = makeSeededRng(seed + 1); // Different seed from safe-f calculation
  const numDays = Math.floor(horizonYears * TRADING_DAYS_PER_YEAR);
  const initialEquity = 100000;

  const terminalValues: number[] = [];

  // Run Monte Carlo at safe-f
  for (let sim = 0; sim < simulations; sim++) {
    const bootstrapReturns = blockBootstrapReturns(dailyReturns, numDays, blockSize, rng);
    const equity = simulateAtFraction(bootstrapReturns, initialEquity, safeF);
    terminalValues.push(equity[equity.length - 1]);
  }

  // Sort to get percentiles
  terminalValues.sort((a, b) => a - b);

  const p25Index = Math.floor(simulations * 0.25);
  const p50Index = Math.floor(simulations * 0.50);
  const p75Index = Math.floor(simulations * 0.75);

  const tv25 = terminalValues[p25Index];
  const tv50 = terminalValues[p50Index];
  const tv75 = terminalValues[p75Index];

  // Terminal Wealth Ratio (TWR) = final / initial
  const twr25 = tv25 / initialEquity;
  const twr50 = tv50 / initialEquity;
  const twr75 = tv75 / initialEquity;

  // Annualize: CAR = TWR^(1/years) - 1
  const car25 = Math.pow(twr25, 1 / horizonYears) - 1;
  const car50 = Math.pow(twr50, 1 / horizonYears) - 1;
  const car75 = Math.pow(twr75, 1 / horizonYears) - 1;

  return {
    car25,
    car50,
    car75,
    twr25,
    twr50,
    twr75,
    safeF,
    finalEquity25: tv25,
    finalEquity50: tv50,
    finalEquity75: tv75,
  };
}

/**
 * Calculate correlation between portfolio returns and benchmark (e.g., SPY)
 */
export function calculateMarketCorrelation(
  portfolioReturns: number[],
  benchmarkReturns: number[],
): MarketCorrelationResult {
  // Align to common length
  const n = Math.min(portfolioReturns.length, benchmarkReturns.length);
  const p = portfolioReturns.slice(-n);
  const b = benchmarkReturns.slice(-n);

  // Calculate means
  const meanP = p.reduce((a, v) => a + v, 0) / n;
  const meanB = b.reduce((a, v) => a + v, 0) / n;

  // Calculate Pearson correlation
  let num = 0;
  let denP = 0;
  let denB = 0;

  for (let i = 0; i < n; i++) {
    const diffP = p[i] - meanP;
    const diffB = b[i] - meanB;
    num += diffP * diffB;
    denP += diffP * diffP;
    denB += diffB * diffB;
  }

  const correlation = num / Math.sqrt(denP * denB);

  // Classification
  let classification: 'low' | 'moderate' | 'high';
  const absCorr = Math.abs(correlation);
  if (absCorr < 0.3) classification = 'low';
  else if (absCorr < 0.7) classification = 'moderate';
  else classification = 'high';

  return {
    correlation,
    classification,
    commonDays: n,
  };
}

/**
 * Full CAR25 analysis: safe-f + CAR25 + correlation
 */
export function analyzeCAR25(
  portfolioReturns: number[],
  benchmarkReturns: number[] | null,
  portfolioName: string,
  config: CAR25Config = {},
): CAR25FullResult {
  // Stage 1: Find safe-f
  const safeF = calculateSafeF(portfolioReturns, config);

  // Stage 2: Calculate CAR25 at safe-f
  const car25 = calculateCAR25(portfolioReturns, safeF.safeF, config);

  // Companion metric: correlation to benchmark
  let correlation: MarketCorrelationResult;
  if (benchmarkReturns && benchmarkReturns.length > 0) {
    correlation = calculateMarketCorrelation(portfolioReturns, benchmarkReturns);
  } else {
    correlation = {
      correlation: 0,
      classification: 'low',
      commonDays: portfolioReturns.length,
    };
  }

  const fullConfig: Required<CAR25Config> = {
    simulations: config.simulations ?? DEFAULT_SIMULATIONS,
    horizonYears: config.horizonYears ?? DEFAULT_HORIZON_YEARS,
    riskTolerance: config.riskTolerance ?? DEFAULT_RISK_TOLERANCE,
    confidenceLevel: config.confidenceLevel ?? DEFAULT_CONFIDENCE,
    blockSize: config.blockSize ?? DEFAULT_BLOCK_SIZE,
    seed: config.seed ?? 42,
  };

  return {
    portfolio: portfolioName,
    safeF,
    car25,
    correlation,
    config: fullConfig,
    inputDays: portfolioReturns.length,
  };
}

/**
 * Price data to daily returns converter
 */
export function pricesToReturns(priceData: PriceData[], symbol: string): number[] {
  const symbolData = priceData
    .filter(p => p.symbol === symbol)
    .sort((a, b) => a.date.localeCompare(b.date));
  
  const prices = symbolData.map(p => p.price);
  return calculateDailyReturns(prices);
}

/**
 * Backtest results to daily returns (simulated from CAGR and volatility)
 * For use when we only have summary stats, not daily data
 */
export function simulateDailyReturnsFromStats(
  cagr: number,
  volatility: number,
  days: number,
  seed: number = 42,
): number[] {
  const rng = makeSeededRng(seed);
  const dailyReturn = cagr / TRADING_DAYS_PER_YEAR;
  const dailyVol = volatility / Math.sqrt(TRADING_DAYS_PER_YEAR);

  const returns: number[] = [];
  for (let i = 0; i < days; i++) {
    // Box-Muller transform for normal distribution
    const u1 = rng();
    const u2 = rng();
    const z = Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
    returns.push(dailyReturn + z * dailyVol);
  }
  return returns;
}

// CLI execution
if (import.meta.main) {
  const args = process.argv.slice(2);
  
  // Parse arguments
  let portfolioName = 'Portfolio';
  let riskTolerance = DEFAULT_RISK_TOLERANCE;
  let horizonYears = DEFAULT_HORIZON_YEARS;
  let compareAll = false;

  for (let i = 0; i < args.length; i++) {
    if (args[i] === '--portfolio' && args[i + 1]) {
      portfolioName = args[i + 1];
      i++;
    } else if (args[i] === '--tolerance' && args[i + 1]) {
      riskTolerance = parseFloat(args[i + 1]);
      i++;
    } else if (args[i] === '--horizon' && args[i + 1]) {
      horizonYears = parseFloat(args[i + 1]);
      i++;
    } else if (args[i] === '--compare-all') {
      compareAll = true;
    }
  }

  if (compareAll) {
    console.log('CAR25 Analysis for All Portfolios');
    console.log('=================================\n');
    console.log('Note: Load price data and run full analysis via TypeScript import');
    console.log('Example:');
    console.log('  import { analyzeCAR25, pricesToReturns } from "./car25";');
    console.log('  const spyReturns = pricesToReturns(priceData, "SPY");');
    console.log('  const result = analyzeCAR25(portfolioReturns, spyReturns, "My Portfolio");');
    process.exit(0);
  }

  console.log(`CAR25 Analysis: ${portfolioName}`);
  console.log(`Risk Tolerance: ${(riskTolerance * 100).toFixed(0)}%`);
  console.log(`Horizon: ${horizonYears} years\n`);
  console.log('Use --compare-all to see all portfolios or import module for full analysis.');
}

export type {
  CAR25Config,
  SafeFResult,
  CAR25Result,
  MarketCorrelationResult,
  CAR25FullResult,
};
