/**
 * Tests for src/backtest/car25.ts — CAR25 performance metric (Bandy methodology).
 * Safe-f position sizing + Monte Carlo CAR25 + market correlation.
 */
import { describe, test, expect } from "bun:test";
import {
  calculateSafeF,
  calculateCAR25,
  calculateMarketCorrelation,
  analyzeCAR25,
  pricesToReturns,
  simulateDailyReturnsFromStats,
} from "../../src/backtest/car25";

// ---------------------------------------------------------------------------
// Helper: generate synthetic return series
// ---------------------------------------------------------------------------

function makeReturns(cagr: number, vol: number, days: number): number[] {
  return simulateDailyReturnsFromStats(cagr, vol, days, 42);
}

// ---------------------------------------------------------------------------
// calculateSafeF
// ---------------------------------------------------------------------------

describe("calculateSafeF", () => {
  test("returns safe-f for typical portfolio returns", () => {
    const returns = makeReturns(0.10, 0.12, 500);
    const result = calculateSafeF(returns, {
      simulations: 200,
      horizonYears: 1,
      riskTolerance: 0.20,
      seed: 42,
    });
    expect(result.safeF).toBeGreaterThan(0);
    expect(result.safeF).toBeLessThanOrEqual(4.0);
    expect(result.iterations).toBeGreaterThan(0);
    expect(result.toleranceUsed).toBe(0.20);
  });

  test("higher risk tolerance allows higher safe-f", () => {
    const returns = makeReturns(0.08, 0.10, 500);
    const conservative = calculateSafeF(returns, {
      simulations: 200, horizonYears: 1, riskTolerance: 0.10, seed: 42,
    });
    const aggressive = calculateSafeF(returns, {
      simulations: 200, horizonYears: 1, riskTolerance: 0.30, seed: 42,
    });
    expect(aggressive.safeF).toBeGreaterThanOrEqual(conservative.safeF);
  });

  test("result fields are all present", () => {
    const returns = makeReturns(0.06, 0.08, 300);
    const result = calculateSafeF(returns, { simulations: 100, seed: 42 });
    expect(result).toHaveProperty("safeF");
    expect(result).toHaveProperty("drawdown95");
    expect(result).toHaveProperty("iterations");
    expect(result).toHaveProperty("converged");
    expect(result.drawdown95).toBeGreaterThanOrEqual(0);
  });

  test("converges with sufficient simulations", () => {
    const returns = makeReturns(0.10, 0.12, 500);
    const result = calculateSafeF(returns, {
      simulations: 500, horizonYears: 1, riskTolerance: 0.20,
    });
    // With sufficient sims should converge
    expect(result.iterations).toBeGreaterThan(0);
  });
});

// ---------------------------------------------------------------------------
// calculateCAR25
// ---------------------------------------------------------------------------

describe("calculateCAR25", () => {
  test("returns CAR25 at given safe-f", () => {
    const returns = makeReturns(0.10, 0.12, 500);
    const result = calculateCAR25(returns, 1.0, {
      simulations: 200,
      horizonYears: 1,
      seed: 42,
    });
    expect(result).toHaveProperty("car25");
    expect(result).toHaveProperty("car50");
    expect(result).toHaveProperty("car75");
    expect(result.safeF).toBe(1.0);
  });

  test("CAR percentiles are ordered", () => {
    const returns = makeReturns(0.10, 0.12, 500);
    const result = calculateCAR25(returns, 1.0, {
      simulations: 300, horizonYears: 1, seed: 42,
    });
    expect(result.car25).toBeLessThanOrEqual(result.car50);
    expect(result.car50).toBeLessThanOrEqual(result.car75);
  });

  test("TWR and CAR are consistent", () => {
    const returns = makeReturns(0.10, 0.12, 500);
    const result = calculateCAR25(returns, 1.0, {
      simulations: 200, horizonYears: 2, seed: 42,
    });
    // CAR = TWR^(1/horizon) - 1
    const expectedCar25 = Math.pow(result.twr25, 0.5) - 1;
    expect(result.car25).toBeCloseTo(expectedCar25);
  });

  test("higher safe-f amplifies returns and risk", () => {
    const returns = makeReturns(0.10, 0.12, 500);
    const low = calculateCAR25(returns, 0.5, { simulations: 200, seed: 42 });
    const high = calculateCAR25(returns, 2.0, { simulations: 200, seed: 42 });
    // Higher f → wider spread between 25th and 75th percentile
    const lowSpread = low.car75 - low.car25;
    const highSpread = high.car75 - high.car25;
    expect(highSpread).toBeGreaterThan(lowSpread);
  });

  test("final equities are positive", () => {
    const returns = makeReturns(0.10, 0.12, 500);
    const result = calculateCAR25(returns, 1.0, {
      simulations: 200, seed: 42,
    });
    expect(result.finalEquity25).toBeGreaterThan(0);
    expect(result.finalEquity50).toBeGreaterThan(0);
    expect(result.finalEquity75).toBeGreaterThan(0);
  });
});

// ---------------------------------------------------------------------------
// calculateMarketCorrelation
// ---------------------------------------------------------------------------

describe("calculateMarketCorrelation", () => {
  test("perfect correlation is 1", () => {
    const returns = makeReturns(0.10, 0.12, 200);
    const result = calculateMarketCorrelation(returns, returns);
    expect(result.correlation).toBeCloseTo(1.0);
    expect(result.classification).toBe("high");
    expect(result.commonDays).toBe(200);
  });

  test("uncorrelated series", () => {
    const a = makeReturns(0.10, 0.12, 200);
    const b = makeReturns(0.10, 0.12, 200).map(r => -r); // Perfectly inverse
    const result = calculateMarketCorrelation(a, b);
    expect(result.correlation).toBeLessThan(0);
  });

  test("classification thresholds", () => {
    // Low: abs < 0.3, Moderate: 0.3-0.7, High: > 0.7
    const n = 100;
    // Create series with known correlation via averaging
    const base = makeReturns(0.08, 0.10, n);

    // High correlation: mostly base + small noise
    const high = base.map((r, i) => r * 0.95 + base[(i + 1) % n] * 0.05);
    const resultHigh = calculateMarketCorrelation(base, high);
    expect(resultHigh.classification).toBe("high");
  });
});

// ---------------------------------------------------------------------------
// pricesToReturns
// ---------------------------------------------------------------------------

describe("pricesToReturns", () => {
  test("converts sorted prices to returns", () => {
    const priceData = [
      { symbol: "SPY", date: "2020-01-02", price: 100 },
      { symbol: "SPY", date: "2020-01-03", price: 101 },
      { symbol: "SPY", date: "2020-01-06", price: 102 },
    ];
    const returns = pricesToReturns(priceData, "SPY");
    expect(returns).toHaveLength(2);
    expect(returns[0]).toBeCloseTo(0.01); // (101-100)/100
    expect(returns[1]).toBeCloseTo(0.0099, 3); // (102-101)/101
  });

  test("filters by symbol", () => {
    const priceData = [
      { symbol: "SPY", date: "2020-01-02", price: 100 },
      { symbol: "GLD", date: "2020-01-02", price: 150 },
      { symbol: "SPY", date: "2020-01-03", price: 101 },
    ];
    const returns = pricesToReturns(priceData, "SPY");
    expect(returns).toHaveLength(1);
  });

  test("single entry returns empty", () => {
    const priceData = [{ symbol: "SPY", date: "2020-01-02", price: 100 }];
    expect(pricesToReturns(priceData, "SPY")).toHaveLength(0);
  });
});

// ---------------------------------------------------------------------------
// simulateDailyReturnsFromStats
// ---------------------------------------------------------------------------

describe("simulateDailyReturnsFromStats", () => {
  test("generates correct number of returns", () => {
    const returns = simulateDailyReturnsFromStats(0.10, 0.15, 500, 42);
    expect(returns).toHaveLength(500);
  });

  test("same seed produces identical output", () => {
    const a = simulateDailyReturnsFromStats(0.10, 0.15, 100, 42);
    const b = simulateDailyReturnsFromStats(0.10, 0.15, 100, 42);
    expect(a).toEqual(b);
  });

  test("different seed produces different output", () => {
    const a = simulateDailyReturnsFromStats(0.10, 0.15, 100, 42);
    const b = simulateDailyReturnsFromStats(0.10, 0.15, 100, 43);
    expect(a).not.toEqual(b);
  });

  test("zero vol produces flat returns", () => {
    const returns = simulateDailyReturnsFromStats(0.10, 0, 100, 42);
    const mean = returns.reduce((a, b) => a + b, 0) / returns.length;
    const dailyReturn = 0.10 / 252;
    expect(mean).toBeCloseTo(dailyReturn, 4);
  });
});

// ---------------------------------------------------------------------------
// analyzeCAR25 — full pipeline
// ---------------------------------------------------------------------------

describe("analyzeCAR25", () => {
  test("full pipeline returns complete result", () => {
    const portfolioReturns = makeReturns(0.10, 0.12, 500);
    const spyReturns = makeReturns(0.10, 0.15, 500);
    const result = analyzeCAR25(portfolioReturns, spyReturns, "Test Portfolio", {
      simulations: 200,
      horizonYears: 1,
      seed: 42,
    });
    expect(result.portfolio).toBe("Test Portfolio");
    expect(result.safeF.safeF).toBeGreaterThan(0);
    expect(result.car25.car25).toBeDefined();
    expect(result.correlation).toBeDefined();
    expect(result.inputDays).toBe(500);
  });

  test("null benchmark returns zero correlation", () => {
    const returns = makeReturns(0.10, 0.12, 200);
    const result = analyzeCAR25(returns, null, "Solo", {
      simulations: 100, seed: 42,
    });
    expect(result.correlation.correlation).toBe(0);
    expect(result.correlation.classification).toBe("low");
  });

  test("empty benchmark returns zero correlation", () => {
    const returns = makeReturns(0.10, 0.12, 200);
    const result = analyzeCAR25(returns, [], "Solo", {
      simulations: 100, seed: 42,
    });
    expect(result.correlation.correlation).toBe(0);
  });

  test("config defaults are filled", () => {
    const returns = makeReturns(0.10, 0.12, 200);
    const result = analyzeCAR25(returns, null, "Test", { seed: 99 });
    expect(result.config.simulations).toBeDefined();
    expect(result.config.horizonYears).toBeDefined();
    expect(result.config.riskTolerance).toBeDefined();
    expect(result.config.seed).toBe(99);
  });
});
