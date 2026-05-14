/**
 * Tests for src/utils/dsr-calculator.ts — Deflated Sharpe Ratio calculator.
 * Pure math, no I/O deps. First TypeScript test in portfolio-lab.
 */
import { describe, test, expect } from "bun:test";
import {
  calculateDSR,
  estimateIndependentTrials,
  batchCalculateDSR,
  flagOverfitConfigs,
} from "../../src/utils/dsr-calculator";

// ---------------------------------------------------------------------------
// calculateDSR
// ---------------------------------------------------------------------------

describe("calculateDSR", () => {
  test("valid DSR for typical grid search", () => {
    const result = calculateDSR({
      sharpe: 0.79,
      nTrials: 94,
      skewness: -0.3,
      kurtosis: 4.0,
      nObservations: 5371,
    });
    expect(result.dsr).toBeGreaterThan(0);
    expect(result.pValue).toBeLessThan(0.05);
    expect(result.isSignificant).toBe(true);
    expect(result.estimatedTrials).toBeGreaterThan(1);
  });

  test("low sharpe with many trials is not significant", () => {
    const result = calculateDSR({
      sharpe: 0.2,
      nTrials: 1000,
      skewness: 0,
      kurtosis: 3,
      nObservations: 500,
    });
    expect(result.isSignificant).toBe(false);
  });

  test("high sharpe with few trials is significant", () => {
    const result = calculateDSR({
      sharpe: 1.5,
      nTrials: 5,
      skewness: 0,
      kurtosis: 3,
      nObservations: 1000,
    });
    expect(result.isSignificant).toBe(true);
    expect(result.dsr).toBeGreaterThan(1.0);
  });

  test("negative sharpe gives negative DSR", () => {
    const result = calculateDSR({
      sharpe: -0.5,
      nTrials: 10,
      skewness: 0,
      kurtosis: 3,
      nObservations: 500,
    });
    expect(result.dsr).toBeLessThan(0);
    expect(result.isSignificant).toBe(false);
  });

  test("zero sharpe with trials", () => {
    const result = calculateDSR({
      sharpe: 0,
      nTrials: 100,
      skewness: 0,
      kurtosis: 3,
      nObservations: 500,
    });
    expect(result.dsr).toBeLessThan(0);
  });

  test("skewness affects DSR", () => {
    const pos = calculateDSR({
      sharpe: 0.8, nTrials: 10, skewness: 0.5, kurtosis: 3, nObservations: 1000,
    });
    const neg = calculateDSR({
      sharpe: 0.8, nTrials: 10, skewness: -0.5, kurtosis: 3, nObservations: 1000,
    });
    // Positive skewness should give different DSR than negative
    expect(pos.dsr).not.toEqual(neg.dsr);
  });

  test("high kurtosis penalizes DSR", () => {
    const normal = calculateDSR({
      sharpe: 0.8, nTrials: 10, skewness: 0, kurtosis: 3, nObservations: 1000,
    });
    const fat = calculateDSR({
      sharpe: 0.8, nTrials: 10, skewness: 0, kurtosis: 6, nObservations: 1000,
    });
    // Fat tails should reduce DSR
    expect(fat.dsr).toBeLessThan(normal.dsr);
  });

  test("more trials reduces significance", () => {
    const few = calculateDSR({
      sharpe: 0.8, nTrials: 10, skewness: 0, kurtosis: 3, nObservations: 1000,
    });
    const many = calculateDSR({
      sharpe: 0.8, nTrials: 1000, skewness: 0, kurtosis: 3, nObservations: 1000,
    });
    expect(many.dsr).toBeLessThan(few.dsr);
  });

  test("insufficient observations throws", () => {
    expect(() =>
      calculateDSR({
        sharpe: 0.5, nTrials: 10, skewness: 0, kurtosis: 3, nObservations: 1,
      })
    ).toThrow("Insufficient");
  });

  test("output fields are all present", () => {
    const result = calculateDSR({
      sharpe: 0.6, nTrials: 50, skewness: 0, kurtosis: 3, nObservations: 500,
    });
    expect(result).toHaveProperty("dsr");
    expect(result).toHaveProperty("pValue");
    expect(result).toHaveProperty("isSignificant");
    expect(result).toHaveProperty("estimatedTrials");
    expect(result).toHaveProperty("confidence95");
    expect(result.confidence95).toBeGreaterThan(0);
    expect(result.pValue).toBeGreaterThanOrEqual(0);
    expect(result.pValue).toBeLessThanOrEqual(1);
  });

  test("edge: large observation count", () => {
    const result = calculateDSR({
      sharpe: 1.0, nTrials: 10, skewness: 0, kurtosis: 3, nObservations: 100000,
    });
    expect(result.isSignificant).toBe(true);
  });
});

// ---------------------------------------------------------------------------
// estimateIndependentTrials
// ---------------------------------------------------------------------------

describe("estimateIndependentTrials", () => {
  test("single config returns 1", () => {
    expect(estimateIndependentTrials(1)).toBe(1);
  });

  test("zero configs returns 1", () => {
    expect(estimateIndependentTrials(0)).toBe(1);
  });

  test("94 configs with default similarity", () => {
    const trials = estimateIndependentTrials(94);
    expect(trials).toBeLessThan(94);
    expect(trials).toBeGreaterThan(1);
  });

  test("higher similarity reduces effective trials", () => {
    const low = estimateIndependentTrials(100, 0.5);
    const high = estimateIndependentTrials(100, 0.9);
    expect(high).toBeGreaterThan(low);
  });

  test("similarity of 1 gives full trials", () => {
    const trials = estimateIndependentTrials(100, 1.0);
    expect(trials).toBe(100);
  });
});

// ---------------------------------------------------------------------------
// batchCalculateDSR
// ---------------------------------------------------------------------------

describe("batchCalculateDSR", () => {
  test("batch processes multiple configs", () => {
    const results = batchCalculateDSR([
      { name: "config_a", sharpe: 0.8, returns: Array(500).fill(0).map((_, i) => (i % 2 ? 0.02 : -0.01)) },
      { name: "config_b", sharpe: 0.4, returns: Array(500).fill(0).map((_, i) => (i % 2 ? 0.01 : -0.005)) },
    ]);
    expect(results).toHaveLength(2);
    expect(results[0].name).toBe("config_a");
    expect(results[1].name).toBe("config_b");
  });

  test("batch output has required fields", () => {
    const results = batchCalculateDSR([
      { name: "test", sharpe: 0.7, returns: [0.01, -0.005, 0.02, 0.0, 0.01] },
    ]);
    const r = results[0];
    expect(r).toHaveProperty("dsr");
    expect(r).toHaveProperty("pValue");
    expect(r).toHaveProperty("isSignificant");
  });

  test("batch with explicit totalTrials", () => {
    const returns = Array(100).fill(0).map(() => 0.01);
    const results = batchCalculateDSR(
      [{ name: "single", sharpe: 0.5, returns }],
      1000
    );
    expect(results[0].dsr).toBeDefined();
  });

  test("empty batch returns empty array", () => {
    const results = batchCalculateDSR([]);
    expect(results).toHaveLength(0);
  });
});

// ---------------------------------------------------------------------------
// flagOverfitConfigs
// ---------------------------------------------------------------------------

describe("flagOverfitConfigs", () => {
  test("separates overfit from validated", () => {
    const results = [
      { name: "good", sharpe: 0.9, dsr: 2.0, pValue: 0.01 },
      { name: "overfit", sharpe: 0.8, dsr: -0.5, pValue: 0.3 },
      { name: "marginal", sharpe: 0.7, dsr: 0.1, pValue: 0.06 },
    ];
    const flagged = flagOverfitConfigs(results);
    expect(flagged.likelyOverfit).toContain("overfit");
    expect(flagged.likelyOverfit).toContain("marginal"); // pValue >= 0.05
    expect(flagged.validated).toContain("good");
    expect(flagged.overfitRatio).toBe(2 / 3);
  });

  test("all validated when all significant", () => {
    const results = [
      { name: "a", sharpe: 1.0, dsr: 3.0, pValue: 0.001 },
      { name: "b", sharpe: 0.9, dsr: 2.5, pValue: 0.005 },
    ];
    const flagged = flagOverfitConfigs(results);
    expect(flagged.validated).toHaveLength(2);
    expect(flagged.likelyOverfit).toHaveLength(0);
    expect(flagged.overfitRatio).toBe(0);
  });

  test("custom DSR threshold", () => {
    const results = [
      { name: "a", sharpe: 0.5, dsr: 0.5, pValue: 0.02 },
      { name: "b", sharpe: 0.4, dsr: -0.3, pValue: 0.01 },
    ];
    const flagged = flagOverfitConfigs(results, -0.5);
    // With threshold -0.5, only dsr < -0.5 is overfit
    expect(flagged.validated).toHaveLength(2);
  });

  test("empty results", () => {
    const flagged = flagOverfitConfigs([]);
    expect(flagged.likelyOverfit).toHaveLength(0);
    expect(flagged.validated).toHaveLength(0);
    expect(flagged.overfitRatio).toBe(0);
  });
});
