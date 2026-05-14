/**
 * Tests for src/backtest/leveraged-treasury-backtest.ts
 */
import { describe, test, expect } from "bun:test";
import {
  analyzeScenarios,
  generateSampleBacktest,
} from "../../src/backtest/leveraged-treasury-backtest";

describe("generateSampleBacktest", () => {
  test("returns array of results", () => {
    const results = generateSampleBacktest();
    expect(Array.isArray(results)).toBe(true);
    expect(results.length).toBeGreaterThan(0);
  });

  test("each result has required fields", () => {
    const results = generateSampleBacktest();
    for (const r of results) {
      expect(r.scenario.length).toBeGreaterThan(0);
      expect(typeof r.cagr).toBe("number");
      expect(typeof r.sharpe).toBe("number");
      expect(r.maxDrawdown).toBeLessThanOrEqual(0);
    }
  });

  test("includes TLT and levered scenarios", () => {
    const results = generateSampleBacktest();
    const scenarios = results.map(r => r.scenario);
    expect(scenarios.some(s => s.includes("TLT") || s.includes("Unlever"))).toBe(true);
  });
});

describe("analyzeScenarios", () => {
  test("returns recommendation and metrics", () => {
    const results = generateSampleBacktest();
    const analysis = analyzeScenarios(results);
    expect(analysis.recommended.length).toBeGreaterThan(0);
    expect(typeof analysis.reasoning).toBe("string");
    expect(Object.keys(analysis.metrics).length).toBe(results.length);
  });

  test("metrics contain stats for each scenario", () => {
    const results = generateSampleBacktest();
    const analysis = analyzeScenarios(results);
    for (const r of results) {
      expect(analysis.metrics[r.scenario]).toBeDefined();
      expect(typeof analysis.metrics[r.scenario].cagr).toBe("number");
      expect(typeof analysis.metrics[r.scenario].sharpe).toBe("number");
    }
  });

  test("recommended scenario is in metrics", () => {
    const results = generateSampleBacktest();
    const analysis = analyzeScenarios(results);
    expect(analysis.metrics[analysis.recommended]).toBeDefined();
  });

  test("empty results handled gracefully", () => {
    const analysis = analyzeScenarios([]);
    expect(typeof analysis.recommended).toBe("string");
    expect(Object.keys(analysis.metrics).length).toBe(0);
  });
});
