/**
 * Tests for src/backtest/stress-validation.ts — Stress period validation.
 * Pure data validation and report-generation functions.
 */
import { describe, test, expect } from "bun:test";
import {
  STRESS_PERIODS,
  generateStressReport,
} from "../../src/backtest/stress-validation";
import type { StressValidationResult } from "../../src/backtest/stress-validation";

// ---------------------------------------------------------------------------
// STRESS_PERIODS
// ---------------------------------------------------------------------------

describe("STRESS_PERIODS", () => {
  test("contains 7 crisis periods", () => {
    expect(STRESS_PERIODS).toHaveLength(7);
  });

  test("all periods have required fields", () => {
    for (const period of STRESS_PERIODS) {
      expect(period.name.length).toBeGreaterThan(0);
      expect(period.startDate.length).toBeGreaterThan(0);
      expect(period.endDate.length).toBeGreaterThan(0);
      expect(period.description.length).toBeGreaterThan(0);
      expect(period.maxDDThreshold).toBeLessThan(0);
    }
  });

  test("GFC 2008 has strictest threshold", () => {
    const gfc = STRESS_PERIODS.find(p => p.name === "GFC 2008")!;
    expect(gfc.maxDDThreshold).toBe(-0.30);
  });

  test("all dates are valid and chronological", () => {
    for (const period of STRESS_PERIODS) {
      const start = new Date(period.startDate);
      const end = new Date(period.endDate);
      expect(isNaN(start.getTime())).toBe(false);
      expect(isNaN(end.getTime())).toBe(false);
      expect(end.getTime()).toBeGreaterThan(start.getTime());
    }
  });

  test("no duplicate period names", () => {
    const names = STRESS_PERIODS.map(p => p.name);
    expect(new Set(names).size).toBe(names.length);
  });

  test("thresholds are within valid range", () => {
    for (const period of STRESS_PERIODS) {
      expect(period.maxDDThreshold).toBeGreaterThanOrEqual(-0.30);
      expect(period.maxDDThreshold).toBeLessThanOrEqual(0);
    }
  });
});

// ---------------------------------------------------------------------------
// generateStressReport
// ---------------------------------------------------------------------------

describe("generateStressReport", () => {
  function makeResult(overrides: Partial<StressValidationResult> = {}): StressValidationResult {
    return {
      portfolioName: "Test",
      stressPeriod: "GFC 2008",
      return: -0.15,
      maxDrawdown: -0.35,
      passesThreshold: false,
      recoveryDays: 120,
      ...overrides,
    };
  }

  test("empty results returns empty report", () => {
    const report = generateStressReport([]);
    expect(Object.keys(report.summary)).toHaveLength(0);
    expect(report.failures).toHaveLength(0);
  });

  test("single passing result", () => {
    const results = [makeResult({ passesThreshold: true, maxDrawdown: -0.10 })];
    const report = generateStressReport(results);
    expect(report.summary["GFC 2008"].passed).toBe(1);
    expect(report.summary["GFC 2008"].failed).toBe(0);
  });

  test("single failing result", () => {
    const results = [makeResult({ passesThreshold: false, maxDrawdown: -0.40 })];
    const report = generateStressReport(results);
    expect(report.summary["GFC 2008"].failed).toBe(1);
    expect(report.failures).toHaveLength(1);
  });

  test("mixed pass/fail across periods", () => {
    const results = [
      makeResult({ stressPeriod: "GFC 2008", passesThreshold: true }),
      makeResult({ stressPeriod: "COVID 2020", passesThreshold: false }),
      makeResult({ portfolioName: "Other", stressPeriod: "COVID 2020", passesThreshold: true }),
    ];
    const report = generateStressReport(results);
    expect(Object.keys(report.summary)).toHaveLength(2);
    expect(report.summary["COVID 2020"].passed).toBe(1);
    expect(report.summary["COVID 2020"].failed).toBe(1);
  });

  test("avg DD is computed correctly", () => {
    const results = [
      makeResult({ stressPeriod: "GFC 2008", maxDrawdown: -0.30 }),
      makeResult({ stressPeriod: "GFC 2008", maxDrawdown: -0.10 }),
    ];
    const report = generateStressReport(results);
    expect(report.summary["GFC 2008"].avgDD).toBeCloseTo(-0.20);
  });

  test("multiple failures trigger recommendations", () => {
    const results = [
      makeResult({ portfolioName: "Risky", passesThreshold: false }),
      makeResult({ portfolioName: "Risky", stressPeriod: "COVID 2020", passesThreshold: false }),
    ];
    const report = generateStressReport(results);
    expect(report.recommendations.some(r => r.includes("Risky"))).toBe(true);
  });

  test("GFC failure triggers specific recommendation", () => {
    const results = [makeResult({ stressPeriod: "GFC 2008", passesThreshold: false })];
    const report = generateStressReport(results);
    expect(report.recommendations.some(r => r.includes("GFC 2008"))).toBe(true);
  });

  test("all passing yields no failures", () => {
    const results = [
      makeResult({ passesThreshold: true }),
      makeResult({ stressPeriod: "Rate Hikes 2022", passesThreshold: true }),
    ];
    const report = generateStressReport(results);
    expect(report.failures).toHaveLength(0);
  });

  test("recommendations are unique strings", () => {
    const results = [
      makeResult({ portfolioName: "A", passesThreshold: false }),
      makeResult({ portfolioName: "B", passesThreshold: false }),
      makeResult({ portfolioName: "A", stressPeriod: "COVID 2020", passesThreshold: false }),
    ];
    const report = generateStressReport(results);
    const uniqueRecs = new Set(report.recommendations);
    expect(uniqueRecs.size).toBe(report.recommendations.length);
  });
});
