/**
 * Tests for src/analytics/sector_attribution.ts — Sector performance attribution.
 * Pure calculation functions with minimal mock inputs.
 */
import { describe, test, expect } from "bun:test";
import {
  calculateSectorReturn,
  calculateSingleAttribution,
  compareToStaticAllocation,
  initializeRollingPerformance,
  updateRollingPerformance,
  calculateRolling90DayAlpha,
} from "../../src/analytics/sector_attribution";

function mockPosition(overrides: Record<string, unknown> = {}) {
  return {
    symbol: "XLK", name: "Technology",
    entryPrice: 100, entryDate: "2024-01-01",
    currentAllocation: 0.15, shares: 10, ...overrides,
  };
}

function mockPerformance(overrides: Partial<Record<string, unknown>> = {}) {
  return {
    periodStart: "2024-01-01", periodEnd: "2024-03-31", daysInPeriod: 90,
    portfolioReturn: 0.05, spyReturn: 0.04, gldReturn: 0.02, tltReturn: 0.01,
    coreSpyContribution: 0.015, sectorOverlayContribution: 0.035,
    totalAlpha: 0.01,
    sectorAttributions: [], portfolioVolatility: 0.11,
    trackingErrorVsSpy: 0.02, informationRatio: 0.5,
    maxDrawdown: -0.05, totalTransactionCosts: 0.001, turnover: 0.3,
    ...overrides,
  };
}

describe("calculateSectorReturn", () => {
  test("positive return", () => {
    expect(calculateSectorReturn(mockPosition({ entryPrice: 100 }), 110)).toBeCloseTo(0.10);
  });
  test("negative return", () => {
    expect(calculateSectorReturn(mockPosition({ entryPrice: 100 }), 90)).toBeCloseTo(-0.10);
  });
  test("exit price overrides current price", () => {
    expect(calculateSectorReturn(mockPosition({ entryPrice: 100 }), 90, 120)).toBeCloseTo(0.20);
  });
  test("zero return", () => {
    expect(calculateSectorReturn(mockPosition({ entryPrice: 100 }), 100)).toBeCloseTo(0);
  });
});

describe("calculateSingleAttribution", () => {
  test("positive alpha when beating SPY", () => {
    const attr = calculateSingleAttribution(
      mockPosition({ entryPrice: 100 }), { XLK: 115 }, 0.05, "2024-06-30"
    );
    expect(attr.sectorReturn).toBeCloseTo(0.15);
    expect(attr.alpha).toBeCloseTo(0.10);
  });

  test("negative alpha when trailing SPY", () => {
    const attr = calculateSingleAttribution(
      mockPosition({ entryPrice: 100 }), { XLK: 102 }, 0.08, "2024-06-30"
    );
    expect(attr.alpha).toBeCloseTo(-0.06);
  });

  test("return contribution uses allocation", () => {
    const attr = calculateSingleAttribution(
      mockPosition({ entryPrice: 100, currentAllocation: 0.20 }),
      { XLK: 110 }, 0.05, "2024-06-30"
    );
    expect(attr.returnContribution).toBeCloseTo(0.20 * 0.10);
  });

  test("average allocation override", () => {
    const attr = calculateSingleAttribution(
      mockPosition({ entryPrice: 100 }), { XLK: 110 }, 0.05, "2024-06-30", 0.25
    );
    expect(attr.allocation).toBe(0.25);
  });

  test("days held calculated correctly", () => {
    const attr = calculateSingleAttribution(
      mockPosition({ entryPrice: 100, entryDate: "2024-01-01" }),
      { XLK: 110 }, 0.05, "2024-01-31"
    );
    expect(attr.daysHeld).toBe(30);
  });

  test("missing price falls back to entryPrice", () => {
    const attr = calculateSingleAttribution(
      mockPosition({ entryPrice: 100 }), {}, 0.05, "2024-06-30"
    );
    expect(attr.sectorReturn).toBeCloseTo(0);
  });

  test("fields are present and finite", () => {
    const attr = calculateSingleAttribution(
      mockPosition(), { XLK: 110 }, 0.05, "2024-06-30"
    );
    expect(attr.symbol).toBe("XLK");
    expect(attr.name).toBe("Technology");
    expect(Number.isFinite(attr.informationRatio)).toBe(true);
    expect(attr.daysHeld).toBeGreaterThan(0);
  });
});

describe("compareToStaticAllocation", () => {
  test("returns comparison array", () => {
    const perf = mockPerformance({ portfolioReturn: 0.12, spyReturn: 0.08,
      gldReturn: 0.15, tltReturn: 0.05 });
    const result = compareToStaticAllocation(perf);
    expect(Array.isArray(result)).toBe(true);
    expect(result.length).toBeGreaterThan(0);
  });

  test("each comparison has required fields", () => {
    const perf = mockPerformance();
    const result = compareToStaticAllocation(perf);
    for (const c of result) {
      expect(c).toHaveProperty("metric");
      expect(c).toHaveProperty("sectorRotation");
      expect(c).toHaveProperty("staticSpyGldTlt");
      expect(c).toHaveProperty("difference");
      expect(c).toHaveProperty("winner");
    }
  });
});

describe("initializeRollingPerformance", () => {
  test("creates empty tracker", () => {
    const rp = initializeRollingPerformance(90);
    expect(rp.windowDays).toBe(90);
    expect(rp.windows).toHaveLength(0);
    expect(rp.currentWindow).toBeNull();
  });

  test("default window is 90 days", () => {
    expect(initializeRollingPerformance().windowDays).toBe(90);
  });
});

describe("updateRollingPerformance", () => {
  test("appends to windows", () => {
    const rp = initializeRollingPerformance(90);
    const perf = mockPerformance();
    const updated = updateRollingPerformance(rp, perf);
    expect(updated.windows).toHaveLength(1);
    expect(updated.currentWindow).toBe(perf);
  });
});

describe("calculateRolling90DayAlpha", () => {
  test("returns zeros for empty array", () => {
    const result = calculateRolling90DayAlpha([]);
    expect(result.avgAlpha).toBe(0);
    expect(result.consistency).toBe(0);
    expect(result.best).toBe(0);
    expect(result.worst).toBe(0);
  });

  test("averages alphas across windows", () => {
    const windows = [
      mockPerformance({ totalAlpha: 0.04 }),
      mockPerformance({ totalAlpha: 0.02 }),
    ];
    const result = calculateRolling90DayAlpha(windows);
    expect(result.avgAlpha).toBeCloseTo(0.03);
    expect(result.best).toBeCloseTo(0.04);
    expect(result.worst).toBeCloseTo(0.02);
  });

  test("consistency is fraction of positive-alpha windows", () => {
    const windows = [
      mockPerformance({ totalAlpha: 0.03 }),
      mockPerformance({ totalAlpha: -0.01 }),
      mockPerformance({ totalAlpha: 0.02 }),
    ];
    const result = calculateRolling90DayAlpha(windows);
    expect(result.consistency).toBeCloseTo(2 / 3);
  });
});
