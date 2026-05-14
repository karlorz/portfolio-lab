/**
 * Tests for src/utils/duration-signals.ts — Yield curve regime and duration allocation.
 * Pure functions, no I/O. Uses bun native test runner.
 */
import { describe, test, expect } from "bun:test";
import {
  classifyRegime,
  getBaseAllocation,
  getRegimeDescription,
  getExpectedAlpha,
  convertToLeveragedAllocation,
  adjustForMomentum,
  adjustForRealYield,
  calculateDurationAllocation,
  LEVERAGED_ETF_REGISTRY,
} from "../../src/utils/duration-signals";

// ---------------------------------------------------------------------------
// classifyRegime
// ---------------------------------------------------------------------------

describe("classifyRegime", () => {
  test("steep above 100bps", () => {
    expect(classifyRegime(150)).toBe("steep");
    expect(classifyRegime(101)).toBe("steep");
  });

  test("normal between 50 and 100", () => {
    expect(classifyRegime(75)).toBe("normal");
    expect(classifyRegime(51)).toBe("normal");
  });

  test("flat between 0 and 50", () => {
    expect(classifyRegime(25)).toBe("flat");
    expect(classifyRegime(1)).toBe("flat");
  });

  test("inverted at or below 0", () => {
    expect(classifyRegime(0)).toBe("inverted");
    expect(classifyRegime(-50)).toBe("inverted");
  });

  test("boundary: exactly 100", () => {
    expect(classifyRegime(100)).toBe("normal");
  });

  test("boundary: exactly 50", () => {
    expect(classifyRegime(50)).toBe("flat");
  });
});

// ---------------------------------------------------------------------------
// getBaseAllocation
// ---------------------------------------------------------------------------

describe("getBaseAllocation", () => {
  test("steep favors long duration", () => {
    const a = getBaseAllocation("steep");
    expect(a.tlt).toBe(0.70);
    expect(a.ief).toBe(0.25);
    expect(a.tlt + a.ief + a.shy + a.bil).toBeCloseTo(1.0);
  });

  test("normal is balanced", () => {
    const a = getBaseAllocation("normal");
    expect(a.tlt).toBe(0.50);
  });

  test("flat shifts to intermediates", () => {
    const a = getBaseAllocation("flat");
    expect(a.tlt).toBe(0.30);
    expect(a.bil).toBeGreaterThan(0);
  });

  test("inverted shifts to short duration", () => {
    const a = getBaseAllocation("inverted");
    expect(a.tlt).toBe(0.15);
    expect(a.bil).toBe(0.25);
  });

  test("all regimes sum to 1", () => {
    for (const r of ["steep", "normal", "flat", "inverted"] as const) {
      const a = getBaseAllocation(r);
      expect(a.tlt + a.ief + a.shy + a.bil).toBeCloseTo(1.0);
    }
  });
});

// ---------------------------------------------------------------------------
// getRegimeDescription & getExpectedAlpha
// ---------------------------------------------------------------------------

describe("getRegimeDescription", () => {
  test("all regimes return non-empty strings", () => {
    for (const r of ["steep", "normal", "flat", "inverted"] as const) {
      expect(getRegimeDescription(r).length).toBeGreaterThan(0);
    }
  });
});

describe("getExpectedAlpha", () => {
  test("steep has positive alpha", () => {
    expect(getExpectedAlpha("steep")).toBeGreaterThan(0);
  });

  test("all regimes return finite numbers", () => {
    for (const r of ["steep", "normal", "flat", "inverted"] as const) {
      expect(Number.isFinite(getExpectedAlpha(r))).toBe(true);
    }
  });

  test("different regimes give different alphas", () => {
    const alphas = new Set(
      ["steep", "normal", "flat", "inverted"].map(getExpectedAlpha)
    );
    expect(alphas.size).toBeGreaterThan(1);
  });
});

// ---------------------------------------------------------------------------
// convertToLeveragedAllocation
// ---------------------------------------------------------------------------

describe("convertToLeveragedAllocation", () => {
  test("none preference keeps unlevered TLT", () => {
    const base = { tlt: 0.50, ief: 0.35, shy: 0.15, bil: 0.00 };
    const lev = convertToLeveragedAllocation(base, "none");
    expect(lev.tlt).toBeCloseTo(0.50);
    expect(lev.ubt).toBe(0);
    expect(lev.tmf).toBe(0);
  });

  test("UBT reduces capital needed vs unlevered", () => {
    const base = { tlt: 0.16, ief: 0.40, shy: 0.25, bil: 0.19 };
    const lev = convertToLeveragedAllocation(base, "ubt");
    // UBT capital < TLT target because of 2x leverage
    expect(lev.ubt).toBeGreaterThan(0);
    expect(lev.ubt).toBeLessThan(base.tlt);
  });

  test("UBT respects max limit", () => {
    const base = { tlt: 0.30, ief: 0.40, shy: 0.25, bil: 0.05 };
    const lev = convertToLeveragedAllocation(base, "ubt", 0.05);
    expect(lev.ubt).toBeLessThanOrEqual(0.06); // near max limit
  });

  test("TMF uses highest leverage", () => {
    const base = { tlt: 0.15, ief: 0.40, shy: 0.35, bil: 0.10 };
    const lev = convertToLeveragedAllocation(base, "tmf");
    expect(lev.tmf).toBeGreaterThan(0);
  });

  test("zero TLT target returns zeros", () => {
    const base = { tlt: 0, ief: 0.50, shy: 0.50, bil: 0.00 };
    const lev = convertToLeveragedAllocation(base, "optimal");
    expect(lev.ubt).toBe(0);
    expect(lev.tmf).toBe(0);
    expect(lev.tlt).toBe(0);
  });
});

// ---------------------------------------------------------------------------
// adjustForMomentum & adjustForRealYield
// ---------------------------------------------------------------------------

describe("adjustForMomentum", () => {
  test("rising rates reduce TLT", () => {
    const base = { tlt: 0.50, ief: 0.35, shy: 0.15, bil: 0.00 };
    const adj = adjustForMomentum(base, 75);
    expect(adj.tlt).toBeLessThan(base.tlt);
  });

  test("falling rates increase TLT", () => {
    const base = { tlt: 0.30, ief: 0.40, shy: 0.25, bil: 0.05 };
    const adj = adjustForMomentum(base, -75);
    expect(adj.tlt).toBeGreaterThan(base.tlt);
  });

  test("neutral momentum unchanged", () => {
    const base = { tlt: 0.50, ief: 0.35, shy: 0.15, bil: 0.00 };
    const adj = adjustForMomentum(base, 0);
    expect(adj.tlt).toBeCloseTo(base.tlt);
  });

  test("sum stays at 1 after adjustment", () => {
    const base = { tlt: 0.50, ief: 0.35, shy: 0.15, bil: 0.00 };
    const adj = adjustForMomentum(base, 75);
    expect(adj.tlt + adj.ief + adj.shy + adj.bil).toBeCloseTo(1.0);
  });
});

describe("adjustForRealYield", () => {
  test("high real yield increases TLT", () => {
    const base = { tlt: 0.50, ief: 0.35, shy: 0.15, bil: 0.00 };
    const adj = adjustForRealYield(base, 2.5);
    expect(adj.tlt).toBeGreaterThan(base.tlt);
  });

  test("low real yield unchanged", () => {
    const base = { tlt: 0.50, ief: 0.35, shy: 0.15, bil: 0.00 };
    const adj = adjustForRealYield(base, 1.0);
    expect(adj.tlt).toBeCloseTo(base.tlt);
  });

  test("boundary at exactly 2.0 unchanged", () => {
    const base = { tlt: 0.50, ief: 0.35, shy: 0.15, bil: 0.00 };
    const adj = adjustForRealYield(base, 2.0);
    expect(adj.tlt).toBeCloseTo(base.tlt);
  });
});

// ---------------------------------------------------------------------------
// calculateDurationAllocation
// ---------------------------------------------------------------------------

describe("calculateDurationAllocation", () => {
  test("returns allocation with regime", () => {
    const result = calculateDurationAllocation(75, 0, 1.5);
    expect(result.regime).toBe("normal");
    expect(result.tlt).toBeGreaterThan(0);
  });

  test("inverted with falling rates adjusts up", () => {
    const result = calculateDurationAllocation(-20, -75, 1.0);
    expect(result.regime).toBe("inverted");
    expect(result.tlt).toBeGreaterThan(0.15);
  });

  test("steep with high real yield maximizes TLT", () => {
    const result = calculateDurationAllocation(150, 0, 2.5);
    expect(result.regime).toBe("steep");
    expect(result.tlt).toBeGreaterThanOrEqual(0.70);
  });
});

// ---------------------------------------------------------------------------
// LEVERAGED_ETF_REGISTRY
// ---------------------------------------------------------------------------

describe("LEVERAGED_ETF_REGISTRY", () => {
  test("contains all expected ETFs", () => {
    expect(LEVERAGED_ETF_REGISTRY).toHaveProperty("TLT");
    expect(LEVERAGED_ETF_REGISTRY).toHaveProperty("UBT");
    expect(LEVERAGED_ETF_REGISTRY).toHaveProperty("TMF");
    expect(LEVERAGED_ETF_REGISTRY).toHaveProperty("IEF");
    expect(LEVERAGED_ETF_REGISTRY).toHaveProperty("SHY");
    expect(LEVERAGED_ETF_REGISTRY).toHaveProperty("BIL");
  });

  test("UBT has 2x leverage", () => {
    expect(LEVERAGED_ETF_REGISTRY.UBT.leverage).toBe(2.0);
  });

  test("TMF has 3x leverage", () => {
    expect(LEVERAGED_ETF_REGISTRY.TMF.leverage).toBe(3.0);
  });

  test("TLT has 1x leverage", () => {
    expect(LEVERAGED_ETF_REGISTRY.TLT.leverage).toBe(1.0);
  });

  test("leveraged ETFs have higher expense ratios", () => {
    expect(LEVERAGED_ETF_REGISTRY.UBT.expenseRatio).toBeGreaterThan(
      LEVERAGED_ETF_REGISTRY.TLT.expenseRatio
    );
  });
});
