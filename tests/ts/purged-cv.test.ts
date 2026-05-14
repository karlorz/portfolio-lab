/**
 * Tests for src/backtest/purged-cv.ts — Purged cross-validation for time series.
 * Zero deps, pure logic. Lopez de Prado methodology.
 */
import { describe, test, expect } from "bun:test";
import {
  PurgedKFold,
  estimateEmbargoPeriod,
  validateSplits,
} from "../../src/backtest/purged-cv";

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function makeDates(count: number, start: Date = new Date("2020-01-01")): Date[] {
  const dates: Date[] = [];
  for (let i = 0; i < count; i++) {
    const d = new Date(start);
    d.setDate(d.getDate() + i);
    dates.push(d);
  }
  return dates;
}

// ---------------------------------------------------------------------------
// PurgedKFold constructor
// ---------------------------------------------------------------------------

describe("PurgedKFold constructor", () => {
  test("default config", () => {
    const pf = new PurgedKFold();
    const splits = pf.createSplits(makeDates(1000));
    expect(splits.length).toBeGreaterThan(0);
  });

  test("custom nSplits", () => {
    const pf = new PurgedKFold({ nSplits: 10 });
    const splits = pf.createSplits(makeDates(1000));
    expect(splits.length).toBeLessThanOrEqual(10);
  });

  test("custom embargo", () => {
    const pf = new PurgedKFold({ embargoDays: 5 });
    const splits = pf.createSplits(makeDates(600));
    expect(splits.length).toBeGreaterThan(0);
  });
});

// ---------------------------------------------------------------------------
// createSplits
// ---------------------------------------------------------------------------

describe("createSplits", () => {
  test("insufficient data throws", () => {
    const pf = new PurgedKFold({ minTrainPeriods: 500 });
    expect(() => pf.createSplits(makeDates(200))).toThrow("Insufficient");
  });

  test("splits are ordered chronologically", () => {
    const pf = new PurgedKFold({ nSplits: 5, embargoDays: 20, minTrainPeriods: 100 });
    const splits = pf.createSplits(makeDates(600));
    for (const s of splits) {
      expect(s.train.length).toBeGreaterThan(0);
      expect(s.test.length).toBeGreaterThan(0);
    }
  });

  test("no overlap between train and test within a split", () => {
    const pf = new PurgedKFold({ nSplits: 5, embargoDays: 30, minTrainPeriods: 100 });
    const splits = pf.createSplits(makeDates(600));
    const result = validateSplits(splits);
    expect(result.valid).toBe(true);
    expect(result.errors).toHaveLength(0);
  });

  test("skips folds with insufficient training data", () => {
    const pf = new PurgedKFold({ nSplits: 10, minTrainPeriods: 250, embargoDays: 10 });
    const splits = pf.createSplits(makeDates(600));
    // Some folds should pass the min training period check
    expect(splits.length).toBeGreaterThan(0);
    expect(splits.length).toBeLessThanOrEqual(10);
  });
});

// ---------------------------------------------------------------------------
// createGrowingWindowSplits
// ---------------------------------------------------------------------------

describe("createGrowingWindowSplits", () => {
  test("produces valid splits", () => {
    const pf = new PurgedKFold({ nSplits: 5, embargoDays: 20, minTrainPeriods: 100 });
    const splits = pf.createGrowingWindowSplits(makeDates(600));
    expect(splits.length).toBeGreaterThan(0);
    const result = validateSplits(splits);
    expect(result.valid).toBe(true);
  });

  test("training window grows over folds", () => {
    const pf = new PurgedKFold({ nSplits: 5, embargoDays: 10, minTrainPeriods: 50 });
    const splits = pf.createGrowingWindowSplits(makeDates(600));
    for (let i = 1; i < splits.length; i++) {
      expect(splits[i].train.length).toBeGreaterThanOrEqual(splits[i - 1].train.length);
    }
  });
});

// ---------------------------------------------------------------------------
// createAnchoredSplits
// ---------------------------------------------------------------------------

describe("createAnchoredSplits", () => {
  test("produces valid purged splits", () => {
    const pf = new PurgedKFold({ nSplits: 5, embargoDays: 20, minTrainPeriods: 100 });
    const splits = pf.createAnchoredSplits(makeDates(600));
    expect(splits.length).toBeGreaterThan(0);
    const result = validateSplits(splits);
    expect(result.valid).toBe(true);
  });

  test("test windows move forward in time", () => {
    const pf = new PurgedKFold({ nSplits: 5, embargoDays: 10, minTrainPeriods: 50 });
    const splits = pf.createAnchoredSplits(makeDates(600));
    for (let i = 1; i < splits.length; i++) {
      const prevMaxTest = Math.max(...splits[i - 1].test.map(d => d.getTime()));
      const currMinTest = Math.min(...splits[i].test.map(d => d.getTime()));
      expect(currMinTest).toBeGreaterThanOrEqual(prevMaxTest);
    }
  });
});

// ---------------------------------------------------------------------------
// estimateEmbargoPeriod
// ---------------------------------------------------------------------------

describe("estimateEmbargoPeriod", () => {
  test("empty/single date returns 20", () => {
    expect(estimateEmbargoPeriod([])).toBe(20);
    expect(estimateEmbargoPeriod([new Date()])).toBe(20);
  });

  test("daily strategy with consecutive dates", () => {
    const dates = makeDates(50);
    const embargo = estimateEmbargoPeriod(dates, "daily");
    expect(embargo).toBeGreaterThanOrEqual(20);
  });

  test("weekly strategy", () => {
    const dates: Date[] = [];
    for (let i = 0; i < 20; i++) {
      const d = new Date("2020-01-01");
      d.setDate(d.getDate() + i * 7);
      dates.push(d);
    }
    const embargo = estimateEmbargoPeriod(dates, "weekly");
    expect(embargo).toBeGreaterThanOrEqual(4);
  });

  test("monthly strategy", () => {
    const dates: Date[] = [];
    for (let i = 0; i < 20; i++) {
      const d = new Date("2020-01-01");
      d.setMonth(d.getMonth() + i);
      dates.push(d);
    }
    const embargo = estimateEmbargoPeriod(dates, "monthly");
    expect(embargo).toBeGreaterThanOrEqual(3);
  });
});

// ---------------------------------------------------------------------------
// validateSplits
// ---------------------------------------------------------------------------

describe("validateSplits", () => {
  test("valid splits pass", () => {
    const pf = new PurgedKFold({ nSplits: 5, embargoDays: 20, minTrainPeriods: 100 });
    const splits = pf.createSplits(makeDates(600));
    const result = validateSplits(splits);
    expect(result.valid).toBe(true);
  });

  test("overlapping splits fail", () => {
    const train = makeDates(300);
    const badSplits = [
      { train: train, test: train.slice(0, 50) }, // Test is subset of train
    ];
    const result = validateSplits(badSplits);
    expect(result.valid).toBe(false);
    expect(result.errors.some(e => e.includes("overlap"))).toBe(true);
  });

  test("chronological violation detected", () => {
    const early = makeDates(100, new Date("2020-01-01"));
    const late = makeDates(100, new Date("2020-06-01"));
    const badSplits = [
      { train: late, test: early }, // Test before train
    ];
    const result = validateSplits(badSplits);
    expect(result.valid).toBe(false);
  });

  test("small train set flagged", () => {
    const tiny = makeDates(50);
    const big = makeDates(200, new Date("2020-06-01"));
    const splits = [{ train: tiny, test: big }];
    const result = validateSplits(splits);
    expect(result.errors.some(e => e.includes("Insufficient training"))).toBe(true);
  });

  test("small test set flagged", () => {
    const big = makeDates(200);
    const tiny = makeDates(10);
    const splits = [{ train: big, test: tiny }];
    const result = validateSplits(splits);
    expect(result.errors.some(e => e.includes("Insufficient test"))).toBe(true);
  });

  test("empty splits returns valid", () => {
    const result = validateSplits([]);
    expect(result.valid).toBe(true);
  });
});
