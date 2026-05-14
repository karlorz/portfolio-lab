/**
 * Tests for src/strategy/sector_momentum.ts — Sector momentum scoring.
 */
import { describe, test, expect } from "bun:test";
import {
  calculateSectorMomentum,
  getTopSectors,
  adjustForRegime,
  SECTOR_ETF_DEFINITIONS,
  SECTOR_ETF_MAP,
  DEFAULT_MOMENTUM_CONFIG,
} from "../../src/strategy/sector_momentum";
import type { HistoricalPrice } from "../../src/data/fetcher";

function makePrices(startPrice: number, days: number, dailyReturn: number): HistoricalPrice[] {
  const prices: HistoricalPrice[] = [];
  for (let i = 0; i < days; i++) {
    const price = startPrice * Math.pow(1 + dailyReturn, i);
    prices.push({
      date: `2024-${String(Math.floor(i / 30) + 1).padStart(2, "0")}-${String((i % 30) + 1).padStart(2, "0")}`,
      open: price * 0.999, high: price * 1.001, low: price * 0.998,
      close: price, adjClose: price, volume: 1000000,
    });
  }
  return prices;
}

describe("SECTOR_ETF_DEFINITIONS", () => {
  test("contains 11 sectors", () => {
    expect(SECTOR_ETF_DEFINITIONS).toHaveLength(11);
  });
  test("all sectors have required fields", () => {
    for (const s of SECTOR_ETF_DEFINITIONS) {
      expect(s.symbol.length).toBeGreaterThan(0);
      expect(s.name.length).toBeGreaterThan(0);
      expect(s.beta).toBeGreaterThan(0);
      expect(["cyclical", "defensive", "sensitive"]).toContain(s.sectorGroup);
    }
  });
  test("SECTOR_ETF_MAP has all entries", () => {
    expect(SECTOR_ETF_MAP.size).toBe(11);
    expect(SECTOR_ETF_MAP.get("XLK")?.name).toBe("Technology");
  });
});

describe("DEFAULT_MOMENTUM_CONFIG", () => {
  test("has expected defaults", () => {
    expect(DEFAULT_MOMENTUM_CONFIG.longLookback).toBe(252);
    expect(DEFAULT_MOMENTUM_CONFIG.shortLookback).toBe(63);
    expect(DEFAULT_MOMENTUM_CONFIG.useDualMomentum).toBe(true);
  });
});

describe("calculateSectorMomentum", () => {
  test("positive trend returns positive momentum", () => {
    const prices = makePrices(100, 300, 0.0005);
    const result = calculateSectorMomentum(prices);
    expect(result.longMomentum).toBeGreaterThan(0);
  });

  test("negative trend returns negative momentum", () => {
    const prices = makePrices(100, 300, -0.0005);
    const result = calculateSectorMomentum(prices);
    expect(result.longMomentum).toBeLessThan(0);
  });

  test("insufficient data returns zeros", () => {
    const prices = makePrices(100, 50, 0.001);
    const result = calculateSectorMomentum(prices);
    expect(result.longMomentum).toBe(0);
    expect(result.composite).toBe(0);
  });

  test("volatility is non-negative", () => {
    const prices = makePrices(100, 300, 0.0005);
    const result = calculateSectorMomentum(prices);
    expect(result.volatility).toBeGreaterThanOrEqual(0);
  });

  test("dual momentum config changes composite", () => {
    const prices = makePrices(100, 300, 0.0005);
    const r1 = calculateSectorMomentum(prices, { ...DEFAULT_MOMENTUM_CONFIG, useDualMomentum: true });
    const r2 = calculateSectorMomentum(prices, { ...DEFAULT_MOMENTUM_CONFIG, useDualMomentum: false });
    expect(r1.composite).not.toEqual(r2.composite);
  });
});

describe("getTopSectors", () => {
  function makeS(symbol: string, composite: number) {
    return { symbol, name: symbol, longMomentum: composite, shortMomentum: composite,
      compositeMomentum: composite, volatility: 0.15, riskAdjustedMomentum: composite/0.15, rank: 0, percentile: 0 };
  }
  test("returns top N by momentum", () => {
    const s = [makeS("XLK",0.20), makeS("XLV",0.05), makeS("XLF",0.15)];
    const top = getTopSectors(s, 2);
    expect(top).toHaveLength(2);
    expect(top[0].symbol).toBe("XLK");
  });
  test("respects min momentum", () => {
    const s = [makeS("XLK",0.20), makeS("XLV",-0.01)];
    expect(getTopSectors(s, 5, 0)).toHaveLength(1);
  });
});

describe("adjustForRegime", () => {
  function makeM(composite: number, risk: number, sym = "XLK", name = "Tech") {
    return {
      symbol: sym, name, longMomentum: composite, shortMomentum: composite,
      compositeMomentum: composite, volatility: 0.15, riskAdjustedMomentum: risk,
      rank: 1, percentile: 100,
    };
  }
  test("early expansion boosts preferred sectors (XLK)", () => {
    // XLK is in preferred for early_expansion
    const scores = [makeM(0.15, 1.0)];
    const adj = adjustForRegime(scores, "early_expansion");
    expect(adj[0].compositeMomentum).toBeGreaterThan(0.15); // boosted
  });
  test("contraction boosts defensive sectors", () => {
    // XLP is in preferred for contraction
    const scores = [makeM(0.10, 0.8, "XLP", "Staples")];
    const adj = adjustForRegime(scores, "contraction");
    expect(adj[0].compositeMomentum).toBeGreaterThan(0.10);
  });
  test("unknown regime falls back to neutral", () => {
    const scores = [makeM(0.15, 1.0)];
    const adj = adjustForRegime(scores, "unknown");
    expect(adj[0].compositeMomentum).toBeCloseTo(0.15); // no change
  });
  test("returns same length array", () => {
    const scores = [makeM(0.15, 1.0), makeM(0.10, 0.8)];
    expect(adjustForRegime(scores, "bull")).toHaveLength(2);
  });
});
