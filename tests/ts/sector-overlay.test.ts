/**
 * Tests for src/strategy/sector_overlay.ts — Sector rotation overlay strategy.
 */
import { describe, test, expect } from "bun:test";
import {
  checkRebalanceNeeded,
  generateSectorSignalOutput,
  formatSectorOverlay,
} from "../../src/strategy/sector_overlay";
import type { SectorMomentum } from "../../src/strategy/sector_momentum";
import type { SectorAllocationResult, SectorAllocationItem } from "../../src/types/sector";

function mockAllocation(symbols: string[]): SectorAllocationResult {
  const items: SectorAllocationItem[] = symbols.map((sym, i) => ({
    symbol: sym, name: sym, allocation: 1 / symbols.length,
    momentumScore: 0.5, riskAdjusted: 1.0,
    beta: 1.0, sectorGroup: "cyclical" as const,
  }));
  return {
    timestamp: new Date().toISOString(),
    spyAllocation: 0.46, gldAllocation: 0.38, tltAllocation: 0.16,
    sectorAllocations: items,
    totalSectorWeight: 0.30, cashBuffer: 0.05,
    sharpeEstimate: 0.75, turnover: 0.15,
    warnings: [],
  };
}

function mockMomentum(): SectorMomentum[] {
  return [
    { symbol: "XLK", name: "Tech", longMomentum: 0.20, shortMomentum: 0.15,
      compositeMomentum: 0.18, volatility: 0.18, riskAdjustedMomentum: 1.0,
      rank: 1, percentile: 100 },
    { symbol: "XLV", name: "Healthcare", longMomentum: 0.05, shortMomentum: 0.03,
      compositeMomentum: 0.04, volatility: 0.12, riskAdjustedMomentum: 0.33,
      rank: 2, percentile: 80 },
  ];
}

describe("checkRebalanceNeeded", () => {
  test("same sectors no rebalance", () => {
    const a = mockAllocation(["XLK", "XLV", "XLF"]);
    expect(checkRebalanceNeeded(a, a).needed).toBe(false);
  });

  test("different sectors triggers rebalance", () => {
    const a = mockAllocation(["XLK", "XLV", "XLF"]);
    const b = mockAllocation(["XLK", "XLV", "XLE"]);
    expect(checkRebalanceNeeded(a, b).needed).toBe(true);
  });

  test("added sector triggers rebalance", () => {
    const a = mockAllocation(["XLK", "XLV"]);
    const b = mockAllocation(["XLK", "XLV", "XLF"]);
    expect(checkRebalanceNeeded(a, b).needed).toBe(true);
  });

  test("removed sector triggers rebalance", () => {
    const a = mockAllocation(["XLK", "XLV", "XLF"]);
    const b = mockAllocation(["XLK", "XLV"]);
    expect(checkRebalanceNeeded(a, b).needed).toBe(true);
  });

  test("provides reason when rebalance needed", () => {
    const a = mockAllocation(["XLK"]);
    const b = mockAllocation(["XLV"]);
    const check = checkRebalanceNeeded(a, b);
    expect(check.needed).toBe(true);
    expect(check.reason).not.toBeNull();
  });
});

describe("generateSectorSignalOutput", () => {
  test("returns top N sectors", () => {
    const alloc = mockAllocation(["XLK", "XLV"]);
    const output = generateSectorSignalOutput(alloc, mockMomentum(), 1);
    expect(output.topSectors).toHaveLength(1);
    expect(output.topSectors[0].symbol).toBe("XLK");
  });

  test("includes timestamp", () => {
    const output = generateSectorSignalOutput(mockAllocation(["XLK"]), mockMomentum(), 5);
    expect(typeof output.timestamp).toBe("string");
  });
});

describe("formatSectorOverlay", () => {
  test("returns non-empty string", () => {
    const result = formatSectorOverlay(mockAllocation(["XLK", "XLV"]));
    expect(result.length).toBeGreaterThan(0);
    expect(result).toInclude("XLK");
  });
});
