/**
 * Tests for ModelValidationPanel utility logic.
 * Pure logic tests — no React rendering (bun test runner).
 */
import { describe, test, expect } from "bun:test";

// ---------------------------------------------------------------------------
// DSR confidence classification logic (extracted for testability)
// ---------------------------------------------------------------------------

function getDsrConfidenceLabel(dsr: number): string {
  if (dsr >= 0.95) return "Champion validated";
  if (dsr >= 0.50) return "Borderline";
  return "Not significant";
}

function getDsrColor(dsr: number): string {
  if (dsr >= 0.95) return "#10b981";
  if (dsr >= 0.50) return "#f59e0b";
  return "#ef4444";
}

function formatWeight(w: number): string {
  return `${(w * 100).toFixed(1)}%`;
}

function computeWeightDelta(blWeight: number, overlayWeight: number): number {
  return blWeight - overlayWeight;
}

describe("DSR confidence labels", () => {
  test("DSR >= 0.95 is validated", () => {
    expect(getDsrConfidenceLabel(0.979)).toBe("Champion validated");
    expect(getDsrConfidenceLabel(0.95)).toBe("Champion validated");
  });

  test("DSR 0.50-0.94 is borderline", () => {
    expect(getDsrConfidenceLabel(0.75)).toBe("Borderline");
    expect(getDsrConfidenceLabel(0.50)).toBe("Borderline");
  });

  test("DSR < 0.50 is not significant", () => {
    expect(getDsrConfidenceLabel(0.30)).toBe("Not significant");
    expect(getDsrConfidenceLabel(0.0)).toBe("Not significant");
  });
});

describe("DSR color mapping", () => {
  test("high confidence is green", () => {
    expect(getDsrColor(0.979)).toBe("#10b981");
  });

  test("moderate confidence is amber", () => {
    expect(getDsrColor(0.75)).toBe("#f59e0b");
  });

  test("low confidence is red", () => {
    expect(getDsrColor(0.30)).toBe("#ef4444");
  });
});

describe("Weight formatting", () => {
  test("formats decimal weights as percentages", () => {
    expect(formatWeight(0.46)).toBe("46.0%");
    expect(formatWeight(0.38)).toBe("38.0%");
    expect(formatWeight(0.16)).toBe("16.0%");
  });

  test("handles small weights", () => {
    expect(formatWeight(0.005)).toBe("0.5%");
  });
});

describe("Weight delta computation", () => {
  test("positive delta when BL > overlay", () => {
    expect(computeWeightDelta(0.50, 0.46)).toBeCloseTo(0.04);
  });

  test("negative delta when BL < overlay", () => {
    expect(computeWeightDelta(0.14, 0.16)).toBeCloseTo(-0.02);
  });

  test("zero delta when equal", () => {
    expect(computeWeightDelta(0.38, 0.38)).toBe(0);
  });
});
