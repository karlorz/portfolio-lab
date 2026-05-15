/**
 * Tests for overlay dashboard panel utility functions.
 * Pure logic tests — no React rendering (bun test runner).
 */
import { describe, test, expect } from "bun:test";

// ---------------------------------------------------------------------------
// Collar panel logic (extracted for testability)
// ---------------------------------------------------------------------------

function getCollarRegimeColor(regime: string): string {
  const colors: Record<string, string> = {
    normal: '#10b981',
    elevated: '#f59e0b',
    stress: '#ef4444',
    crisis: '#dc2626',
  };
  return colors[regime] || '#6b7280';
}

function computeCollarOtm(callStrike: number, putStrike: number, spot: number) {
  const callOtmPct = ((callStrike / spot) - 1) * 100;
  const putOtmPct = ((spot - putStrike) / spot) * 100;
  return { callOtmPct: Math.round(callOtmPct * 10) / 10, putOtmPct: Math.round(putOtmPct * 10) / 10 };
}

describe("Collar panel logic", () => {
  test("normal regime is green", () => {
    expect(getCollarRegimeColor("normal")).toBe("#10b981");
  });

  test("crisis regime is red", () => {
    expect(getCollarRegimeColor("crisis")).toBe("#dc2626");
  });

  test("unknown regime is gray", () => {
    expect(getCollarRegimeColor("unknown")).toBe("#6b7280");
  });

  test("OTM calculations", () => {
    const { callOtmPct, putOtmPct } = computeCollarOtm(566, 534, 550);
    expect(callOtmPct).toBeGreaterThan(0);    // Call above spot = OTM
    expect(putOtmPct).toBeGreaterThan(0);      // Put below spot = OTM
  });

  test("ITM call produces negative OTM pct", () => {
    const { callOtmPct } = computeCollarOtm(530, 534, 550);
    expect(callOtmPct).toBeLessThan(0);  // Call below spot = ITM
  });
});

// ---------------------------------------------------------------------------
// Crypto panel logic
// ---------------------------------------------------------------------------

function getCryptoVolColor(regime: string): string {
  const colors: Record<string, string> = {
    low: '#10b981',
    normal: '#3b82f6',
    high: '#f59e0b',
    extreme: '#ef4444',
  };
  return colors[regime] || '#6b7280';
}

function formatMomentum(mom: number): string {
  const sign = mom >= 0 ? '+' : '';
  return `${sign}${mom.toFixed(1)}%`;
}

describe("Crypto panel logic", () => {
  test("low vol is green", () => {
    expect(getCryptoVolColor("low")).toBe("#10b981");
  });

  test("extreme vol is red", () => {
    expect(getCryptoVolColor("extreme")).toBe("#ef4444");
  });

  test("positive momentum has plus sign", () => {
    expect(formatMomentum(25.5)).toBe("+25.5%");
  });

  test("negative momentum shown correctly", () => {
    expect(formatMomentum(-10.3)).toBe("-10.3%");
  });

  test("zero momentum has plus sign", () => {
    expect(formatMomentum(0)).toBe("+0.0%");
  });
});

// ---------------------------------------------------------------------------
// Calendar panel logic
// ---------------------------------------------------------------------------

const WINDOW_LABELS: Record<string, string> = {
  tom_window: 'Turn-of-Month',
  pre_holiday: 'Pre-Holiday',
  post_holiday: 'Post-Holiday',
  quarter_end: 'Quarter-End',
  monday: 'Monday',
  pre_fomc: 'Pre-FOMC',
  december: 'December',
  options_expiry: 'OPEX',
};

function getWindowLabel(key: string): string {
  return WINDOW_LABELS[key] || key;
}

function getEffectColor(effect: string): string {
  const colors: Record<string, string> = {
    positive: '#10b981',
    neutral: '#6b7280',
    negative: '#f59e0b',
    avoid: '#ef4444',
  };
  return colors[effect] || '#6b7280';
}

describe("Calendar panel logic", () => {
  test("TOM window label", () => {
    expect(getWindowLabel("tom_window")).toBe("Turn-of-Month");
  });

  test("pre-holiday label", () => {
    expect(getWindowLabel("pre_holiday")).toBe("Pre-Holiday");
  });

  test("unknown window returns key", () => {
    expect(getWindowLabel("unknown_window")).toBe("unknown_window");
  });

  test("all known windows have labels", () => {
    const keys = Object.keys(WINDOW_LABELS);
    expect(keys.length).toBe(8);
    for (const key of keys) {
      expect(getWindowLabel(key)).not.toBe(key);
    }
  });

  test("positive effect is green", () => {
    expect(getEffectColor("positive")).toBe("#10b981");
  });

  test("avoid effect is red", () => {
    expect(getEffectColor("avoid")).toBe("#ef4444");
  });
});

// ---------------------------------------------------------------------------
// Kurtosis panel logic
// ---------------------------------------------------------------------------

function getKurtosisRegimeColor(regime: string): string {
  const colors: Record<string, string> = {
    low_kurtosis: '#10b981',
    normal: '#3b82f6',
    high_kurtosis: '#f59e0b',
    extreme_kurtosis: '#ef4444',
  };
  return colors[regime] || '#6b7280';
}

function getPreferenceLabel(pref: string): string {
  const labels: Record<string, string> = {
    trend_following: 'Trend (TSMOM)',
    mean_reversion: 'Mean-Reversion',
    balanced: 'Balanced',
    defensive: 'Defensive',
  };
  return labels[pref] || pref;
}

function getFatTailRiskColor(risk: number): string {
  if (risk > 0.5) return '#ef4444';
  if (risk > 0.3) return '#f59e0b';
  return '#10b981';
}

describe("Kurtosis panel logic", () => {
  test("low kurtosis is green", () => {
    expect(getKurtosisRegimeColor("low_kurtosis")).toBe("#10b981");
  });

  test("extreme kurtosis is red", () => {
    expect(getKurtosisRegimeColor("extreme_kurtosis")).toBe("#ef4444");
  });

  test("trend following label", () => {
    expect(getPreferenceLabel("trend_following")).toBe("Trend (TSMOM)");
  });

  test("defensive label", () => {
    expect(getPreferenceLabel("defensive")).toBe("Defensive");
  });

  test("low fat tail risk is green", () => {
    expect(getFatTailRiskColor(0.1)).toBe("#10b981");
  });

  test("medium fat tail risk is yellow", () => {
    expect(getFatTailRiskColor(0.4)).toBe("#f59e0b");
  });

  test("high fat tail risk is red", () => {
    expect(getFatTailRiskColor(0.8)).toBe("#ef4444");
  });

  test("boundary values", () => {
    expect(getFatTailRiskColor(0.3)).toBe("#10b981");   // Not > 0.3
    expect(getFatTailRiskColor(0.5)).toBe("#f59e0b");    // > 0.3, not > 0.5
    expect(getFatTailRiskColor(0.51)).toBe("#ef4444");   // > 0.5
  });
});
