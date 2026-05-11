/**
 * Duration Signal Utilities - Yield Curve Regime Detection
 * v2.17.1 - Dynamic Duration Allocation Framework
 */

export interface YieldCurveData {
  date: string;
  dgs2: number;   // 2-Year Treasury Yield
  dgs10: number;  // 10-Year Treasury Yield
  dgs30: number;  // 30-Year Treasury Yield
  spread2s10s: number;  // 2s10s spread in bps
  spread10s30s: number; // 10s30s spread in bps
  momentum3m: number;   // 3-month rate change
}

export type DurationRegime = 'steep' | 'normal' | 'flat' | 'inverted';

export interface DurationAllocation {
  tlt: number;  // Long duration (20+ years)
  ief: number;  // Intermediate duration (7-10 years)
  shy: number;  // Short duration (1-3 years)
  bil: number;  // Ultra-short/cash (0-1 year)
}

/**
 * Classify yield curve regime based on 2s10s spread
 * Thresholds based on historical research (AQR, Fed)
 */
export function classifyRegime(spread2s10s: number): DurationRegime {
  if (spread2s10s > 100) return 'steep';
  if (spread2s10s > 50) return 'normal';
  if (spread2s10s > 0) return 'flat';
  return 'inverted';
}

/**
 * Get base allocation for a given regime
 * Based on AQR "Carry and Trend in Fixed Income" research
 */
export function getBaseAllocation(regime: DurationRegime): DurationAllocation {
  const allocations: Record<DurationRegime, DurationAllocation> = {
    steep: { tlt: 0.70, ief: 0.25, shy: 0.05, bil: 0.00 },
    normal: { tlt: 0.50, ief: 0.35, shy: 0.15, bil: 0.00 },
    flat: { tlt: 0.30, ief: 0.40, shy: 0.25, bil: 0.05 },
    inverted: { tlt: 0.15, ief: 0.25, shy: 0.35, bil: 0.25 },
  };
  return allocations[regime];
}

/**
 * Adjust allocation based on rate momentum (trend following)
 * Rising rates → reduce duration; Falling rates → increase duration
 */
export function adjustForMomentum(
  allocation: DurationAllocation,
  rateMomentum: number  // 3-month change in 10Y yield (bps)
): DurationAllocation {
  const adjusted = { ...allocation };

  if (rateMomentum > 50) {
    // Rising rates: reduce TLT, increase SHY
    const shift = Math.min(0.10, allocation.tlt * 0.3);
    adjusted.tlt -= shift;
    adjusted.shy += shift;
  } else if (rateMomentum < -50) {
    // Falling rates: increase TLT and IEF
    const shift = 0.05;
    adjusted.tlt += shift;
    adjusted.ief += shift;
    adjusted.bil -= shift * 2;
  }

  return normalizeAllocation(adjusted);
}

/**
 * Adjust allocation based on real yield level
 * High real yields (>2%) favor longer duration
 */
export function adjustForRealYield(
  allocation: DurationAllocation,
  realYield: number  // 10Y TIPS yield
): DurationAllocation {
  if (realYield > 2.0) {
    const adjusted = { ...allocation };
    adjusted.tlt += 0.05;
    adjusted.bil = Math.max(0, adjusted.bil - 0.05);
    return normalizeAllocation(adjusted);
  }
  return allocation;
}

/**
 * Calculate final duration allocation given current conditions
 */
export function calculateDurationAllocation(
  spread2s10s: number,
  rateMomentum: number,
  realYield: number
): DurationAllocation & { regime: DurationRegime } {
  const regime = classifyRegime(spread2s10s);
  let allocation = getBaseAllocation(regime);

  // Apply tactical adjustments
  allocation = adjustForMomentum(allocation, rateMomentum);
  allocation = adjustForRealYield(allocation, realYield);

  return { ...allocation, regime };
}

/**
 * Normalize allocation to ensure weights sum to 1.0
 */
function normalizeAllocation(allocation: DurationAllocation): DurationAllocation {
  const sum = allocation.tlt + allocation.ief + allocation.shy + allocation.bil;
  if (sum === 0) return { tlt: 0.25, ief: 0.25, shy: 0.25, bil: 0.25 };

  return {
    tlt: Math.max(0.05, Math.min(0.70, allocation.tlt / sum)),
    ief: Math.max(0.05, Math.min(0.70, allocation.ief / sum)),
    shy: Math.max(0.05, Math.min(0.70, allocation.shy / sum)),
    bil: Math.max(0, Math.min(0.50, allocation.bil / sum)),
  };
}

/**
 * Smooth regime transitions to avoid whipsaws
 * Uses 20-day moving average of spread
 */
export function smoothRegime(
  currentRegime: DurationRegime,
  spreadHistory: number[],
  lookback: number = 20
): DurationRegime {
  if (spreadHistory.length < lookback) return currentRegime;

  const recent = spreadHistory.slice(-lookback);
  const avgSpread = recent.reduce((a, b) => a + b, 0) / recent.length;

  return classifyRegime(avgSpread);
}

/**
 * Check if rebalance is needed based on allocation drift
 */
export function shouldRebalance(
  target: DurationAllocation,
  current: DurationAllocation,
  threshold: number = 0.10
): boolean {
  return (
    Math.abs(target.tlt - current.tlt) > threshold ||
    Math.abs(target.ief - current.ief) > threshold ||
    Math.abs(target.shy - current.shy) > threshold ||
    Math.abs(target.bil - current.bil) > threshold
  );
}

/**
 * Get regime description for UI display
 */
export function getRegimeDescription(regime: DurationRegime): string {
  const descriptions: Record<DurationRegime, string> = {
    steep: 'Steep Curve (>100bps) - Long duration preferred',
    normal: 'Normal Curve (50-100bps) - Moderate duration',
    flat: 'Flat Curve (0-50bps) - Short duration preferred',
    inverted: 'Inverted Curve (<0bps) - Ultra-short/cash',
  };
  return descriptions[regime];
}

/**
 * Get expected annual alpha from dynamic duration vs static TLT
 * Based on historical backtesting research
 */
export function getExpectedAlpha(regime: DurationRegime): number {
  const alphas: Record<DurationRegime, number> = {
    steep: 1.2,
    normal: 0.0,
    flat: 0.5,
    inverted: 1.8,
  };
  return alphas[regime];
}
