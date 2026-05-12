/**
 * Duration Signal Utilities - Yield Curve Regime Detection
 * v2.35 - Capital Efficiency via Leveraged Treasury ETFs (UBT/TMF)
 * 
 * Extended to support leveraged ETFs for capital-efficient duration exposure:
 * - UBT: 2x leveraged 20+ Year Treasuries (ProShares)
 * - TMF: 3x leveraged 20+ Year Treasuries (Direxion)
 * 
 * Benefits: 50% capital requirement for same duration exposure
 * Risks: Volatility decay, tracking error, amplified drawdowns
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
  tlt: number;  // Long duration (20+ years) - unlevered
  ief: number;  // Intermediate duration (7-10 years)
  shy: number;  // Short duration (1-3 years)
  bil: number;  // Ultra-short/cash (0-1 year)
}

/**
 * Leveraged Treasury ETF allocation interface
 * Allows capital-efficient duration exposure
 */
export interface LeveragedDurationAllocation {
  tlt: number;   // Unlevered: 1.0x exposure
  ubt: number;   // 2x leveraged: 2.0x exposure with 0.50x capital
  tmf: number;   // 3x leveraged: 3.0x exposure with 0.33x capital
  ief: number;   // Intermediate (no levered equivalent)
  shy: number;   // Short duration
  bil: number;   // Ultra-short/cash
}

/**
 * ETF metadata for backtesting and simulation
 */
export interface LeveragedETFMetadata {
  symbol: string;
  leverage: number;
  expenseRatio: number;  // Annual fee
  trackingError: number; // Estimated annual tracking vs theoretical
  volatilityDecay: number; // Estimated annual decay in choppy markets
}

export const LEVERAGED_ETF_REGISTRY: Record<string, LeveragedETFMetadata> = {
  TLT: { symbol: 'TLT', leverage: 1.0, expenseRatio: 0.0015, trackingError: 0.0005, volatilityDecay: 0.0 },
  UBT: { symbol: 'UBT', leverage: 2.0, expenseRatio: 0.0080, trackingError: 0.0015, volatilityDecay: 0.008 },
  TMF: { symbol: 'TMF', leverage: 3.0, expenseRatio: 0.0091, trackingError: 0.0025, volatilityDecay: 0.015 },
  IEF: { symbol: 'IEF', leverage: 1.0, expenseRatio: 0.0015, trackingError: 0.0005, volatilityDecay: 0.0 },
  SHY: { symbol: 'SHY', leverage: 1.0, expenseRatio: 0.0015, trackingError: 0.0003, volatilityDecay: 0.0 },
  BIL: { symbol: 'BIL', leverage: 1.0, expenseRatio: 0.0014, trackingError: 0.0003, volatilityDecay: 0.0 },
};

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
 * Convert base allocation to leveraged equivalent for capital efficiency
 * 
 * Example: 16% TLT target → 8% UBT (2x) or 5.33% TMF (3x)
 * Frees up 8% or 10.67% capital for other allocations
 */
export function convertToLeveragedAllocation(
  baseAllocation: DurationAllocation,
  leveragePreference: 'none' | 'ubt' | 'tmf' | 'optimal' = 'optimal',
  maxUbtPct: number = 0.10,  // Max 10% portfolio in UBT
  maxTmfPct: number = 0.05   // Max 5% portfolio in TMF (higher risk)
): LeveragedDurationAllocation {
  const leveraged: LeveragedDurationAllocation = {
    tlt: 0, ubt: 0, tmf: 0, ief: baseAllocation.ief, shy: baseAllocation.shy, bil: baseAllocation.bil
  };

  const longDurationTarget = baseAllocation.tlt;

  if (leveragePreference === 'none' || longDurationTarget === 0) {
    leveraged.tlt = longDurationTarget;
    return leveraged;
  }

  if (leveragePreference === 'ubt' || leveragePreference === 'optimal') {
    // Use UBT for capital efficiency, respecting max limit
    const ubtCapitalNeeded = longDurationTarget / 2.0; // 2x leverage
    if (ubtCapitalNeeded <= maxUbtPct) {
      leveraged.ubt = ubtCapitalNeeded;
    } else {
      // Cap at max, fill remainder with TLT
      leveraged.ubt = maxUbtPct;
      leveraged.tlt = longDurationTarget - (maxUbtPct * 2.0);
    }
  }

  if (leveragePreference === 'tmf') {
    // Use TMF (higher risk, only if explicitly requested)
    const tmfCapitalNeeded = longDurationTarget / 3.0; // 3x leverage
    if (tmfCapitalNeeded <= maxTmfPct) {
      leveraged.tmf = tmfCapitalNeeded;
    } else {
      leveraged.tmf = maxTmfPct;
      // Remainder goes to UBT first, then TLT
      const remainder = longDurationTarget - (maxTmfPct * 3.0);
      const ubtForRemainder = remainder / 2.0;
      if (ubtForRemainder <= maxUbtPct) {
        leveraged.ubt = ubtForRemainder;
      } else {
        leveraged.ubt = maxUbtPct;
        leveraged.tlt = remainder - (maxUbtPct * 2.0);
      }
    }
  }

  return normalizeLeveragedAllocation(leveraged);
}

/**
 * Calculate capital freed by using leveraged ETFs
 */
export function calculateCapitalFreed(
  baseAllocation: DurationAllocation,
  leveragedAllocation: LeveragedDurationAllocation
): number {
  const baseCapital = baseAllocation.tlt;
  const leveragedCapital = leveragedAllocation.ubt + leveragedAllocation.tmf + leveragedAllocation.tlt;
  return baseCapital - leveragedCapital;
}

/**
 * Simulate leveraged ETF returns with volatility decay adjustment
 * 
 * Formula: DailyReturn_levered = (DailyReturn_underlying × Leverage) - ExpenseRatio/252
 * Volatility decay penalty applied based on market regime (trending vs choppy)
 */
export function simulateLeveragedReturn(
  underlyingDailyReturn: number,
  underlyingAnnualVolatility: number,
  etfMetadata: LeveragedETFMetadata,
  isTrending: boolean = false  // Trending markets have less decay
): number {
  const { leverage, expenseRatio, volatilityDecay } = etfMetadata;
  
  // Base leveraged return
  const grossReturn = underlyingDailyReturn * leverage;
  
  // Expense drag (daily)
  const expenseDrag = expenseRatio / 252;
  
  // Volatility decay (only in choppy markets)
  const decayFactor = isTrending ? 0.3 : 1.0;  // Reduced decay in trending markets
  const decayPenalty = volatilityDecay * decayFactor / 252;
  
  // Tracking error (random noise, mean 0)
  const trackingNoise = (Math.random() - 0.5) * 2 * etfMetadata.trackingError / Math.sqrt(252);
  
  return grossReturn - expenseDrag - decayPenalty + trackingNoise;
}

/**
 * Calculate expected total expense and decay drag for a leveraged allocation
 */
export function calculateLeveragedDrag(
  allocation: LeveragedDurationAllocation,
  daysHeld: number = 252,
  isTrending: boolean = false
): { totalDrag: number; expenseDrag: number; decayDrag: number } {
  let totalExpenseDrag = 0;
  let totalDecayDrag = 0;

  const applyDrag = (symbol: keyof typeof LEVERAGED_ETF_REGISTRY, weight: number) => {
    if (weight <= 0) return;
    const meta = LEVERAGED_ETF_REGISTRY[symbol];
    const expenseDrag = weight * (meta.expenseRatio * daysHeld / 252);
    const decayDrag = weight * (meta.volatilityDecay * daysHeld / 252 * (isTrending ? 0.3 : 1.0));
    totalExpenseDrag += expenseDrag;
    totalDecayDrag += decayDrag;
  };

  applyDrag('TLT', allocation.tlt);
  applyDrag('UBT', allocation.ubt);
  applyDrag('TMF', allocation.tmf);
  applyDrag('IEF', allocation.ief);
  applyDrag('SHY', allocation.shy);
  applyDrag('BIL', allocation.bil);

  return {
    totalDrag: totalExpenseDrag + totalDecayDrag,
    expenseDrag: totalExpenseDrag,
    decayDrag: totalDecayDrag,
  };
}

/**
 * Backtest scenario definitions for leveraged Treasury allocation
 */
export interface LeveragedBacktestScenario {
  name: string;
  description: string;
  allocationFn: (regime: DurationRegime) => LeveragedDurationAllocation;
}

export const LEVERAGED_BACKTEST_SCENARIOS: LeveragedBacktestScenario[] = [
  {
    name: 'Baseline_TLT',
    description: '100% TLT for long duration (no leverage)',
    allocationFn: (regime) => {
      const base = getBaseAllocation(regime);
      return { tlt: base.tlt, ubt: 0, tmf: 0, ief: base.ief, shy: base.shy, bil: base.bil };
    }
  },
  {
    name: 'Capital_Efficient_UBT',
    description: '50% UBT + 50% TLT for long duration (frees 4-8% capital)',
    allocationFn: (regime) => {
      const base = getBaseAllocation(regime);
      const halfUbt = base.tlt / 4;  // Half the exposure via UBT (2x)
      return { 
        tlt: base.tlt / 2, 
        ubt: halfUbt, 
        tmf: 0, 
        ief: base.ief, 
        shy: base.shy, 
        bil: base.bil 
      };
    }
  },
  {
    name: 'Full_UBT_Replacement',
    description: '100% UBT for long duration (frees 8% capital, max efficiency)',
    allocationFn: (regime) => {
      const base = getBaseAllocation(regime);
      return { 
        tlt: 0, 
        ubt: base.tlt / 2,  // Half the capital for 2x exposure
        tmf: 0, 
        ief: base.ief, 
        shy: base.shy, 
        bil: base.bil 
      };
    }
  },
  {
    name: 'Duration_Barbell_UBT_IEF',
    description: 'UBT for long + IEF for intermediate (barbell strategy)',
    allocationFn: (regime) => {
      const base = getBaseAllocation(regime);
      // Shift some IEF to long duration via UBT
      const ubtAllocation = (base.tlt + base.ief * 0.3) / 2;
      return { 
        tlt: 0, 
        ubt: ubtAllocation, 
        tmf: 0, 
        ief: base.ief * 0.7, 
        shy: base.shy + base.ief * 0.3, 
        bil: base.bil 
      };
    }
  },
  {
    name: 'RiskParity_Levered',
    description: 'Heavy UBT weight for risk-parity style bond allocation',
    allocationFn: (regime) => {
      // Risk parity: equal risk contribution → more bonds (levered)
      return { 
        tlt: 0, 
        ubt: 0.15,  // Fixed 15% UBT = 30% TLT equivalent
        tmf: 0, 
        ief: 0.10, 
        shy: 0.10, 
        bil: 0.05 
      };
    }
  }
];

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
 * Calculate leveraged duration allocation with capital efficiency
 */
export function calculateLeveragedDurationAllocation(
  spread2s10s: number,
  rateMomentum: number,
  realYield: number,
  leveragePreference: 'none' | 'ubt' | 'tmf' | 'optimal' = 'optimal'
): LeveragedDurationAllocation & { 
  regime: DurationRegime; 
  capitalFreed: number;
  expectedDrag: { totalDrag: number; expenseDrag: number; decayDrag: number };
} {
  const base = calculateDurationAllocation(spread2s10s, rateMomentum, realYield);
  const leveraged = convertToLeveragedAllocation(base, leveragePreference);
  const capitalFreed = calculateCapitalFreed(base, leveraged);
  const expectedDrag = calculateLeveragedDrag(leveraged, 252, false);

  return {
    ...leveraged,
    regime: base.regime,
    capitalFreed,
    expectedDrag,
  };
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
 * Normalize leveraged allocation
 */
function normalizeLeveragedAllocation(allocation: LeveragedDurationAllocation): LeveragedDurationAllocation {
  const sum = allocation.tlt + allocation.ubt + allocation.tmf + allocation.ief + allocation.shy + allocation.bil;
  if (sum === 0) return { tlt: 0.20, ubt: 0, tmf: 0, ief: 0.20, shy: 0.20, bil: 0.20 };

  return {
    tlt: Math.max(0, allocation.tlt / sum),
    ubt: Math.max(0, allocation.ubt / sum),
    tmf: Math.max(0, allocation.tmf / sum),
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
 * Check if leveraged rebalance is needed
 */
export function shouldRebalanceLeveraged(
  target: LeveragedDurationAllocation,
  current: LeveragedDurationAllocation,
  threshold: number = 0.10
): boolean {
  return (
    Math.abs(target.tlt - current.tlt) > threshold ||
    Math.abs(target.ubt - current.ubt) > threshold ||
    Math.abs(target.tmf - current.tmf) > threshold ||
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

/**
 * Get expected benefit from capital efficiency (freed capital deployment)
 */
export function getCapitalEfficiencyBenefit(
  capitalFreed: number,
  expectedReturnOnFreedCapital: number = 0.06  // Assume 6% return on freed capital
): number {
  return capitalFreed * expectedReturnOnFreedCapital;
}

/**
 * Calculate net expected benefit after drag costs
 */
export function getNetCapitalEfficiencyBenefit(
  capitalFreed: number,
  expectedReturnOnFreedCapital: number = 0.06,
  totalDrag: number = 0.01
): number {
  const benefit = getCapitalEfficiencyBenefit(capitalFreed, expectedReturnOnFreedCapital);
  return benefit - totalDrag;
}
