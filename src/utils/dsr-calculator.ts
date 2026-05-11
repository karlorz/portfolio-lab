/**
 * Deflated Sharpe Ratio (DSR) Calculator
 * Adjusts Sharpe ratio for multiple trials and non-normality
 * Reference: Lopez de Prado & Bailey, "The Sharpe Ratio Demystified" (2014)
 */

export interface DSRInput {
  sharpe: number;           // Observed Sharpe ratio
  nTrials: number;          // Number of independent trials/configurations tested
  skewness: number;         // Return distribution skewness
  kurtosis: number;         // Return distribution excess kurtosis
  nObservations: number;    // Number of observations (trading periods)
}

export interface DSROutput {
  dsr: number;              // Deflated Sharpe Ratio
  pValue: number;           // Probability under null hypothesis
  isSignificant: boolean;   // DSR > threshold (typically 0)
  estimatedTrials: number;  // Estimated effective trials
  confidence95: number;     // 95% confidence threshold
}

/**
 * Calculate Deflated Sharpe Ratio
 * 
 * DSR = Sharpe × f(skewness, kurtosis, nTrials, nObservations)
 * 
 * The deflation factor accounts for:
 * 1. Multiple testing (nTrials)
 * 2. Non-normality (skewness, kurtosis)
 * 3. Sample size (nObservations)
 */
export function calculateDSR(input: DSRInput): DSROutput {
  const { sharpe, nTrials, skewness, kurtosis, nObservations } = input;

  if (nObservations < 2) {
    throw new Error('Insufficient observations for DSR calculation');
  }

  // Variance of Sharpe ratio under normality
  // Var(SR) ≈ (1 + 0.5 × SR²) / (n - 1)
  const varNormal = (1 + 0.5 * sharpe * sharpe) / (nObservations - 1);
  const stdNormal = Math.sqrt(varNormal);

  // Adjust for skewness and kurtosis (Bao's adjustment)
  // Var(SR) adjusted = Var(SR) × [1 + (skewness × SR) + ((kurtosis - 3) × SR² / 4)]
  const skewnessAdjustment = 1 + skewness * sharpe;
  const kurtosisAdjustment = 1 + ((kurtosis - 3) * sharpe * sharpe) / 4;
  const variance = varNormal * skewnessAdjustment * kurtosisAdjustment;
  const stdDev = Math.sqrt(Math.max(0, variance));

  // Multiple testing adjustment using Sidak correction
  // Find probability that best of nTrials exceeds observed Sharpe
  const confidenceLevel = 0.95;
  const perTrialConfidence = Math.pow(confidenceLevel, 1 / nTrials);
  const zScore = standardNormalInverseCDF(perTrialConfidence);

  // Critical Sharpe ratio threshold
  const criticalSharpe = zScore * stdNormal;

  // Deflated Sharpe Ratio
  // DSR = (Observed Sharpe - Expected Max Sharpe under null) / Adjusted Std Dev
  const dsr = (sharpe - criticalSharpe) / stdDev;

  // P-value calculation
  const pValue = 1 - standardNormalCDF(dsr);
  const isSignificant = dsr > 0 && pValue < 0.05;

  // Effective trials (accounting for correlation between strategies)
  // Simple estimate: assume 50% correlation reduces effective trials
  const estimatedTrials = Math.max(1, nTrials * 0.7);

  return {
    dsr,
    pValue,
    isSignificant,
    estimatedTrials,
    confidence95: criticalSharpe,
  };
}

/**
 * Estimate number of independent trials from grid search results
 * Accounts for correlation between similar configurations
 */
export function estimateIndependentTrials(
  totalConfigs: number,
  similarityThreshold: number = 0.7
): number {
  // Empirical factor: similar allocations have ~70% correlation
  // Effective trials = total / (1 + (n-1) × correlation)
  if (totalConfigs <= 1) return 1;
  
  const effectiveTrials = totalConfigs / (1 + (totalConfigs - 1) * (1 - similarityThreshold));
  return Math.round(effectiveTrials);
}

/**
 * Batch calculate DSR for multiple configurations
 */
export function batchCalculateDSR(
  results: Array<{
    name: string;
    sharpe: number;
    returns: number[];
  }>,
  totalTrials?: number
): Array<{
  name: string;
  sharpe: number;
  dsr: number;
  pValue: number;
  isSignificant: boolean;
}> {
  const trials = totalTrials ?? results.length;

  return results.map(r => {
    const stats = calculateReturnStatistics(r.returns);
    const dsrResult = calculateDSR({
      sharpe: r.sharpe,
      nTrials: trials,
      skewness: stats.skewness,
      kurtosis: stats.kurtosis,
      nObservations: r.returns.length,
    });

    return {
      name: r.name,
      sharpe: r.sharpe,
      dsr: dsrResult.dsr,
      pValue: dsrResult.pValue,
      isSignificant: dsrResult.isSignificant,
    };
  });
}

/**
 * Calculate skewness and kurtosis from returns
 */
function calculateReturnStatistics(returns: number[]): {
  skewness: number;
  kurtosis: number;
  mean: number;
  std: number;
} {
  const n = returns.length;
  if (n < 3) return { skewness: 0, kurtosis: 3, mean: 0, std: 1 };

  const mean = returns.reduce((a, b) => a + b, 0) / n;
  const variance = returns.reduce((sum, r) => sum + Math.pow(r - mean, 2), 0) / n;
  const std = Math.sqrt(variance);

  if (std === 0) return { skewness: 0, kurtosis: 3, mean, std: 1 };

  // Skewness
  const skewness = returns.reduce((sum, r) => sum + Math.pow((r - mean) / std, 3), 0) / n;

  // Kurtosis (excess)
  const kurtosis = returns.reduce((sum, r) => sum + Math.pow((r - mean) / std, 4), 0) / n;

  return { skewness, kurtosis, mean, std };
}

/**
 * Standard normal CDF approximation (Abramowitz and Stegun)
 */
function standardNormalCDF(x: number): number {
  const a1 = 0.254829592;
  const a2 = -0.284496736;
  const a3 = 1.421413741;
  const a4 = -1.453152027;
  const a5 = 1.061405429;
  const p = 0.3275911;

  const sign = x >= 0 ? 1 : -1;
  const absX = Math.abs(x) / Math.sqrt(2);

  const t = 1 / (1 + p * absX);
  const y = 1 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * Math.exp(-absX * absX);

  return 0.5 * (1 + sign * y);
}

/**
 * Standard normal inverse CDF approximation
 */
function standardNormalInverseCDF(p: number): number {
  if (p <= 0) return -Infinity;
  if (p >= 1) return Infinity;

  const a1 = -3.969683028665376e+01;
  const a2 = 2.209460984245205e+02;
  const a3 = -2.759285104469687e+02;
  const a4 = 1.383577518672690e+02;
  const a5 = -3.066479806614716e+01;
  const a6 = 2.506628277459239e+00;

  const b1 = -5.447609879822406e+01;
  const b2 = 1.615858368580409e+02;
  const b3 = -1.556989798598866e+02;
  const b4 = 6.680131188771972e+01;
  const b5 = -1.328068155288572e+01;

  const c1 = -7.784894002430293e-03;
  const c2 = -3.223964580411365e-01;
  const c3 = -2.400758277161838e+00;
  const c4 = -2.549732539343734e+00;
  const c5 = 4.374664141464968e+00;
  const c6 = 2.938163982698783e+00;

  const d1 = 7.784695709041462e-03;
  const d2 = 3.224671290700398e-01;
  const d3 = 2.445134137142996e+00;
  const d4 = 3.754408661907416e+00;

  const p_low = 0.02425;
  const p_high = 1 - p_low;

  let q, r;

  if (p < p_low) {
    q = Math.sqrt(-2 * Math.log(p));
    return (((((c1 * q + c2) * q + c3) * q + c4) * q + c5) * q + c6) /
           ((((d1 * q + d2) * q + d3) * q + d4) * q + 1);
  } else if (p <= p_high) {
    q = p - 0.5;
    r = q * q;
    return (((((a1 * r + a2) * r + a3) * r + a4) * r + a5) * r + a6) * q /
           (((((b1 * r + b2) * r + b3) * r + b4) * r + b5) * r + 1);
  } else {
    q = Math.sqrt(-2 * Math.log(1 - p));
    return -(((((c1 * q + c2) * q + c3) * q + c4) * q + c5) * q + c6) /
            ((((d1 * q + d2) * q + d3) * q + d4) * q + 1);
  }
}

/**
 * Flag overfit configurations based on DSR threshold
 */
export function flagOverfitConfigs(
  results: Array<{
    name: string;
    sharpe: number;
    dsr: number;
    pValue: number;
  }>,
  dsrThreshold: number = 0
): {
  likelyOverfit: string[];
  validated: string[];
  overfitRatio: number;
} {
  const likelyOverfit = results
    .filter(r => r.dsr < dsrThreshold || r.pValue >= 0.05)
    .map(r => r.name);

  const validated = results
    .filter(r => r.dsr >= dsrThreshold && r.pValue < 0.05)
    .map(r => r.name);

  const overfitRatio = results.length > 0 ? likelyOverfit.length / results.length : 0;

  return { likelyOverfit, validated, overfitRatio };
}
