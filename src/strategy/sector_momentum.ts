/**
 * Sector Momentum Calculator
 * v2.40 - Sector Rotation Momentum Infrastructure
 * 
 * Implements momentum scoring for SPDR sector ETFs
 * based on 12-month lookback (252 trading days)
 */

import { HistoricalPrice } from '../data/fetcher';

// Sector ETF definitions with metadata
export interface SectorETF {
  symbol: string;
  name: string;
  beta: number;
  sectorGroup: 'cyclical' | 'defensive' | 'sensitive';
}

export const SECTOR_ETF_DEFINITIONS: SectorETF[] = [
  { symbol: 'XLK', name: 'Technology', beta: 1.10, sectorGroup: 'sensitive' },
  { symbol: 'XLV', name: 'Healthcare', beta: 0.85, sectorGroup: 'defensive' },
  { symbol: 'XLF', name: 'Financials', beta: 1.05, sectorGroup: 'cyclical' },
  { symbol: 'XLY', name: 'Consumer Discretionary', beta: 1.15, sectorGroup: 'cyclical' },
  { symbol: 'XLI', name: 'Industrials', beta: 1.00, sectorGroup: 'cyclical' },
  { symbol: 'XLE', name: 'Energy', beta: 0.95, sectorGroup: 'sensitive' },
  { symbol: 'XLP', name: 'Consumer Staples', beta: 0.65, sectorGroup: 'defensive' },
  { symbol: 'XLU', name: 'Utilities', beta: 0.55, sectorGroup: 'defensive' },
  { symbol: 'XLB', name: 'Materials', beta: 1.05, sectorGroup: 'sensitive' },
  { symbol: 'XLRE', name: 'Real Estate', beta: 0.75, sectorGroup: 'sensitive' },
  { symbol: 'XLC', name: 'Communication Services', beta: 1.00, sectorGroup: 'sensitive' },
];

// Create a lookup map
export const SECTOR_ETF_MAP = new Map(
  SECTOR_ETF_DEFINITIONS.map(s => [s.symbol, s])
);

// Momentum calculation parameters
export const DEFAULT_MOMENTUM_LOOKBACK = 252; // 12 months (trading days)
export const DEFAULT_SHORT_LOOKBACK = 63;     // 3 months (quarterly view)
export const DEFAULT_MIN_MOMENTUM = 0;      // Minimum momentum threshold (positive)

export interface SectorMomentum {
  symbol: string;
  name: string;
  longMomentum: number;     // 12-month momentum
  shortMomentum: number;    // 3-month momentum
  compositeMomentum: number; // Dual momentum (long > 0 && short > 0 ? avg : min)
  volatility: number;       // Annualized volatility
  riskAdjustedMomentum: number; // Momentum / volatility
  rank: number;            // Cross-sectional rank (1 = best)
  percentile: number;      // 0-100 score
}

export interface MomentumConfig {
  longLookback: number;
  shortLookback: number;
  minMomentum: number;
  useDualMomentum: boolean; // Require both long and short positive
  riskAdjust: boolean;
}

export const DEFAULT_MOMENTUM_CONFIG: MomentumConfig = {
  longLookback: DEFAULT_MOMENTUM_LOOKBACK,
  shortLookback: DEFAULT_SHORT_LOOKBACK,
  minMomentum: DEFAULT_MIN_MOMENTUM,
  useDualMomentum: true,
  riskAdjust: true,
};

/**
 * Calculate momentum for a single sector ETF
 */
export function calculateSectorMomentum(
  prices: HistoricalPrice[],
  config: MomentumConfig = DEFAULT_MOMENTUM_CONFIG
): { longMomentum: number; shortMomentum: number; composite: number; volatility: number } {
  if (prices.length < config.longLookback) {
    return { longMomentum: 0, shortMomentum: 0, composite: 0, volatility: 0 };
  }

  const sorted = [...prices].sort((a, b) => 
    new Date(a.date).getTime() - new Date(b.date).getTime()
  );

  // Get prices
  const currentPrice = sorted[sorted.length - 1].adjClose;
  const longPrice = sorted[sorted.length - 1 - config.longLookback].adjClose;
  const shortPrice = sorted[sorted.length - 1 - config.shortLookback].adjClose;

  // Calculate returns
  const longMomentum = (currentPrice / longPrice) - 1;
  const shortMomentum = (currentPrice / shortPrice) - 1;

  // Calculate volatility (annualized)
  const returns: number[] = [];
  for (let i = sorted.length - config.longLookback; i < sorted.length; i++) {
    const dailyReturn = (sorted[i].adjClose / sorted[i - 1].adjClose) - 1;
    returns.push(dailyReturn);
  }
  
  const mean = returns.reduce((a, b) => a + b, 0) / returns.length;
  const variance = returns.reduce((sum, r) => sum + Math.pow(r - mean, 2), 0) / returns.length;
  const dailyVol = Math.sqrt(variance);
  const volatility = dailyVol * Math.sqrt(252); // Annualized

  // Composite momentum logic
  let composite: number;
  if (config.useDualMomentum) {
    // Dual momentum: only positive if both long and short are positive
    if (longMomentum > 0 && shortMomentum > 0) {
      composite = (longMomentum + shortMomentum) / 2;
    } else {
      composite = Math.min(longMomentum, shortMomentum);
    }
  } else {
    composite = longMomentum;
  }

  return { longMomentum, shortMomentum, composite, volatility };
}

/**
 * Calculate momentum for all sector ETFs
 */
export function calculateAllSectorMomentum(
  sectorData: { [symbol: string]: HistoricalPrice[] },
  config: MomentumConfig = DEFAULT_MOMENTUM_CONFIG
): SectorMomentum[] {
  const results: SectorMomentum[] = [];

  for (const [symbol, prices] of Object.entries(sectorData)) {
    const def = SECTOR_ETF_MAP.get(symbol);
    if (!def) continue;

    const momentum = calculateSectorMomentum(prices, config);
    
    results.push({
      symbol,
      name: def.name,
      longMomentum: momentum.longMomentum,
      shortMomentum: momentum.shortMomentum,
      compositeMomentum: momentum.composite,
      volatility: momentum.volatility,
      riskAdjustedMomentum: momentum.volatility > 0 
        ? momentum.composite / momentum.volatility 
        : 0,
      rank: 0, // Set later
      percentile: 0, // Set later
    });
  }

  // Sort by composite momentum and assign ranks
  const sorted = [...results].sort((a, b) => b.compositeMomentum - a.compositeMomentum);
  
  for (let i = 0; i < sorted.length; i++) {
    sorted[i].rank = i + 1;
    sorted[i].percentile = Math.round(((sorted.length - i) / sorted.length) * 100);
  }

  return sorted;
}

/**
 * Get top N sectors by momentum
 */
export function getTopSectors(
  momentumScores: SectorMomentum[],
  n: number = 3,
  minMomentum: number = DEFAULT_MIN_MOMENTUM
): SectorMomentum[] {
  return momentumScores
    .filter(s => s.compositeMomentum >= minMomentum)
    .slice(0, n);
}

/**
 * Regime-aware sector preferences
 * Maps economic regimes to preferred/avoided sectors
 */
export interface RegimeSectorPreferences {
  preferred: string[];
  avoid: string[];
  neutral: string[];
}

export const REGIME_SECTOR_PREFERENCES: Record<string, RegimeSectorPreferences> = {
  early_expansion: {
    preferred: ['XLK', 'XLY', 'XLF'],
    avoid: ['XLU', 'XLP'],
    neutral: ['XLI', 'XLB', 'XLV', 'XLE', 'XLRE', 'XLC'],
  },
  late_expansion: {
    preferred: ['XLE', 'XLB', 'XLI'],
    avoid: ['XLK', 'XLY'],
    neutral: ['XLF', 'XLV', 'XLU', 'XLP', 'XLRE', 'XLC'],
  },
  contraction: {
    preferred: ['XLP', 'XLV', 'XLU'],
    avoid: ['XLY', 'XLB', 'XLE'],
    neutral: ['XLK', 'XLF', 'XLI', 'XLRE', 'XLC'],
  },
  recovery: {
    preferred: ['XLF', 'XLRE', 'XLK'],
    avoid: ['XLU', 'XLP'],
    neutral: ['XLY', 'XLI', 'XLV', 'XLE', 'XLB', 'XLC'],
  },
  neutral: {
    preferred: [],
    avoid: [],
    neutral: SECTOR_ETF_DEFINITIONS.map(s => s.symbol),
  },
};

/**
 * Adjust sector rankings based on regime preferences
 */
export function adjustForRegime(
  momentumScores: SectorMomentum[],
  regime: string,
  preferenceBoost: number = 0.02 // 200bp boost for preferred sectors
): SectorMomentum[] {
  const prefs = REGIME_SECTOR_PREFERENCES[regime] || REGIME_SECTOR_PREFERENCES.neutral;
  
  return momentumScores.map(score => {
    let adjustedMomentum = score.compositeMomentum;
    
    if (prefs.preferred.includes(score.symbol)) {
      adjustedMomentum += preferenceBoost;
    } else if (prefs.avoid.includes(score.symbol)) {
      adjustedMomentum -= preferenceBoost;
    }
    
    return {
      ...score,
      compositeMomentum: adjustedMomentum,
      riskAdjustedMomentum: score.volatility > 0 
        ? adjustedMomentum / score.volatility 
        : 0,
    };
  }).sort((a, b) => b.compositeMomentum - a.compositeMomentum)
    .map((s, i) => ({ ...s, rank: i + 1 }));
}

/**
 * Get sector allocation weights based on momentum
 */
export interface SectorAllocation {
  symbol: string;
  name: string;
  weight: number;        // Portfolio weight (sum of top sectors)
  momentum: number;      // Momentum score
  regimeAdjusted: boolean;
}

export function getSectorAllocation(
  momentumScores: SectorMomentum[],
  topN: number = 3,
  overlayPct: number = 0.25, // 25% of equity allocation
  spyWeight: number = 0.46,  // Base SPY allocation
  minMomentum: number = DEFAULT_MIN_MOMENTUM
): { spAllocation: number; sectorAllocations: SectorAllocation[]; totalWeight: number } {
  const topSectors = getTopSectors(momentumScores, topN, minMomentum);
  
  if (topSectors.length === 0) {
    return {
      spAllocation: spyWeight, // Full SPY if no positive momentum
      sectorAllocations: [],
      totalWeight: spyWeight,
    };
  }

  // Calculate sector portion
  const sectorPortion = spyWeight * overlayPct;
  const spAllocation = spyWeight - sectorPortion;
  
  // Equal weight among top sectors
  const sectorWeight = sectorPortion / topSectors.length;
  
  const sectorAllocations: SectorAllocation[] = topSectors.map(s => ({
    symbol: s.symbol,
    name: s.name,
    weight: sectorWeight,
    momentum: s.compositeMomentum,
    regimeAdjusted: false,
  }));

  return {
    spAllocation,
    sectorAllocations,
    totalWeight: spAllocation + sectorAllocations.reduce((sum, s) => sum + s.weight, 0),
  };
}

/**
 * Format momentum scores for display
 */
export function formatMomentumReport(
  momentumScores: SectorMomentum[],
  topN: number = 5
): string {
  const lines: string[] = [];
  lines.push('='.repeat(60));
  lines.push('SECTOR MOMENTUM REPORT');
  lines.push('='.repeat(60));
  lines.push(`Rank | Symbol | Name              | 12M Mom | 3M Mom | Composite | Vol% | Risk-Adj`);
  lines.push('-'.repeat(60));
  
  for (const s of momentumScores.slice(0, topN)) {
    lines.push(
      `${s.rank.toString().padStart(4)} | ${s.symbol.padEnd(6)} | ${s.name.padEnd(17)} | ` +
      `${(s.longMomentum * 100).toFixed(1)}% | ${(s.shortMomentum * 100).toFixed(1)}% | ` +
      `${(s.compositeMomentum * 100).toFixed(1)}% | ${(s.volatility * 100).toFixed(1)}% | ` +
      `${s.riskAdjustedMomentum.toFixed(2)}`
    );
  }
  
  lines.push('-'.repeat(60));
  
  // Show allocation
  const alloc = getSectorAllocation(momentumScores);
  lines.push(`\nRECOMMENDED ALLOCATION (25% Sector Overlay):`);
  lines.push(`  SPY: ${(alloc.spAllocation * 100).toFixed(1)}%`);
  for (const s of alloc.sectorAllocations) {
    lines.push(`  ${s.symbol}: ${(s.weight * 100).toFixed(1)}% (${s.name})`);
  }
  lines.push(`  Total Equity: ${(alloc.totalWeight * 100).toFixed(1)}%`);
  
  return lines.join('\n');
}

// CLI usage
if (import.meta.main) {
  // Example usage demonstration
  console.log('Sector Momentum Calculator v2.40');
  console.log('=' .repeat(50));
  console.log('Available sectors:');
  for (const sector of SECTOR_ETF_DEFINITIONS) {
    console.log(`  ${sector.symbol}: ${sector.name} (β=${sector.beta})`);
  }
  console.log('\nRegime Preferences:');
  for (const [regime, prefs] of Object.entries(REGIME_SECTOR_PREFERENCES)) {
    if (prefs.preferred.length > 0) {
      console.log(`  ${regime}: Prefer [${prefs.preferred.join(', ')}], Avoid [${prefs.avoid.join(', ')}]`);
    }
  }
}