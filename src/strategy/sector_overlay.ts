/**
 * Sector Overlay Allocation Engine
 * v2.40 - Sector Rotation Momentum Infrastructure
 * 
 * Combines base portfolio allocation with sector momentum overlay
 */

import { 
  SectorMomentum, 
  calculateAllSectorMomentum, 
  getTopSectors, 
  adjustForRegime,
  SectorAllocation,
  SECTOR_ETF_MAP,
} from './sector_momentum';

import {
  SectorOverlayConfig,
  DEFAULT_SECTOR_OVERLAY_CONFIG,
  SectorAllocationResult,
  SectorAllocationItem,
  SectorSignalOutput,
} from '../types/sector';

import { HistoricalPrice } from '../data/fetcher';

/**
 * Calculate sector overlay allocation
 */
export function calculateSectorOverlay(
  sectorData: { [symbol: string]: HistoricalPrice[] },
  baseAllocation: { spy: number; gld: number; tlt: number },
  regime: string | null = null,
  vix: number = 0,
  config: SectorOverlayConfig = DEFAULT_SECTOR_OVERLAY_CONFIG
): SectorAllocationResult {
  // Step 1: Calculate momentum scores
  let momentumScores = calculateAllSectorMomentum(sectorData, {
    longLookback: config.momentumLookback,
    shortLookback: Math.floor(config.momentumLookback / 4), // ~quarterly
    minMomentum: config.minMomentum,
    useDualMomentum: config.useDualMomentum,
    riskAdjust: config.riskAdjustMomentum,
  });

  // Step 2: Apply regime adjustment if regime provided
  const regimeAdjusted = regime !== null;
  if (regime && regime !== 'neutral') {
    momentumScores = adjustForRegime(
      momentumScores, 
      regime, 
      config.regimePreferenceBoost
    );
  }

  // Step 3: Check VIX threshold - disable sector rotation in high vol
  if (vix > config.vixThresholdForFallback) {
    return {
      spAllocation: baseAllocation.spy,
      sectorAllocations: [],
      totalEquityWeight: baseAllocation.spy,
      regimeAdjusted: false,
      regime: null,
      timestamp: new Date().toISOString(),
      rebalanceRecommended: false,
      rebalanceReason: `VIX ${vix.toFixed(1)} > threshold ${config.vixThresholdForFallback} - sector rotation disabled`,
    };
  }

  // Step 4: Get top sectors meeting minimum momentum
  const topSectors = getTopSectors(momentumScores, config.numTopSectors, config.minMomentum);
  
  // Check if we have minimum sectors required
  if (topSectors.length < config.minSectorsRequired) {
    return {
      spAllocation: baseAllocation.spy,
      sectorAllocations: [],
      totalEquityWeight: baseAllocation.spy,
      regimeAdjusted,
      regime,
      timestamp: new Date().toISOString(),
      rebalanceRecommended: false,
      rebalanceReason: `Only ${topSectors.length} sectors meet momentum threshold (need ${config.minSectorsRequired})`,
    };
  }

  // Step 5: Calculate allocations
  const sectorPortion = baseAllocation.spy * config.sectorOverlayPct;
  const spAllocation = baseAllocation.spy - sectorPortion;
  
  // Equal weight among top sectors, respecting max sector weight
  const numSectors = topSectors.length;
  let sectorWeight = sectorPortion / numSectors;
  
  // Cap at max sector weight
  if (sectorWeight > config.maxSectorWeight) {
    sectorWeight = config.maxSectorWeight;
  }

  // Build allocation items
  const sectorAllocations: SectorAllocationItem[] = topSectors.map(s => {
    const def = SECTOR_ETF_MAP.get(s.symbol)!;
    const regimeBoost = regime && regime !== 'neutral' ? config.regimePreferenceBoost : 0;
    
    return {
      symbol: s.symbol,
      name: def.name,
      weight: sectorWeight,
      momentum: s.compositeMomentum,
      rank: s.rank,
      volatility: s.volatility,
      regimeBoost,
    };
  });

  const totalWeight = spAllocation + sectorAllocations.reduce((sum, s) => sum + s.weight, 0);

  return {
    spAllocation,
    sectorAllocations,
    totalEquityWeight: totalWeight,
    regimeAdjusted,
    regime,
    timestamp: new Date().toISOString(),
    rebalanceRecommended: topSectors.length > 0 && topSectors[0].compositeMomentum > 0.10,
    rebalanceReason: topSectors.length > 0 && topSectors[0].compositeMomentum > 0.10 
      ? `Top momentum sector ${topSectors[0].symbol} at ${(topSectors[0].compositeMomentum * 100).toFixed(1)}%`
      : null,
  };
}

/**
 * Generate sector signal output for dashboard/signals.json
 */
export function generateSectorSignalOutput(
  allocation: SectorAllocationResult,
  momentumScores: SectorMomentum[],
  topN: number = 5
): SectorSignalOutput {
  return {
    timestamp: new Date().toISOString(),
    topSectors: momentumScores.slice(0, topN).map(s => ({
      symbol: s.symbol,
      name: SECTOR_ETF_MAP.get(s.symbol)?.name || s.symbol,
      momentumScore: s.compositeMomentum,
      allocation: 0, // Will be set if in allocation
      rank: s.rank,
    })).map(s => ({
      ...s,
      allocation: allocation.sectorAllocations.find(a => a.symbol === s.symbol)?.weight || 0,
    })),
    allocation: {
      spWeight: allocation.spAllocation,
      sectorWeights: Object.fromEntries(
        allocation.sectorAllocations.map(a => [a.symbol, a.weight])
      ),
      totalEquity: allocation.totalEquityWeight,
    },
    regime: allocation.regime,
    regimeAdjusted: allocation.regimeAdjusted,
    rebalanceRecommended: allocation.rebalanceRecommended,
  };
}

/**
 * Check if rebalance is recommended based on momentum changes
 */
export function checkRebalanceNeeded(
  currentAllocation: SectorAllocationResult,
  newAllocation: SectorAllocationResult,
  threshold: number = 0.05 // 5% drift threshold
): { needed: boolean; reason: string | null } {
  // Check if sector set has changed
  const currentSectors = new Set(currentAllocation.sectorAllocations.map(s => s.symbol));
  const newSectors = new Set(newAllocation.sectorAllocations.map(s => s.symbol));
  
  const addedSectors = [...newSectors].filter(s => !currentSectors.has(s));
  const removedSectors = [...currentSectors].filter(s => !newSectors.has(s));
  
  if (addedSectors.length > 0 || removedSectors.length > 0) {
    return {
      needed: true,
      reason: `Sector rotation: +[${addedSectors.join(',')}] -[${removedSectors.join(',')}]`,
    };
  }

  // Check allocation drift
  for (const newSector of newAllocation.sectorAllocations) {
    const currentSector = currentAllocation.sectorAllocations.find(
      s => s.symbol === newSector.symbol
    );
    
    if (!currentSector) continue;
    
    const drift = Math.abs(newSector.weight - currentSector.weight);
    if (drift > threshold) {
      return {
        needed: true,
        reason: `${newSector.symbol} drift: ${(drift * 100).toFixed(1)}% > ${(threshold * 100).toFixed(0)}%`,
      };
    }
  }

  // Check SPY allocation drift
  const spyDrift = Math.abs(newAllocation.spAllocation - currentAllocation.spAllocation);
  if (spyDrift > threshold) {
    return {
      needed: true,
      reason: `SPY drift: ${(spyDrift * 100).toFixed(1)}% > ${(threshold * 100).toFixed(0)}%`,
    };
  }

  return { needed: false, reason: null };
}

/**
 * Generate full portfolio allocation with sector overlay
 */
export interface FullPortfolioAllocation {
  equity: {
    spy: number;
    sectors: { [symbol: string]: number };
  };
  gold: number;  // GLD
  bonds: number; // TLT
  cash: number;
  total: number;
}

export function generateFullAllocation(
  sectorResult: SectorAllocationResult,
  baseAllocation: { spy: number; gld: number; tlt: number },
  cashBuffer: number = 0
): FullPortfolioAllocation {
  const sectors: { [symbol: string]: number } = {};
  for (const s of sectorResult.sectorAllocations) {
    sectors[s.symbol] = s.weight;
  }

  return {
    equity: {
      spy: sectorResult.spAllocation,
      sectors,
    },
    gold: baseAllocation.gld,
    bonds: baseAllocation.tlt,
    cash: cashBuffer,
    total: sectorResult.totalEquityWeight + baseAllocation.gld + baseAllocation.tlt + cashBuffer,
  };
}

/**
 * Format allocation for display
 */
export function formatSectorOverlay(allocation: SectorAllocationResult): string {
  const lines: string[] = [];
  lines.push('=' .repeat(70));
  lines.push('SECTOR ROTATION ALLOCATION');
  lines.push('=' .repeat(70));
  lines.push(`Regime: ${allocation.regime || 'N/A'} (${allocation.regimeAdjusted ? 'adjusted' : 'raw momentum'})`);
  lines.push(`Timestamp: ${allocation.timestamp}`);
  lines.push('');
  lines.push('EQUITY ALLOCATION:');
  lines.push(`  SPY (Core):       ${(allocation.spAllocation * 100).toFixed(2)}%`);
  
  if (allocation.sectorAllocations.length > 0) {
    lines.push('  Sector Overlay:');
    for (const s of allocation.sectorAllocations) {
      const regimeTag = s.regimeBoost > 0 ? ' [regime boost]' : '';
      lines.push(
        `    ${s.symbol.padEnd(4)} (${s.name.padEnd(18)}): ${(s.weight * 100).toFixed(2)}%` +
        ` (rank #${s.rank}, mom: ${(s.momentum * 100).toFixed(1)}%)${regimeTag}`
      );
    }
    const totalSector = allocation.sectorAllocations.reduce((sum, s) => sum + s.weight, 0);
    lines.push(`    ${' '.repeat(4)} ${'Total Sector'.padEnd(18)}: ${(totalSector * 100).toFixed(2)}%`);
  }
  
  lines.push(`  ${'Total Equity'.padEnd(24)}: ${(allocation.totalEquityWeight * 100).toFixed(2)}%`);
  lines.push('');
  lines.push('REBALANCE STATUS:');
  if (allocation.rebalanceRecommended) {
    lines.push(`  RECOMMENDED: ${allocation.rebalanceReason}`);
  } else {
    lines.push(`  ${allocation.rebalanceReason || 'Hold current allocation'}`);
  }
  lines.push('=' .repeat(70));
  
  return lines.join('\n');
}
