/**
 * Sector Rotation Strategy Types
 * v2.40 - Sector Rotation Momentum Infrastructure
 */

import { SectorMomentum } from './sector_momentum';

/**
 * Sector overlay configuration interface
 */
export interface SectorOverlayConfig {
  /** Base SPY weight (e.g., 0.46 for 46%) */
  spyBaseWeight: number;
  
  /** Percentage of equity allocation to overlay with sector rotation (e.g., 0.25 for 25%) */
  sectorOverlayPct: number;
  
  /** Number of trading days for long-term momentum lookback (default: 252) */
  momentumLookback: number;
  
  /** Number of sectors to select (default: 3) */
  numTopSectors: number;
  
  /** Minimum momentum threshold to include a sector (default: 0.0) */
  minMomentum: number;
  
  /** Rebalancing frequency */
  rebalanceFreq: 'monthly' | 'quarterly';
  
  /** Momentum threshold to trigger fallback to SPY (default: -0.05 = -5%) */
  cashThreshold: number;
  
  /** Boost to apply to preferred sectors in a regime (default: 0.02 = 2%) */
  regimePreferenceBoost: number;
  
  /** Use dual momentum (require both long and short positive) */
  useDualMomentum: boolean;
  
  /** Risk-adjust momentum scores by volatility */
  riskAdjustMomentum: boolean;
  
  /** Maximum weight per sector (default: 0.15 = 15%) */
  maxSectorWeight: number;
  
  /** Minimum sectors required for overlay (default: 2) */
  minSectorsRequired: number;
  
  /** VIX threshold to disable sector rotation and fall back to SPY (default: 30) */
  vixThresholdForFallback: number;
}

/**
 * Default sector overlay configuration
 */
export const DEFAULT_SECTOR_OVERLAY_CONFIG: SectorOverlayConfig = {
  spyBaseWeight: 0.46,
  sectorOverlayPct: 0.25,
  momentumLookback: 252,
  numTopSectors: 3,
  minMomentum: 0.0,
  rebalanceFreq: 'quarterly',
  cashThreshold: -0.05,
  regimePreferenceBoost: 0.02,
  useDualMomentum: true,
  riskAdjustMomentum: true,
  maxSectorWeight: 0.15,
  minSectorsRequired: 2,
  vixThresholdForFallback: 30,
};

/**
 * Sector allocation result
 */
export interface SectorAllocationResult {
  /** SPY allocation after sector overlay applied */
  spAllocation: number;
  
  /** Individual sector allocations */
  sectorAllocations: SectorAllocationItem[];
  
  /** Total equity weight (SPY + sectors) */
  totalEquityWeight: number;
  
  /** Whether allocation was adjusted for regime */
  regimeAdjusted: boolean;
  
  /** Current regime used for adjustment (if any) */
  regime: string | null;
  
  /** Timestamp of calculation */
  timestamp: string;
  
  /** Rebalance recommendation */
  rebalanceRecommended: boolean;
  
  /** Reason for rebalance (if recommended) */
  rebalanceReason: string | null;
}

/**
 * Individual sector allocation item
 */
export interface SectorAllocationItem {
  symbol: string;
  name: string;
  weight: number;
  momentum: number;
  rank: number;
  volatility: number;
  regimeBoost: number;
}

/**
 * Sector signal output for dashboard/signals.json
 */
export interface SectorSignalOutput {
  timestamp: string;
  topSectors: {
    symbol: string;
    name: string;
    momentumScore: number;
    allocation: number;
    rank: number;
  }[];
  allocation: {
    spWeight: number;
    sectorWeights: { [symbol: string]: number };
    totalEquity: number;
  };
  regime: string | null;
  regimeAdjusted: boolean;
  rebalanceRecommended: boolean;
}

/**
 * Sector momentum report for UI display
 */
export interface SectorMomentumReport {
  calculatedAt: string;
  allSectors: SectorMomentum[];
  topSectors: SectorMomentum[];
  bottomSectors: SectorMomentum[];
  regime?: string;
  summary: {
    avgMomentum: number;
    bestSector: string;
    worstSector: string;
    momentumDispersion: number; // Std dev of momentum scores
    regimeConsensus: number;   // % of sectors aligned with regime preference
  };
}

/**
 * Sector rotation performance metrics
 */
export interface SectorRotationMetrics {
  lookbackPeriod: string;
  totalRebalances: number;
  avgMomentumOfSelected: number;
  momentumHitRate: number;     // % of time selected sectors outperform SPY
  turnover: number;           // Average annual turnover
  trackingError: number;      // vs pure SPY allocation
  informationRatio: number;   // Alpha / tracking error
}

/**
 * Comparison between sector rotation and SPY-only
 */
export interface SectorVsSpyComparison {
  sectorStrategy: {
    cagr: number;
    volatility: number;
    sharpe: number;
    maxDrawdown: number;
  };
  spyOnly: {
    cagr: number;
    volatility: number;
    sharpe: number;
    maxDrawdown: number;
  };
  difference: {
    cagr: number;
    sharpe: number;
  };
  winner: 'sector' | 'spy' | 'tie';
}
