/**
 * Sector Position Tracker
 * v2.41 - Sector Signal Trading Integration (Phase 1)
 * 
 * Tracks sector ETF holdings vs target allocation
 * Calculates drift, generates rebalance signals
 * Records entry/exit timestamps and momentum scores
 */

import { SectorMomentum } from '../strategy/sector_momentum';
import { SectorAllocationResult, SectorAllocationItem } from '../types/sector';

/**
 * Individual sector position tracking
 */
export interface SectorPosition {
  symbol: string;
  name: string;
  targetAllocation: number;      // Target portfolio weight
  currentAllocation: number;       // Actual current weight
  shares: number;                  // Number of shares held
  entryPrice: number;              // Price at entry
  entryMomentumScore: number;      // Momentum score when entered
  entryDate: string;               // ISO timestamp of entry
  lastUpdated: string;            // Last position update
  unrealizedPnL: number;         // Unrealized profit/loss
  realizedPnL: number;            // Realized profit/loss
  totalReturn: number;           // Total return since entry
  daysHeld: number;              // Days since entry
}

/**
 * Sector portfolio summary
 */
export interface SectorPortfolio {
  spyCore: number;                // SPY core allocation (target 34.5%)
  sectorOverlay: number;        // Total sector overlay allocation
  cash: number;                 // Cash buffer
  lastRebalance: string;         // Last rebalance timestamp
  nextScheduledRebalance: string; // Next quarterly rebalance
  positions: SectorPosition[];   // All sector positions
  isRebalanceRecommended: boolean;
  rebalanceReason: string | null;
}

/**
 * Drift calculation result
 */
export interface DriftResult {
  symbol: string;
  targetWeight: number;
  currentWeight: number;
  drift: number;                 // Absolute drift from target
  driftPct: number;              // Percentage drift
  needsRebalance: boolean;
}

/**
 * Rebalance recommendation
 */
export interface RebalanceSignal {
  timestamp: string;
  triggered: boolean;
  triggerType: 'quarterly' | 'drift' | 'momentum_drop' | 'none';
  reason: string | null;
  drifts: DriftResult[];
  recommendedActions: RebalanceAction[];
}

/**
 * Rebalance action
 */
export interface RebalanceAction {
  action: 'buy' | 'sell' | 'hold';
  symbol: string;
  currentShares: number;
  targetShares: number;
  sharesDelta: number;
  estimatedValue: number;
  urgency: 'immediate' | 'normal' | 'low';
}

// Rebalancing configuration
export const REBALANCE_CONFIG = {
  quarterlyMonths: [2, 5, 8, 11], // March, June, September, December (0-indexed)
  quarterlyDay: 15,             // 15th of month
  driftThreshold: 0.05,          // 5% drift triggers rebalance
  momentumDropThreshold: 0.10,   // Top sector below 10% momentum triggers review
  vixMax: 30,                   // Disable rebalancing if VIX > 30
  maxSectors: 3,                // Max sectors to hold
  minSectors: 2,                // Min sectors required
  targetPerSector: 0.0383,      // 3.83% per sector (11.5% / 3)
  spyCoreTarget: 0.345,        // 34.5% SPY core
  totalEquityTarget: 0.46,       // 46% total equity
};

/**
 * Calculate drift for all sector positions
 */
export function calculateDrift(
  positions: SectorPosition[],
  targetAllocations: SectorAllocationItem[],
  portfolioValue: number
): DriftResult[] {
  const results: DriftResult[] = [];
  
  // Calculate current weights based on portfolio value
  for (const position of positions) {
    const currentValue = position.shares * position.entryPrice; // Simplified - should use live price
    const currentWeight = portfolioValue > 0 ? currentValue / portfolioValue : 0;
    
    const target = targetAllocations.find(t => t.symbol === position.symbol);
    const targetWeight = target?.weight || 0;
    
    const drift = Math.abs(currentWeight - targetWeight);
    
    results.push({
      symbol: position.symbol,
      targetWeight,
      currentWeight,
      drift,
      driftPct: targetWeight > 0 ? drift / targetWeight : 0,
      needsRebalance: drift > REBALANCE_CONFIG.driftThreshold,
    });
  }
  
  // Check for new positions not yet held
  for (const target of targetAllocations) {
    const existing = results.find(r => r.symbol === target.symbol);
    if (!existing) {
      results.push({
        symbol: target.symbol,
        targetWeight: target.weight,
        currentWeight: 0,
        drift: target.weight,
        driftPct: 1.0,
        needsRebalance: target.weight > REBALANCE_CONFIG.driftThreshold,
      });
    }
  }
  
  return results;
}

/**
 * Check if quarterly rebalance is due
 */
export function isQuarterlyRebalanceDue(
  lastRebalance: string,
  currentDate: Date = new Date()
): boolean {
  const last = new Date(lastRebalance);
  
  // Check if we're in a rebalance month and past the rebalance day
  const currentMonth = currentDate.getMonth();
  const currentDay = currentDate.getDate();
  
  const isRebalanceMonth = REBALANCE_CONFIG.quarterlyMonths.includes(currentMonth);
  const isPastRebalanceDay = currentDay >= REBALANCE_CONFIG.quarterlyDay;
  
  if (!isRebalanceMonth || !isPastRebalanceDay) {
    return false;
  }
  
  // Check if we've already rebalanced this quarter
  const lastMonth = last.getMonth();
  const lastYear = last.getFullYear();
  const currentYear = currentDate.getFullYear();
  
  // Find which quarter last rebalance was in
  const lastQuarter = Math.floor(lastMonth / 3);
  const currentQuarter = Math.floor(currentMonth / 3);
  
  return !(lastYear === currentYear && lastQuarter === currentQuarter);
}

/**
 * Generate rebalance signal based on current state
 */
export function generateRebalanceSignal(
  portfolio: SectorPortfolio,
  targetAllocations: SectorAllocationItem[],
  vix: number,
  topMomentum: SectorMomentum[],
  portfolioValue: number,
  currentDate: Date = new Date()
): RebalanceSignal {
  const drifts = calculateDrift(portfolio.positions, targetAllocations, portfolioValue);
  const maxDrift = Math.max(...drifts.map(d => d.drift), 0);
  
  // Check VIX threshold - no rebalancing in high volatility
  if (vix > REBALANCE_CONFIG.vixMax) {
    return {
      timestamp: currentDate.toISOString(),
      triggered: false,
      triggerType: 'none',
      reason: `VIX ${vix.toFixed(1)} exceeds threshold ${REBALANCE_CONFIG.vixMax}`,
      drifts,
      recommendedActions: [],
    };
  }
  
  // Check quarterly rebalance
  if (isQuarterlyRebalanceDue(portfolio.lastRebalance, currentDate)) {
    return {
      timestamp: currentDate.toISOString(),
      triggered: true,
      triggerType: 'quarterly',
      reason: `Quarterly rebalancing due (${currentDate.toISOString().split('T')[0]})`,
      drifts,
      recommendedActions: calculateRebalanceActions(portfolio.positions, targetAllocations, portfolioValue, 'normal'),
    };
  }
  
  // Check drift threshold
  if (maxDrift > REBALANCE_CONFIG.driftThreshold) {
    const worstDrift = drifts.find(d => d.drift === maxDrift);
    return {
      timestamp: currentDate.toISOString(),
      triggered: true,
      triggerType: 'drift',
      reason: `${worstDrift?.symbol} drift ${(maxDrift * 100).toFixed(1)}% exceeds threshold`,
      drifts,
      recommendedActions: calculateRebalanceActions(portfolio.positions, targetAllocations, portfolioValue, 'normal'),
    };
  }
  
  // Check momentum drop - top sector below threshold
  if (topMomentum.length > 0 && topMomentum[0].compositeMomentum < REBALANCE_CONFIG.momentumDropThreshold) {
    return {
      timestamp: currentDate.toISOString(),
      triggered: true,
      triggerType: 'momentum_drop',
      reason: `Top sector ${topMomentum[0].symbol} momentum ${(topMomentum[0].compositeMomentum * 100).toFixed(1)}% below threshold`,
      drifts,
      recommendedActions: calculateRebalanceActions(portfolio.positions, targetAllocations, portfolioValue, 'low'),
    };
  }
  
  // No rebalance needed
  return {
    timestamp: currentDate.toISOString(),
    triggered: false,
    triggerType: 'none',
    reason: null,
    drifts,
    recommendedActions: [],
  };
}

/**
 * Calculate rebalance actions
 */
function calculateRebalanceActions(
  positions: SectorPosition[],
  targetAllocations: SectorAllocationItem[],
  portfolioValue: number,
  urgency: 'immediate' | 'normal' | 'low'
): RebalanceAction[] {
  const actions: RebalanceAction[] = [];
  const allSymbols: string[] = [];
  for (const p of positions) {
    if (!allSymbols.includes(p.symbol)) allSymbols.push(p.symbol);
  }
  for (const t of targetAllocations) {
    if (!allSymbols.includes(t.symbol)) allSymbols.push(t.symbol);
  }
  
  // Mock price lookup - in production would fetch live prices
  const mockPrices: { [symbol: string]: number } = {
    XLK: 200, XLE: 95, XLI: 140, XLY: 180, XLF: 45,
    XLV: 140, XLP: 75, XLU: 65, XLB: 85, XLRE: 40, XLC: 85,
  };
  
  for (const symbol of allSymbols) {
    const position = positions.find(p => p.symbol === symbol);
    const target = targetAllocations.find(t => t.symbol === symbol);
    
    const currentShares = position?.shares || 0;
    const targetValue = target ? target.weight * portfolioValue : 0;
    const price = mockPrices[symbol] || 100;
    const targetShares = Math.floor(targetValue / price);
    const sharesDelta = targetShares - currentShares;
    
    let action: 'buy' | 'sell' | 'hold' = 'hold';
    if (sharesDelta > 0) action = 'buy';
    else if (sharesDelta < 0) action = 'sell';
    
    if (action !== 'hold' || target) {
      actions.push({
        action,
        symbol,
        currentShares,
        targetShares,
        sharesDelta,
        estimatedValue: Math.abs(sharesDelta) * price,
        urgency: Math.abs(sharesDelta * price / portfolioValue) > 0.02 ? 'immediate' : urgency,
      });
    }
  }
  
  return actions.sort((a, b) => b.estimatedValue - a.estimatedValue);
}

/**
 * Create a new sector position
 */
export function createSectorPosition(
  symbol: string,
  name: string,
  shares: number,
  entryPrice: number,
  entryMomentumScore: number,
  targetAllocation: number
): SectorPosition {
  const now = new Date().toISOString();
  return {
    symbol,
    name,
    targetAllocation,
    currentAllocation: 0, // Calculated later
    shares,
    entryPrice,
    entryMomentumScore,
    entryDate: now,
    lastUpdated: now,
    unrealizedPnL: 0,
    realizedPnL: 0,
    totalReturn: 0,
    daysHeld: 0,
  };
}

/**
 * Initialize empty sector portfolio
 */
export function initializeSectorPortfolio(): SectorPortfolio {
  const now = new Date();
  const nextQuarter = new Date(now);
  const currentMonth = now.getMonth();
  const nextQuarterMonth = REBALANCE_CONFIG.quarterlyMonths.find(m => m > currentMonth) || 2;
  nextQuarter.setMonth(nextQuarterMonth);
  nextQuarter.setDate(REBALANCE_CONFIG.quarterlyDay);
  
  return {
    spyCore: REBALANCE_CONFIG.spyCoreTarget,
    sectorOverlay: 0,
    cash: 0,
    lastRebalance: now.toISOString(),
    nextScheduledRebalance: nextQuarter.toISOString(),
    positions: [],
    isRebalanceRecommended: false,
    rebalanceReason: null,
  };
}

/**
 * Update sector portfolio with new allocation
 */
export function updateSectorPortfolio(
  portfolio: SectorPortfolio,
  allocation: SectorAllocationResult,
  portfolioValue: number,
  momentumScores: SectorMomentum[]
): SectorPortfolio {
  const now = new Date();
  const updatedPositions: SectorPosition[] = [];
  
  // Update existing positions
  for (const pos of portfolio.positions) {
    const newAlloc = allocation.sectorAllocations.find(a => a.symbol === pos.symbol);
    const momentum = momentumScores.find(m => m.symbol === pos.symbol);
    
    if (newAlloc) {
      // Position continues - update
      const daysHeld = Math.floor((now.getTime() - new Date(pos.entryDate).getTime()) / (1000 * 60 * 60 * 24));
      updatedPositions.push({
        ...pos,
        targetAllocation: newAlloc.weight,
        // currentAllocation calculated elsewhere with live prices
        currentAllocation: (pos.shares * pos.entryPrice) / portfolioValue,
        lastUpdated: now.toISOString(),
        daysHeld,
        // PnL calculations would use live prices
      });
    }
    // Positions not in new allocation are dropped (realized PnL recorded elsewhere)
  }
  
  // Add new positions
  for (const alloc of allocation.sectorAllocations) {
    const existing = portfolio.positions.find(p => p.symbol === alloc.symbol);
    if (!existing) {
      const momentum = momentumScores.find(m => m.symbol === alloc.symbol);
      updatedPositions.push(
        createSectorPosition(
          alloc.symbol,
          alloc.name,
          0, // Shares calculated based on allocation
          100, // Mock price - would be live price
          momentum?.compositeMomentum || 0,
          alloc.weight
        )
      );
    }
  }
  
  return {
    ...portfolio,
    spyCore: allocation.spAllocation,
    sectorOverlay: allocation.sectorAllocations.reduce((sum, s) => sum + s.weight, 0),
    positions: updatedPositions,
    isRebalanceRecommended: allocation.rebalanceRecommended,
    rebalanceReason: allocation.rebalanceReason,
  };
}

/**
 * Format portfolio for display
 */
export function formatSectorPortfolio(portfolio: SectorPortfolio): string {
  const lines: string[] = [];
  lines.push('=' .repeat(70));
  lines.push('SECTOR PORTFOLIO POSITIONS');
  lines.push('=' .repeat(70));
  lines.push(`SPY Core: ${(portfolio.spyCore * 100).toFixed(1)}%`);
  lines.push(`Sector Overlay: ${(portfolio.sectorOverlay * 100).toFixed(1)}%`);
  lines.push(`Last Rebalance: ${portfolio.lastRebalance.split('T')[0]}`);
  lines.push(`Next Scheduled: ${portfolio.nextScheduledRebalance.split('T')[0]}`);
  lines.push('');
  
  if (portfolio.positions.length === 0) {
    lines.push('No sector positions currently held.');
  } else {
    lines.push('POSITIONS:');
    lines.push(`${'Symbol'.padEnd(8)} ${'Name'.padEnd(20)} ${'Target%'.padEnd(8)} ${'Current%'.padEnd(8)} ${'Shares'.padEnd(8)} ${'Entry'.padEnd(12)} ${'Days'.padEnd(5)}`);
    lines.push('-'.repeat(70));
    for (const pos of portfolio.positions) {
      lines.push(
        `${pos.symbol.padEnd(8)} ${pos.name.slice(0, 18).padEnd(20)} ` +
        `${(pos.targetAllocation * 100).toFixed(1).padEnd(8)} ` +
        `${(pos.currentAllocation * 100).toFixed(1).padEnd(8)} ` +
        `${pos.shares.toString().padEnd(8)} ` +
        `${pos.entryDate.split('T')[0].padEnd(12)} ` +
        `${pos.daysHeld.toString().padEnd(5)}`
      );
    }
  }
  
  lines.push('=' .repeat(70));
  return lines.join('\n');
}

/**
 * Format rebalance signal for display
 */
export function formatRebalanceSignal(signal: RebalanceSignal): string {
  const lines: string[] = [];
  lines.push('=' .repeat(70));
  lines.push('REBALANCE SIGNAL');
  lines.push('=' .repeat(70));
  lines.push(`Timestamp: ${signal.timestamp}`);
  lines.push(`Triggered: ${signal.triggered ? 'YES' : 'NO'}`);
  if (signal.triggered) {
    lines.push(`Type: ${signal.triggerType}`);
    lines.push(`Reason: ${signal.reason}`);
  }
  
  if (signal.drifts.length > 0) {
    lines.push('');
    lines.push('DRIFT ANALYSIS:');
    lines.push(`${'Symbol'.padEnd(8)} ${'Target%'.padEnd(8)} ${'Current%'.padEnd(8)} ${'Drift%'.padEnd(8)} ${'Status'.padEnd(12)}`);
    lines.push('-'.repeat(70));
    for (const d of signal.drifts) {
      const status = d.needsRebalance ? 'REBALANCE' : 'OK';
      lines.push(
        `${d.symbol.padEnd(8)} ${(d.targetWeight * 100).toFixed(1).padEnd(8)} ` +
        `${(d.currentWeight * 100).toFixed(1).padEnd(8)} ${(d.drift * 100).toFixed(1).padEnd(8)} ` +
        `${status.padEnd(12)}`
      );
    }
  }
  
  if (signal.recommendedActions.length > 0) {
    lines.push('');
    lines.push('RECOMMENDED ACTIONS:');
    for (const action of signal.recommendedActions.slice(0, 5)) {
      lines.push(
        `  ${action.action.toUpperCase().padEnd(4)} ${action.symbol.padEnd(6)} ` +
        `${action.sharesDelta > 0 ? '+' : ''}${action.sharesDelta} shares ` +
        `($${action.estimatedValue.toFixed(0)}) [${action.urgency}]`
      );
    }
  }
  
  lines.push('=' .repeat(70));
  return lines.join('\n');
}

// CLI usage demonstration
if (import.meta.main) {
  console.log('Sector Position Tracker v2.41 - Phase 1');
  console.log('=' .repeat(50));
  console.log('Configuration:');
  console.log(`  Quarterly Rebalance: ${REBALANCE_CONFIG.quarterlyMonths.map(m => ['Mar', 'Jun', 'Sep', 'Dec'][m/3]).join(', ')}`);
  console.log(`  Drift Threshold: ${(REBALANCE_CONFIG.driftThreshold * 100).toFixed(0)}%`);
  console.log(`  VIX Max: ${REBALANCE_CONFIG.vixMax}`);
  console.log(`  Target per Sector: ${(REBALANCE_CONFIG.targetPerSector * 100).toFixed(2)}%`);
  console.log('');
  console.log('Example portfolio initialization:');
  const portfolio = initializeSectorPortfolio();
  console.log(formatSectorPortfolio(portfolio));
}
