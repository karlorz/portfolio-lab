/**
 * Sector Attribution Analytics
 * v2.41 - Sector Signal Trading Integration (Phase 1.3)
 * 
 * Performance attribution analysis for sector rotation overlay
 * Tracks alpha vs SPY-only benchmark and contribution to return
 */

import { SectorPosition, SectorPortfolio } from '../trading/sector_positions';
import { PaperSectorPosition } from '../trading/sector_rebalancer';

/**
 * Performance attribution by sector
 */
export interface SectorAttribution {
  symbol: string;
  name: string;
  allocation: number;              // Average allocation during period
  returnContribution: number;      // Contribution to portfolio return
  sectorReturn: number;            // Sector ETF total return
  spyBenchmark: number;            // SPY return over same period
  alpha: number;                   // Excess return vs SPY
  informationRatio: number;         // Alpha / tracking error
  entryDate: string;
  exitDate: string | null;
  daysHeld: number;
  turnover: number;               // Turnover % for this sector
}

/**
 * Aggregated sector rotation performance
 */
export interface SectorRotationPerformance {
  periodStart: string;
  periodEnd: string;
  daysInPeriod: number;
  
  // Portfolio returns
  portfolioReturn: number;
  spyReturn: number;
  gldReturn: number;
  tltReturn: number;
  
  // Attribution
  coreSpyContribution: number;      // SPY core contribution
  sectorOverlayContribution: number; // Sector rotation contribution
  totalAlpha: number;              // Total alpha vs static allocation
  
  // Sector breakdown
  sectorAttributions: SectorAttribution[];
  
  // Risk metrics
  portfolioVolatility: number;
  trackingErrorVsSpy: number;
  informationRatio: number;
  maxDrawdown: number;
  
  // Costs
  totalTransactionCosts: number;
  turnover: number;                // Annualized turnover
}

/**
 * Rolling performance window
 */
export interface RollingPerformance {
  windowDays: number;
  windows: SectorRotationPerformance[];
  currentWindow: SectorRotationPerformance | null;
}

/**
 * Sector rotation vs static comparison
 */
export interface ComparisonResult {
  metric: string;
  sectorRotation: number;
  staticSpyGldTlt: number;
  difference: number;
  winner: 'sector' | 'static' | 'tie';
}

/**
 * Calculate sector return from entry to current (or exit)
 */
export function calculateSectorReturn(
  position: SectorPosition | PaperSectorPosition,
  currentPrice: number,
  exitPrice?: number
): number {
  const exit = exitPrice || currentPrice;
  return (exit - position.entryPrice) / position.entryPrice;
}

/**
 * Calculate attribution for a single sector position
 */
export function calculateSingleAttribution(
  position: SectorPosition | PaperSectorPosition,
  currentPrices: { [symbol: string]: number },
  spyReturn: number,
  endDate: string,
  avgAllocation?: number
): SectorAttribution {
  const currentPrice = currentPrices[position.symbol] || position.entryPrice;
  const sectorReturn = calculateSectorReturn(position, currentPrice);
  const alpha = sectorReturn - spyReturn;
  const daysHeld = Math.floor(
    (new Date(endDate).getTime() - new Date(position.entryDate).getTime()) / (1000 * 60 * 60 * 24)
  );
  
  // Simplified tracking error calculation
  const trackingError = Math.abs(alpha) * 0.5; // Simplified estimate
  const informationRatio = trackingError > 0 ? alpha / trackingError : 0;
  
  // Allocation (use average if provided, else current)
  const allocation = avgAllocation || position.currentAllocation;
  
  return {
    symbol: position.symbol,
    name: position.name,
    allocation,
    returnContribution: allocation * sectorReturn,
    sectorReturn,
    spyBenchmark: spyReturn,
    alpha,
    informationRatio,
    entryDate: position.entryDate,
    exitDate: null,
    daysHeld,
    turnover: 0, // Calculated separately
  };
}

/**
 * Calculate full sector rotation performance attribution
 */
export function calculateSectorRotationPerformance(
  portfolio: SectorPortfolio,
  positions: (SectorPosition | PaperSectorPosition)[],
  currentPrices: { [symbol: string]: number },
  spyReturn: number,
  gldReturn: number,
  tltReturn: number,
  transactionCosts: number,
  periodStart: string,
  periodEnd: string
): SectorRotationPerformance {
  // Calculate sector attributions
  const sectorAttributions = positions.map(pos => 
    calculateSingleAttribution(pos, currentPrices, spyReturn, periodEnd)
  );
  
  // Sum contributions
  const sectorOverlayContribution = sectorAttributions.reduce(
    (sum, s) => sum + s.returnContribution, 0
  );
  
  // Core SPY contribution (34.5% of portfolio at SPY return)
  const coreSpyContribution = portfolio.spyCore * spyReturn;
  
  // Static allocation (46% SPY / 38% GLD / 16% TLT) would have returned:
  const staticAllocationReturn = 0.46 * spyReturn + 0.38 * gldReturn + 0.16 * tltReturn;
  
  // Actual portfolio return
  const portfolioReturn = coreSpyContribution + sectorOverlayContribution;
  
  // Alpha vs static allocation
  const totalAlpha = portfolioReturn - staticAllocationReturn;
  
  // Days in period
  const daysInPeriod = Math.floor(
    (new Date(periodEnd).getTime() - new Date(periodStart).getTime()) / (1000 * 60 * 60 * 24)
  );
  
  // Simplified volatility (would use actual returns series in production)
  const portfolioVolatility = 0.111; // Placeholder - 11.1% annual vol
  const trackingErrorVsSpy = Math.abs(portfolioReturn - spyReturn);
  const informationRatio = trackingErrorVsSpy > 0 ? totalAlpha / trackingErrorVsSpy : 0;
  
  return {
    periodStart,
    periodEnd,
    daysInPeriod,
    portfolioReturn,
    spyReturn,
    gldReturn,
    tltReturn,
    coreSpyContribution,
    sectorOverlayContribution,
    totalAlpha,
    sectorAttributions,
    portfolioVolatility,
    trackingErrorVsSpy,
    informationRatio,
    maxDrawdown: 0, // Would calculate from series
    totalTransactionCosts: transactionCosts,
    turnover: 0, // Would calculate from rebalancing activity
  };
}

/**
 * Compare sector rotation vs static allocation
 */
export function compareToStaticAllocation(
  performance: SectorRotationPerformance
): ComparisonResult[] {
  // Static 46/38/16 allocation returns
  const staticReturn = 0.46 * performance.spyReturn + 
                       0.38 * performance.gldReturn + 
                       0.16 * performance.tltReturn;
  
  return [
    {
      metric: 'Period Return',
      sectorRotation: performance.portfolioReturn,
      staticSpyGldTlt: staticReturn,
      difference: performance.portfolioReturn - staticReturn,
      winner: performance.portfolioReturn > staticReturn ? 'sector' : 
              performance.portfolioReturn < staticReturn ? 'static' : 'tie',
    },
    {
      metric: 'SPY Core Contribution',
      sectorRotation: performance.coreSpyContribution,
      staticSpyGldTlt: 0.46 * performance.spyReturn,
      difference: performance.coreSpyContribution - (0.46 * performance.spyReturn),
      winner: performance.coreSpyContribution > (0.46 * performance.spyReturn) ? 'sector' : 'tie',
    },
    {
      metric: 'Sector Overlay Alpha',
      sectorRotation: performance.sectorOverlayContribution,
      staticSpyGldTlt: 0,
      difference: performance.sectorOverlayContribution,
      winner: performance.sectorOverlayContribution > 0 ? 'sector' : 
              performance.sectorOverlayContribution < 0 ? 'static' : 'tie',
    },
    {
      metric: 'Total Alpha vs Static',
      sectorRotation: performance.totalAlpha,
      staticSpyGldTlt: 0,
      difference: performance.totalAlpha,
      winner: performance.totalAlpha > 0 ? 'sector' : 
              performance.totalAlpha < 0 ? 'static' : 'tie',
    },
  ];
}

/**
 * Generate attribution report
 */
export function generateAttributionReport(
  performance: SectorRotationPerformance
): string {
  const lines: string[] = [];
  
  lines.push('=' .repeat(80));
  lines.push('SECTOR ROTATION PERFORMANCE ATTRIBUTION');
  lines.push('=' .repeat(80));
  lines.push(`Period: ${performance.periodStart} to ${performance.periodEnd}`);
  lines.push(`Days: ${performance.daysInPeriod}`);
  lines.push('');
  
  // Benchmark returns
  lines.push('BENCHMARK RETURNS:');
  lines.push(`  SPY: ${(performance.spyReturn * 100).toFixed(2)}%`);
  lines.push(`  GLD: ${(performance.gldReturn * 100).toFixed(2)}%`);
  lines.push(`  TLT: ${(performance.tltReturn * 100).toFixed(2)}%`);
  lines.push(`  Static 46/38/16: ${((0.46 * performance.spyReturn + 0.38 * performance.gldReturn + 0.16 * performance.tltReturn) * 100).toFixed(2)}%`);
  lines.push('');
  
  // Portfolio return breakdown
  lines.push('PORTFOLIO RETURN BREAKDOWN:');
  lines.push(`  SPY Core (${(performance.coreSpyContribution / performance.spyReturn * 100).toFixed(1)}% weight): ${(performance.coreSpyContribution * 100).toFixed(2)}% contribution`);
  lines.push(`  Sector Overlay: ${(performance.sectorOverlayContribution * 100).toFixed(2)}% contribution`);
  lines.push(`  Total Portfolio: ${(performance.portfolioReturn * 100).toFixed(2)}%`);
  lines.push(`  Transaction Costs: ${(performance.totalTransactionCosts * 100).toFixed(2)}%`);
  lines.push(`  Net Return: ${((performance.portfolioReturn - performance.totalTransactionCosts) * 100).toFixed(2)}%`);
  lines.push('');
  
  // Alpha analysis
  lines.push('ALPHA ANALYSIS:');
  lines.push(`  Total Alpha vs Static: ${(performance.totalAlpha * 100).toFixed(2)}%`);
  lines.push(`  Information Ratio: ${performance.informationRatio.toFixed(2)}`);
  lines.push(`  Tracking Error vs SPY: ${(performance.trackingErrorVsSpy * 100).toFixed(2)}%`);
  lines.push('');
  
  // Sector breakdown
  if (performance.sectorAttributions.length > 0) {
    lines.push('SECTOR ATTRIBUTION:');
    lines.push(`${'Symbol'.padEnd(8)} ${'Alloc%'.padEnd(8)} ${'SectorRtn'.padEnd(10)} ${'SpyBench'.padEnd(10)} ${'Alpha'.padEnd(10)} ${'Contrib'.padEnd(10)} ${'Days'.padEnd(6)}`);
    lines.push('-'.repeat(80));
    
    // Sort by absolute contribution
    const sorted = [...performance.sectorAttributions].sort(
      (a, b) => Math.abs(b.returnContribution) - Math.abs(a.returnContribution)
    );
    
    for (const s of sorted) {
      const alphaStr = s.alpha >= 0 ? `+${(s.alpha * 100).toFixed(2)}%` : `${(s.alpha * 100).toFixed(2)}%`;
      lines.push(
        `${s.symbol.padEnd(8)} ` +
        `${(s.allocation * 100).toFixed(1).padEnd(8)} ` +
        `${(s.sectorReturn * 100).toFixed(2).padEnd(10)} ` +
        `${(s.spyBenchmark * 100).toFixed(2).padEnd(10)} ` +
        `${alphaStr.padEnd(10)} ` +
        `${(s.returnContribution * 100).toFixed(2).padEnd(10)} ` +
        `${s.daysHeld.toString().padEnd(6)}`
      );
    }
    lines.push('');
  }
  
  // Comparison summary
  lines.push('VS STATIC ALLOCATION:');
  const comparisons = compareToStaticAllocation(performance);
  for (const comp of comparisons) {
    const winnerIcon = comp.winner === 'sector' ? '✓' : comp.winner === 'static' ? '✗' : '=';
    const diffStr = comp.difference >= 0 ? `+${(comp.difference * 100).toFixed(2)}%` : `${(comp.difference * 100).toFixed(2)}%`;
    lines.push(`  ${winnerIcon} ${comp.metric}: ${diffStr}`);
  }
  lines.push('');
  
  lines.push('=' .repeat(80));
  
  return lines.join('\n');
}

/**
 * Initialize rolling performance tracker
 */
export function initializeRollingPerformance(windowDays: number = 90): RollingPerformance {
  return {
    windowDays,
    windows: [],
    currentWindow: null,
  };
}

/**
 * Update rolling performance with new data
 */
export function updateRollingPerformance(
  rolling: RollingPerformance,
  newPerformance: SectorRotationPerformance
): RollingPerformance {
  const windows = [...rolling.windows, newPerformance];
  
  // Keep only windows that fit within reasonable history
  // In production, would age out old windows
  const maxWindows = 52; // ~1 year of weekly windows
  if (windows.length > maxWindows) {
    windows.shift();
  }
  
  return {
    ...rolling,
    windows,
    currentWindow: newPerformance,
  };
}

/**
 * Calculate 90-day rolling alpha
 */
export function calculateRolling90DayAlpha(
  windows: SectorRotationPerformance[]
): { avgAlpha: number; consistency: number; best: number; worst: number } {
  if (windows.length === 0) {
    return { avgAlpha: 0, consistency: 0, best: 0, worst: 0 };
  }
  
  const alphas = windows.map(w => w.totalAlpha);
  const avgAlpha = alphas.reduce((a, b) => a + b, 0) / alphas.length;
  const best = Math.max(...alphas);
  const worst = Math.min(...alphas);
  
  // Consistency = % of positive alpha periods
  const positivePeriods = alphas.filter(a => a > 0).length;
  const consistency = positivePeriods / alphas.length;
  
  return { avgAlpha, consistency, best, worst };
}

/**
 * Generate dashboard-compatible attribution summary
 */
export function generateDashboardAttribution(
  performance: SectorRotationPerformance,
  rollingAlpha: { avgAlpha: number; consistency: number; best: number; worst: number }
): object {
  return {
    period: {
      start: performance.periodStart,
      end: performance.periodEnd,
      days: performance.daysInPeriod,
    },
    returns: {
      portfolio: performance.portfolioReturn,
      spy: performance.spyReturn,
      gld: performance.gldReturn,
      tlt: performance.tltReturn,
    },
    attribution: {
      coreSpy: performance.coreSpyContribution,
      sectorOverlay: performance.sectorOverlayContribution,
      transactionCosts: performance.totalTransactionCosts,
      totalAlpha: performance.totalAlpha,
    },
    sectors: performance.sectorAttributions.map(s => ({
      symbol: s.symbol,
      allocation: s.allocation,
      contribution: s.returnContribution,
      alpha: s.alpha,
      daysHeld: s.daysHeld,
    })),
    rolling90day: {
      avgAlpha: rollingAlpha.avgAlpha,
      consistency: rollingAlpha.consistency,
      bestPeriod: rollingAlpha.best,
      worstPeriod: rollingAlpha.worst,
    },
    comparison: {
      vsStaticAllocation: performance.totalAlpha,
      vsSpy: performance.portfolioReturn - performance.spyReturn,
    },
  };
}

// CLI demonstration
if (import.meta.main) {
  console.log('Sector Attribution Analytics v2.41 - Phase 1.3');
  console.log('=' .repeat(50));
  console.log('Attribution tracking for sector rotation overlay');
  console.log('');
  console.log('Key Metrics:');
  console.log('  - Sector alpha vs SPY');
  console.log('  - Contribution to portfolio return');
  console.log('  - Information ratio');
  console.log('  - 90-day rolling alpha');
  console.log('');
  console.log('Benchmark: Static 46% SPY / 38% GLD / 16% TLT');
}
