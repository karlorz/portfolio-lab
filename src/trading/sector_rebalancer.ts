/**
 * Sector Rebalancer Engine
 * v2.41 - Sector Signal Trading Integration (Phase 1.2)
 * 
 * Implements rebalancing logic for sector ETF positions
 * Quarterly calendar rebalancing with drift threshold monitoring
 * No live trading - signal generation only
 */

import { 
  SectorPortfolio, 
  SectorPosition, 
  RebalanceSignal, 
  RebalanceAction,
  REBALANCE_CONFIG,
  calculateDrift,
  isQuarterlyRebalanceDue,
  generateRebalanceSignal
} from './sector_positions';
import { SectorAllocationItem } from '../types/sector';
import { SectorMomentum } from '../strategy/sector_momentum';

/**
 * Rebalance execution record (paper trading log)
 */
export interface RebalanceExecution {
  id: string;
  timestamp: string;
  triggerType: 'quarterly' | 'drift' | 'momentum_drop' | 'manual';
  status: 'pending' | 'simulated' | 'confirmed' | 'cancelled';
  actions: RebalanceAction[];
  portfolioValue: number;
  vixAtExecution: number;
  transactionCosts: number;
  notes: string;
}

/**
 * Rebalance history tracker
 */
export interface RebalanceHistory {
  executions: RebalanceExecution[];
  totalRebalances: number;
  quarterlyRebalances: number;
  driftRebalances: number;
  lastExecutionId: string | null;
  averageTransactionCost: number;
  totalCostsYTD: number;
}

/**
 * Sector performance vs SPY baseline
 */
export interface SectorAttribution {
  symbol: string;
  allocation: number;
  returnContribution: number;
  spyBenchmark: number;
  alpha: number;
  entryDate: string;
  exitDate: string | null;
  daysHeld: number;
}

/**
 * Paper trading position (simulated)
 */
export interface PaperSectorPosition extends SectorPosition {
  simulatedEntryPrice: number;
  simulatedCurrentPrice: number;
  simulatedValue: number;
  transactionLog: TransactionRecord[];
}

/**
 * Transaction record for paper trading
 */
export interface TransactionRecord {
  timestamp: string;
  action: 'buy' | 'sell';
  shares: number;
  price: number;
  value: number;
  commission: number;
  slippage: number;
  totalCost: number;
}

// Transaction cost assumptions
const TRANSACTION_COSTS = {
  commission: 0.0001,  // 0.01% commission
  slippage: 0.0002,    // 2 basis points slippage
  minCommission: 0.01, // $0.01 minimum
};

/**
 * Initialize empty rebalance history
 */
export function initializeRebalanceHistory(): RebalanceHistory {
  return {
    executions: [],
    totalRebalances: 0,
    quarterlyRebalances: 0,
    driftRebalances: 0,
    lastExecutionId: null,
    averageTransactionCost: 0,
    totalCostsYTD: 0,
  };
}

/**
 * Calculate transaction costs for a rebalance action
 */
export function calculateTransactionCost(action: RebalanceAction): number {
  const grossValue = Math.abs(action.estimatedValue);
  const commission = Math.max(
    grossValue * TRANSACTION_COSTS.commission,
    TRANSACTION_COSTS.minCommission
  );
  const slippage = grossValue * TRANSACTION_COSTS.slippage;
  return commission + slippage;
}

/**
 * Calculate total transaction costs for a set of actions
 */
export function calculateTotalTransactionCosts(actions: RebalanceAction[]): number {
  return actions.reduce((total, action) => total + calculateTransactionCost(action), 0);
}

/**
 * Generate unique execution ID
 */
function generateExecutionId(): string {
  return `rebal_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
}

/**
 * Create a new rebalance execution record
 */
export function createRebalanceExecution(
  signal: RebalanceSignal,
  portfolioValue: number,
  vix: number,
  manualTrigger: boolean = false
): RebalanceExecution {
  const costs = calculateTotalTransactionCosts(signal.recommendedActions);
  
  // Map signal trigger type to execution type
  // 'none' is not a valid execution type - use 'manual' if explicitly triggered or if triggered without type
  const triggerType: 'quarterly' | 'drift' | 'momentum_drop' | 'manual' = 
    signal.triggerType === 'none' 
      ? (manualTrigger ? 'manual' : 'quarterly') // Default to quarterly for scheduled rebalances
      : signal.triggerType;
  
  return {
    id: generateExecutionId(),
    timestamp: signal.timestamp,
    triggerType,
    status: 'pending',
    actions: signal.recommendedActions,
    portfolioValue,
    vixAtExecution: vix,
    transactionCosts: costs,
    notes: signal.reason || '',
  };
}

/**
 * Check if rebalance should proceed based on cost-benefit analysis
 */
export function shouldRebalanceProceed(
  signal: RebalanceSignal,
  portfolioValue: number,
  minBenefitThreshold: number = 0.001 // 0.1% minimum benefit
): { proceed: boolean; reason: string } {
  const costs = calculateTotalTransactionCosts(signal.recommendedActions);
  const costPct = portfolioValue > 0 ? costs / portfolioValue : 0;
  
  // Check if costs exceed benefit threshold
  if (costPct > minBenefitThreshold) {
    return {
      proceed: false,
      reason: `Transaction costs (${(costPct * 100).toFixed(3)}%) exceed threshold (${(minBenefitThreshold * 100).toFixed(2)}%)`,
    };
  }
  
  // Check if VIX is too high for rebalancing
  // Note: This should be checked before calling this function, but double-check
  
  // Check if there are any actions to take
  const meaningfulActions = signal.recommendedActions.filter(
    a => a.action !== 'hold' && Math.abs(a.sharesDelta) > 0
  );
  
  if (meaningfulActions.length === 0) {
    return {
      proceed: false,
      reason: 'No meaningful rebalancing actions required',
    };
  }
  
  return {
    proceed: true,
    reason: `Costs ${(costPct * 100).toFixed(3)}% within threshold, ${meaningfulActions.length} actions pending`,
  };
}

/**
 * Execute rebalance in paper trading mode (simulation only)
 */
export function executePaperRebalance(
  execution: RebalanceExecution,
  currentPositions: PaperSectorPosition[],
  currentPrices: { [symbol: string]: number }
): { updatedPositions: PaperSectorPosition[]; execution: RebalanceExecution } {
  const updatedPositions: PaperSectorPosition[] = [...currentPositions];
  const transactionLog: TransactionRecord[] = [];
  
  for (const action of execution.actions) {
    if (action.action === 'hold' || action.sharesDelta === 0) continue;
    
    const price = currentPrices[action.symbol] || 100;
    const shares = Math.abs(action.sharesDelta);
    const value = shares * price;
    const commission = Math.max(value * TRANSACTION_COSTS.commission, TRANSACTION_COSTS.minCommission);
    const slippage = value * TRANSACTION_COSTS.slippage;
    const totalCost = commission + slippage;
    
    const transaction: TransactionRecord = {
      timestamp: execution.timestamp,
      action: action.action,
      shares: action.action === 'buy' ? shares : -shares,
      price,
      value: action.action === 'buy' ? -value : value,
      commission,
      slippage,
      totalCost: action.action === 'buy' ? -(value + totalCost) : value - totalCost,
    };
    
    transactionLog.push(transaction);
    
    // Update position
    const existingIndex = updatedPositions.findIndex(p => p.symbol === action.symbol);
    
    if (action.action === 'buy') {
      if (existingIndex >= 0) {
        const pos = updatedPositions[existingIndex];
        const newShares = pos.shares + shares;
        const newCostBasis = ((pos.shares * pos.entryPrice) + (shares * price)) / newShares;
        updatedPositions[existingIndex] = {
          ...pos,
          shares: newShares,
          entryPrice: newCostBasis,
          simulatedEntryPrice: newCostBasis,
          simulatedCurrentPrice: price,
          simulatedValue: newShares * price,
          transactionLog: [...pos.transactionLog, transaction],
        };
      } else {
        updatedPositions.push({
          symbol: action.symbol,
          name: action.symbol,
          targetAllocation: action.targetShares * price / execution.portfolioValue,
          currentAllocation: action.targetShares * price / execution.portfolioValue,
          shares: shares,
          entryPrice: price,
          entryMomentumScore: 0,
          entryDate: execution.timestamp,
          lastUpdated: execution.timestamp,
          unrealizedPnL: 0,
          realizedPnL: 0,
          totalReturn: 0,
          daysHeld: 0,
          simulatedEntryPrice: price,
          simulatedCurrentPrice: price,
          simulatedValue: shares * price,
          transactionLog: [transaction],
        });
      }
    } else if (action.action === 'sell') {
      if (existingIndex >= 0) {
        const pos = updatedPositions[existingIndex];
        const newShares = pos.shares - shares;
        
        if (newShares <= 0) {
          // Close position
          const realizedPnL = (price - pos.entryPrice) * pos.shares;
          updatedPositions.splice(existingIndex, 1);
        } else {
          updatedPositions[existingIndex] = {
            ...pos,
            shares: newShares,
            simulatedCurrentPrice: price,
            simulatedValue: newShares * price,
            realizedPnL: pos.realizedPnL + ((price - pos.entryPrice) * shares),
            transactionLog: [...pos.transactionLog, transaction],
          };
        }
      }
    }
  }
  
  const completedExecution: RebalanceExecution = {
    ...execution,
    status: 'simulated',
  };
  
  return { updatedPositions, execution: completedExecution };
}

/**
 * Update rebalance history with new execution
 */
export function updateRebalanceHistory(
  history: RebalanceHistory,
  execution: RebalanceExecution
): RebalanceHistory {
  const executions = [...history.executions, execution];
  
  const quarterlyRebalances = executions.filter(e => e.triggerType === 'quarterly').length;
  const driftRebalances = executions.filter(e => e.triggerType === 'drift').length;
  
  const totalCosts = executions.reduce((sum, e) => sum + e.transactionCosts, 0);
  const avgCost = executions.length > 0 ? totalCosts / executions.length : 0;
  
  return {
    executions,
    totalRebalances: executions.length,
    quarterlyRebalances,
    driftRebalances,
    lastExecutionId: execution.id,
    averageTransactionCost: avgCost,
    totalCostsYTD: totalCosts,
  };
}

/**
 * Calculate sector attribution vs SPY benchmark
 */
export function calculateSectorAttribution(
  positions: SectorPosition[],
  spyReturn: number,
  currentPrices: { [symbol: string]: number }
): SectorAttribution[] {
  return positions.map(pos => {
    const currentPrice = currentPrices[pos.symbol] || pos.entryPrice;
    const positionReturn = (currentPrice - pos.entryPrice) / pos.entryPrice;
    const alpha = positionReturn - spyReturn;
    
    return {
      symbol: pos.symbol,
      allocation: pos.currentAllocation,
      returnContribution: pos.currentAllocation * positionReturn,
      spyBenchmark: spyReturn,
      alpha,
      entryDate: pos.entryDate,
      exitDate: null,
      daysHeld: pos.daysHeld,
    };
  });
}

/**
 * Generate rebalance summary report
 */
export function generateRebalanceReport(
  signal: RebalanceSignal,
  history: RebalanceHistory,
  portfolio: SectorPortfolio
): string {
  const lines: string[] = [];
  
  lines.push('=' .repeat(80));
  lines.push('SECTOR REBALANCER REPORT');
  lines.push('=' .repeat(80));
  lines.push(`Generated: ${new Date().toISOString()}`);
  lines.push('');
  
  // Current portfolio state
  lines.push('PORTFOLIO STATE:');
  lines.push(`  SPY Core: ${(portfolio.spyCore * 100).toFixed(1)}%`);
  lines.push(`  Sector Overlay: ${(portfolio.sectorOverlay * 100).toFixed(1)}%`);
  lines.push(`  Last Rebalance: ${portfolio.lastRebalance.split('T')[0]}`);
  lines.push(`  Next Scheduled: ${portfolio.nextScheduledRebalance.split('T')[0]}`);
  lines.push('');
  
  // Signal status
  lines.push('REBALANCE SIGNAL:');
  lines.push(`  Triggered: ${signal.triggered ? 'YES' : 'NO'}`);
  if (signal.triggered) {
    lines.push(`  Type: ${signal.triggerType}`);
    lines.push(`  Reason: ${signal.reason}`);
  }
  lines.push('');
  
  // Drift analysis
  if (signal.drifts.length > 0) {
    lines.push('DRIFT ANALYSIS:');
    lines.push(`${'Symbol'.padEnd(8)} ${'Target%'.padEnd(10)} ${'Current%'.padEnd(10)} ${'Drift%'.padEnd(10)} ${'Action'.padEnd(12)}`);
    lines.push('-'.repeat(80));
    for (const d of signal.drifts) {
      const needsAction = d.needsRebalance ? 'REBALANCE' : 'HOLD';
      lines.push(
        `${d.symbol.padEnd(8)} ${(d.targetWeight * 100).toFixed(2).padEnd(10)} ` +
        `${(d.currentWeight * 100).toFixed(2).padEnd(10)} ${(d.drift * 100).toFixed(2).padEnd(10)} ` +
        `${needsAction.padEnd(12)}`
      );
    }
    lines.push('');
  }
  
  // Recommended actions
  if (signal.recommendedActions.length > 0) {
    const meaningfulActions = signal.recommendedActions.filter(a => a.action !== 'hold');
    
    if (meaningfulActions.length > 0) {
      lines.push('RECOMMENDED ACTIONS:');
      lines.push(`${'Action'.padEnd(8)} ${'Symbol'.padEnd(8)} ${'Shares'.padEnd(10)} ${'Value ($)'.padEnd(12)} ${'Cost ($)'.padEnd(10)} ${'Urgency'.padEnd(10)}`);
      lines.push('-'.repeat(80));
      
      for (const action of meaningfulActions) {
        const cost = calculateTransactionCost(action);
        lines.push(
          `${action.action.toUpperCase().padEnd(8)} ${action.symbol.padEnd(8)} ` +
          `${(action.sharesDelta > 0 ? '+' : '') + action.sharesDelta.toString().padEnd(9)} ` +
          `${action.estimatedValue.toFixed(2).padEnd(12)} ${cost.toFixed(2).padEnd(10)} ` +
          `${action.urgency.padEnd(10)}`
        );
      }
      
      const totalCost = calculateTotalTransactionCosts(signal.recommendedActions);
      lines.push('-'.repeat(80));
      lines.push(`Total Transaction Costs: $${totalCost.toFixed(2)}`);
      lines.push('');
    }
  }
  
  // History summary
  lines.push('REBALANCE HISTORY:');
  lines.push(`  Total Executions: ${history.totalRebalances}`);
  lines.push(`  Quarterly: ${history.quarterlyRebalances} | Drift: ${history.driftRebalances}`);
  lines.push(`  Avg Transaction Cost: $${history.averageTransactionCost.toFixed(2)}`);
  lines.push(`  Total Costs YTD: $${history.totalCostsYTD.toFixed(2)}`);
  lines.push('');
  
  lines.push('=' .repeat(80));
  
  return lines.join('\n');
}

/**
 * Full rebalancer state container
 */
export interface RebalancerState {
  portfolio: SectorPortfolio;
  history: RebalanceHistory;
  paperPositions: PaperSectorPosition[];
  currentPrices: { [symbol: string]: number };
  vix: number;
  lastSignal: RebalanceSignal | null;
}

/**
 * Initialize full rebalancer state
 */
export function initializeRebalancerState(
  initialPrices: { [symbol: string]: number } = {},
  initialVix: number = 20
): RebalancerState {
  return {
    portfolio: {
      spyCore: REBALANCE_CONFIG.spyCoreTarget,
      sectorOverlay: 0,
      cash: 0,
      lastRebalance: new Date().toISOString(),
      nextScheduledRebalance: calculateNextQuarterlyDate(),
      positions: [],
      isRebalanceRecommended: false,
      rebalanceReason: null,
    },
    history: initializeRebalanceHistory(),
    paperPositions: [],
    currentPrices: initialPrices,
    vix: initialVix,
    lastSignal: null,
  };
}

/**
 * Calculate next quarterly rebalance date
 */
function calculateNextQuarterlyDate(): string {
  const now = new Date();
  const currentMonth = now.getMonth();
  
  // Find next quarter month
  let nextQuarterMonth = REBALANCE_CONFIG.quarterlyMonths.find(m => m > currentMonth);
  let nextYear = now.getFullYear();
  
  if (!nextQuarterMonth) {
    nextQuarterMonth = REBALANCE_CONFIG.quarterlyMonths[0]; // Wrap to March
    nextYear++;
  }
  
  const nextDate = new Date(nextYear, nextQuarterMonth, REBALANCE_CONFIG.quarterlyDay);
  return nextDate.toISOString();
}

/**
 * Run full rebalancer cycle
 */
export function runRebalancerCycle(
  state: RebalancerState,
  targetAllocations: SectorAllocationItem[],
  momentumScores: SectorMomentum[],
  portfolioValue: number,
  newPrices?: { [symbol: string]: number },
  newVix?: number
): { newState: RebalancerState; report: string } {
  // Update market data
  const updatedState: RebalancerState = {
    ...state,
    currentPrices: newPrices || state.currentPrices,
    vix: newVix !== undefined ? newVix : state.vix,
  };
  
  // Generate rebalance signal
  const signal = generateRebalanceSignal(
    updatedState.portfolio,
    targetAllocations,
    updatedState.vix,
    momentumScores,
    portfolioValue
  );
  
  updatedState.lastSignal = signal;
  
  // Check if we should proceed
  const { proceed, reason } = shouldRebalanceProceed(signal, portfolioValue);
  
  if (signal.triggered && proceed) {
    // Create execution record
    const execution = createRebalanceExecution(signal, portfolioValue, updatedState.vix);
    
    // Execute in paper trading mode
    const { updatedPositions, execution: completedExecution } = executePaperRebalance(
      execution,
      updatedState.paperPositions,
      updatedState.currentPrices
    );
    
    // Update state
    updatedState.paperPositions = updatedPositions;
    updatedState.history = updateRebalanceHistory(updatedState.history, completedExecution);
    
    // Update portfolio metadata
    updatedState.portfolio = {
      ...updatedState.portfolio,
      lastRebalance: execution.timestamp,
      nextScheduledRebalance: calculateNextQuarterlyDate(),
    };
  }
  
  // Generate report
  const report = generateRebalanceReport(signal, updatedState.history, updatedState.portfolio);
  
  return { newState: updatedState, report };
}

// CLI demonstration
if (import.meta.main) {
  console.log('Sector Rebalancer Engine v2.41 - Phase 1.2');
  console.log('=' .repeat(50));
  console.log('Configuration:');
  console.log(`  Commission: ${(TRANSACTION_COSTS.commission * 100).toFixed(2)}%`);
  console.log(`  Slippage: ${(TRANSACTION_COSTS.slippage * 100).toFixed(2)}%`);
  console.log(`  Min Commission: $${TRANSACTION_COSTS.minCommission}`);
  console.log('');
  
  // Demo initialization
  const state = initializeRebalancerState({
    XLK: 200, XLE: 95, XLI: 140, XLY: 180, XLF: 45,
    XLV: 140, XLP: 75, XLU: 65, XLB: 85, XLRE: 40, XLC: 85,
  }, 18.5);
  
  console.log('Rebalancer initialized with demo prices');
  console.log(`VIX: ${state.vix}`);
  console.log(`Next Quarterly: ${state.portfolio.nextScheduledRebalance.split('T')[0]}`);
}
