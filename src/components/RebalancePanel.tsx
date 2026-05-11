import React, { useMemo } from 'react';
import type { SignalsData } from '../types/live';

interface RebalancePanelProps {
  signals: SignalsData | null;
  onRebalanceRequest?: () => void;
  readOnly?: boolean;
}

interface AllocationRow {
  symbol: string;
  current: number;
  target: number;
  drift: number;
  driftPct: number;
  needsRebalance: boolean;
  action: 'hold' | 'buy' | 'sell';
  sharesNeeded: number;
  estimatedValue: number;
}

const REBALANCE_THRESHOLD = 0.10; // 10% drift triggers rebalance

export function RebalancePanel({ signals, onRebalanceRequest, readOnly = true }: RebalancePanelProps) {
  const allocationData = useMemo(() => {
    if (!signals?.portfolio || !signals?.target_allocation) {
      return [];
    }

    const { positions, total_value } = signals.portfolio;
    const targets = signals.target_allocation;

    // Build allocation rows
    const rows: AllocationRow[] = [];

    // Current weights from positions
    const currentWeights: Record<string, number> = {};
    positions.forEach(pos => {
      currentWeights[pos.symbol] = pos.weight || (pos.value / total_value);
    });

    // Include all target symbols (even with 0 current weight)
    const allSymbols = new Set([...Object.keys(targets), ...Object.keys(currentWeights)]);

    allSymbols.forEach(symbol => {
      const current = currentWeights[symbol] || 0;
      const target = targets[symbol] || 0;
      const drift = current - target;
      const driftPct = Math.abs(drift);
      const needsRebalance = driftPct > REBALANCE_THRESHOLD;

      // Calculate shares needed (rough estimate)
      const price = signals.latest_prices?.[symbol] || 0;
      const targetValue = total_value * target;
      const currentValue = total_value * current;
      const deltaValue = targetValue - currentValue;
      const sharesNeeded = price > 0 ? Math.abs(deltaValue / price) : 0;

      rows.push({
        symbol,
        current,
        target,
        drift,
        driftPct,
        needsRebalance,
        action: drift > 0.01 ? 'sell' : drift < -0.01 ? 'buy' : 'hold',
        sharesNeeded,
        estimatedValue: Math.abs(deltaValue)
      });
    });

    // Sort by drift (largest first)
    return rows.sort((a, b) => b.driftPct - a.driftPct);
  }, [signals]);

  const summary = useMemo(() => {
    const needsRebalance = allocationData.filter(r => r.needsRebalance).length;
    const maxDrift = allocationData.length > 0 
      ? Math.max(...allocationData.map(r => r.driftPct)) 
      : 0;
    const totalBuyValue = allocationData
      .filter(r => r.action === 'buy')
      .reduce((sum, r) => sum + r.estimatedValue, 0);

    return { needsRebalance, maxDrift, totalBuyValue };
  }, [allocationData]);

  if (allocationData.length === 0) {
    return (
      <div className="rebalance-panel empty">
        <p>No allocation data available</p>
        <small>Waiting for portfolio data...</small>
      </div>
    );
  }

  const formatPct = (v: number) => `${(v * 100).toFixed(1)}%`;
  const formatCurrency = (v: number) => 
    new Intl.NumberFormat('en-US', { style: 'currency', currency: 'USD' }).format(v);

  const hasDrift = summary.needsRebalance > 0;

  return (
    <div className="rebalance-panel">
      <div className="rebalance-header">
        <h3>Allocation Drift Monitor</h3>
        <div className="rebalance-summary">
          <div className={`drift-indicator ${hasDrift ? 'alert' : 'ok'}`}>
            <span className="drift-status">{hasDrift ? '⚠️ REBALANCE NEEDED' : '✓ WITHIN TOLERANCE'}</span>
            <span className="drift-detail">{summary.needsRebalance} assets, max drift: {formatPct(summary.maxDrift)}</span>
          </div>
          {!readOnly && hasDrift && (
            <button className="rebalance-btn" onClick={onRebalanceRequest}>
              Rebalance Now
            </button>
          )}
        </div>
      </div>

      <div className="allocation-table-container">
        <table className="allocation-table">
          <thead>
            <tr>
              <th>Symbol</th>
              <th>Current</th>
              <th>Target</th>
              <th>Drift</th>
              <th>Status</th>
              <th>Action</th>
            </tr>
          </thead>
          <tbody>
            {allocationData.map(row => (
              <tr key={row.symbol} className={row.needsRebalance ? 'needs-rebalance' : ''}>
                <td>
                  <strong>{row.symbol}</strong>
                </td>
                <td>{formatPct(row.current)}</td>
                <td>{formatPct(row.target)}</td>
                <td className={row.drift > 0 ? 'overweight' : row.drift < 0 ? 'underweight' : ''}>
                  {row.drift > 0 ? '+' : ''}{formatPct(row.drift)}
                  <small className="drift-pct">({formatPct(row.driftPct)} off)</small>
                </td>
                <td>
                  <span className={`status-badge ${row.needsRebalance ? 'alert' : 'ok'}`}>
                    {row.needsRebalance ? 'DRIFT' : 'OK'}
                  </span>
                </td>
                <td>
                  {row.action !== 'hold' && (
                    <div className={`action-${row.action}`}>
                      <span className="action-label">{row.action.toUpperCase()}</span>
                      {!readOnly && (
                        <small>{row.sharesNeeded.toFixed(2)} shares</small>
                      )}
                    </div>
                  )}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {summary.totalBuyValue > 0 && !readOnly && (
        <div className="rebalance-estimate">
          <p>Estimated buy orders: {formatCurrency(summary.totalBuyValue)}</p>
          <small>Cash available: {formatCurrency(signals?.portfolio?.cash || 0)}</small>
        </div>
      )}

      {readOnly && (
        <div className="readonly-notice">
          <p>📊 Read-only mode. Rebalancing runs automatically via evaluator cron.</p>
          <small>Threshold: ±{REBALANCE_THRESHOLD * 100}% drift triggers rebalance</small>
        </div>
      )}
    </div>
  );
}
