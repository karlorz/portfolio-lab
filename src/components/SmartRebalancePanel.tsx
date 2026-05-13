import React, { useMemo } from 'react';
import type { SmartRebalanceData } from '../types/live';

interface SmartRebalancePanelProps {
  data: SmartRebalanceData | null | undefined;
}

const URGENCY_COLORS: Record<string, string> = {
  emergency: '#ef4444',
  high: '#f59e0b',
  moderate: '#3b82f6',
  low: '#10b981',
};

const DECISION_LABELS: Record<string, string> = {
  execute: 'EXECUTE',
  override_emergency: 'EMERGENCY OVERRIDE',
  defer_toxicity: 'DEFERRED (VPIN)',
  defer_timing: 'DEFERRED (Timing)',
  defer_budget: 'DEFERRED (Budget)',
  skip_low_drift: 'SKIP (Low Drift)',
  no_positions: 'NO POSITIONS',
};

function formatBps(v: number): string {
  return `${v.toFixed(1)} bps`;
}

function formatPct(v: number): string {
  return `${(v * 100).toFixed(1)}%`;
}

export function SmartRebalancePanel({ data }: SmartRebalancePanelProps) {
  const driftEntries = useMemo(() => {
    if (!data?.drift_details) return [];
    return Object.entries(data.drift_details)
      .map(([sym, drift]) => ({ sym, drift: drift as number }))
      .sort((a, b) => b.drift - a.drift);
  }, [data]);

  if (!data) {
    return (
      <div className="smart-rebalance-panel empty">
        <h3>Smart Rebalance Controller</h3>
        <p>No data available</p>
        <small>Smart rebalancing requires active positions</small>
      </div>
    );
  }

  const urgencyColor = URGENCY_COLORS[data.urgency] || '#6b7280';
  const decisionLabel = DECISION_LABELS[data.decision] || data.decision.toUpperCase();
  const budgetUsedPct = data.status?.ytd_cost_pct ?? 0;
  const budgetLimit = data.status?.config?.annual_cost_limit ?? '0.5%';
  const budgetWarning = data.status?.is_warning ?? false;
  const budgetOver = data.status?.is_over_budget ?? false;

  return (
    <div className="smart-rebalance-panel">
      <div className="sr-header">
        <h3>Smart Rebalance Controller</h3>
        <span className="sr-version">v2.90</span>
      </div>

      {/* Decision Status */}
      <div className="sr-decision-row">
        <div className="sr-decision-badge" style={{ borderColor: urgencyColor }}>
          <span className="sr-decision-label">{decisionLabel}</span>
          <span className="sr-urgency" style={{ color: urgencyColor }}>
            {data.urgency.toUpperCase()}
          </span>
        </div>
        <div className="sr-reason">{data.reason}</div>
      </div>

      {/* Key Metrics Grid */}
      <div className="sr-metrics-grid">
        <div className="sr-metric">
          <label>Max Drift</label>
          <span className={`sr-value ${data.max_drift > 0.15 ? 'alert' : data.max_drift > 0.10 ? 'warning' : ''}`}>
            {formatPct(data.max_drift)}
          </span>
        </div>
        <div className="sr-metric">
          <label>VPIN</label>
          <span className={`sr-value ${data.vpin > 0.50 ? 'alert' : data.vpin > 0.35 ? 'warning' : ''}`}>
            {data.vpin.toFixed(2)}
          </span>
          <small>{data.vpin > 0.50 ? 'HIGH TOXICITY' : data.vpin > 0.35 ? 'MODERATE' : 'LOW'}</small>
        </div>
        <div className="sr-metric">
          <label>Est. Cost</label>
          <span className="sr-value">{formatBps(data.estimated_cost_bps)}</span>
        </div>
        <div className="sr-metric">
          <label>Window</label>
          <span className={`sr-value ${data.in_optimal_window ? 'positive' : 'muted'}`}>
            {data.in_optimal_window ? 'OPTIMAL' : 'OUTSIDE'}
          </span>
          <small>11:00-14:00 ET</small>
        </div>
      </div>

      {/* Drift by Asset */}
      {driftEntries.length > 0 && (
        <div className="sr-drift-section">
          <h4>Drift by Asset</h4>
          <div className="sr-drift-bars">
            {driftEntries.map(({ sym, drift }) => {
              const driftPct = drift * 100;
              const threshold = (data.status?.config?.drift_threshold ?? 0.10) * 100;
              const barWidth = Math.min(driftPct / 30 * 100, 100); // Scale to 30% max
              const isOver = driftPct > threshold;
              return (
                <div key={sym} className="sr-drift-row">
                  <span className="sr-drift-sym">{sym}</span>
                  <div className="sr-drift-bar-container">
                    <div
                      className={`sr-drift-bar ${isOver ? 'over' : ''}`}
                      style={{ width: `${barWidth}%` }}
                    />
                    <div
                      className="sr-drift-threshold"
                      style={{ left: `${(threshold / 30) * 100}%` }}
                    />
                  </div>
                  <span className={`sr-drift-val ${isOver ? 'alert' : ''}`}>
                    {driftPct.toFixed(1)}%
                  </span>
                </div>
              );
            })}
          </div>
          <small className="sr-drift-legend">
            Threshold: {((data.status?.config?.drift_threshold ?? 0.10) * 100).toFixed(0)}% | Red line = trigger
          </small>
        </div>
      )}

      {/* Cost Budget Gauge */}
      <div className="sr-budget-section">
        <h4>YTD Cost Budget</h4>
        <div className="sr-budget-gauge">
          <div className="sr-budget-bar-container">
            <div
              className={`sr-budget-bar ${budgetOver ? 'over' : budgetWarning ? 'warning' : ''}`}
              style={{ width: `${Math.min(budgetUsedPct / 0.6 * 100, 100)}%` }}
            />
            <div className="sr-budget-limit" style={{ left: `${(0.5 / 0.6) * 100}%` }} />
          </div>
          <div className="sr-budget-labels">
            <span>{budgetUsedPct.toFixed(2)}% used</span>
            <span className="sr-budget-limit-label">Limit: {budgetLimit}</span>
          </div>
        </div>
        <div className="sr-budget-stats">
          <span>YTD: {formatBps(data.ytd_cost_bps)}</span>
          <span>Remaining: {data.remaining_budget_pct.toFixed(1)}%</span>
        </div>
      </div>

      {/* Status Footer */}
      <div className="sr-footer">
        {data.status?.last_rebalance && (
          <small>Last rebalance: {new Date(data.status.last_rebalance).toLocaleDateString()}</small>
        )}
        {data.status?.deferred_until && (
          <small className="sr-deferred">Deferred until: {new Date(data.status.deferred_until).toLocaleString()}</small>
        )}
        <small className="sr-config">
          VPIN threshold: {data.status?.config?.vpin_threshold ?? 0.50} | Window: {data.status?.config?.optimal_window ?? '11:00-14:00 ET'}
        </small>
      </div>
    </div>
  );
}
