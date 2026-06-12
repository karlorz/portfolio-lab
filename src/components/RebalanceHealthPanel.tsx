import React, { useMemo } from 'react';
import type { SmartRebalanceData } from '../types/live';

// ── Rebalance Health Types ────────────────────────────────────────────

export interface RebalanceHealthData {
  generated?: string;
  next_rebalance?: {
    date: string;
    days_until: number;
    frequency: string;
  };
  schedule_compliance?: {
    on_time: number;
    delayed: number;
    total: number;
    compliance_pct: number;
  };
  execution_history?: Array<{
    date: string;
    time: string;
    orders: number;
    total_value: number;
    symbols: string[];
  }>;
  total_executions?: number;
  market_data_consistency?: {
    status: string;
    reason?: string;
    checked_at?: string;
    rows?: Array<Record<string, unknown>>;
    warnings?: string[];
  };
  alpaca_feed_entitlement?: {
    configured_feed: string;
    effective_feed: string;
    entitlement: string;
    delayed: boolean;
    acceptable_for_live: boolean;
    policy_decision: string;
    reason?: string;
  };
}

interface RebalanceHealthPanelProps {
  rebalanceData: SmartRebalanceData | null | undefined;
  healthData: RebalanceHealthData | null | undefined;
}

export interface RebalanceLiveDiagnosticSummary {
  hasDiagnostics: boolean;
  feedEntitlement: {
    status: string;
    label: string;
    detail: string;
    acceptableForLive: boolean;
  } | null;
  marketDataConsistency: {
    status: string;
    label: string;
    detail: string;
  } | null;
}

// ── Helpers ───────────────────────────────────────────────────────────

function formatCurrency(v: number): string {
  if (v >= 1000000) return `$${(v / 1000000).toFixed(1)}M`;
  if (v >= 1000) return `$${(v / 1000).toFixed(0)}K`;
  return `$${v.toFixed(0)}`;
}

function daysUntilColor(days: number): string {
  if (days <= 3) return '#ef4444';
  if (days <= 7) return '#f59e0b';
  if (days <= 14) return '#3b82f6';
  return '#10b981';
}

function complianceColor(pct: number): string {
  if (pct >= 90) return '#10b981';
  if (pct >= 70) return '#f59e0b';
  return '#ef4444';
}

function formatStatus(status: string): string {
  return status.replace(/_/g, ' ');
}

function diagnosticUrgency(status: string, ok: boolean): 'low' | 'moderate' | 'high' {
  if (ok || status === 'ok' || status === 'accept') return 'low';
  if (status === 'unavailable' || status === 'warning') return 'moderate';
  return 'high';
}

export function summarizeRebalanceLiveDiagnostics(
  healthData: RebalanceHealthData | null | undefined
): RebalanceLiveDiagnosticSummary {
  const feed = healthData?.alpaca_feed_entitlement
    ? {
        status: healthData.alpaca_feed_entitlement.policy_decision,
        label: `Feed ${healthData.alpaca_feed_entitlement.policy_decision}: `
          + `${healthData.alpaca_feed_entitlement.effective_feed} / `
          + `${healthData.alpaca_feed_entitlement.entitlement}`,
        detail: healthData.alpaca_feed_entitlement.reason
          ?? (healthData.alpaca_feed_entitlement.acceptable_for_live ? 'live acceptable' : 'not live acceptable'),
        acceptableForLive: healthData.alpaca_feed_entitlement.acceptable_for_live,
      }
    : null;

  const consistency = healthData?.market_data_consistency
    ? {
        status: healthData.market_data_consistency.status,
        label: `Market data ${formatStatus(healthData.market_data_consistency.status)}`,
        detail: healthData.market_data_consistency.reason
          ?? (healthData.market_data_consistency.warnings?.[0] || 'no consistency warnings'),
      }
    : null;

  return {
    hasDiagnostics: Boolean(feed || consistency),
    feedEntitlement: feed,
    marketDataConsistency: consistency,
  };
}

// ── Component ─────────────────────────────────────────────────────────

export function RebalanceHealthPanel({ rebalanceData, healthData }: RebalanceHealthPanelProps) {
  const nextDate = useMemo(() => {
    if (healthData?.next_rebalance?.date) {
      return new Date(healthData.next_rebalance.date);
    }
    return null;
  }, [healthData]);
  const liveDiagnostics = useMemo(
    () => summarizeRebalanceLiveDiagnostics(healthData),
    [healthData]
  );

  if (!rebalanceData && !healthData) {
    return (
      <div className="rebalance-health-panel empty">
        <h3>Rebalance Health</h3>
        <p>No data available</p>
        <small>Run rebalance health exporter to populate</small>
      </div>
    );
  }

  const driftThreshold = rebalanceData?.status?.config?.drift_threshold ?? 0.10;
  const lastRebalance = rebalanceData?.status?.last_rebalance ?? null;
  const daysUntil = healthData?.next_rebalance?.days_until ?? 0;
  const urgency = rebalanceData?.urgency ?? 'low';
  const compliance = healthData?.schedule_compliance?.compliance_pct ?? 0;

  return (
    <div className="rebalance-health-panel">
      <div className="rh-header">
        <h3>Rebalance Health</h3>
        <span className="rh-version">v9.00</span>
      </div>

      {/* ── Next Scheduled Rebalance ─────────────────────────────── */}
      <div className="rh-next-section">
        <h4>Next Scheduled Rebalance</h4>
        <div className="rh-next-card" style={{ borderColor: daysUntilColor(daysUntil) }}>
          <div className="rh-next-date">
            {nextDate ? nextDate.toLocaleDateString('en-US', {
              weekday: 'long',
              year: 'numeric',
              month: 'long',
              day: 'numeric',
            }) : 'Unknown'}
          </div>
          <div className="rh-next-meta">
            <span className="rh-days-badge" style={{ backgroundColor: daysUntilColor(daysUntil) }}>
              {daysUntil > 0 ? `${daysUntil} days` : 'TODAY'}
            </span>
            <span className="rh-frequency">{healthData?.next_rebalance?.frequency ?? 'monthly (~30 days)'}</span>
          </div>
        </div>
        {lastRebalance && (
          <small className="rh-last-rebalance">
            Last rebalance: {new Date(lastRebalance).toLocaleDateString()}
          </small>
        )}
      </div>

      {/* ── Schedule Compliance ──────────────────────────────────── */}
      {healthData?.schedule_compliance && healthData.schedule_compliance.total > 0 && (
        <div className="rh-compliance-section">
          <h4>Schedule Compliance</h4>
          <div className="rh-compliance-bar-container">
            <div className="rh-compliance-labels">
              <span>On-time</span>
              <span style={{ color: complianceColor(compliance) }}>{compliance.toFixed(0)}%</span>
            </div>
            <div className="rh-compliance-bar">
              <div
                className="rh-compliance-fill"
                style={{
                  width: `${compliance}%`,
                  backgroundColor: complianceColor(compliance),
                }}
              />
            </div>
            <div className="rh-compliance-breakdown">
              <span className="rh-on-time">
                {healthData.schedule_compliance.on_time} on-time
              </span>
              <span className="rh-delayed">
                {healthData.schedule_compliance.delayed} delayed
              </span>
            </div>
          </div>
        </div>
      )}

      {/* ── Live Data Diagnostics ───────────────────────────────── */}
      {liveDiagnostics.hasDiagnostics && (
        <div className="rh-state-section">
          <h4>Live Data Diagnostics</h4>
          <div className="rh-state-grid">
            {liveDiagnostics.feedEntitlement && (
              <div className="rh-state-item">
                <label>Feed Policy</label>
                <span
                  className={`rh-badge rh-urgency-${diagnosticUrgency(
                    liveDiagnostics.feedEntitlement.status,
                    liveDiagnostics.feedEntitlement.acceptableForLive
                  )}`}
                >
                  {liveDiagnostics.feedEntitlement.status.toUpperCase()}
                </span>
                <small>{liveDiagnostics.feedEntitlement.label}</small>
                <small>{liveDiagnostics.feedEntitlement.detail}</small>
              </div>
            )}
            {liveDiagnostics.marketDataConsistency && (
              <div className="rh-state-item">
                <label>Market Data</label>
                <span
                  className={`rh-badge rh-urgency-${diagnosticUrgency(
                    liveDiagnostics.marketDataConsistency.status,
                    liveDiagnostics.marketDataConsistency.status === 'ok'
                  )}`}
                >
                  {liveDiagnostics.marketDataConsistency.status.toUpperCase()}
                </span>
                <small>{liveDiagnostics.marketDataConsistency.label}</small>
                <small>{liveDiagnostics.marketDataConsistency.detail}</small>
              </div>
            )}
          </div>
        </div>
      )}

      {/* ── Current State Summary ────────────────────────────────── */}
      {rebalanceData && (
        <div className="rh-state-section">
          <h4>Current State</h4>
          <div className="rh-state-grid">
            <div className="rh-state-item">
              <label>Urgency</label>
              <span className={`rh-badge rh-urgency-${urgency}`}>{urgency.toUpperCase()}</span>
            </div>
            <div className="rh-state-item">
              <label>Max Drift</label>
              <span className={rebalanceData.max_drift > driftThreshold ? 'rh-alert' : ''}>
                {(rebalanceData.max_drift * 100).toFixed(1)}%
              </span>
              <small>Threshold: {(driftThreshold * 100).toFixed(0)}%</small>
            </div>
            <div className="rh-state-item">
              <label>Decision</label>
              <span className="rh-decision">{rebalanceData.decision.replace(/_/g, ' ').toUpperCase()}</span>
            </div>
            <div className="rh-state-item">
              <label>YTD Cost</label>
              <span>{rebalanceData.ytd_cost_bps.toFixed(1)} bps</span>
              <small>{rebalanceData.remaining_budget_pct.toFixed(1)}% budget remaining</small>
            </div>
          </div>
        </div>
      )}

      {/* ── Execution History ────────────────────────────────────── */}
      {healthData?.execution_history && healthData.execution_history.length > 0 && (
        <div className="rh-history-section">
          <h4>
            Execution History
            <small>{healthData.total_executions} total</small>
          </h4>
          <div className="rh-history-list">
            {healthData.execution_history.slice(0, 5).map((exec, i) => (
              <div key={i} className="rh-history-row">
                <div className="rh-history-date">
                  <span className="rh-hist-date">{exec.date}</span>
                  <span className="rh-hist-time">{exec.time}</span>
                </div>
                <div className="rh-history-detail">
                  <span className="rh-hist-orders">{exec.orders} orders</span>
                  <span className="rh-hist-value">{formatCurrency(exec.total_value)}</span>
                </div>
                <div className="rh-history-symbols">
                  {exec.symbols.slice(0, 4).map(s => (
                    <span key={s} className="rh-hist-sym">{s}</span>
                  ))}
                  {exec.symbols.length > 4 && (
                    <span className="rh-hist-more">+{exec.symbols.length - 4}</span>
                  )}
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* ── Footer ───────────────────────────────────────────────── */}
      {healthData?.generated && (
        <div className="rh-footer">
          <small>Data generated: {new Date(healthData.generated).toLocaleString()}</small>
        </div>
      )}
    </div>
  );
}
