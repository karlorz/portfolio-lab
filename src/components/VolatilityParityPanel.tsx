import React from 'react';

interface VolParityAllocation {
  date: string;
  target_volatility: number;
  spy_pct: number;
  gld_pct: number;
  tlt_pct: number;
  core_vol_contribution: number;
  vix_short_pct: number;
  vix_tail_pct: number;
  vix_vol_contribution: number;
  cash_pct: number;
  expected_portfolio_vol: number;
  expected_max_dd: number;
  rebalance_triggered: boolean;
  rebalance_reason: string | null;
}

interface VolParitySummary {
  total_capital_allocation: number;
  total_vol_contribution: number;
  target_vol: number;
  vol_gap: number;
  vix_regime: string;
}

interface VolatilityParityData {
  allocation: VolParityAllocation;
  summary: VolParitySummary;
}

interface VolatilityParityPanelProps {
  data: VolatilityParityData | null;
}

function AllocationBar({ label, pct, color, maxPct = 60 }: {
  label: string; pct: number; color: string; maxPct?: number;
}) {
  const width = Math.min(Math.abs(pct) / maxPct * 100, 100);
  return (
    <div style={{ marginBottom: 4 }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 2 }}>
        <span style={{ fontSize: 11, color: '#e2e8f0' }}>{label}</span>
        <span style={{ fontSize: 11, color: '#94a3b8' }}>{pct.toFixed(1)}%</span>
      </div>
      <div style={{ height: 6, background: '#1e293b', borderRadius: 3, overflow: 'hidden' }}>
        <div style={{ width: `${width}%`, height: '100%', background: color, borderRadius: 3 }} />
      </div>
    </div>
  );
}

export function VolatilityParityPanel({ data }: VolatilityParityPanelProps) {
  if (!data || !data.allocation) {
    return (
      <div className="panel">
        <h3>Volatility Parity</h3>
        <p className="muted">No volatility parity data available</p>
      </div>
    );
  }

  const a = data.allocation;
  const s = data.summary;
  const volGapColor = Math.abs(s.vol_gap) < 0.02 ? '#10b981' : '#f59e0b';

  return (
    <div className="panel">
      <h3>Volatility Parity</h3>

      {/* Summary metrics */}
      <div className="panel-grid">
        <div className="metric">
          <span className="label">Target Vol</span>
          <span className="value">{(a.target_volatility * 100).toFixed(0)}%</span>
        </div>
        <div className="metric">
          <span className="label">Expected Vol</span>
          <span className="value" style={{ color: volGapColor }}>
            {(a.expected_portfolio_vol * 100).toFixed(1)}%
          </span>
        </div>
        <div className="metric">
          <span className="label">Vol Gap</span>
          <span className="value" style={{ color: volGapColor }}>
            {s.vol_gap >= 0 ? '+' : ''}{(s.vol_gap * 100).toFixed(1)}%
          </span>
        </div>
        <div className="metric">
          <span className="label">VIX Regime</span>
          <span className="value" style={{
            color: s.vix_regime === 'contango' ? '#10b981' : '#ef4444',
          }}>
            {s.vix_regime.charAt(0).toUpperCase() + s.vix_regime.slice(1)}
          </span>
        </div>
      </div>

      {/* Core allocation bars */}
      <div style={{ marginTop: 10 }}>
        <span className="label" style={{ display: 'block', marginBottom: 6 }}>Core Allocation</span>
        <AllocationBar label="SPY" pct={a.spy_pct * 100} color="#3b82f6" />
        <AllocationBar label="GLD" pct={a.gld_pct * 100} color="#f59e0b" />
        <AllocationBar label="TLT" pct={a.tlt_pct * 100} color="#8b5cf6" />
      </div>

      {/* Convexity allocation */}
      <div style={{ marginTop: 8 }}>
        <span className="label" style={{ display: 'block', marginBottom: 6 }}>Convexity</span>
        <AllocationBar label="Short VIX" pct={a.vix_short_pct * 100} color="#10b981" maxPct={10} />
        <AllocationBar label="Tail Protection" pct={a.vix_tail_pct * 100} color="#ef4444" maxPct={10} />
      </div>

      {/* Risk metrics */}
      <div className="panel-grid" style={{ marginTop: 8 }}>
        <div className="metric">
          <span className="label">Expected Max DD</span>
          <span className="value" style={{ color: '#f59e0b' }}>
            -{(a.expected_max_dd * 100).toFixed(0)}%
          </span>
        </div>
        <div className="metric">
          <span className="label">Cash</span>
          <span className="value">{(a.cash_pct * 100).toFixed(1)}%</span>
        </div>
      </div>

      {/* Rebalance alert */}
      {a.rebalance_triggered && (
        <div style={{
          marginTop: 8, padding: '6px 8px', borderRadius: 4,
          background: '#1e3a5f', color: '#93c5fd', fontSize: 12,
        }}>
          Rebalance: {a.rebalance_reason || 'Triggered'}
        </div>
      )}
    </div>
  );
}
