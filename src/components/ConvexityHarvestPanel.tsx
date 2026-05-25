import React from 'react';

interface ConvexityHarvestData {
  date: string;
  allocation_pct: number;
  position_type: string;
  vix_level: number;
  contango_pct: number;
  expected_roll_yield: number;
  risk_score: number;
  exit_triggered: boolean;
  exit_reason: string | null;
}

interface ConvexityHarvestPanelProps {
  data: ConvexityHarvestData | null;
}

const POSITION_COLORS: Record<string, string> = {
  short_vix: '#10b981',
  long_vix: '#ef4444',
  flat: '#6b7280',
};

const POSITION_LABELS: Record<string, string> = {
  short_vix: 'Short VIX',
  long_vix: 'Long VIX',
  flat: 'Flat',
};

export function ConvexityHarvestPanel({ data }: ConvexityHarvestPanelProps) {
  if (!data) {
    return (
      <div className="panel">
        <h3>Convexity Harvest</h3>
        <p className="muted">No convexity harvest data available</p>
      </div>
    );
  }

  const posColor = POSITION_COLORS[data.position_type] || '#6b7280';
  const posLabel = POSITION_LABELS[data.position_type] || data.position_type;

  const riskColor = data.risk_score < 0.3 ? '#10b981'
    : data.risk_score < 0.6 ? '#f59e0b'
    : '#ef4444';

  const contangoColor = data.contango_pct > 5 ? '#10b981'
    : data.contango_pct > 0 ? '#f59e0b'
    : '#ef4444';

  return (
    <div className="panel">
      <h3>Convexity Harvest</h3>

      {/* Position status */}
      <div className="panel-grid">
        <div className="metric">
          <span className="label">Position</span>
          <span className="value" style={{ color: posColor }}>
            {posLabel}
          </span>
        </div>
        <div className="metric">
          <span className="label">Allocation</span>
          <span className="value">{data.allocation_pct.toFixed(1)}%</span>
        </div>
        <div className="metric">
          <span className="label">VIX Level</span>
          <span className="value" style={{
            color: data.vix_level < 15 ? '#10b981' : data.vix_level < 25 ? '#f59e0b' : '#ef4444',
          }}>
            {data.vix_level.toFixed(1)}
          </span>
        </div>
        <div className="metric">
          <span className="label">Risk Score</span>
          <span className="value" style={{ color: riskColor }}>
            {(data.risk_score * 100).toFixed(0)}%
          </span>
        </div>
      </div>

      {/* Roll yield metrics */}
      <div className="panel-grid">
        <div className="metric">
          <span className="label">Contango</span>
          <span className="value" style={{ color: contangoColor }}>
            {data.contango_pct.toFixed(1)}%
          </span>
        </div>
        <div className="metric">
          <span className="label">Exp. Roll Yield</span>
          <span className="value" style={{
            color: data.expected_roll_yield > 0 ? '#10b981' : '#ef4444',
          }}>
            {data.expected_roll_yield >= 0 ? '+' : ''}{data.expected_roll_yield.toFixed(1)}%
          </span>
        </div>
      </div>

      {/* Risk bar */}
      <div style={{ marginTop: 8 }}>
        <span className="label" style={{ display: 'block', marginBottom: 4 }}>Risk Level</span>
        <div style={{
          height: 8, background: '#1e293b', borderRadius: 4, overflow: 'hidden',
        }}>
          <div style={{
            width: `${Math.min(data.risk_score * 100, 100)}%`,
            height: '100%',
            background: riskColor,
            borderRadius: 4,
            transition: 'width 0.3s',
          }} />
        </div>
      </div>

      {/* Exit alert */}
      {data.exit_triggered && (
        <div style={{
          marginTop: 8, padding: '6px 8px', borderRadius: 4,
          background: '#7f1d1d', color: '#fca5a5', fontSize: 12,
        }}>
          EXIT: {data.exit_reason || 'Risk threshold breached'}
        </div>
      )}
    </div>
  );
}
