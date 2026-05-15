import React from 'react';

interface KurtosisData {
  active: boolean;
  kurtosis_20d: number;
  kurtosis_60d: number;
  ker_ratio: number;
  regime: string;
  transitioning: boolean;
  strategy_preference: string;
  tsom_weight: number;
  mr_weight: number;
  fat_tail_risk: number;
}

interface KurtosisRegimePanelProps {
  data: KurtosisData | null;
}

const REGIME_COLORS: Record<string, string> = {
  low_kurtosis: '#10b981',
  normal: '#3b82f6',
  high_kurtosis: '#f59e0b',
  extreme_kurtosis: '#ef4444',
};

const PREFERENCE_LABELS: Record<string, string> = {
  trend_following: 'Trend (TSMOM)',
  mean_reversion: 'Mean-Reversion',
  balanced: 'Balanced',
  defensive: 'Defensive',
};

export function KurtosisRegimePanel({ data }: KurtosisRegimePanelProps) {
  if (!data || !data.active) {
    return (
      <div className="panel">
        <h3>Kurtosis Regime (v4.91)</h3>
        <p className="muted">No data available</p>
      </div>
    );
  }

  const regimeColor = REGIME_COLORS[data.regime] || '#6b7280';
  const riskColor = data.fat_tail_risk > 0.5 ? '#ef4444' : data.fat_tail_risk > 0.3 ? '#f59e0b' : '#10b981';

  return (
    <div className="panel">
      <h3>Kurtosis Regime (v4.91)</h3>
      <div className="panel-grid">
        <div className="metric">
          <span className="label">Regime</span>
          <span className="value" style={{ color: regimeColor }}>
            {data.regime.replace('_', ' ').toUpperCase()}
            {data.transitioning && ' ⚡'}
          </span>
        </div>
        <div className="metric">
          <span className="label">Kurtosis 20d</span>
          <span className="value">{data.kurtosis_20d.toFixed(2)}</span>
        </div>
        <div className="metric">
          <span className="label">Kurtosis 60d</span>
          <span className="value">{data.kurtosis_60d.toFixed(2)}</span>
        </div>
        <div className="metric">
          <span className="label">KER Ratio</span>
          <span className="value" style={{ color: data.ker_ratio > 1.2 ? '#f59e0b' : '#10b981' }}>
            {data.ker_ratio.toFixed(2)}
          </span>
        </div>
        <div className="metric">
          <span className="label">Strategy</span>
          <span className="value">
            {PREFERENCE_LABELS[data.strategy_preference] || data.strategy_preference}
          </span>
        </div>
        <div className="metric">
          <span className="label">TSMOM / MR</span>
          <span className="value">
            {data.tsom_weight.toFixed(0)}% / {data.mr_weight.toFixed(0)}%
          </span>
        </div>
        <div className="metric">
          <span className="label">Fat Tail Risk</span>
          <span className="value" style={{ color: riskColor }}>
            {data.fat_tail_risk.toFixed(1)}%
          </span>
        </div>
      </div>
    </div>
  );
}
