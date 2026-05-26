import React from 'react';

export interface LLMSentimentData {
  timestamp: string;
  technical_regime: string;
  technical_confidence: number;
  sentiment_regime: string;
  sentiment_confidence: number;
  combined_score: number;
  combined_regime: string;
  technical_weight: number;
  sentiment_weight: number;
  circuit_breaker_level: string;
  position_scaling_factor: number;
  equity_tilt: number;
  bond_duration_tilt: number;
  gold_tilt: number;
}

interface LLMSentimentPanelProps {
  data: LLMSentimentData | null;
}

const REGIME_COLORS: Record<string, string> = {
  extreme_risk_off: '#dc2626',
  risk_off: '#ef4444',
  neutral: '#6b7280',
  risk_on: '#10b981',
  extreme_risk_on: '#059669',
};

const CIRCUIT_BREAKER_COLORS: Record<string, string> = {
  green: '#10b981',
  yellow: '#f59e0b',
  orange: '#f97316',
  red: '#ef4444',
};

function TiltBar({ value, label }: { value: number; label: string }) {
  const pct = Math.min(Math.abs(value) / 1 * 100, 100);
  const isPositive = value >= 0;
  return (
    <div style={{ marginBottom: 6 }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 2 }}>
        <span className="label">{label}</span>
        <span style={{
          fontSize: 11,
          color: isPositive ? '#10b981' : '#ef4444',
          minWidth: 36,
          textAlign: 'right',
        }}>
          {value >= 0 ? '+' : ''}{value.toFixed(2)}
        </span>
      </div>
      <div style={{
        height: 6, background: '#1e293b', borderRadius: 3,
        position: 'relative', overflow: 'hidden',
      }}>
        <div style={{
          position: 'absolute',
          left: isPositive ? '50%' : `${50 - pct}%`,
          width: `${pct / 2}%`,
          height: '100%',
          background: isPositive ? '#10b981' : '#ef4444',
          borderRadius: 3,
        }} />
      </div>
    </div>
  );
}

export function LLMSentimentPanel({ data }: LLMSentimentPanelProps) {
  if (!data) {
    return (
      <div className="panel">
        <h3>LLM Regime Sentiment</h3>
        <p className="muted">No sentiment data available</p>
      </div>
    );
  }

  const regimeColor = REGIME_COLORS[data.combined_regime] || '#6b7280';
  const cbColor = CIRCUIT_BREAKER_COLORS[data.circuit_breaker_level] || '#6b7280';
  const scoreColor = data.combined_score > 0.2 ? '#10b981'
    : data.combined_score < -0.2 ? '#ef4444' : '#f59e0b';

  return (
    <div className="panel">
      <h3>LLM Regime Sentiment</h3>

      {/* Combined regime */}
      <div className="panel-grid">
        <div className="metric">
          <span className="label">Combined Regime</span>
          <span className="value" style={{ color: regimeColor }}>
            {data.combined_regime.replace(/_/g, ' ')}
          </span>
        </div>
        <div className="metric">
          <span className="label">Circuit Breaker</span>
          <span className="value" style={{ color: cbColor }}>
            {data.circuit_breaker_level.toUpperCase()}
          </span>
        </div>
        <div className="metric">
          <span className="label">Combined Score</span>
          <span className="value" style={{ color: scoreColor }}>
            {data.combined_score >= 0 ? '+' : ''}{(data.combined_score * 100).toFixed(0)}%
          </span>
        </div>
        <div className="metric">
          <span className="label">Position Scale</span>
          <span className="value">{(data.position_scaling_factor * 100).toFixed(0)}%</span>
        </div>
      </div>

      {/* Source breakdown */}
      <div style={{ marginTop: 10 }}>
        <span className="label" style={{ display: 'block', marginBottom: 6 }}>Signal Sources</span>
        <div style={{
          display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 6,
        }}>
          <div style={{ padding: '4px 8px', background: '#0f172a', borderRadius: 4 }}>
            <span className="label">Technical</span>
            <div style={{ fontSize: 12, color: '#e2e8f0' }}>
              {data.technical_regime} ({(data.technical_confidence * 100).toFixed(0)}%)
            </div>
            <div style={{ fontSize: 11, color: '#94a3b8' }}>
              Weight: {(data.technical_weight * 100).toFixed(0)}%
            </div>
          </div>
          <div style={{ padding: '4px 8px', background: '#0f172a', borderRadius: 4 }}>
            <span className="label">Sentiment</span>
            <div style={{ fontSize: 12, color: '#e2e8f0' }}>
              {data.sentiment_regime} ({(data.sentiment_confidence * 100).toFixed(0)}%)
            </div>
            <div style={{ fontSize: 11, color: '#94a3b8' }}>
              Weight: {(data.sentiment_weight * 100).toFixed(0)}%
            </div>
          </div>
        </div>
      </div>

      {/* Asset tilts */}
      <div style={{ marginTop: 10 }}>
        <span className="label" style={{ display: 'block', marginBottom: 6 }}>Asset Tilts</span>
        <TiltBar value={data.equity_tilt} label="Equity (SPY)" />
        <TiltBar value={data.bond_duration_tilt} label="Duration (TLT)" />
        <TiltBar value={data.gold_tilt} label="Gold (GLD)" />
      </div>
    </div>
  );
}
