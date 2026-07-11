import React from 'react';

export interface StackingEnsembleData {
  active: boolean;
  stacking_available: boolean;      // XGBoost model loaded
  runtime_role: 'research_dormant' | 'model_backed_advisory';
  runtime_status: 'unavailable_no_model' | 'model_loaded';
  live_authoritative: boolean;
  routed: boolean;
  routed_by: string | null;
  prediction_available: boolean;
  prediction_direction: string;     // bullish, bearish, neutral
  confidence: number;               // 0-1
  probability_bullish: number;
  probability_bearish: number;
  probability_neutral: number;
  fallback_used: boolean;           // True if weighted voting fallback
  model_version: string;
  voting_accuracy: number | null;   // Baseline 65%, unavailable without a model
  stacking_accuracy: number | null; // Target 76%, unavailable without a model
  accuracy_metrics_available: boolean;
  feature_count: number | null;     // Model-metadata-backed feature vector size
  feature_count_metadata_available: boolean;
  feature_count_source: 'model_metadata' | 'unavailable_no_model' | 'unavailable_missing_metadata';
  source_roster: string[];          // Empty when feature metadata is unavailable
  source_roster_version: string;
  fallback_semantics: string;
  latency_ms: number;               // Inference latency
  status_reason: string;
  operator_message: string;
  top_features?: Array<{ name: string; importance: number }>;
  backtest_finding?: string;
}

interface StackingEnsemblePanelProps {
  data: StackingEnsembleData | null;
}

function ConfidenceBar({ value, color = '#3b82f6' }: { value: number; color?: string }) {
  const pct = Math.max(0, Math.min(100, value * 100));
  return (
    <div className="confidence-bar">
      <div className="bar-fill" style={{ width: `${pct}%`, backgroundColor: color }} />
      <span className="bar-label">{(value * 100).toFixed(0)}%</span>
    </div>
  );
}

function DirectionBadge({ direction, fallback }: { direction: string; fallback: boolean }) {
  const colors: Record<string, string> = {
    bullish: '#22c55e',
    bearish: '#ef4444',
    neutral: '#6b7280',
    unavailable: '#6b7280',
  };
  const color = colors[direction] || '#6b7280';
  const label = direction.toUpperCase();
  return (
    <span className="badge" style={{ backgroundColor: color, color: '#fff' }}>
      {label}{fallback ? ' (fallback)' : ''}
    </span>
  );
}

function formatRuntimeRole(role: StackingEnsembleData['runtime_role']) {
  return role === 'research_dormant' ? 'Research dormant' : 'Model-backed advisory';
}

export function StackingEnsemblePanel({ data }: StackingEnsemblePanelProps) {
  if (!data) {
    return (
      <div className="panel">
        <h3>Stacking Ensemble (v3.10)</h3>
        <p className="muted">No stacking ensemble data available</p>
      </div>
    );
  }

  const predictionVisible = data.prediction_available;
  const accuracyMetricsVisible = (
    data.accuracy_metrics_available
    && data.voting_accuracy !== null
    && data.stacking_accuracy !== null
  );
  const votingAccuracy = data.voting_accuracy ?? 0;
  const stackingAccuracy = data.stacking_accuracy ?? 0;

  return (
    <div className="panel">
      <h3>Stacking Ensemble (v3.10)</h3>

      {/* Model status */}
      <div className="panel-section">
        <div className="metric-row">
          <span className="label">Model</span>
          <span className="value">
            {data.stacking_available ? data.model_version : 'Unavailable (no model)'}
          </span>
        </div>
        <div className="metric-row">
          <span className="label">Runtime role</span>
          <span className="value">
            {formatRuntimeRole(data.runtime_role)}
          </span>
        </div>
        <div className="metric-row">
          <span className="label">Order routed</span>
          <span className="value">
            {data.routed ? `Yes (${data.routed_by ?? 'unknown'})` : 'No'}
          </span>
        </div>
        {data.stacking_available && (
          <div className="metric-row">
            <span className="label">Features</span>
            <span className="value">{data.feature_count ?? 'Unavailable'}</span>
          </div>
        )}
        {data.stacking_available && (
          <div className="metric-row">
            <span className="label">Latency</span>
            <span className="value" style={{
              color: data.latency_ms < 5 ? '#22c55e' : data.latency_ms < 10 ? '#f59e0b' : '#ef4444',
            }}>
              {data.latency_ms.toFixed(1)}ms
            </span>
          </div>
        )}
      </div>

      {!predictionVisible && (
        <div className="panel-section">
          <h4>{formatRuntimeRole(data.runtime_role)}</h4>
          <p className="muted small">{data.status_reason}</p>
          <p className="muted small">{data.operator_message}</p>
        </div>
      )}

      {/* Prediction */}
      {predictionVisible && (
        <div className="panel-section">
          <h4>Current Prediction</h4>
          <DirectionBadge direction={data.prediction_direction} fallback={data.fallback_used} />
          <div className="panel-grid" style={{ marginTop: '0.5rem' }}>
            <div className="metric">
              <span className="label">Bullish</span>
              <ConfidenceBar value={data.probability_bullish} color="#22c55e" />
            </div>
            <div className="metric">
              <span className="label">Bearish</span>
              <ConfidenceBar value={data.probability_bearish} color="#ef4444" />
            </div>
            <div className="metric">
              <span className="label">Neutral</span>
              <ConfidenceBar value={data.probability_neutral} color="#6b7280" />
            </div>
          </div>
          <div className="metric-row">
            <span className="label">Overall Confidence</span>
            <span className="value">{(data.confidence * 100).toFixed(0)}%</span>
          </div>
        </div>
      )}

      {/* Accuracy comparison */}
      {accuracyMetricsVisible && (
        <div className="panel-section">
          <h4>Directional Accuracy</h4>
          <div className="panel-grid">
            <div className="metric">
              <span className="label">Voting (baseline)</span>
              <span className="value" style={{ color: '#f97316' }}>
                {(votingAccuracy * 100).toFixed(0)}%
              </span>
            </div>
            <div className="metric">
              <span className="label">Stacking (target)</span>
              <span className="value" style={{ color: '#22c55e' }}>
                {(stackingAccuracy * 100).toFixed(0)}%
              </span>
            </div>
            <div className="metric">
              <span className="label">Delta</span>
              <span className="value" style={{ color: '#3b82f6' }}>
                +{((stackingAccuracy - votingAccuracy) * 100).toFixed(0)}pp
              </span>
            </div>
          </div>
        </div>
      )}

      {/* Top features */}
      {data.top_features && data.top_features.length > 0 && (
        <div className="panel-section">
          <h4>Top Features</h4>
          <div className="feature-list">
            {data.top_features.slice(0, 5).map((f, i) => (
              <div key={i} className="feature-row">
                <span className="feature-name">{f.name}</span>
                <div className="confidence-bar" style={{ flex: 1, margin: '0 0.5rem' }}>
                  <div
                    className="bar-fill"
                    style={{ width: `${(f.importance * 100).toFixed(0)}%`, backgroundColor: '#8b5cf6' }}
                  />
                </div>
                <span className="feature-value">{(f.importance * 100).toFixed(1)}%</span>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Backtest finding */}
      {accuracyMetricsVisible && data.backtest_finding && (
        <div className="panel-section">
          <h4>Backtest Finding (Phase 5)</h4>
          <p className="muted small">{data.backtest_finding}</p>
        </div>
      )}
    </div>
  );
}
