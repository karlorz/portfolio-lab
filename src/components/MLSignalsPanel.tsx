import React from 'react';
import type { SignalsData } from '../types/live';

type MLSignalsData = SignalsData['ml_signals'];

interface MLSignalsPanelProps {
  data: MLSignalsData | null;
}

const SYMBOL_COLORS: Record<string, string> = {
  SPY: '#3b82f6',
  GLD: '#f59e0b',
  TLT: '#8b5cf6',
  EFA: '#10b981',
  VXUS: '#06b6d4',
};

function DirectionIndicator({ direction }: { direction: string | null | undefined }) {
  if (!direction) return <span style={{ color: '#6b7280' }}>—</span>;
  const isUp = direction === 'up' || direction === 'bullish';
  const isDown = direction === 'down' || direction === 'bearish';
  return (
    <span style={{
      fontSize: 11,
      color: isUp ? '#10b981' : isDown ? '#ef4444' : '#6b7280',
    }}>
      {isUp ? '▲' : isDown ? '▼' : '◆'} {direction}
    </span>
  );
}

function RegimeIndicator({ regime }: { regime: string | null | undefined }) {
  if (!regime) return <span style={{ color: '#6b7280' }}>—</span>;
  const normalized = regime.toLowerCase();
  const color = normalized === 'bull'
    ? '#10b981'
    : normalized === 'bear'
      ? '#ef4444'
      : '#94a3b8';

  return (
    <span style={{ fontSize: 11, color }}>
      {regime}
    </span>
  );
}

function formatProbabilitySummary(probabilities: Record<string, number> | null | undefined) {
  const keys = Object.keys(probabilities ?? {});
  if (!probabilities || keys.length === 0) return '—';
  const preferredOrder = ['bear', 'neutral', 'bull'];
  const orderedKeys = [
    ...preferredOrder.filter((key) => key in probabilities),
    ...keys.filter((key) => !preferredOrder.includes(key)).sort(),
  ];

  return orderedKeys
    .map((key) => `${key} ${(probabilities[key] * 100).toFixed(0)}%`)
    .join(', ');
}

function formatDate(value: string | null | undefined) {
  if (!value) return '—';
  const parsed = new Date(value);
  if (!Number.isNaN(parsed.getTime())) return parsed.toISOString().slice(0, 10);
  return value.slice(0, 10);
}

export function MLSignalsPanel({ data }: MLSignalsPanelProps) {
  if (!data || !data.available) {
    return (
      <div className="panel">
        <h3>ML Signals</h3>
        <p className="muted">
          {data ? 'ML features not yet generated' : 'No ML signals data available'}
        </p>
      </div>
    );
  }

  const featureSymbols = Object.keys(data.features);
  const predictionSymbols = Object.keys(data.predictions);
  const hasGridSearch = data.grid_search?.sharpe != null;
  const gridSearchStatus = data.grid_search?.freshness_status === 'frozen_benchmark'
    ? 'Frozen benchmark'
    : data.grid_search?.freshness_status;

  return (
    <div className="panel">
      <h3>ML Signals</h3>

      <div style={{
        display: 'flex',
        flexWrap: 'wrap',
        gap: '6px 12px',
        marginBottom: 8,
        fontSize: 11,
        color: '#94a3b8',
      }}>
        <span>Generated {formatDate(data.generated_at ?? data.timestamp)}</span>
        <span>
          Feature as of {formatDate(data.feature_as_of)}
          {data.feature_freshness_status ? ` · ${data.feature_freshness_status}` : ''}
        </span>
        {data.feature_source_artifact && <span>{data.feature_source_artifact}</span>}
      </div>

      {/* Grid search summary */}
      {hasGridSearch && (
        <div>
          <div className="panel-grid">
            <div className="metric">
              <span className="label">Sharpe</span>
              <span className="value" style={{
                color: (data.grid_search.sharpe ?? 0) > 0.5 ? '#10b981' : '#f59e0b',
              }}>
                {(data.grid_search.sharpe ?? 0).toFixed(2)}
              </span>
            </div>
            <div className="metric">
              <span className="label">Volatility</span>
              <span className="value">
                {data.grid_search.volatility != null
                  ? `${(data.grid_search.volatility * 100).toFixed(1)}%`
                  : '—'}
              </span>
            </div>
          </div>
          <div style={{ marginTop: 6, fontSize: 11, color: '#94a3b8' }}>
            {gridSearchStatus && <span>{gridSearchStatus}</span>}
            {data.grid_search.benchmark_timestamp && (
              <span> · {formatDate(data.grid_search.benchmark_timestamp)}</span>
            )}
            {data.grid_search.source_artifact && (
              <span> · {data.grid_search.source_artifact}</span>
            )}
          </div>
        </div>
      )}

      {/* Predictions table */}
      {predictionSymbols.length > 0 && (
        <div style={{ marginTop: 8 }}>
          <span className="label" style={{ display: 'block', marginBottom: 6 }}>Predictions</span>
          <table style={{ width: '100%', fontSize: 11, borderCollapse: 'collapse' }}>
            <thead>
              <tr style={{ color: '#94a3b8', borderBottom: '1px solid #1e293b' }}>
                <th style={{ textAlign: 'left', padding: '2px 4px' }}>Symbol</th>
                <th style={{ textAlign: 'center', padding: '2px 4px' }}>Predicted Regime</th>
                <th style={{ textAlign: 'left', padding: '2px 4px' }}>Probabilities</th>
                <th style={{ textAlign: 'right', padding: '2px 4px' }}>Conf</th>
                <th style={{ textAlign: 'right', padding: '2px 4px' }}>Source</th>
              </tr>
            </thead>
            <tbody>
              {predictionSymbols.map((sym) => {
                const pred = data.predictions[sym];
                return (
                  <tr key={sym} style={{ borderBottom: '1px solid #0f172a' }}>
                    <td style={{ padding: '2px 4px', color: SYMBOL_COLORS[sym] || '#e2e8f0' }}>
                      {sym}
                    </td>
                    <td style={{ textAlign: 'center', padding: '2px 4px' }}>
                      <RegimeIndicator regime={pred?.predicted_regime} />
                    </td>
                    <td style={{ textAlign: 'left', padding: '2px 4px', color: '#cbd5e1' }}>
                      {formatProbabilitySummary(pred?.probabilities)}
                    </td>
                    <td style={{ textAlign: 'right', padding: '2px 4px', color: '#94a3b8' }}>
                      {pred?.confidence != null ? `${(pred.confidence * 100).toFixed(0)}%` : '—'}
                    </td>
                    <td style={{ textAlign: 'right', padding: '2px 4px', color: '#94a3b8' }}>
                      {pred?.heuristic == null ? '—' : pred.heuristic ? 'Heuristic' : 'Model'}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      )}

      {/* Features table */}
      {featureSymbols.length > 0 && (
        <div style={{ marginTop: 10 }}>
          <span className="label" style={{ display: 'block', marginBottom: 6 }}>Latest Features</span>
          <table style={{ width: '100%', fontSize: 11, borderCollapse: 'collapse' }}>
            <thead>
              <tr style={{ color: '#94a3b8', borderBottom: '1px solid #1e293b' }}>
                <th style={{ textAlign: 'left', padding: '2px 4px' }}>Symbol</th>
                <th style={{ textAlign: 'center', padding: '2px 4px' }}>VIX</th>
                <th style={{ textAlign: 'center', padding: '2px 4px' }}>Trend</th>
                <th style={{ textAlign: 'right', padding: '2px 4px' }}>5d Ret</th>
              </tr>
            </thead>
            <tbody>
              {featureSymbols.map((sym) => {
                const feat = data.features[sym];
                return (
                  <tr key={sym} style={{ borderBottom: '1px solid #0f172a' }}>
                    <td style={{ padding: '2px 4px', color: SYMBOL_COLORS[sym] || '#e2e8f0' }}>
                      {sym}
                    </td>
                    <td style={{ textAlign: 'center', padding: '2px 4px', color: '#e2e8f0' }}>
                      {feat?.vix_level != null ? feat.vix_level.toFixed(1) : '—'}
                    </td>
                    <td style={{ textAlign: 'center', padding: '2px 4px' }}>
                      <DirectionIndicator direction={
                        feat?.trend_direction != null
                          ? (feat.trend_direction > 0 ? 'up' : 'down')
                          : null
                      } />
                    </td>
                    <td style={{
                      textAlign: 'right', padding: '2px 4px',
                      color: (feat?.return_5d || 0) > 0 ? '#10b981' : '#ef4444',
                    }}>
                      {feat?.return_5d != null
                        ? `${feat.return_5d >= 0 ? '+' : ''}${(feat.return_5d * 100).toFixed(1)}%`
                        : '—'}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}
