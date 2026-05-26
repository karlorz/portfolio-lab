import React from 'react';

interface FeatureData {
  vix_level?: number | null;
  trend_direction?: number | null;
  price_vs_sma20?: number | null;
  return_5d?: number | null;
  spy_correlation?: number | null;
}

interface PredictionData {
  predicted_return?: number | null;
  confidence?: number | null;
  direction?: string | null;
}

interface GridSearchData {
  best_sharpe?: number | null;
  best_params?: Record<string, number | string> | null;
  configs_tested?: number | null;
}

export interface MLSignalsData {
  available: boolean;
  timestamp: string | null;
  predictions: Record<string, PredictionData>;
  features: Record<string, FeatureData>;
  grid_search: GridSearchData;
}

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
  const hasGridSearch = data.grid_search?.best_sharpe != null;

  return (
    <div className="panel">
      <h3>ML Signals</h3>

      {/* Grid search summary */}
      {hasGridSearch && (
        <div className="panel-grid">
          <div className="metric">
            <span className="label">Best Sharpe</span>
            <span className="value" style={{
              color: (data.grid_search.best_sharpe || 0) > 0.5 ? '#10b981' : '#f59e0b',
            }}>
              {(data.grid_search.best_sharpe || 0).toFixed(2)}
            </span>
          </div>
          <div className="metric">
            <span className="label">Configs Tested</span>
            <span className="value">{data.grid_search.configs_tested || 0}</span>
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
                <th style={{ textAlign: 'center', padding: '2px 4px' }}>Dir</th>
                <th style={{ textAlign: 'right', padding: '2px 4px' }}>Return</th>
                <th style={{ textAlign: 'right', padding: '2px 4px' }}>Conf</th>
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
                      <DirectionIndicator direction={pred?.direction} />
                    </td>
                    <td style={{
                      textAlign: 'right', padding: '2px 4px',
                      color: (pred?.predicted_return || 0) > 0 ? '#10b981' : '#ef4444',
                    }}>
                      {pred?.predicted_return != null
                        ? `${pred.predicted_return >= 0 ? '+' : ''}${(pred.predicted_return * 100).toFixed(1)}%`
                        : '—'}
                    </td>
                    <td style={{ textAlign: 'right', padding: '2px 4px', color: '#94a3b8' }}>
                      {pred?.confidence != null ? `${(pred.confidence * 100).toFixed(0)}%` : '—'}
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
