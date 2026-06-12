import React from 'react';

export interface BLView {
  signal_name: string;
  asset: string;
  direction: 'bullish' | 'bearish' | 'neutral';
  confidence: number;  // 0-1
  expected_return_delta: number;
}

export interface BlackLittermanMapperData {
  prior_weights: Record<string, number>;
  posterior_weights: Record<string, number>;
  views: BLView[];
  tau: number;
  view_confidence_method: string;  // "idzorek"
}

interface BlackLittermanMapperPanelProps {
  data: BlackLittermanMapperData | null;
}

const ASSET_COLORS: Record<string, string> = {
  SPY: '#3b82f6',
  GLD: '#f59e0b',
  TLT: '#8b5cf6',
  IEF: '#06b6d4',
  EFA: '#10b981',
  VXUS: '#ec4899',
  DBC: '#f97316',
};

const DEFAULT_ASSET_COLOR = '#6b7280';

function directionBadge(direction: 'bullish' | 'bearish' | 'neutral') {
  const styles: Record<string, { tone: string; label: string }> = {
    bullish: { tone: 'alc-chip-success', label: 'Bullish' },
    bearish: { tone: 'alc-chip-danger', label: 'Bearish' },
    neutral: { tone: 'alc-chip-neutral', label: 'Neutral' },
  };
  const s = styles[direction];
  return (
    <span className={`alc-chip-small ${s.tone}`}>
      {direction === 'bullish' ? '\u2191' : direction === 'bearish' ? '\u2193' : '\u2192'} {s.label}
    </span>
  );
}

function ConfidenceBar({ value, label }: { value: number; label?: string }) {
  const pct = Math.min(Math.max(value * 100, 0), 100);
  const color = value >= 0.8 ? '#10b981' : value >= 0.5 ? '#f59e0b' : '#ef4444';

  return (
    <div className="alc-row">
      {label && (
        <span className="alc-small alc-fixed-lg">{label}</span>
      )}
      <div className="alc-progress alc-grow">
        <div
          className="alc-progress-fill"
          style={{ width: `${pct}%`, backgroundColor: color }}
        />
      </div>
      <span className="alc-small alc-mono alc-align-right alc-fixed-sm">
        {pct.toFixed(0)}%
      </span>
    </div>
  );
}

function AssetBar({ label, pct, color }: { label: string; pct: number; color: string }) {
  const width = Math.min(Math.abs(pct) * 100, 100);
  return (
    <div className="alc-row">
      <span className="alc-small alc-strong alc-fixed-sm">{label}</span>
      <div className="alc-progress-tall alc-grow">
        <div
          className="alc-progress-fill"
          style={{ width: `${width}%`, backgroundColor: color }}
        />
      </div>
      <span className="alc-small alc-mono alc-align-right alc-fixed-md">
        {(pct * 100).toFixed(1)}%
      </span>
    </div>
  );
}

export function BlackLittermanMapperPanel({ data }: BlackLittermanMapperPanelProps) {
  if (!data) {
    return (
      <div className="alc-panel alc-panel-muted">
        <h3 className="alc-title">Black-Litterman Mapper</h3>
        <p className="alc-muted">No Black-Litterman data available</p>
      </div>
    );
  }

  const { prior_weights, posterior_weights, views, tau, view_confidence_method } = data;

  // Collect all unique assets from both weight objects, sorted
  const allAssets = Array.from(
    new Set([...Object.keys(prior_weights), ...Object.keys(posterior_weights)])
  ).sort();

  // Group views by signal_name for Idzorek confidence display
  const viewSignals = views.reduce<Map<string, { views: BLView[]; avgConfidence: number }>>((acc, v) => {
    if (!acc.has(v.signal_name)) {
      acc.set(v.signal_name, { views: [], avgConfidence: 0 });
    }
    acc.get(v.signal_name)!.views.push(v);
    return acc;
  }, new Map());

  viewSignals.forEach((entry) => {
    const total = entry.views.reduce((sum, v) => sum + v.confidence, 0);
    entry.avgConfidence = entry.views.length > 0 ? total / entry.views.length : 0;
  });

  const confidenceMethodLabel =
    view_confidence_method === 'idzorek' ? 'Idzorek' : view_confidence_method || 'Unknown';

  return (
    <div className="alc-panel alc-panel-muted">
      {/* Header */}
      <div className="alc-header">
        <h3 className="alc-title">Black-Litterman Mapper</h3>
        <span className="alc-chip-small alc-chip-neutral alc-mono">
          tau={tau.toFixed(2)} &middot; {confidenceMethodLabel}
        </span>
      </div>

      {/* 1. Prior Weights */}
      <div className="alc-section">
        <h4 className="alc-section-title">
          Prior Weights (Equilibrium)
        </h4>
        <div className="alc-stack-xs">
          {allAssets.map((asset) => {
            const pct = prior_weights[asset] ?? 0;
            const color = ASSET_COLORS[asset] || DEFAULT_ASSET_COLOR;
            return <AssetBar key={`prior-${asset}`} label={asset} pct={pct} color={color} />;
          })}
        </div>
      </div>

      {/* 2. Active Views */}
      <div className="alc-section">
        <h4 className="alc-section-title">
          Active Views ({views.length})
        </h4>
        {views.length === 0 ? (
          <p className="alc-muted">No active views</p>
        ) : (
          <div className="alc-table-wrap">
            <table className="alc-table">
              <thead>
                <tr>
                  <th>Signal</th>
                  <th>Asset</th>
                  <th>Direction</th>
                  <th className="alc-cell-right">Confidence</th>
                  <th className="alc-cell-right">Return Delta</th>
                </tr>
              </thead>
              <tbody>
                {views.map((view, i) => {
                  const deltaColor =
                    view.direction === 'bullish'
                      ? 'alc-text-success'
                      : view.direction === 'bearish'
                        ? 'alc-text-danger'
                        : 'alc-text-muted';
                  return (
                    <tr
                      key={`${view.signal_name}-${view.asset}-${i}`}
                    >
                      <td className="alc-strong">
                        {view.signal_name}
                      </td>
                      <td className="alc-mono">
                        {view.asset}
                      </td>
                      <td>
                        {directionBadge(view.direction)}
                      </td>
                      <td className="alc-cell-right alc-mono">
                        {(view.confidence * 100).toFixed(0)}%
                      </td>
                      <td className={`alc-cell-right alc-mono ${deltaColor}`}>
                        {view.expected_return_delta >= 0 ? '+' : ''}
                        {(view.expected_return_delta * 100).toFixed(2)}%
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        )}
      </div>

      {/* 3. Posterior Weights vs Prior */}
      <div className="alc-section">
        <h4 className="alc-section-title">
          Posterior Weights (BL-Adjusted)
        </h4>
        <div className="alc-table-wrap">
          <table className="alc-table">
            <thead>
              <tr>
                <th>Asset</th>
                <th className="alc-cell-right">Prior</th>
                <th className="alc-cell-right">Posterior</th>
                <th className="alc-cell-right">Shift</th>
              </tr>
            </thead>
            <tbody>
              {allAssets.map((asset) => {
                const prior = prior_weights[asset] ?? 0;
                const posterior = posterior_weights[asset] ?? 0;
                const shift = posterior - prior;
                const shiftAbs = Math.abs(shift);
                const shiftColor =
                  shiftAbs > 0.05
                    ? 'alc-text-warning'
                    : shiftAbs > 0.02
                      ? 'alc-text-info'
                      : 'alc-text-muted';

                return (
                  <tr
                    key={`post-${asset}`}
                  >
                    <td className="alc-strong">
                      {asset}
                    </td>
                    <td className="alc-cell-right alc-mono">
                      {(prior * 100).toFixed(1)}%
                    </td>
                    <td className="alc-cell-right alc-mono alc-strong">
                      {(posterior * 100).toFixed(1)}%
                    </td>
                    <td className={`alc-cell-right alc-mono ${shiftColor}`}>
                      {shift >= 0 ? '+' : ''}
                      {(shift * 100).toFixed(1)}pp
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </div>

      {/* 4. Idzorek Confidence */}
      <div className="alc-section">
        <h4 className="alc-section-title">
          Idzorek Confidence per Signal
        </h4>
        {viewSignals.size === 0 ? (
          <p className="alc-muted">No signals with confidence data</p>
        ) : (
          <div className="alc-stack-sm">
            {Array.from(viewSignals.entries()).map(([name, entry]) => (
              <div key={name}>
                <div className="alc-row">
                  <span className="alc-small alc-strong">{name}</span>
                  <span className="alc-small alc-mono">
                    {entry.views.length} view{entry.views.length !== 1 ? 's' : ''}
                  </span>
                </div>
                <ConfidenceBar value={entry.avgConfidence} />
                {entry.views.length > 1 && (
                  <div className="alc-cluster">
                    {entry.views.map((v, j) => (
                      <span
                        key={j}
                        className="alc-chip-small alc-chip-neutral alc-mono"
                      >
                        {v.asset}: {(v.confidence * 100).toFixed(0)}%
                      </span>
                    ))}
                  </div>
                )}
              </div>
            ))}
          </div>
        )}
      </div>

      {/* 5. Tau parameter detail */}
      <div className="alc-card alc-card-compact">
        <div className="alc-row">
          <div>
            <span className="alc-label">Uncertainty Parameter (&tau;)</span>
            <p className="alc-small">
              Controls how much posterior weights deviate from prior. Lower &tau; = stronger view influence.
            </p>
          </div>
          <span className="alc-chip alc-chip-neutral alc-mono">
            {tau.toFixed(2)}
          </span>
        </div>
      </div>
    </div>
  );
}
