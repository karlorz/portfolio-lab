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
  const styles: Record<string, { bg: string; text: string; label: string }> = {
    bullish: { bg: 'bg-emerald-900/40', text: 'text-emerald-400', label: 'Bullish' },
    bearish: { bg: 'bg-red-900/40', text: 'text-red-400', label: 'Bearish' },
    neutral: { bg: 'bg-gray-700/50', text: 'text-gray-400', label: 'Neutral' },
  };
  const s = styles[direction];
  return (
    <span className={`inline-flex items-center px-2 py-0.5 rounded text-xs font-medium ${s.bg} ${s.text}`}>
      {direction === 'bullish' ? '\u2191' : direction === 'bearish' ? '\u2193' : '\u2192'} {s.label}
    </span>
  );
}

function ConfidenceBar({ value, label }: { value: number; label?: string }) {
  const pct = Math.min(Math.max(value * 100, 0), 100);
  const color = value >= 0.8 ? '#10b981' : value >= 0.5 ? '#f59e0b' : '#ef4444';

  return (
    <div className="flex items-center gap-2 w-full">
      {label && (
        <span className="text-xs text-gray-400 min-w-[80px] truncate shrink-0">{label}</span>
      )}
      <div className="flex-1 h-2 bg-gray-800 rounded-full overflow-hidden">
        <div
          className="h-full rounded-full transition-all duration-300"
          style={{ width: `${pct}%`, backgroundColor: color }}
        />
      </div>
      <span className="text-xs font-mono text-gray-400 min-w-[36px] text-right shrink-0">
        {pct.toFixed(0)}%
      </span>
    </div>
  );
}

function AssetBar({ label, pct, color }: { label: string; pct: number; color: string }) {
  const width = Math.min(Math.abs(pct) * 100, 100);
  return (
    <div className="flex items-center gap-2 w-full">
      <span className="text-xs font-semibold text-gray-200 min-w-[36px] shrink-0">{label}</span>
      <div className="flex-1 h-5 bg-gray-800 rounded overflow-hidden relative">
        <div
          className="h-full rounded transition-all duration-300 flex items-center justify-end pr-1"
          style={{ width: `${width}%`, backgroundColor: color }}
        >
          {width > 15 && (
            <span className="text-[10px] font-bold text-white drop-shadow-sm">
              {(pct * 100).toFixed(0)}%
            </span>
          )}
        </div>
      </div>
      <span className="text-xs font-mono text-gray-400 min-w-[48px] text-right shrink-0">
        {(pct * 100).toFixed(1)}%
      </span>
    </div>
  );
}

export function BlackLittermanMapperPanel({ data }: BlackLittermanMapperPanelProps) {
  if (!data) {
    return (
      <div className="bg-gray-900/50 rounded-lg border border-gray-700/50 p-4">
        <h3 className="text-sm font-semibold text-gray-100 mb-2">Black-Litterman Mapper</h3>
        <p className="text-xs text-gray-500">No Black-Litterman data available</p>
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
    <div className="bg-gray-900/50 rounded-lg border border-gray-700/50 p-4 space-y-4">
      {/* Header */}
      <div className="flex items-center justify-between">
        <h3 className="text-sm font-semibold text-gray-100">Black-Litterman Mapper</h3>
        <span className="text-[10px] font-mono text-gray-500 bg-gray-800/60 px-2 py-0.5 rounded">
          tau={tau.toFixed(2)} &middot; {confidenceMethodLabel}
        </span>
      </div>

      {/* 1. Prior Weights */}
      <div>
        <h4 className="text-xs font-medium text-gray-300 uppercase tracking-wider mb-2">
          Prior Weights (Equilibrium)
        </h4>
        <div className="space-y-1.5">
          {allAssets.map((asset) => {
            const pct = prior_weights[asset] ?? 0;
            const color = ASSET_COLORS[asset] || DEFAULT_ASSET_COLOR;
            return <AssetBar key={`prior-${asset}`} label={asset} pct={pct} color={color} />;
          })}
        </div>
      </div>

      {/* 2. Active Views */}
      <div>
        <h4 className="text-xs font-medium text-gray-300 uppercase tracking-wider mb-2">
          Active Views ({views.length})
        </h4>
        {views.length === 0 ? (
          <p className="text-xs text-gray-500 italic">No active views</p>
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full text-xs border-collapse">
              <thead>
                <tr className="text-gray-500 border-b border-gray-700/50">
                  <th className="text-left py-1.5 pr-2 font-medium">Signal</th>
                  <th className="text-left py-1.5 pr-2 font-medium">Asset</th>
                  <th className="text-left py-1.5 pr-2 font-medium">Direction</th>
                  <th className="text-right py-1.5 pr-2 font-medium">Confidence</th>
                  <th className="text-right py-1.5 font-medium">Return Delta</th>
                </tr>
              </thead>
              <tbody>
                {views.map((view, i) => {
                  const deltaColor =
                    view.direction === 'bullish'
                      ? 'text-emerald-400'
                      : view.direction === 'bearish'
                        ? 'text-red-400'
                        : 'text-gray-400';
                  return (
                    <tr
                      key={`${view.signal_name}-${view.asset}-${i}`}
                      className="border-b border-gray-800/50 hover:bg-gray-800/30"
                    >
                      <td className="py-1.5 pr-2 text-gray-200 font-medium whitespace-nowrap">
                        {view.signal_name}
                      </td>
                      <td className="py-1.5 pr-2 text-gray-300 font-mono whitespace-nowrap">
                        {view.asset}
                      </td>
                      <td className="py-1.5 pr-2 whitespace-nowrap">
                        {directionBadge(view.direction)}
                      </td>
                      <td className="py-1.5 pr-2 text-right font-mono text-gray-300 whitespace-nowrap">
                        {(view.confidence * 100).toFixed(0)}%
                      </td>
                      <td className={`py-1.5 text-right font-mono ${deltaColor} whitespace-nowrap`}>
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
      <div>
        <h4 className="text-xs font-medium text-gray-300 uppercase tracking-wider mb-2">
          Posterior Weights (BL-Adjusted)
        </h4>
        <div className="overflow-x-auto">
          <table className="w-full text-xs border-collapse">
            <thead>
              <tr className="text-gray-500 border-b border-gray-700/50">
                <th className="text-left py-1.5 pr-2 font-medium">Asset</th>
                <th className="text-right py-1.5 pr-2 font-medium">Prior</th>
                <th className="text-right py-1.5 pr-2 font-medium">Posterior</th>
                <th className="text-right py-1.5 font-medium">Shift</th>
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
                    ? 'text-amber-400'
                    : shiftAbs > 0.02
                      ? 'text-yellow-400'
                      : 'text-gray-400';

                return (
                  <tr
                    key={`post-${asset}`}
                    className="border-b border-gray-800/50 hover:bg-gray-800/30"
                  >
                    <td className="py-1.5 pr-2 text-gray-200 font-semibold whitespace-nowrap">
                      {asset}
                    </td>
                    <td className="py-1.5 pr-2 text-right font-mono text-gray-300 whitespace-nowrap">
                      {(prior * 100).toFixed(1)}%
                    </td>
                    <td className="py-1.5 pr-2 text-right font-mono text-gray-100 whitespace-nowrap">
                      {(posterior * 100).toFixed(1)}%
                    </td>
                    <td className={`py-1.5 text-right font-mono ${shiftColor} whitespace-nowrap`}>
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
      <div>
        <h4 className="text-xs font-medium text-gray-300 uppercase tracking-wider mb-2">
          Idzorek Confidence per Signal
        </h4>
        {viewSignals.size === 0 ? (
          <p className="text-xs text-gray-500 italic">No signals with confidence data</p>
        ) : (
          <div className="space-y-2">
            {Array.from(viewSignals.entries()).map(([name, entry]) => (
              <div key={name}>
                <div className="flex items-center justify-between mb-0.5">
                  <span className="text-xs text-gray-200 font-medium">{name}</span>
                  <span className="text-[10px] text-gray-500 font-mono">
                    {entry.views.length} view{entry.views.length !== 1 ? 's' : ''}
                  </span>
                </div>
                <ConfidenceBar value={entry.avgConfidence} />
                {entry.views.length > 1 && (
                  <div className="flex flex-wrap gap-1.5 mt-0.5 ml-1">
                    {entry.views.map((v, j) => (
                      <span
                        key={j}
                        className="text-[10px] text-gray-500 bg-gray-800/40 px-1.5 py-0.5 rounded"
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
      <div className="bg-gray-800/40 rounded-md p-2.5 border border-gray-700/30">
        <div className="flex items-center justify-between">
          <div>
            <span className="text-xs font-medium text-gray-400">Uncertainty Parameter (&tau;)</span>
            <p className="text-[10px] text-gray-600 mt-0.5">
              Controls how much posterior weights deviate from prior. Lower &tau; = stronger view influence.
            </p>
          </div>
          <span className="text-sm font-bold font-mono text-gray-100 bg-gray-900/60 px-3 py-1 rounded">
            {tau.toFixed(2)}
          </span>
        </div>
      </div>
    </div>
  );
}
