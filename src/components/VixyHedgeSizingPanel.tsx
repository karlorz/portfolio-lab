import React from 'react';

// ── Types ──────────────────────────────────────────────────────────────────

export interface HedgePerformance {
  period: string;
  portfolio_return: number;
  hedge_return: number;
  combined_return: number;
}

export interface VixyHedgeSizingData {
  vixy_position: number;
  hedge_ratio: number;
  target_volatility: number;
  vix_level: number;
  vix_zone: 'LOW' | 'NORMAL' | 'ELEVATED' | 'HIGH' | 'CRISIS';
  zone_thresholds: Record<string, [number, number]>;
  costs: {
    daily_carry: number;
    roll_cost: number;
    total_pct: number;
  };
  crisis_performance: HedgePerformance[];
  recommendation: {
    allocation: number;
    reasoning: string;
  };
}

interface VixyHedgeSizingPanelProps {
  data?: VixyHedgeSizingData | null;
}

// ── Zone Config ────────────────────────────────────────────────────────────

const ZONE_META: Record<string, { color: string; bg: string; border: string; badge: string }> = {
  LOW:      { color: '#10b981', bg: 'bg-emerald-900/30',    border: 'border-emerald-700/40', badge: 'bg-emerald-600/20 text-emerald-400' },
  NORMAL:   { color: '#3b82f6', bg: 'bg-blue-900/30',       border: 'border-blue-700/40',    badge: 'bg-blue-600/20 text-blue-400' },
  ELEVATED: { color: '#f59e0b', bg: 'bg-amber-900/30',      border: 'border-amber-700/40',   badge: 'bg-amber-600/20 text-amber-400' },
  HIGH:     { color: '#f97316', bg: 'bg-orange-900/30',     border: 'border-orange-700/40',  badge: 'bg-orange-600/20 text-orange-400' },
  CRISIS:   { color: '#ef4444', bg: 'bg-red-900/30',        border: 'border-red-700/40',     badge: 'bg-red-600/20 text-red-400' },
};

// ── Helpers ────────────────────────────────────────────────────────────────

function formatPct(v: number): string {
  return `${(v * 100).toFixed(1)}%`;
}

function formatPctFull(v: number): string {
  return `${(v * 100).toFixed(2)}%`;
}

// ── Component ──────────────────────────────────────────────────────────────

export function VixyHedgeSizingPanel({ data }: VixyHedgeSizingPanelProps) {
  if (!data) {
    return (
      <div className="bg-gray-900/50 border border-gray-700/50 rounded-lg p-4">
        <div className="flex items-center justify-between mb-2">
          <h3 className="text-base font-semibold text-gray-100">VIXY Hedge Sizing</h3>
        </div>
        <p className="text-sm text-gray-500">Hedge sizing data not available</p>
        <p className="text-xs text-gray-600 mt-1">Run VIXY hedge sizing exporter to populate</p>
      </div>
    );
  }

  const zoneMeta = ZONE_META[data.vix_zone] ?? ZONE_META.NORMAL;
  const zoneColor = zoneMeta.color;

  return (
    <div className="bg-gray-900/50 border border-gray-700/50 rounded-lg p-4 space-y-5 text-gray-100">

      {/* ── Header ─────────────────────────────────────────────── */}
      <div className="flex items-center justify-between">
        <h3 className="text-base font-semibold">VIXY Hedge Sizing</h3>
        <span className={`px-2.5 py-0.5 rounded text-xs font-semibold ${zoneMeta.badge}`}>
          {data.vix_zone}
        </span>
      </div>

      {/* ── Section 1: Current Hedge Allocation ───────────────── */}
      <div>
        <h4 className="text-sm font-medium text-gray-400 mb-3">Current Hedge Allocation</h4>
        <div className="grid grid-cols-3 gap-3">
          <div className="bg-gray-800/60 rounded-lg p-3 border border-gray-700/40">
            <p className="text-xs text-gray-500 mb-1">VIXY Position</p>
            <p className="text-lg font-mono font-bold text-gray-100">
              {formatPct(data.vixy_position)}
            </p>
          </div>
          <div className="bg-gray-800/60 rounded-lg p-3 border border-gray-700/40">
            <p className="text-xs text-gray-500 mb-1">Hedge Ratio</p>
            <p className="text-lg font-mono font-bold text-gray-100">
              {data.hedge_ratio.toFixed(2)}x
            </p>
          </div>
          <div className="bg-gray-800/60 rounded-lg p-3 border border-gray-700/40">
            <p className="text-xs text-gray-500 mb-1">Target Volatility</p>
            <p className="text-lg font-mono font-bold text-gray-100">
              {formatPct(data.target_volatility)}
            </p>
          </div>
        </div>
      </div>

      {/* ── Section 2: VIX Regime Zone ─────────────────────────── */}
      <div>
        <h4 className="text-sm font-medium text-gray-400 mb-3">VIX Regime Zone</h4>
        <div className={`${zoneMeta.bg} ${zoneMeta.border} rounded-lg p-3 border`}>
          <div className="flex items-center justify-between mb-2">
            <span className="text-xs text-gray-400">Current VIX Level</span>
            <span className="text-2xl font-mono font-bold" style={{ color: zoneColor }}>
              {data.vix_level.toFixed(2)}
            </span>
          </div>
          <div className="flex items-center justify-between text-xs text-gray-500">
            <span>Zone: <span className="font-semibold" style={{ color: zoneColor }}>{data.vix_zone}</span></span>
          </div>
        </div>

        {/* Zone threshold bar */}
        <div className="mt-2">
          {(() => {
            const totalRange = 60; // visual max scale
            return (
              <div>
                <div className="relative h-2 bg-gray-800 rounded-full overflow-hidden">
                  {/* Colored segments for each zone */}
                  {(['LOW', 'NORMAL', 'ELEVATED', 'HIGH', 'CRISIS'] as const).map((z) => {
                    const meta = ZONE_META[z];
                    const thresholds = data.zone_thresholds[z];
                    if (!thresholds) return null;
                    const [lo, hi] = thresholds;
                    const width = z === 'CRISIS'
                      ? Math.max(0, 100 - (lo / totalRange) * 100)
                      : ((hi - lo) / totalRange) * 100;
                    const left = z === 'CRISIS' ? (lo / totalRange) * 100 : (lo / totalRange) * 100;
                    const isActive = data.vix_zone === z;
                    return (
                      <div
                        key={z}
                        className={`absolute top-0 h-full transition-all duration-300 ${isActive ? 'opacity-100' : 'opacity-30'}`}
                        style={{
                          left: `${Math.max(0, left)}%`,
                          width: `${Math.max(2, width)}%`,
                          backgroundColor: meta.color,
                        }}
                        title={`${z}: ${lo}-${hi}`}
                      />
                    );
                  })}
                  {/* VIX marker */}
                  <div
                    className="absolute top-0 h-full w-0.5 bg-white shadow-md z-10"
                    style={{
                      left: `${Math.min((data.vix_level / totalRange) * 100, 97)}%`,
                    }}
                  />
                </div>
                <div className="flex justify-between text-[10px] text-gray-600 mt-1">
                  <span>0</span>
                  <span>LOW</span>
                  <span>NORMAL</span>
                  <span>ELEVATED</span>
                  <span>HIGH</span>
                  <span>CRISIS</span>
                  <span>60+</span>
                </div>
              </div>
            );
          })()}
        </div>

        {/* Zone threshold detail */}
        <div className="grid grid-cols-5 gap-1 mt-2">
          {(['LOW', 'NORMAL', 'ELEVATED', 'HIGH', 'CRISIS'] as const).map((z) => {
            const meta = ZONE_META[z];
            const t = data.zone_thresholds[z];
            if (!t) return null;
            const label = z === 'LOW'
              ? `< ${t[1]}`
              : z === 'CRISIS'
                ? `> ${t[0]}`
                : `${t[0]}-${t[1]}`;
            const isActive = data.vix_zone === z;
            return (
              <div
                key={z}
                className={`text-center rounded py-1 px-0.5 transition-all ${
                  isActive ? `${meta.bg} ${meta.border} border` : 'opacity-40'
                }`}
              >
                <p className="text-[10px] font-semibold" style={{ color: meta.color }}>{z}</p>
                <p className="text-[10px] text-gray-500">{label}</p>
              </div>
            );
          })}
        </div>
      </div>

      {/* ── Section 3: Hedge Cost Analysis ─────────────────────── */}
      <div>
        <h4 className="text-sm font-medium text-gray-400 mb-3">Hedge Cost Analysis</h4>
        <div className="grid grid-cols-3 gap-3 mb-2">
          <div className="bg-gray-800/60 rounded-lg p-3 border border-gray-700/40">
            <p className="text-xs text-gray-500 mb-1">Daily Carry</p>
            <p className="text-lg font-mono font-bold text-amber-400">
              {formatPctFull(data.costs.daily_carry)}
            </p>
          </div>
          <div className="bg-gray-800/60 rounded-lg p-3 border border-gray-700/40">
            <p className="text-xs text-gray-500 mb-1">Roll Cost</p>
            <p className="text-lg font-mono font-bold text-orange-400">
              {formatPctFull(data.costs.roll_cost)}
            </p>
          </div>
          <div className="bg-gray-800/60 rounded-lg p-3 border border-gray-700/40">
            <p className="text-xs text-gray-500 mb-1">Total Cost</p>
            <p className={`text-lg font-mono font-bold ${
              data.costs.total_pct > 0.08 ? 'text-red-400' :
              data.costs.total_pct > 0.05 ? 'text-amber-400' :
              'text-emerald-400'
            }`}>
              {formatPctFull(data.costs.total_pct)}
            </p>
          </div>
        </div>
        {/* Cost gauge bar */}
        <div className="bg-gray-800/60 rounded-lg p-2 border border-gray-700/40">
          <div className="flex items-center justify-between text-xs text-gray-500 mb-1">
            <span>% of portfolio</span>
            <span>{formatPctFull(data.costs.total_pct)}</span>
          </div>
          <div className="relative h-2 bg-gray-800 rounded-full overflow-hidden">
            <div
              className="absolute top-0 left-0 h-full rounded-full transition-all duration-500"
              style={{
                width: `${Math.min((data.costs.total_pct / 0.20) * 100, 100)}%`,
                backgroundColor:
                  data.costs.total_pct > 0.08 ? '#ef4444' :
                  data.costs.total_pct > 0.05 ? '#f59e0b' :
                  data.costs.total_pct > 0.03 ? '#3b82f6' :
                  '#10b981',
              }}
            />
          </div>
          <div className="flex justify-between text-[10px] text-gray-600 mt-1">
            <span>0%</span>
            <span>3%</span>
            <span>5%</span>
            <span>8%</span>
            <span>20%+</span>
          </div>
        </div>
      </div>

      {/* ── Section 4: Historical Crisis Performance ───────────── */}
      <div>
        <h4 className="text-sm font-medium text-gray-400 mb-3">Crisis Performance</h4>
        {data.crisis_performance.length === 0 ? (
          <p className="text-xs text-gray-500">No crisis performance data available</p>
        ) : (
          <div className="overflow-hidden rounded-lg border border-gray-700/40">
            <table className="w-full text-xs">
              <thead>
                <tr className="bg-gray-800/80 text-gray-500">
                  <th className="text-left py-2 px-3 font-medium">Period</th>
                  <th className="text-right py-2 px-3 font-medium">Portfolio</th>
                  <th className="text-right py-2 px-3 font-medium">Hedge (VIXY)</th>
                  <th className="text-right py-2 px-3 font-medium">Combined</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-gray-700/30">
                {data.crisis_performance.map((cp, i) => (
                  <tr key={i} className="bg-gray-800/40 hover:bg-gray-700/30 transition-colors">
                    <td className="py-2 px-3 text-gray-300 font-medium">{cp.period}</td>
                    <td className={`py-2 px-3 text-right font-mono font-semibold ${
                      cp.portfolio_return < 0 ? 'text-red-400' : 'text-emerald-400'
                    }`}>
                      {formatPctFull(cp.portfolio_return)}
                    </td>
                    <td className={`py-2 px-3 text-right font-mono font-semibold ${
                      cp.hedge_return < 0 ? 'text-red-400' : 'text-emerald-400'
                    }`}>
                      {formatPctFull(cp.hedge_return)}
                    </td>
                    <td className={`py-2 px-3 text-right font-mono font-semibold ${
                      cp.combined_return < 0 ? 'text-red-400' : 'text-emerald-400'
                    }`}>
                      {formatPctFull(cp.combined_return)}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>

      {/* ── Section 5: Sizing Recommendation ───────────────────── */}
      <div>
        <h4 className="text-sm font-medium text-gray-400 mb-3">Sizing Recommendation</h4>
        <div className={`rounded-lg p-4 border ${
          data.recommendation.allocation > 0.08 ? 'bg-red-900/20 border-red-700/40' :
          data.recommendation.allocation > 0.05 ? 'bg-amber-900/20 border-amber-700/40' :
          data.recommendation.allocation > 0.02 ? 'bg-blue-900/20 border-blue-700/40' :
          'bg-emerald-900/20 border-emerald-700/40'
        }`}>
          <div className="flex items-baseline justify-between mb-3">
            <span className="text-xs text-gray-400">Recommended VIXY Allocation</span>
            <span className={`text-2xl font-mono font-bold ${
              data.recommendation.allocation > 0.08 ? 'text-red-400' :
              data.recommendation.allocation > 0.05 ? 'text-amber-400' :
              data.recommendation.allocation > 0.02 ? 'text-blue-400' :
              'text-emerald-400'
            }`}>
              {formatPct(data.recommendation.allocation)}
            </span>
          </div>
          <p className="text-sm text-gray-300 leading-relaxed">
            {data.recommendation.reasoning}
          </p>
        </div>
        {/* Visual sizing gauge */}
        {data.recommendation.allocation > 0 && (
          <div className="mt-2 bg-gray-800/60 rounded-lg p-3 border border-gray-700/40">
            <div className="flex items-center justify-between text-xs text-gray-500 mb-2">
              <span>Allocation gauge</span>
              <span>Current: {formatPct(data.vixy_position)}</span>
            </div>
            <div className="relative h-3 bg-gray-800 rounded-full overflow-hidden">
              {/* Recommended allocation fill */}
              <div
                className="absolute top-0 left-0 h-full rounded-full opacity-60 transition-all duration-500"
                style={{
                  width: `${Math.min(data.recommendation.allocation * 100, 100)}%`,
                  backgroundColor:
                    data.recommendation.allocation > 0.08 ? '#ef4444' :
                    data.recommendation.allocation > 0.05 ? '#f59e0b' :
                    data.recommendation.allocation > 0.02 ? '#3b82f6' :
                    '#10b981',
                }}
              />
              {/* Current position marker */}
              <div
                className="absolute top-0 h-full w-0.5 bg-white shadow-md z-10"
                style={{
                  left: `${Math.min(data.vixy_position * 100, 100)}%`,
                }}
              />
            </div>
            <div className="flex justify-between text-[10px] text-gray-600 mt-1">
              <span>0%</span>
              <span>5%</span>
              <span>10%</span>
              <span>15%+</span>
            </div>
            <div className="flex items-center gap-2 mt-2 text-[10px] text-gray-500">
              <span className="inline-block w-2 h-2 rounded-full bg-white/80" />
              <span>Current position</span>
              <span className="inline-block w-2 h-2 rounded-full bg-gray-500/60 ml-2" />
              <span>Recommended</span>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
