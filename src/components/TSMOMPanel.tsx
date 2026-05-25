import React from 'react';

// ── Types ──────────────────────────────────────────────────────────────────

export interface TSMOMSpeed {
  label: string;
  weight: number;
  signal: number;
  asset_signals: Record<string, number>;
}

export interface TSMOMData {
  composite_signal: number;
  speed_breakdown: TSMOMSpeed[];
  position_recommendation: 'long' | 'short' | 'neutral';
  confidence: number;
  standalone_sharpe: number;
  overlay_sharpe: number;
  health_score: number;
  is_gated_off: boolean;
  generated_at: string;
}

interface TSMOMPanelProps {
  data: TSMOMData | null;
}

// ── Constants ──────────────────────────────────────────────────────────────

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

const BASELINE_SHARPE = 0.79; // Champion portfolio Sharpe

// ── Helpers ────────────────────────────────────────────────────────────────

function signalColor(signal: number): string {
  if (signal > 0.3) return '#10b981';
  if (signal > 0) return '#34d399';
  if (signal === 0) return '#94a3b8';
  if (signal > -0.3) return '#fbbf24';
  return '#ef4444';
}

function signalArrow(signal: number): string {
  if (signal > 0) return '\u2191';
  if (signal < 0) return '\u2193';
  return '\u2192';
}

function healthColor(score: number): string {
  if (score >= 0.80) return '#10b981';
  if (score >= 0.60) return '#f59e0b';
  return '#ef4444';
}

function positionBadge(
  recommendation: 'long' | 'short' | 'neutral',
  compositeSignal: number,
) {
  const styles: Record<string, { bg: string; text: string; label: string }> = {
    long: { bg: 'bg-emerald-900/50', text: 'text-emerald-400', label: 'LONG' },
    short: { bg: 'bg-red-900/50', text: 'text-red-400', label: 'SHORT' },
    neutral: { bg: 'bg-gray-700/50', text: 'text-gray-400', label: 'NEUTRAL' },
  };
  const s = styles[recommendation];
  return (
    <span
      className={`inline-flex items-center px-2.5 py-0.5 rounded text-xs font-semibold tracking-wide ${s.bg} ${s.text}`}
    >
      {signalArrow(compositeSignal)} {s.label}
    </span>
  );
}

// ── Sub-Components ─────────────────────────────────────────────────────────

function SignalGauge({ signal }: { signal: number }) {
  // Normalise -1..+1 to 0..100 for the bar position
  const pct = ((signal + 1) / 2) * 100;

  return (
    <div className="bg-gray-800/60 rounded-lg p-3 border border-gray-700/40">
      <div className="flex items-center justify-between mb-1">
        <span className="text-xs text-gray-500 uppercase tracking-wider font-medium">Composite Signal</span>
        <span
          className="text-2xl font-mono font-bold"
          style={{ color: signalColor(signal) }}
        >
          {signal >= 0 ? '+' : ''}
          {signal.toFixed(2)}
        </span>
      </div>

      {/* Gauge track */}
      <div className="relative h-3 bg-gray-800 rounded-full overflow-hidden">
        {/* Red zone (negative) */}
        <div
          className="absolute top-0 left-0 h-full"
          style={{ width: '50%', background: 'linear-gradient(to right, #ef4444, #fbbf24)' }}
        />
        {/* Green zone (positive) */}
        <div
          className="absolute top-0 right-0 h-full"
          style={{ width: '50%', background: 'linear-gradient(to left, #10b981, #34d399)' }}
        />
        {/* Center divider */}
        <div className="absolute top-0 left-1/2 h-full w-0.5 bg-gray-600 z-10" />
        {/* Signal marker */}
        <div
          className="absolute top-0 h-full w-0.5 bg-white shadow-md z-20 transition-all duration-500"
          style={{ left: `calc(${pct}% - 1px)` }}
        />
      </div>

      {/* Scale labels */}
      <div className="flex justify-between text-[10px] text-gray-600 mt-1">
        <span>-1.0</span>
        <span className="text-gray-500 font-medium">0.0</span>
        <span>+1.0</span>
      </div>
    </div>
  );
}

function SpeedCard({ speed }: { speed: TSMOMSpeed }) {
  const direction = speed.signal > 0 ? 'text-emerald-400' : speed.signal < 0 ? 'text-red-400' : 'text-gray-400';
  const weightPct = `${(speed.weight * 100).toFixed(0)}%`;
  const assetEntries = Object.entries(speed.asset_signals).slice(0, 6);

  return (
    <div className="bg-gray-800/60 rounded-lg p-3 border border-gray-700/40 space-y-2.5">
      {/* Header: label + weight */}
      <div className="flex items-center justify-between">
        <span className="text-sm font-semibold text-gray-100">{speed.label}</span>
        <span className="text-[10px] font-mono text-gray-500 bg-gray-900/60 px-2 py-0.5 rounded">
          {weightPct} weight
        </span>
      </div>

      {/* Weight bar */}
      <div>
        <div className="flex items-center justify-between text-[10px] text-gray-500 mb-0.5">
          <span>Weight in composite</span>
          <span>{weightPct}</span>
        </div>
        <div className="h-2 bg-gray-800 rounded-full overflow-hidden">
          <div
            className="h-full rounded-full transition-all duration-300"
            style={{ width: weightPct, backgroundColor: '#3b82f6' }}
          />
        </div>
      </div>

      {/* Signal value with arrow */}
      <div className="flex items-center gap-2">
        <span className="text-xs text-gray-500">Signal</span>
        <span className={`text-lg font-mono font-bold ${direction}`}>
          {signalArrow(speed.signal)} {speed.signal >= 0 ? '+' : ''}
          {speed.signal.toFixed(2)}
        </span>
      </div>

      {/* Per-asset signals */}
      {assetEntries.length > 0 && (
        <div>
          <span className="text-[10px] text-gray-500 uppercase tracking-wider block mb-1.5">
            Per-Asset Signals
          </span>
          <div className="space-y-1">
            {assetEntries.map(([asset, sig]) => {
              const assetColor = ASSET_COLORS[asset] || DEFAULT_ASSET_COLOR;
              // Normalise -1..+1 to 0..100 for bar width, centre at 50%
              const barPct = ((sig + 1) / 2) * 100;
              return (
                <div key={asset} className="flex items-center gap-1.5">
                  <span
                    className="text-[10px] font-semibold min-w-[32px] text-right shrink-0"
                    style={{ color: assetColor }}
                  >
                    {asset}
                  </span>
                  <div className="flex-1 h-3 bg-gray-800 rounded-full overflow-hidden relative">
                    {/* Negative fill (left half) */}
                    {sig < 0 && (
                      <div
                        className="absolute top-0 h-full rounded-l-full"
                        style={{
                          left: `${barPct}%`,
                          width: `${50 - barPct}%`,
                          backgroundColor: '#ef4444',
                        }}
                      />
                    )}
                    {/* Positive fill (right half) */}
                    {sig > 0 && (
                      <div
                        className="absolute top-0 h-full rounded-r-full"
                        style={{
                          left: '50%',
                          width: `${barPct - 50}%`,
                          backgroundColor: '#10b981',
                        }}
                      />
                    )}
                    {/* Center line */}
                    <div className="absolute top-0 left-1/2 h-full w-0.5 bg-gray-600 z-10" />
                  </div>
                  <span className="text-[10px] font-mono text-gray-400 min-w-[32px] text-right shrink-0">
                    {sig >= 0 ? '+' : ''}
                    {sig.toFixed(2)}
                  </span>
                </div>
              );
            })}
          </div>
        </div>
      )}
    </div>
  );
}

function HealthGatingCard({
  healthScore,
  isGatedOff,
}: {
  healthScore: number;
  isGatedOff: boolean;
}) {
  const color = healthColor(healthScore);
  const pct = Math.min(healthScore * 100, 100);
  const viabilityFloor = 0.60;
  const belowFloor = healthScore < viabilityFloor;

  return (
    <div className="bg-gray-800/60 rounded-lg p-3 border border-gray-700/40 space-y-3">
      <h4 className="text-xs font-medium text-gray-300 uppercase tracking-wider">
        Health &amp; Gating
      </h4>

      {/* Health score bar */}
      <div>
        <div className="flex items-center justify-between mb-1">
          <span className="text-xs text-gray-500">Health Score</span>
          <span className="text-sm font-mono font-bold" style={{ color }}>
            {healthScore.toFixed(2)}
          </span>
        </div>
        <div className="relative h-3 bg-gray-800 rounded-full overflow-hidden">
          {/* Viability floor line */}
          <div
            className="absolute top-0 w-0.5 bg-red-500 z-10 transition-all"
            style={{
              left: `${viabilityFloor * 100}%`,
              height: '100%',
              opacity: 0.8,
            }}
            title={`Viability floor: ${(viabilityFloor * 100).toFixed(0)}%`}
          />
          {/* Score bar */}
          <div
            className="h-full rounded-full transition-all duration-300"
            style={{ width: `${pct}%`, backgroundColor: color }}
          />
        </div>
        <span className="text-[10px] text-gray-600 mt-0.5 block">
          Viability floor: {(viabilityFloor * 100).toFixed(0)}%
          {belowFloor && (
            <span className="text-red-400 ml-1 font-semibold">
              &mdash; Below floor
            </span>
          )}
        </span>
      </div>

      {/* Gate status */}
      <div>
        <span className="text-xs text-gray-500 block mb-1">Gate Status</span>
        {isGatedOff ? (
          <div className="bg-red-900/20 border border-red-700/40 rounded-md px-3 py-2">
            <span className="text-xs font-semibold text-red-400">
              GATED OFF in HIGH_VOL / CRISIS
            </span>
            <p className="text-[10px] text-gray-500 mt-0.5">
              TSMOM disabled in high volatility and crisis regimes via RegimeGate.
            </p>
          </div>
        ) : (
          <div className="bg-emerald-900/20 border border-emerald-700/40 rounded-md px-3 py-2">
            <span className="text-xs font-semibold text-emerald-400">
              ACTIVE
            </span>
            <p className="text-[10px] text-gray-500 mt-0.5">
              Signal operational under current regime conditions.
            </p>
          </div>
        )}
      </div>
    </div>
  );
}

function PerformanceRow({
  standaloneSharpe,
  overlaySharpe,
}: {
  standaloneSharpe: number;
  overlaySharpe: number;
}) {
  const overlayColor =
    overlaySharpe >= standaloneSharpe
      ? '#10b981'
      : overlaySharpe >= standaloneSharpe * 0.8
        ? '#f59e0b'
        : '#ef4444';

  const isNetNegative = overlaySharpe < BASELINE_SHARPE;
  const netImpactLabel = isNetNegative
    ? 'Net-negative as overlay'
    : overlaySharpe >= standaloneSharpe
      ? 'Neutral or additive'
      : 'Slightly degraded';

  return (
    <div className="bg-gray-800/60 rounded-lg p-3 border border-gray-700/40">
      <h4 className="text-xs font-medium text-gray-300 uppercase tracking-wider mb-3">
        Performance Comparison
      </h4>

      <div className="grid grid-cols-2 gap-3 mb-3">
        <div className="bg-gray-900/60 rounded-md p-2.5 border border-gray-700/30">
          <span className="text-[10px] text-gray-500 uppercase tracking-wider block mb-1">
            Standalone Sharpe
          </span>
          <span className="text-lg font-mono font-bold text-emerald-400">
            {standaloneSharpe.toFixed(2)}
          </span>
        </div>
        <div className="bg-gray-900/60 rounded-md p-2.5 border border-gray-700/30">
          <span className="text-[10px] text-gray-500 uppercase tracking-wider block mb-1">
            Overlay Sharpe
          </span>
          <span className="text-lg font-mono font-bold" style={{ color: overlayColor }}>
            {overlaySharpe.toFixed(2)}
          </span>
        </div>
      </div>

      {/* Net impact badge */}
      <div
        className={`rounded-md px-3 py-2 border ${
          isNetNegative
            ? 'bg-red-900/20 border-red-700/40'
            : overlaySharpe >= standaloneSharpe
              ? 'bg-emerald-900/20 border-emerald-700/40'
              : 'bg-amber-900/20 border-amber-700/40'
        }`}
      >
        <div className="flex items-center justify-between">
          <span className="text-xs text-gray-400">Net Impact vs Baseline ({BASELINE_SHARPE.toFixed(2)})</span>
          <span
            className={`text-xs font-semibold ${
              isNetNegative ? 'text-red-400' : overlaySharpe >= standaloneSharpe ? 'text-emerald-400' : 'text-amber-400'
            }`}
          >
            {netImpactLabel}
          </span>
        </div>
        <p className="text-[10px] text-gray-500 mt-0.5">
          {isNetNegative
            ? 'Signal conflicts erode alpha when used as overlay — standalone use preferred.'
            : 'Overlay integration does not materially degrade standalone performance.'}
        </p>
      </div>
    </div>
  );
}

// ── Main Component ─────────────────────────────────────────────────────────

export function TSMOMPanel({ data }: TSMOMPanelProps) {
  if (!data) {
    return (
      <div className="bg-gray-900/50 border border-gray-700/50 rounded-lg p-4">
        <h3 className="text-sm font-semibold text-gray-100 mb-2">TSMOM Overlay</h3>
        <p className="text-sm text-gray-500">No TSMOM data available</p>
      </div>
    );
  }

  const {
    composite_signal,
    speed_breakdown,
    position_recommendation,
    confidence,
    standalone_sharpe,
    overlay_sharpe,
    health_score,
    is_gated_off,
    generated_at,
  } = data;

  return (
    <div className="bg-gray-900/50 border border-gray-700/50 rounded-lg p-4 space-y-4 text-gray-100">
      {/* ── Header ─────────────────────────────────────────── */}
      <div className="flex items-center justify-between">
        <div>
          <h3 className="text-sm font-semibold">TSMOM Overlay</h3>
          <p className="text-[10px] text-gray-600 mt-0.5">
            Time-Series Momentum &middot; Generated {generated_at}
          </p>
        </div>
        <div className="flex items-center gap-2">
          {positionBadge(position_recommendation, composite_signal)}
          <span className="text-[10px] font-mono text-gray-500 bg-gray-800/60 px-2 py-0.5 rounded">
            {(confidence * 100).toFixed(0)}% conf
          </span>
        </div>
      </div>

      {/* ── Section 1: Signal Strength Gauge ──────────────── */}
      <div>
        <SignalGauge signal={composite_signal} />
      </div>

      {/* ── Section 2: Speed Breakdown ─────────────────────── */}
      <div>
        <h4 className="text-xs font-medium text-gray-300 uppercase tracking-wider mb-2">
          Speed Breakdown ({speed_breakdown.length})
        </h4>
        {speed_breakdown.length === 0 ? (
          <p className="text-xs text-gray-500 italic">No speed breakdown available</p>
        ) : (
          <div className="grid grid-cols-1 sm:grid-cols-3 gap-2">
            {speed_breakdown.map((speed) => (
              <SpeedCard key={speed.label} speed={speed} />
            ))}
          </div>
        )}
      </div>

      {/* ── Section 3: Health & Gating ─────────────────────── */}
      <HealthGatingCard healthScore={health_score} isGatedOff={is_gated_off} />

      {/* ── Section 4: Performance Comparison ──────────────── */}
      <PerformanceRow standaloneSharpe={standalone_sharpe} overlaySharpe={overlay_sharpe} />
    </div>
  );
}
