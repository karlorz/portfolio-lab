import React from 'react';

// ── Types ──────────────────────────────────────────────────────────────────

export interface GateRule {
  signal_name: string;
  off_regimes: string[];
  is_active: boolean;
}

export interface RegimeGateData {
  current_regime: string;
  regime_confidence: number;
  gate_rules: GateRule[];
  active_signals: string[];
  inactive_signals: string[];
  min_dwell_days: number;
  generated_at: string;
}

interface RegimeGatePanelProps {
  data: RegimeGateData | null;
}

// ── Constants ──────────────────────────────────────────────────────────────

const ALL_REGIMES = ['CRISIS', 'HIGH_VOL', 'NORMAL', 'LOW_VOL', 'RECOVERY'] as const;

const REGIME_META: Record<string, {
  color: string; bg: string; border: string; badge: string; headerBg: string; colBg: string
}> = {
  CRISIS:   { color: '#ef4444', bg: 'bg-red-900/20',   border: 'border-red-700/40',  badge: 'bg-red-600/20 text-red-400',    headerBg: 'bg-red-900/30',  colBg: 'bg-red-900/15' },
  HIGH_VOL: { color: '#f97316', bg: 'bg-orange-900/20', border: 'border-orange-700/40', badge: 'bg-orange-600/20 text-orange-400', headerBg: 'bg-orange-900/30', colBg: 'bg-orange-900/15' },
  NORMAL:   { color: '#3b82f6', bg: 'bg-blue-900/20',  border: 'border-blue-700/40',  badge: 'bg-blue-600/20 text-blue-400',  headerBg: 'bg-blue-900/30',  colBg: 'bg-blue-900/15' },
  LOW_VOL:  { color: '#10b981', bg: 'bg-emerald-900/20', border: 'border-emerald-700/40', badge: 'bg-emerald-600/20 text-emerald-400', headerBg: 'bg-emerald-900/30', colBg: 'bg-emerald-900/15' },
  RECOVERY: { color: '#14b8a6', bg: 'bg-teal-900/20',  border: 'border-teal-700/40',  badge: 'bg-teal-600/20 text-teal-400',  headerBg: 'bg-teal-900/30',  colBg: 'bg-teal-900/15' },
};

const DEFAULT_REGIME_META = REGIME_META.NORMAL;

function humanizeSignalName(name: string): string {
  const map: Record<string, string> = {
    multi_speed_momentum: 'Multi-Speed Momentum',
    cross_asset_relative_value: 'Cross-Asset RV',
    cross_asset_rv: 'Cross-Asset RV',
    international_momentum: 'International Momentum',
    alternative_data: 'Alternative Data',
    cross_asset_regime_arb: 'Regime Arbitrage',
    unified_overlay: 'Unified Overlay',
    behavioral_sentiment: 'Behavioral Sentiment',
    crypto_momentum: 'Crypto Momentum',
    tsmom_overlay: 'TSMOM Overlay',
    collar_signal: 'Collar Signal',
    bond_duration_signal: 'Bond Duration',
    calendar_seasonality: 'Calendar Seasonality',
    vix_term_structure: 'VIX Term Structure',
    fed_policy_overlay: 'Fed Policy Overlay',
    vpin_bvc: 'VPIN / BVC',
    multi_strategy_adapters: 'Multi-Strategy Adapters',
    stacking_integrator: 'Stacking Integrator',
    signal_snapshot: 'Signal Snapshot',
    regime_gate: 'Regime Gate',
    health_tracker: 'Health Tracker',
    credit_spread: 'Credit Spread',
    commodity_curve: 'Commodity Curve',
  };
  return map[name] || name.replace(/_/g, ' ').replace(/\b\w/g, (c) => c.toUpperCase());
}

export function makeSignalListKey(name: string, index: number): string {
  return `${name}-${index}`;
}

// ── Sub-components ─────────────────────────────────────────────────────────

function RegimeDot({ active, color }: { active: boolean; color: string }) {
  return (
    <span
      className="inline-block w-2.5 h-2.5 rounded-full"
      style={{ backgroundColor: active ? color : '#374151' }}
      title={active ? 'ON' : 'OFF'}
    />
  );
}

function ConfidenceBar({ value }: { value: number }) {
  const pct = Math.min(Math.max(value * 100, 0), 100);
  const color = value >= 0.8 ? '#10b981' : value >= 0.5 ? '#f59e0b' : '#ef4444';
  return (
    <div className="flex items-center gap-2 w-full">
      <div className="flex-1 h-2 bg-gray-800 rounded-full overflow-hidden">
        <div
          className="h-full rounded-full transition-all duration-300"
          style={{ width: `${pct}%`, backgroundColor: color }}
        />
      </div>
      <span className="text-xs font-mono text-gray-400 min-w-[40px] text-right shrink-0">
        {pct.toFixed(0)}%
      </span>
    </div>
  );
}

// ── Component ──────────────────────────────────────────────────────────────

export function RegimeGatePanel({ data }: RegimeGatePanelProps) {
  if (!data) {
    return (
      <div className="bg-gray-900/50 border border-gray-700/50 rounded-lg p-4">
        <h3 className="text-sm font-semibold text-gray-100 mb-2">Regime Gate (v5.00)</h3>
        <p className="text-xs text-gray-500">No regime gate data available</p>
      </div>
    );
  }

  const regime = data.current_regime?.toUpperCase() || 'NORMAL';
  const meta = REGIME_META[regime] || DEFAULT_REGIME_META;

  // ── Compute per-signal x per-regime state ────────────────────────
  const signalRegimeMap = new Map<string, Record<string, boolean>>();

  for (const rule of data.gate_rules) {
    const row: Record<string, boolean> = {};
    for (const r of ALL_REGIMES) {
      // If a signal has off_regimes listed, it's OFF in those regimes (else ON)
      row[r] = !rule.off_regimes.includes(r);
    }
    signalRegimeMap.set(rule.signal_name, row);
  }

  // ── Rendering ────────────────────────────────────────────────────

  return (
    <div className="bg-gray-900/50 rounded-lg border border-gray-700/50 p-4 space-y-4">
      {/* ── Header ──────────────────────────────────────────────── */}
      <div className="flex items-center justify-between">
        <h3 className="text-sm font-semibold text-gray-100">Regime Gate (v5.00)</h3>
        {data.generated_at && (
          <span className="text-[10px] font-mono text-gray-600 bg-gray-800/60 px-2 py-0.5 rounded">
            {new Date(data.generated_at).toLocaleString()}
          </span>
        )}
      </div>

      {/* ── Section 1: Current Regime ───────────────────────────── */}
      <div>
        <h4 className="text-xs font-medium text-gray-300 uppercase tracking-wider mb-2">
          Current Regime
        </h4>
        <div className={`${meta.headerBg} ${meta.border} rounded-lg p-3 border`}>
          <div className="flex items-center justify-between mb-2">
            <span className="text-xs text-gray-400">Regime</span>
            <span className={`px-2.5 py-0.5 rounded text-xs font-semibold ${meta.badge}`}>
              {regime}
            </span>
          </div>
          <div className="flex items-center justify-between mb-1">
            <span className="text-xs text-gray-400">Confidence</span>
            <span className="text-sm font-mono font-bold text-gray-100">
              {(data.regime_confidence * 100).toFixed(0)}%
            </span>
          </div>
          <ConfidenceBar value={data.regime_confidence} />
        </div>
      </div>

      {/* ── Section 2: Signal Gate Matrix ───────────────────────── */}
      <div>
        <h4 className="text-xs font-medium text-gray-300 uppercase tracking-wider mb-2">
          Signal Gate Matrix
        </h4>
        <div className="overflow-x-auto rounded-lg border border-gray-700/40">
          <table className="w-full text-xs border-collapse">
            <thead>
              <tr className="bg-gray-800/80 text-gray-500">
                <th className="text-left py-2 px-3 font-medium whitespace-nowrap">Signal</th>
                {ALL_REGIMES.map((r) => {
                  const rMeta = REGIME_META[r] || DEFAULT_REGIME_META;
                  const isCurrent = r === regime;
                  return (
                    <th
                      key={r}
                      className={`text-center py-2 px-2 font-medium whitespace-nowrap transition-colors ${
                        isCurrent ? rMeta.colBg + ' text-gray-100' : ''
                      }`}
                    >
                      <span className={isCurrent ? 'font-bold' : ''}>{r}</span>
                    </th>
                  );
                })}
              </tr>
            </thead>
            <tbody className="divide-y divide-gray-700/30">
              {data.gate_rules.map((rule) => (
                <tr
                  key={rule.signal_name}
                  className="bg-gray-800/40 hover:bg-gray-700/30 transition-colors"
                >
                  <td className="py-2 px-3 text-gray-200 font-medium whitespace-nowrap">
                    {humanizeSignalName(rule.signal_name)}
                  </td>
                  {ALL_REGIMES.map((r) => {
                    const rMeta = REGIME_META[r] || DEFAULT_REGIME_META;
                    const isCurrent = r === regime;
                    const active = signalRegimeMap.get(rule.signal_name)?.[r] ?? false;
                    return (
                      <td
                        key={`${rule.signal_name}-${r}`}
                        className={`text-center py-2 px-2 ${
                          isCurrent ? rMeta.colBg + ' rounded-sm' : ''
                        }`}
                      >
                        <div className="flex justify-center">
                          <RegimeDot active={active} color={rMeta.color} />
                        </div>
                      </td>
                    );
                  })}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        {/* Legend */}
        <div className="flex items-center gap-4 mt-2 text-[10px] text-gray-500">
          <div className="flex items-center gap-1.5">
            <span className="inline-block w-2 h-2 rounded-full bg-emerald-500" />
            <span>Active (ON)</span>
          </div>
          <div className="flex items-center gap-1.5">
            <span className="inline-block w-2 h-2 rounded-full bg-gray-600" />
            <span>Inactive (OFF)</span>
          </div>
          <div className="flex items-center gap-1.5">
            <span className="inline-flex items-center text-[10px] font-bold text-gray-400 bg-gray-800/60 px-1.5 py-0.5 rounded">
              {regime}
            </span>
            <span>= Current regime column</span>
          </div>
        </div>
      </div>

      {/* ── Section 3: Active / Inactive Summary ─────────────────── */}
      <div>
        <h4 className="text-xs font-medium text-gray-300 uppercase tracking-wider mb-2">
          Signal Status Summary
        </h4>
        <div className="space-y-2">
          {/* Active signals */}
          <div className="bg-gray-800/40 rounded-lg p-3 border border-gray-700/30">
            <div className="flex items-center gap-2 mb-2">
              <span className="inline-flex items-center justify-center min-w-[20px] h-5 px-1.5 rounded-full bg-emerald-600/20 text-emerald-400 text-[10px] font-bold">
                {data.active_signals.length}
              </span>
              <span className="text-xs font-medium text-gray-300">Active Signals</span>
            </div>
            {data.active_signals.length === 0 ? (
              <p className="text-[11px] text-gray-500 italic">No active signals</p>
            ) : (
              <div className="flex flex-wrap gap-1.5">
                {data.active_signals.map((name, index) => (
                  <span
                    key={makeSignalListKey(name, index)}
                    className="inline-flex items-center px-2 py-0.5 rounded text-[11px] font-medium bg-emerald-900/25 text-emerald-300 border border-emerald-700/30"
                  >
                    {humanizeSignalName(name)}
                  </span>
                ))}
              </div>
            )}
          </div>

          {/* Inactive signals */}
          <div className="bg-gray-800/40 rounded-lg p-3 border border-gray-700/30">
            <div className="flex items-center gap-2 mb-2">
              <span className="inline-flex items-center justify-center min-w-[20px] h-5 px-1.5 rounded-full bg-red-600/20 text-red-400 text-[10px] font-bold">
                {data.inactive_signals.length}
              </span>
              <span className="text-xs font-medium text-gray-300">Inactive Signals</span>
            </div>
            {data.inactive_signals.length === 0 ? (
              <p className="text-[11px] text-gray-500 italic">No inactive signals</p>
            ) : (
              <div className="flex flex-wrap gap-1.5">
                {data.inactive_signals.map((name, index) => (
                  <span
                    key={makeSignalListKey(name, index)}
                    className="inline-flex items-center px-2 py-0.5 rounded text-[11px] font-medium bg-red-900/25 text-red-300 border border-red-700/30"
                  >
                    {humanizeSignalName(name)}
                  </span>
                ))}
              </div>
            )}
          </div>
        </div>
      </div>

      {/* ── Section 4: Hysteresis Info ────────────────────────────── */}
      <div>
        <h4 className="text-xs font-medium text-gray-300 uppercase tracking-wider mb-2">
          Hysteresis
        </h4>
        <div className="bg-gray-800/40 rounded-md p-3 border border-gray-700/30 flex items-center gap-3">
          <span className="text-sm font-mono font-bold text-gray-100 bg-gray-900/60 px-2.5 py-1 rounded">
            {data.min_dwell_days}d
          </span>
          <div>
            <span className="text-xs font-medium text-gray-400">Minimum Dwell Days</span>
            <p className="text-[10px] text-gray-600 mt-0.5">
              Signals must remain in the current regime for at least {data.min_dwell_days} day
              {data.min_dwell_days !== 1 ? 's' : ''} before gate transitions take effect.
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}
