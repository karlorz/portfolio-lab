
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
  color: string; tone: string; chip: string
}> = {
  CRISIS:   { color: '#ef4444', tone: 'alc-tone-danger', chip: 'alc-chip-danger' },
  HIGH_VOL: { color: '#f97316', tone: 'alc-tone-warning', chip: 'alc-chip-warning' },
  NORMAL:   { color: '#3b82f6', tone: 'alc-tone-info', chip: 'alc-chip-info' },
  LOW_VOL:  { color: '#10b981', tone: 'alc-tone-success', chip: 'alc-chip-success' },
  RECOVERY: { color: '#14b8a6', tone: 'alc-tone-teal', chip: 'alc-chip-teal' },
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

export function uniqueSignalNames(names: string[]): string[] {
  return [...new Set(names)];
}

// ── Sub-components ─────────────────────────────────────────────────────────

function RegimeDot({ active, color }: { active: boolean; color: string }) {
  return (
    <span
      className="alc-dot"
      style={{ backgroundColor: active ? color : '#374151' }}
      title={active ? 'ON' : 'OFF'}
    />
  );
}

function ConfidenceBar({ value }: { value: number }) {
  const pct = Math.min(Math.max(value * 100, 0), 100);
  const color = value >= 0.8 ? '#10b981' : value >= 0.5 ? '#f59e0b' : '#ef4444';
  return (
    <div className="alc-row">
      <div className="alc-progress alc-grow">
        <div
          className="alc-progress-fill"
          style={{ width: `${pct}%`, backgroundColor: color }}
        />
      </div>
      <span className="alc-small alc-mono alc-align-right alc-fixed-md">
        {pct.toFixed(0)}%
      </span>
    </div>
  );
}

// ── Component ──────────────────────────────────────────────────────────────

export function RegimeGatePanel({ data }: RegimeGatePanelProps) {
  if (!data) {
    return (
      <div className="alc-panel alc-panel-muted">
        <h3 className="alc-title">Regime Gate (v5.00)</h3>
        <p className="alc-muted">No regime gate data available</p>
      </div>
    );
  }

  const regime = data.current_regime?.toUpperCase() || 'NORMAL';
  const meta = REGIME_META[regime] || DEFAULT_REGIME_META;
  const activeSignals = uniqueSignalNames(data.active_signals);
  const inactiveSignals = uniqueSignalNames(data.inactive_signals);

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
    <div className="alc-panel alc-panel-muted">
      {/* ── Header ──────────────────────────────────────────────── */}
      <div className="alc-header">
        <h3 className="alc-title">Regime Gate (v5.00)</h3>
        {data.generated_at && (
          <span className="alc-chip-small alc-chip-neutral alc-mono">
            {new Date(data.generated_at).toLocaleString()}
          </span>
        )}
      </div>

      {/* ── Section 1: Current Regime ───────────────────────────── */}
      <div className="alc-section">
        <h4 className="alc-section-title">
          Current Regime
        </h4>
        <div className={`alc-card ${meta.tone}`}>
          <div className="alc-row">
            <span className="alc-label">Regime</span>
            <span className={`alc-chip ${meta.chip}`}>
              {regime}
            </span>
          </div>
          <div className="alc-row">
            <span className="alc-label">Confidence</span>
            <span className="alc-value">
              {(data.regime_confidence * 100).toFixed(0)}%
            </span>
          </div>
          <ConfidenceBar value={data.regime_confidence} />
        </div>
      </div>

      {/* ── Section 2: Signal Gate Matrix ───────────────────────── */}
      <div className="alc-section">
        <h4 className="alc-section-title">
          Signal Gate Matrix
        </h4>
        <div className="alc-table-wrap">
          <table className="alc-table">
            <thead>
              <tr>
                <th>Signal</th>
                {ALL_REGIMES.map((r) => {
                  const rMeta = REGIME_META[r] || DEFAULT_REGIME_META;
                  const isCurrent = r === regime;
                  return (
                    <th
                      key={r}
                      className={isCurrent ? rMeta.tone : undefined}
                    >
                      <span className={isCurrent ? 'alc-strong' : undefined}>{r}</span>
                    </th>
                  );
                })}
              </tr>
            </thead>
            <tbody>
              {data.gate_rules.map((rule) => (
                <tr
                  key={rule.signal_name}
                >
                  <td className="alc-strong">
                    {humanizeSignalName(rule.signal_name)}
                  </td>
                  {ALL_REGIMES.map((r) => {
                    const rMeta = REGIME_META[r] || DEFAULT_REGIME_META;
                    const isCurrent = r === regime;
                    const active = signalRegimeMap.get(rule.signal_name)?.[r] ?? false;
                    return (
                      <td
                        key={`${rule.signal_name}-${r}`}
                        className={isCurrent ? rMeta.tone : undefined}
                      >
                        <div className="alc-cluster">
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
        <div className="alc-cluster">
          <div className="alc-cluster">
            <span className="alc-swatch" style={{ backgroundColor: '#10b981' }} />
            <span>Active (ON)</span>
          </div>
          <div className="alc-cluster">
            <span className="alc-swatch" style={{ backgroundColor: '#475569' }} />
            <span>Inactive (OFF)</span>
          </div>
          <div className="alc-cluster">
            <span className="alc-chip-small alc-chip-neutral">
              {regime}
            </span>
            <span>= Current regime column</span>
          </div>
        </div>
      </div>

      {/* ── Section 3: Active / Inactive Summary ─────────────────── */}
      <div className="alc-section">
        <h4 className="alc-section-title">
          Signal Status Summary
        </h4>
        <div className="alc-stack-sm">
          {/* Active signals */}
          <div className="alc-card">
            <div className="alc-cluster">
              <span className="alc-chip-small alc-chip-success">
                {activeSignals.length}
              </span>
              <span className="alc-label">Active Signals</span>
            </div>
            {activeSignals.length === 0 ? (
              <p className="alc-muted">No active signals</p>
            ) : (
              <div className="alc-cluster">
                {activeSignals.map((name, index) => (
                  <span
                    key={makeSignalListKey(name, index)}
                    className="alc-chip-small alc-chip-success"
                  >
                    {humanizeSignalName(name)}
                  </span>
                ))}
              </div>
            )}
          </div>

          {/* Inactive signals */}
          <div className="alc-card">
            <div className="alc-cluster">
              <span className="alc-chip-small alc-chip-danger">
                {inactiveSignals.length}
              </span>
              <span className="alc-label">Inactive Signals</span>
            </div>
            {inactiveSignals.length === 0 ? (
              <p className="alc-muted">No inactive signals</p>
            ) : (
              <div className="alc-cluster">
                {inactiveSignals.map((name, index) => (
                  <span
                    key={makeSignalListKey(name, index)}
                    className="alc-chip-small alc-chip-danger"
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
      <div className="alc-section">
        <h4 className="alc-section-title">
          Hysteresis
        </h4>
        <div className="alc-card">
          <div className="alc-row-top">
          <span className="alc-chip alc-chip-neutral alc-mono">
            {data.min_dwell_days}d
          </span>
          <div>
            <span className="alc-label">Minimum Dwell Days</span>
            <p className="alc-small">
              Signals must remain in the current regime for at least {data.min_dwell_days} day
              {data.min_dwell_days !== 1 ? 's' : ''} before gate transitions take effect.
            </p>
          </div>
          </div>
        </div>
      </div>
    </div>
  );
}
