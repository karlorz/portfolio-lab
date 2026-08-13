
/**
 * Volatility parity panel units.
 *
 * Public producer (`signals.json.volatility_parity` and Python
 * `VolParityAllocation`) stores allocation and risk fields as
 * **percentage points** (spy_pct: 40 means 40%, target_volatility: 10
 * means 10% vol). This panel normalizes producer/flat and nested legacy
 * shapes into a view-model that is already in percentage points — do not
 * multiply by 100 again at render time.
 */

type UnknownRecord = Record<string, unknown>;

interface VolParityAllocation {
  date: string;
  /** Percentage points (10 = 10% target vol). */
  target_volatility: number;
  /** Percentage points (40 = 40% weight). */
  spy_pct: number;
  gld_pct: number;
  tlt_pct: number;
  core_vol_contribution: number;
  vix_short_pct: number;
  vix_tail_pct: number;
  vix_vol_contribution: number;
  cash_pct: number;
  expected_portfolio_vol: number;
  expected_max_dd: number;
  rebalance_triggered: boolean;
  rebalance_reason: string | null;
}

interface VolParitySummary {
  total_capital_allocation: number;
  total_vol_contribution: number;
  /** Percentage points. */
  target_vol: number;
  /** Percentage points (target_vol - expected_portfolio_vol). */
  vol_gap: number;
  vix_regime: string;
}

export interface VolatilityParityData {
  allocation: VolParityAllocation;
  summary: VolParitySummary;
}

interface VolatilityParityPanelProps {
  /** Flat producer payload, nested legacy shape, or null. */
  data: VolatilityParityData | VolParityAllocation | UnknownRecord | null;
}

function isRecord(value: unknown): value is UnknownRecord {
  return typeof value === 'object' && value !== null && !Array.isArray(value);
}

function asFiniteNumber(value: unknown, fallback = 0): number {
  const n = typeof value === 'number' ? value : Number(value);
  return Number.isFinite(n) ? n : fallback;
}

function asBool(value: unknown, fallback = false): boolean {
  return typeof value === 'boolean' ? value : fallback;
}

function asString(value: unknown, fallback = ''): string {
  return typeof value === 'string' ? value : fallback;
}

/**
 * Convert a value that may be either a decimal fraction (0.40) or percentage
 * points (40) into percentage points. Producer contract is percentage points;
 * true fractional fixtures (|x| <= 1.5 for allocation-like fields) are scaled.
 *
 * For vol fields, the producer target is ~10 percentage points, so values
 * |x| <= 1 are treated as fractions (0.10 → 10). Values > 1 are left as-is.
 */
export function toPercentagePoints(
  value: unknown,
  { treatAsFractionBelow = 1.5 }: { treatAsFractionBelow?: number } = {},
): number {
  const n = asFiniteNumber(value, 0);
  if (Math.abs(n) > 0 && Math.abs(n) < treatAsFractionBelow) {
    return n * 100;
  }
  return n;
}

function readAllocation(raw: UnknownRecord): VolParityAllocation | null {
  // Nested legacy: { allocation: {...}, summary: {...} }
  const source = isRecord(raw.allocation) ? raw.allocation : raw;

  // Need at least one core allocation field to be useful.
  if (
    source.spy_pct === undefined
    && source.gld_pct === undefined
    && source.tlt_pct === undefined
    && source.target_volatility === undefined
  ) {
    return null;
  }

  // Allocation weights: producer uses percentage points (0–100).
  // True fractions (0–1) are rare; scale only when |x| < 1.5.
  const spy_pct = toPercentagePoints(source.spy_pct);
  const gld_pct = toPercentagePoints(source.gld_pct);
  const tlt_pct = toPercentagePoints(source.tlt_pct);
  const cash_pct = toPercentagePoints(source.cash_pct);
  const vix_short_pct = toPercentagePoints(source.vix_short_pct);
  const vix_tail_pct = toPercentagePoints(source.vix_tail_pct);

  // Risk/vol fields: producer uses percentage points (~10). Scale only
  // true fractions (|x| < 1) so 0.10 → 10 without touching 10.0.
  const target_volatility = toPercentagePoints(source.target_volatility, {
    treatAsFractionBelow: 1,
  });
  const expected_portfolio_vol = toPercentagePoints(source.expected_portfolio_vol, {
    treatAsFractionBelow: 1,
  });
  const expected_max_dd = toPercentagePoints(source.expected_max_dd, {
    treatAsFractionBelow: 1,
  });
  const core_vol_contribution = toPercentagePoints(source.core_vol_contribution, {
    treatAsFractionBelow: 1,
  });
  // vix_vol_contribution can be large negative percentage points (-30).
  const vix_vol_contribution = toPercentagePoints(source.vix_vol_contribution, {
    treatAsFractionBelow: 1,
  });

  return {
    date: asString(source.date),
    target_volatility,
    spy_pct,
    gld_pct,
    tlt_pct,
    core_vol_contribution,
    vix_short_pct,
    vix_tail_pct,
    vix_vol_contribution,
    cash_pct,
    expected_portfolio_vol,
    expected_max_dd,
    rebalance_triggered: asBool(source.rebalance_triggered),
    rebalance_reason: typeof source.rebalance_reason === 'string'
      ? source.rebalance_reason
      : null,
  };
}

function readSummary(
  raw: UnknownRecord,
  allocation: VolParityAllocation,
): VolParitySummary {
  const summaryRaw = isRecord(raw.summary) ? raw.summary : {};

  const target_vol = summaryRaw.target_vol !== undefined
    ? toPercentagePoints(summaryRaw.target_vol, { treatAsFractionBelow: 1 })
    : allocation.target_volatility;

  let vol_gap: number;
  if (summaryRaw.vol_gap !== undefined) {
    // Producer vol_gap is already percentage points (target - expected ≈ -1.66).
    // Only scale true fractional gaps (|x| < 0.5) from old decimal fixtures.
    vol_gap = toPercentagePoints(summaryRaw.vol_gap, { treatAsFractionBelow: 0.5 });
  } else {
    vol_gap = target_vol - allocation.expected_portfolio_vol;
  }

  const total_capital_allocation = summaryRaw.total_capital_allocation !== undefined
    ? toPercentagePoints(summaryRaw.total_capital_allocation)
    : allocation.spy_pct
      + allocation.gld_pct
      + allocation.tlt_pct
      + allocation.vix_short_pct
      + allocation.vix_tail_pct
      + allocation.cash_pct;

  const total_vol_contribution = summaryRaw.total_vol_contribution !== undefined
    ? toPercentagePoints(summaryRaw.total_vol_contribution, { treatAsFractionBelow: 1 })
    : allocation.core_vol_contribution + allocation.vix_vol_contribution;

  const vix_regime = asString(
    summaryRaw.vix_regime,
    allocation.vix_short_pct > 0 ? 'contango' : 'backwardation',
  );

  return {
    total_capital_allocation,
    total_vol_contribution,
    target_vol,
    vol_gap,
    vix_regime,
  };
}

/**
 * Map producer-shaped `signals.json.volatility_parity` (flat percentage points)
 * and nested legacy `{allocation, summary}` into a percentage-point view-model.
 */
export function normalizeVolatilityParityData(
  raw: unknown,
): VolatilityParityData | null {
  if (!isRecord(raw)) return null;
  const allocation = readAllocation(raw);
  if (!allocation) return null;
  return {
    allocation,
    summary: readSummary(raw, allocation),
  };
}

function AllocationBar({ label, pct, color, maxPct = 60 }: {
  label: string; pct: number; color: string; maxPct?: number;
}) {
  const width = Math.min(Math.abs(pct) / maxPct * 100, 100);
  return (
    <div style={{ marginBottom: 4 }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 2 }}>
        <span style={{ fontSize: 11, color: '#e2e8f0' }}>{label}</span>
        <span style={{ fontSize: 11, color: '#94a3b8' }}>{pct.toFixed(1)}%</span>
      </div>
      <div style={{ height: 6, background: '#1e293b', borderRadius: 3, overflow: 'hidden' }}>
        <div style={{ width: `${width}%`, height: '100%', background: color, borderRadius: 3 }} />
      </div>
    </div>
  );
}

export function VolatilityParityPanel({ data }: VolatilityParityPanelProps) {
  const view = normalizeVolatilityParityData(data);

  if (!view) {
    return (
      <div className="panel">
        <h3>Volatility Parity</h3>
        <p className="muted">No volatility parity data available</p>
      </div>
    );
  }

  const a = view.allocation;
  const s = view.summary;
  // vol_gap is percentage points; ~2pp tolerance for "on target".
  const volGapColor = Math.abs(s.vol_gap) < 2 ? '#10b981' : '#f59e0b';

  return (
    <div className="panel">
      <h3>Volatility Parity</h3>

      {/* Summary metrics — values already percentage points */}
      <div className="panel-grid">
        <div className="metric">
          <span className="label">Target Vol</span>
          <span className="value">{a.target_volatility.toFixed(0)}%</span>
        </div>
        <div className="metric">
          <span className="label">Expected Vol</span>
          <span className="value" style={{ color: volGapColor }}>
            {a.expected_portfolio_vol.toFixed(1)}%
          </span>
        </div>
        <div className="metric">
          <span className="label">Vol Gap</span>
          <span className="value" style={{ color: volGapColor }}>
            {s.vol_gap >= 0 ? '+' : ''}{s.vol_gap.toFixed(1)}%
          </span>
        </div>
        <div className="metric">
          <span className="label">VIX Regime</span>
          <span className="value" style={{
            color: s.vix_regime === 'contango' ? '#10b981' : '#ef4444',
          }}>
            {s.vix_regime.charAt(0).toUpperCase() + s.vix_regime.slice(1)}
          </span>
        </div>
      </div>

      {/* Core allocation bars */}
      <div style={{ marginTop: 10 }}>
        <span className="label" style={{ display: 'block', marginBottom: 6 }}>Core Allocation</span>
        <AllocationBar label="SPY" pct={a.spy_pct} color="#3b82f6" />
        <AllocationBar label="GLD" pct={a.gld_pct} color="#f59e0b" />
        <AllocationBar label="TLT" pct={a.tlt_pct} color="#8b5cf6" />
      </div>

      {/* Convexity allocation */}
      <div style={{ marginTop: 8 }}>
        <span className="label" style={{ display: 'block', marginBottom: 6 }}>Convexity</span>
        <AllocationBar label="Short VIX" pct={a.vix_short_pct} color="#10b981" maxPct={10} />
        <AllocationBar label="Tail Protection" pct={a.vix_tail_pct} color="#ef4444" maxPct={10} />
      </div>

      {/* Risk metrics */}
      <div className="panel-grid" style={{ marginTop: 8 }}>
        <div className="metric">
          <span className="label">Expected Max DD</span>
          <span className="value" style={{ color: '#f59e0b' }}>
            -{a.expected_max_dd.toFixed(0)}%
          </span>
        </div>
        <div className="metric">
          <span className="label">Cash</span>
          <span className="value">{a.cash_pct.toFixed(1)}%</span>
        </div>
      </div>

      {/* Rebalance alert */}
      {a.rebalance_triggered && (
        <div style={{
          marginTop: 8, padding: '6px 8px', borderRadius: 4,
          background: '#1e3a5f', color: '#93c5fd', fontSize: 12,
        }}>
          Rebalance: {a.rebalance_reason || 'Triggered'}
        </div>
      )}
    </div>
  );
}
