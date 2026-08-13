
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

type UnknownRecord = Record<string, unknown>;
type VixZone = VixyHedgeSizingData['vix_zone'];

const DEFAULT_ZONE_THRESHOLDS: Record<string, [number, number]> = {
  LOW: [0, 15],
  NORMAL: [15, 20],
  ELEVATED: [20, 25],
  HIGH: [25, 35],
  CRISIS: [35, 60],
};

// ── Zone Config ────────────────────────────────────────────────────────────

const ZONE_META: Record<string, { color: string; tone: string; chip: string }> = {
  LOW:      { color: '#10b981', tone: 'alc-tone-success', chip: 'alc-chip-success' },
  NORMAL:   { color: '#3b82f6', tone: 'alc-tone-info', chip: 'alc-chip-info' },
  ELEVATED: { color: '#f59e0b', tone: 'alc-tone-warning', chip: 'alc-chip-warning' },
  HIGH:     { color: '#f97316', tone: 'alc-tone-warning', chip: 'alc-chip-warning' },
  CRISIS:   { color: '#ef4444', tone: 'alc-tone-danger', chip: 'alc-chip-danger' },
};

// ── Helpers ────────────────────────────────────────────────────────────────

function formatPct(v: number): string {
  return `${(v * 100).toFixed(1)}%`;
}

function formatPctFull(v: number): string {
  return `${(v * 100).toFixed(2)}%`;
}

function isRecord(value: unknown): value is UnknownRecord {
  return typeof value === 'object' && value !== null && !Array.isArray(value);
}

function asFiniteNumber(value: unknown, fallback = 0): number {
  return typeof value === 'number' && Number.isFinite(value) ? value : fallback;
}

function pctFieldToFraction(value: unknown, fallback = 0): number {
  const numeric = asFiniteNumber(value, fallback);
  return Math.abs(numeric) > 1 ? numeric / 100 : numeric;
}

function bpsFieldToFraction(value: unknown, fallback = 0): number {
  return asFiniteNumber(value, fallback) / 10000;
}

function normalizeZone(value: unknown): VixZone {
  if (typeof value !== 'string') return 'NORMAL';
  const normalized = value.toUpperCase().replace(/[-\s]/g, '_');
  if (normalized === 'HIGH_VOL') return 'HIGH';
  if (normalized === 'LOW_VOL') return 'LOW';
  return normalized in ZONE_META ? normalized as VixZone : 'NORMAL';
}

function normalizeThresholds(value: unknown): Record<string, [number, number]> {
  if (!isRecord(value)) return { ...DEFAULT_ZONE_THRESHOLDS };
  const thresholds: Record<string, [number, number]> = { ...DEFAULT_ZONE_THRESHOLDS };
  for (const [zone, range] of Object.entries(value)) {
    if (!Array.isArray(range) || range.length < 2) continue;
    const lo = asFiniteNumber(range[0], thresholds[zone]?.[0] ?? 0);
    const hi = asFiniteNumber(range[1], thresholds[zone]?.[1] ?? 60);
    thresholds[zone] = [lo, hi];
  }
  return thresholds;
}

function normalizeCosts(value: UnknownRecord): VixyHedgeSizingData['costs'] {
  if (isRecord(value.costs)) {
    return {
      daily_carry: asFiniteNumber(value.costs.daily_carry),
      roll_cost: asFiniteNumber(value.costs.roll_cost),
      total_pct: asFiniteNumber(value.costs.total_pct),
    };
  }

  return {
    daily_carry: bpsFieldToFraction(value.daily_carry_bps),
    roll_cost: bpsFieldToFraction(value.roll_cost_bps),
    total_pct: bpsFieldToFraction(value.ytd_cost_bps),
  };
}

function normalizePerformance(value: unknown): HedgePerformance[] {
  if (!Array.isArray(value)) return [];
  return value
    .filter(isRecord)
    .map((row) => ({
      period: typeof row.period === 'string' ? row.period : 'Unknown',
      portfolio_return: asFiniteNumber(row.portfolio_return),
      hedge_return: asFiniteNumber(row.hedge_return),
      combined_return: asFiniteNumber(row.combined_return),
    }));
}

export function normalizeVixyHedgeSizingData(value: unknown): VixyHedgeSizingData | null {
  if (!isRecord(value)) return null;

  const hasPanelShape = 'vixy_position' in value || 'hedge_ratio' in value || 'target_volatility' in value;
  const hasStatusShape = 'current_allocation_pct' in value || 'target_allocation_pct' in value || 'regime' in value;
  if (!hasPanelShape && !hasStatusShape) return null;

  const vixyPosition = pctFieldToFraction(value.vixy_position ?? value.current_allocation_pct);
  const recommendationAllocation = pctFieldToFraction(
    isRecord(value.recommendation) ? value.recommendation.allocation : value.target_allocation_pct,
    vixyPosition,
  );
  const hedgeRatio = asFiniteNumber(
    value.hedge_ratio,
    recommendationAllocation > 0 ? vixyPosition / recommendationAllocation : 0,
  );
  const vixLevel = asFiniteNumber(value.vix_level);
  const vixZone = normalizeZone(value.vix_zone ?? value.regime);
  const recommendation = isRecord(value.recommendation)
    ? {
      allocation: asFiniteNumber(value.recommendation.allocation, recommendationAllocation),
      reasoning: typeof value.recommendation.reasoning === 'string'
        ? value.recommendation.reasoning
        : 'Recommendation data is partially available.',
    }
    : {
      allocation: recommendationAllocation,
      reasoning: `Target allocation from hedge status. Efficiency ${asFiniteNumber(value.hedge_efficiency).toFixed(2)} across ${asFiniteNumber(value.total_signals)} signals.`,
    };

  return {
    vixy_position: vixyPosition,
    hedge_ratio: hedgeRatio,
    target_volatility: pctFieldToFraction(value.target_volatility, 0.09),
    vix_level: vixLevel,
    vix_zone: vixZone,
    zone_thresholds: normalizeThresholds(value.zone_thresholds),
    costs: normalizeCosts(value),
    crisis_performance: normalizePerformance(value.crisis_performance),
    recommendation,
  };
}

// ── Component ──────────────────────────────────────────────────────────────

export function VixyHedgeSizingPanel({ data }: VixyHedgeSizingPanelProps) {
  const normalizedData = normalizeVixyHedgeSizingData(data);

  if (!normalizedData) {
    return (
      <div className="alc-panel alc-panel-muted">
        <div className="alc-header">
          <h3 className="alc-title">VIXY Hedge Sizing</h3>
        </div>
        <p className="alc-muted">Hedge sizing data not available</p>
        <p className="alc-small">Run VIXY hedge sizing exporter to populate</p>
      </div>
    );
  }

  data = normalizedData;
  const zoneMeta = ZONE_META[data.vix_zone] ?? ZONE_META.NORMAL;
  const zoneColor = zoneMeta.color;
  const costTone =
    data.costs.total_pct > 0.08 ? 'alc-text-danger' :
    data.costs.total_pct > 0.05 ? 'alc-text-warning' :
    'alc-text-success';
  const recommendationTone =
    data.recommendation.allocation > 0.08 ? 'alc-tone-danger' :
    data.recommendation.allocation > 0.05 ? 'alc-tone-warning' :
    data.recommendation.allocation > 0.02 ? 'alc-tone-info' :
    'alc-tone-success';
  const recommendationText =
    data.recommendation.allocation > 0.08 ? 'alc-text-danger' :
    data.recommendation.allocation > 0.05 ? 'alc-text-warning' :
    data.recommendation.allocation > 0.02 ? 'alc-text-info' :
    'alc-text-success';

  return (
    <div className="alc-panel alc-panel-muted">

      {/* ── Header ─────────────────────────────────────────────── */}
      <div className="alc-header">
        <h3 className="alc-title">VIXY Hedge Sizing</h3>
        <span className={`alc-chip ${zoneMeta.chip}`}>
          {data.vix_zone}
        </span>
      </div>

      {/* ── Section 1: Current Hedge Allocation ───────────────── */}
      <div className="alc-section">
        <h4 className="alc-section-title">Current Hedge Allocation</h4>
        <div className="alc-grid alc-grid-three">
          <div className="alc-card">
            <p className="alc-label">VIXY Position</p>
            <p className="alc-value-lg">
              {formatPct(data.vixy_position)}
            </p>
          </div>
          <div className="alc-card">
            <p className="alc-label">Hedge Ratio</p>
            <p className="alc-value-lg">
              {data.hedge_ratio.toFixed(2)}x
            </p>
          </div>
          <div className="alc-card">
            <p className="alc-label">Target Volatility</p>
            <p className="alc-value-lg">
              {formatPct(data.target_volatility)}
            </p>
          </div>
        </div>
      </div>

      {/* ── Section 2: VIX Regime Zone ─────────────────────────── */}
      <div className="alc-section">
        <h4 className="alc-section-title">VIX Regime Zone</h4>
        <div className={`alc-card ${zoneMeta.tone}`}>
          <div className="alc-row">
            <span className="alc-label">Current VIX Level</span>
            <span className="alc-value-xl" style={{ color: zoneColor }}>
              {data.vix_level.toFixed(2)}
            </span>
          </div>
          <div className="alc-row">
            <span className="alc-small">Zone: <span className="alc-strong" style={{ color: zoneColor }}>{data.vix_zone}</span></span>
          </div>
        </div>

        {/* Zone threshold bar */}
        <div className="alc-stack-xs">
          {(() => {
            const totalRange = 60; // visual max scale
            return (
              <div>
                <div className="alc-progress">
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
                        className="alc-progress-segment"
                        style={{
                          left: `${Math.max(0, left)}%`,
                          width: `${Math.max(2, width)}%`,
                          backgroundColor: meta.color,
                          opacity: isActive ? 1 : 0.3,
                        }}
                        title={`${z}: ${lo}-${hi}`}
                      />
                    );
                  })}
                  {/* VIX marker */}
                  <div
                    className="alc-progress-marker"
                    style={{
                      left: `${Math.min((data.vix_level / totalRange) * 100, 97)}%`,
                    }}
                  />
                </div>
                <div className="alc-scale">
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
        <div className="alc-grid alc-grid-five">
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
                className={`alc-card alc-card-compact ${isActive ? meta.tone : 'alc-tone-neutral'}`}
                style={{ opacity: isActive ? 1 : 0.45 }}
              >
                <p className="alc-small alc-strong" style={{ color: meta.color }}>{z}</p>
                <p className="alc-small">{label}</p>
              </div>
            );
          })}
        </div>
      </div>

      {/* ── Section 3: Hedge Cost Analysis ─────────────────────── */}
      <div className="alc-section">
        <h4 className="alc-section-title">Hedge Cost Analysis</h4>
        <div className="alc-grid alc-grid-three">
          <div className="alc-card">
            <p className="alc-label">Daily Carry</p>
            <p className="alc-value-lg alc-text-warning">
              {formatPctFull(data.costs.daily_carry)}
            </p>
          </div>
          <div className="alc-card">
            <p className="alc-label">Roll Cost</p>
            <p className="alc-value-lg alc-text-warning">
              {formatPctFull(data.costs.roll_cost)}
            </p>
          </div>
          <div className="alc-card">
            <p className="alc-label">Total Cost</p>
            <p className={`alc-value-lg ${costTone}`}>
              {formatPctFull(data.costs.total_pct)}
            </p>
          </div>
        </div>
        {/* Cost gauge bar */}
        <div className="alc-card alc-card-compact">
          <div className="alc-row">
            <span>% of portfolio</span>
            <span>{formatPctFull(data.costs.total_pct)}</span>
          </div>
          <div className="alc-progress">
            <div
              className="alc-progress-fill"
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
          <div className="alc-scale">
            <span>0%</span>
            <span>3%</span>
            <span>5%</span>
            <span>8%</span>
            <span>20%+</span>
          </div>
        </div>
      </div>

      {/* ── Section 4: Historical Crisis Performance ───────────── */}
      <div className="alc-section">
        <h4 className="alc-section-title">Crisis Performance</h4>
        {data.crisis_performance.length === 0 ? (
          <p className="alc-muted">No crisis performance data available</p>
        ) : (
          <div className="alc-table-wrap">
            <table className="alc-table">
              <thead>
                <tr>
                  <th>Period</th>
                  <th className="alc-cell-right">Portfolio</th>
                  <th className="alc-cell-right">Hedge (VIXY)</th>
                  <th className="alc-cell-right">Combined</th>
                </tr>
              </thead>
              <tbody>
                {data.crisis_performance.map((cp, i) => (
                  <tr key={i}>
                    <td className="alc-strong">{cp.period}</td>
                    <td className={`alc-cell-right alc-mono alc-strong ${
                      cp.portfolio_return < 0 ? 'alc-text-danger' : 'alc-text-success'
                    }`}>
                      {formatPctFull(cp.portfolio_return)}
                    </td>
                    <td className={`alc-cell-right alc-mono alc-strong ${
                      cp.hedge_return < 0 ? 'alc-text-danger' : 'alc-text-success'
                    }`}>
                      {formatPctFull(cp.hedge_return)}
                    </td>
                    <td className={`alc-cell-right alc-mono alc-strong ${
                      cp.combined_return < 0 ? 'alc-text-danger' : 'alc-text-success'
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
      <div className="alc-section">
        <h4 className="alc-section-title">Sizing Recommendation</h4>
        <div className={`alc-note ${recommendationTone}`}>
          <div className="alc-row-baseline">
            <span className="alc-label">Recommended VIXY Allocation</span>
            <span className={`alc-value-xl ${recommendationText}`}>
              {formatPct(data.recommendation.allocation)}
            </span>
          </div>
          <p className="alc-muted alc-break-anywhere">
            {data.recommendation.reasoning}
          </p>
        </div>
        {/* Visual sizing gauge */}
        {data.recommendation.allocation > 0 && (
          <div className="alc-card">
            <div className="alc-row">
              <span>Allocation gauge</span>
              <span>Current: {formatPct(data.vixy_position)}</span>
            </div>
            <div className="alc-progress-tall">
              {/* Recommended allocation fill */}
              <div
                className="alc-progress-fill"
                style={{
                  width: `${Math.min(data.recommendation.allocation * 100, 100)}%`,
                  opacity: 0.6,
                  backgroundColor:
                    data.recommendation.allocation > 0.08 ? '#ef4444' :
                    data.recommendation.allocation > 0.05 ? '#f59e0b' :
                    data.recommendation.allocation > 0.02 ? '#3b82f6' :
                    '#10b981',
                }}
              />
              {/* Current position marker */}
              <div
                className="alc-progress-marker"
                style={{
                  left: `${Math.min(data.vixy_position * 100, 100)}%`,
                }}
              />
            </div>
            <div className="alc-scale">
              <span>0%</span>
              <span>5%</span>
              <span>10%</span>
              <span>15%+</span>
            </div>
            <div className="alc-cluster">
              <span className="alc-swatch" style={{ backgroundColor: 'rgba(248, 250, 252, 0.8)' }} />
              <span>Current position</span>
              <span className="alc-swatch" style={{ backgroundColor: 'rgba(100, 116, 139, 0.6)' }} />
              <span>Recommended</span>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
