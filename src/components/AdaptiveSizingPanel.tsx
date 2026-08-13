
interface SignalSize {
  name: string;
  current_weight: number;
  target_weight: number;
  health_score: number;
  regime_adjusted: Record<string, number>;
}

interface AllocationArtifactAuthority {
  runtime_role: 'advisory_non_routed';
  live_authoritative: false;
  routed: false;
  routed_by: null;
  canonical_controller: string;
  routed_surface: 'target_allocations';
  description: string;
}

export interface AdaptiveSizingData {
  signals: SignalSize[];
  constraints: {
    max_per_signal: number;
    min_weight: number;
    max_leverage: number;
    viability_floor: number;
  };
  current_regime: string;
  total_weight: number;
  authority?: AllocationArtifactAuthority;
}

interface AdaptiveSizingPanelProps {
  data: AdaptiveSizingData | null;
}

type UnknownRecord = Record<string, unknown>;

const DEFAULT_CONSTRAINTS: AdaptiveSizingData['constraints'] = {
  max_per_signal: 0.5,
  min_weight: 0,
  max_leverage: 1,
  viability_floor: 0.5,
};

const REGIME_COLORS: Record<string, string> = {
  LOW_VOL: '#10b981',
  NORMAL: '#3b82f6',
  HIGH_VOL: '#f59e0b',
  CRISIS: '#ef4444',
  RECOVERY: '#8b5cf6',
};

const REGIME_LABELS: Record<string, string> = {
  LOW_VOL: 'Low Vol',
  NORMAL: 'Normal',
  HIGH_VOL: 'High Vol',
  CRISIS: 'Crisis',
  RECOVERY: 'Recovery',
};

function isRecord(value: unknown): value is UnknownRecord {
  return typeof value === 'object' && value !== null && !Array.isArray(value);
}

function asFiniteNumber(value: unknown, fallback = 0): number {
  return typeof value === 'number' && Number.isFinite(value) ? value : fallback;
}

function clamp01(value: number): number {
  return Math.max(0, Math.min(1, value));
}

function normalizeRegimeName(value: unknown): string {
  if (typeof value !== 'string' || value.length === 0) return 'UNKNOWN';
  return value.replace(/-/g, '_').toUpperCase();
}

function normalizeConstraints(value: unknown): AdaptiveSizingData['constraints'] {
  if (!isRecord(value)) return { ...DEFAULT_CONSTRAINTS };
  return {
    max_per_signal: asFiniteNumber(value.max_per_signal, DEFAULT_CONSTRAINTS.max_per_signal),
    min_weight: asFiniteNumber(value.min_weight, DEFAULT_CONSTRAINTS.min_weight),
    max_leverage: asFiniteNumber(value.max_leverage, DEFAULT_CONSTRAINTS.max_leverage),
    viability_floor: asFiniteNumber(value.viability_floor, DEFAULT_CONSTRAINTS.viability_floor),
  };
}

function normalizeRegimeAdjustments(value: unknown): Record<string, number> {
  if (!isRecord(value)) return {};
  return Object.fromEntries(
    Object.entries(value)
      .filter(([, adjustment]) => typeof adjustment === 'number' && Number.isFinite(adjustment)),
  ) as Record<string, number>;
}

function normalizeSignalSize(value: unknown, fallbackHealth: number): SignalSize | null {
  if (!isRecord(value)) return null;
  const name = typeof value.name === 'string' && value.name.length > 0 ? value.name : null;
  if (!name) return null;

  return {
    name,
    current_weight: asFiniteNumber(value.current_weight),
    target_weight: asFiniteNumber(value.target_weight),
    health_score: clamp01(asFiniteNumber(value.health_score, fallbackHealth)),
    regime_adjusted: normalizeRegimeAdjustments(value.regime_adjusted),
  };
}

function normalizeAuthority(value: unknown): AllocationArtifactAuthority | undefined {
  if (!isRecord(value)) return undefined;
  if (value.runtime_role !== 'advisory_non_routed') return undefined;
  if (value.live_authoritative !== false || value.routed !== false || value.routed_by !== null) return undefined;
  if (value.routed_surface !== 'target_allocations') return undefined;
  if (typeof value.canonical_controller !== 'string') return undefined;
  if (typeof value.description !== 'string') return undefined;

  return {
    runtime_role: 'advisory_non_routed',
    live_authoritative: false,
    routed: false,
    routed_by: null,
    canonical_controller: value.canonical_controller,
    routed_surface: 'target_allocations',
    description: value.description,
  };
}

function normalizeAllocationRows(value: UnknownRecord): SignalSize[] {
  if (!isRecord(value.adjusted_allocation)) return [];

  const baseAllocation = isRecord(value.base_allocation) ? value.base_allocation : {};
  const adjustedAllocation = value.adjusted_allocation;
  const factors = isRecord(value.factors) ? value.factors : {};
  const healthScore = clamp01(asFiniteNumber(factors.regime_confidence, 1));
  const assetNames = Object.keys(adjustedAllocation)
    .filter((asset) => typeof adjustedAllocation[asset] === 'number' && Number.isFinite(adjustedAllocation[asset]));

  return assetNames.map((asset) => ({
    name: asset,
    current_weight: asFiniteNumber(baseAllocation[asset], asFiniteNumber(adjustedAllocation[asset])),
    target_weight: asFiniteNumber(adjustedAllocation[asset]),
    health_score: healthScore,
    regime_adjusted: {},
  }));
}

export function normalizeAdaptiveSizingData(value: unknown): AdaptiveSizingData | null {
  if (!isRecord(value)) return null;

  const factors = isRecord(value.factors) ? value.factors : {};
  const constraints = normalizeConstraints(value.constraints);
  const signalRows = Array.isArray(value.signals)
    ? value.signals
      .map((signal) => normalizeSignalSize(signal, asFiniteNumber(factors.regime_confidence, 1)))
      .filter((signal): signal is SignalSize => signal !== null)
    : normalizeAllocationRows(value);

  if (signalRows.length === 0) return null;

  const totalWeight = asFiniteNumber(
    value.total_weight,
    signalRows.reduce((sum, signal) => sum + signal.target_weight, 0),
  );

  return {
    signals: signalRows,
    constraints,
    current_regime: normalizeRegimeName(value.current_regime ?? factors.regime),
    total_weight: totalWeight,
    authority: normalizeAuthority(value.authority),
  };
}

function AuthorityDisclosure({ authority }: { authority?: AllocationArtifactAuthority }) {
  if (!authority) return null;

  return (
    <div style={{
      marginTop: 8,
      padding: '6px 8px',
      borderRadius: 4,
      background: 'rgba(245, 158, 11, 0.12)',
      border: '1px solid rgba(245, 158, 11, 0.35)',
      color: '#fbbf24',
      fontSize: 11,
      fontWeight: 600,
    }}>
      <span>Not order-routed</span>
      <span style={{ color: '#94a3b8', fontWeight: 400 }}>
        {' '}· live orders use {authority.routed_surface}
      </span>
    </div>
  );
}

function HealthBar({ score, floor }: { score: number; floor: number }) {
  const pct = Math.min(score * 100, 100);
  const belowFloor = score < floor;
  const color = belowFloor ? '#ef4444' : score >= 0.8 ? '#10b981' : '#f59e0b';

  return (
    <div style={{ display: 'flex', alignItems: 'center', gap: 6, width: '100%' }}>
      <div style={{
        flex: 1, height: 8, background: '#1e293b', borderRadius: 4, position: 'relative', overflow: 'visible',
      }}>
        {/* Viability floor line */}
        <div style={{
          position: 'absolute', left: `${floor * 100}%`, top: -2, width: 2, height: 12,
          background: '#ef4444', opacity: 0.7, zIndex: 1,
        }} title={`Viability floor: ${(floor * 100).toFixed(0)}%`} />
        {/* Score bar */}
        <div style={{
          width: `${pct}%`, height: '100%', background: color, borderRadius: 4,
          transition: 'width 0.3s ease',
        }} />
      </div>
      <span style={{
        fontSize: 11, color, minWidth: 36, textAlign: 'right', fontFamily: 'monospace',
      }}>
        {score.toFixed(2)}
      </span>
    </div>
  );
}

function RegimeAdjustmentBars({ adjustments }: { adjustments: Record<string, number> }) {
  const regimes = Object.keys(adjustments).filter(r => r in REGIME_COLORS);

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
      {regimes.length === 0 ? (
        <span style={{ fontSize: 10, color: '#64748b' }}>None</span>
      ) : (
        regimes.map(regime => {
          const value = adjustments[regime];
          const maxAbs = Math.max(...regimes.map(r => Math.abs(adjustments[r])), 0.1);
          const pct = Math.min(Math.abs(value) / maxAbs * 100, 100);
          const isOverweight = value >= 0;

          return (
            <div key={regime} style={{ display: 'flex', alignItems: 'center', gap: 4 }}>
              <span style={{
                fontSize: 9, color: REGIME_COLORS[regime] || '#94a3b8', minWidth: 48,
              }}>
                {REGIME_LABELS[regime] || regime}
              </span>
              <div style={{
                flex: 1, height: 4, background: '#1e293b', borderRadius: 2,
                position: 'relative', overflow: 'hidden',
              }}>
                <div style={{
                  position: 'absolute',
                  left: isOverweight ? '50%' : `${50 - pct}%`,
                  width: `${pct / 2}%`,
                  height: '100%',
                  background: isOverweight ? '#10b981' : '#ef4444',
                  borderRadius: 2,
                }} />
              </div>
              <span style={{
                fontSize: 9, color: '#94a3b8', minWidth: 30, textAlign: 'right', fontFamily: 'monospace',
              }}>
                {value >= 0 ? '+' : ''}{(value * 100).toFixed(0)}%
              </span>
            </div>
          );
        })
      )}
    </div>
  );
}

export function AdaptiveSizingPanel({ data }: AdaptiveSizingPanelProps) {
  const normalizedData = normalizeAdaptiveSizingData(data);

  if (!normalizedData) {
    return (
      <div className="panel">
        <h3>Adaptive Sizing</h3>
        <p className="muted">No adaptive sizing data available</p>
      </div>
    );
  }

  const { signals, constraints, current_regime, total_weight, authority } = normalizedData;
  const floor = constraints.viability_floor;
  const withinLeverage = total_weight <= constraints.max_leverage;
  const totalWeightColor = total_weight > constraints.max_leverage
    ? '#ef4444' : total_weight > constraints.max_leverage * 0.9
    ? '#f59e0b' : '#10b981';

  const totalDeviation = signals.reduce((sum, s) =>
    sum + Math.abs(s.current_weight - s.target_weight), 0
  );

  return (
    <div className="panel">
      <h3>Adaptive Sizing</h3>
      <AuthorityDisclosure authority={authority} />

      {/* Summary metrics */}
      <div className="panel-grid">
        <div className="metric">
          <span className="label">Current Regime</span>
          <span className="value" style={{ color: REGIME_COLORS[current_regime] || '#6b7280', fontSize: '0.9rem' }}>
            {REGIME_LABELS[current_regime] || current_regime}
          </span>
        </div>
        <div className="metric">
          <span className="label">Total Weight</span>
          <span className="value" style={{ color: totalWeightColor }}>
            {(total_weight * 100).toFixed(1)}%
          </span>
        </div>
        <div className="metric">
          <span className="label">Signals</span>
          <span className="value">{signals.length}</span>
        </div>
        <div className="metric">
          <span className="label">Total Deviation</span>
          <span className="value" style={{
            color: totalDeviation > 0.15 ? '#ef4444' : totalDeviation > 0.08 ? '#f59e0b' : '#10b981',
          }}>
            {(totalDeviation * 100).toFixed(1)}%
          </span>
        </div>
      </div>

      {/* Sizing Constraints */}
      <div style={{ marginTop: 10 }}>
        <span className="label" style={{ display: 'block', marginBottom: 6 }}>Sizing Constraints</span>
        <div style={{
          display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 6,
        }}>
          <div style={{
            background: '#0f172a', borderRadius: 4, padding: '6px 8px', textAlign: 'center',
          }}>
            <span style={{ fontSize: 9, color: '#64748b', display: 'block', textTransform: 'uppercase', letterSpacing: '0.5px' }}>
              Max/Signal
            </span>
            <span style={{ fontSize: 13, fontWeight: 700, color: '#e2e8f0', fontFamily: 'monospace' }}>
              {(constraints.max_per_signal * 100).toFixed(0)}%
            </span>
          </div>
          <div style={{
            background: '#0f172a', borderRadius: 4, padding: '6px 8px', textAlign: 'center',
          }}>
            <span style={{ fontSize: 9, color: '#64748b', display: 'block', textTransform: 'uppercase', letterSpacing: '0.5px' }}>
              Min Weight
            </span>
            <span style={{ fontSize: 13, fontWeight: 700, color: '#e2e8f0', fontFamily: 'monospace' }}>
              {(constraints.min_weight * 100).toFixed(1)}%
            </span>
          </div>
          <div style={{
            background: '#0f172a', borderRadius: 4, padding: '6px 8px', textAlign: 'center',
          }}>
            <span style={{ fontSize: 9, color: '#64748b', display: 'block', textTransform: 'uppercase', letterSpacing: '0.5px' }}>
              Max Lev.
            </span>
            <span style={{ fontSize: 13, fontWeight: 700, color: withinLeverage ? '#e2e8f0' : '#ef4444', fontFamily: 'monospace' }}>
              {constraints.max_leverage.toFixed(1)}x
            </span>
          </div>
        </div>
      </div>

      {/* Per-signal sizing table */}
      {signals.length > 0 && (
        <div style={{ marginTop: 10 }}>
          <span className="label" style={{ display: 'block', marginBottom: 6 }}>
            Signal Sizing
          </span>
          <table style={{ width: '100%', fontSize: 11, borderCollapse: 'collapse' }}>
            <thead>
              <tr style={{ color: '#94a3b8', borderBottom: '1px solid #1e293b' }}>
                <th style={{ textAlign: 'left', padding: '2px 4px' }}>Signal</th>
                <th style={{ textAlign: 'right', padding: '2px 4px' }}>Current</th>
                <th style={{ textAlign: 'right', padding: '2px 4px' }}>Target</th>
                <th style={{ textAlign: 'right', padding: '2px 4px' }}>Dev</th>
                <th style={{ textAlign: 'right', padding: '2px 4px' }}>Health</th>
              </tr>
            </thead>
            <tbody>
              {signals.map((sig) => {
                const deviation = sig.current_weight - sig.target_weight;
                const devPct = Math.abs(deviation);
                const devColor = devPct > 0.05 ? '#ef4444' : devPct > 0.02 ? '#f59e0b' : '#10b981';
                const healthColor = sig.health_score < floor ? '#ef4444' : sig.health_score >= 0.8 ? '#10b981' : '#f59e0b';

                return (
                  <tr key={sig.name} style={{ borderBottom: '1px solid #0f172a' }}>
                    <td style={{ padding: '2px 4px', color: '#e2e8f0', fontWeight: 600 }}>
                      {sig.name}
                    </td>
                    <td style={{
                      padding: '2px 4px', textAlign: 'right', color: '#e2e8f0', fontFamily: 'monospace',
                    }}>
                      {(sig.current_weight * 100).toFixed(1)}%
                    </td>
                    <td style={{
                      padding: '2px 4px', textAlign: 'right', color: '#94a3b8', fontFamily: 'monospace',
                    }}>
                      {(sig.target_weight * 100).toFixed(1)}%
                    </td>
                    <td style={{
                      padding: '2px 4px', textAlign: 'right', color: devColor, fontFamily: 'monospace',
                    }}>
                      {deviation >= 0 ? '+' : ''}{(deviation * 100).toFixed(1)}%
                    </td>
                    <td style={{
                      padding: '2px 4px', textAlign: 'right', color: healthColor, fontFamily: 'monospace',
                    }}>
                      {sig.health_score.toFixed(2)}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      )}

      {/* Health Score Visualization */}
      {signals.length > 0 && (
        <div style={{ marginTop: 10 }}>
          <span className="label" style={{ display: 'block', marginBottom: 6 }}>
            Health Scores <span style={{ color: '#64748b', fontWeight: 400, textTransform: 'none' }}>
              (floor: {(floor * 100).toFixed(0)}%)
            </span>
          </span>
          {signals.map((sig) => {
            const belowFloor = sig.health_score < floor;
            return (
              <div key={sig.name} style={{ marginBottom: 5 }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 2 }}>
                  <span style={{
                    fontSize: 10, color: belowFloor ? '#ef4444' : '#e2e8f0',
                  }}>
                    {sig.name}
                    {belowFloor && (
                      <span style={{ color: '#ef4444', marginLeft: 4, fontSize: 9, fontWeight: 700 }}>
                        BELOW FLOOR
                      </span>
                    )}
                  </span>
                </div>
                <HealthBar score={sig.health_score} floor={floor} />
              </div>
            );
          })}
        </div>
      )}

      {/* Regime-Based Sizing Adjustments */}
      {signals.some(s => Object.keys(s.regime_adjusted).length > 0) && (
        <div style={{ marginTop: 10 }}>
          <span className="label" style={{ display: 'block', marginBottom: 6 }}>
            Regime Adjustments
          </span>
          <table style={{ width: '100%', fontSize: 11, borderCollapse: 'collapse' }}>
            <thead>
              <tr style={{ color: '#94a3b8', borderBottom: '1px solid #1e293b' }}>
                <th style={{ textAlign: 'left', padding: '2px 4px' }}>Signal</th>
                <th style={{ textAlign: 'left', padding: '2px 4px' }}>Adjustments</th>
              </tr>
            </thead>
            <tbody>
              {signals.filter(s => Object.keys(s.regime_adjusted).length > 0).map((sig) => (
                <tr key={sig.name} style={{ borderBottom: '1px solid #0f172a' }}>
                  <td style={{ padding: '2px 4px', color: '#e2e8f0', fontWeight: 600, verticalAlign: 'top' }}>
                    {sig.name}
                  </td>
                  <td style={{ padding: '2px 4px' }}>
                    <RegimeAdjustmentBars adjustments={sig.regime_adjusted} />
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      {/* Total weight vs leverage warning */}
      {total_weight > constraints.max_leverage * 0.95 && (
        <div style={{
          marginTop: 8, padding: '6px 8px', borderRadius: 4,
          background: total_weight > constraints.max_leverage
            ? 'rgba(239, 68, 68, 0.15)' : 'rgba(245, 158, 11, 0.15)',
          color: total_weight > constraints.max_leverage ? '#ef4444' : '#f59e0b',
          fontSize: 11, fontWeight: 600,
        }}>
          {total_weight > constraints.max_leverage
            ? `Leverage limit exceeded: ${(total_weight * 100).toFixed(1)}% total vs ${(constraints.max_leverage * 100).toFixed(0)}% max`
            : `Approaching leverage limit: ${(total_weight * 100).toFixed(1)}% / ${(constraints.max_leverage * 100).toFixed(0)}%`
          }
        </div>
      )}
    </div>
  );
}
