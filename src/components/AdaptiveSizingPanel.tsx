import React from 'react';

interface SignalSize {
  name: string;
  current_weight: number;
  target_weight: number;
  health_score: number;
  regime_adjusted: Record<string, number>;
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
}

interface AdaptiveSizingPanelProps {
  data: AdaptiveSizingData | null;
}

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
  if (!data) {
    return (
      <div className="panel">
        <h3>Adaptive Sizing</h3>
        <p className="muted">No adaptive sizing data available</p>
      </div>
    );
  }

  const { signals, constraints, current_regime, total_weight } = data;
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
