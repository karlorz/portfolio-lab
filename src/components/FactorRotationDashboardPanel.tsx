
export interface FactorRotationDashboardData {
  active: boolean;
  selected_factors: string[];
  signal_strength: number;
  factor_allocations: Record<string, number>;
  backtest_finding: string;
}

interface FactorRotationDashboardPanelProps {
  data: FactorRotationDashboardData | null;
}

const FACTOR_COLORS: Record<string, string> = {
  MTUM: '#3b82f6',
  QUAL: '#8b5cf6',
  USMV: '#10b981',
  VLUE: '#f59e0b',
};

const FACTOR_NAMES: Record<string, string> = {
  MTUM: 'Momentum',
  QUAL: 'Quality',
  USMV: 'Low Vol',
  VLUE: 'Value',
};

export function FactorRotationDashboardPanel({ data }: FactorRotationDashboardPanelProps) {
  if (!data) {
    return (
      <div className="panel">
        <h3>Factor Rotation Dashboard</h3>
        <p className="muted">No factor rotation dashboard data available</p>
      </div>
    );
  }

  if (!data.active) {
    return (
      <div className="panel">
        <h3>Factor Rotation Dashboard</h3>
        <p className="muted">Factor rotation inactive</p>
      </div>
    );
  }

  const strengthColor = data.signal_strength > 0.3 ? '#10b981'
    : data.signal_strength > 0 ? '#f59e0b' : '#ef4444';

  const factors = data.selected_factors || [];
  const allocations = data.factor_allocations || {};

  return (
    <div className="panel">
      <h3>Factor Rotation Dashboard</h3>

      {/* Signal overview */}
      <div className="panel-grid">
        <div className="metric">
          <span className="label">Signal Strength</span>
          <span className="value" style={{ color: strengthColor }}>
            {data.signal_strength >= 0 ? '+' : ''}{data.signal_strength.toFixed(2)}
          </span>
        </div>
        <div className="metric">
          <span className="label">Active Factors</span>
          <span className="value">{factors.length}</span>
        </div>
      </div>

      {/* Factor allocation bars */}
      {factors.length > 0 && (
        <div style={{ marginTop: 10 }}>
          <span className="label" style={{ display: 'block', marginBottom: 6 }}>Factor Allocations</span>
          {factors.map((factor) => {
            const pct = allocations[factor.toLowerCase() + '_pct']
              || allocations[factor]
              || 0;
            const color = FACTOR_COLORS[factor] || '#6b7280';
            const name = FACTOR_NAMES[factor] || factor;
            const width = Math.min(pct / 50 * 100, 100);

            return (
              <div key={factor} style={{ marginBottom: 4 }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 2 }}>
                  <span style={{ fontSize: 11, color }}>
                    {name} ({factor})
                  </span>
                  <span style={{ fontSize: 11, color: '#94a3b8' }}>
                    {typeof pct === 'number' ? pct.toFixed(1) : pct}%
                  </span>
                </div>
                <div style={{
                  height: 6, background: '#1e293b', borderRadius: 3, overflow: 'hidden',
                }}>
                  <div style={{
                    width: `${width}%`, height: '100%', background: color, borderRadius: 3,
                  }} />
                </div>
              </div>
            );
          })}
        </div>
      )}

      {/* Backtest finding */}
      {data.backtest_finding && (
        <div style={{ marginTop: 10 }}>
          <span className="label" style={{ display: 'block', marginBottom: 4 }}>Backtest (Phase 3)</span>
          <p className="muted small">{data.backtest_finding}</p>
        </div>
      )}
    </div>
  );
}
