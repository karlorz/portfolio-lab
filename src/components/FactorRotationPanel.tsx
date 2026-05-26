import React from 'react';

interface FactorAllocation {
  mtum_pct: number;
  qual_pct: number;
  usmv_pct: number;
  vlue_pct: number;
}

export interface FactorRotationData {
  active: boolean;
  regime: string;                  // bull, bear, neutral, high_vol, crisis
  quality_momentum_score: number;  // -1 to +1
  confidence: number;              // 0-1
  factor_allocations: FactorAllocation;
  equity_adjustment: number;       // recommended shift from base 46%
  mtum_momentum_6m?: number;
  qual_roe?: number;
  usmv_beta?: number;
  vlue_pe_ratio?: number;
  backtest_finding?: string;       // Phase 3 backtest summary
}

interface FactorRotationPanelProps {
  data: FactorRotationData | null;
}

const REGIME_COLORS: Record<string, string> = {
  bull: '#22c55e',
  bear: '#ef4444',
  neutral: '#6b7280',
  high_vol: '#f59e0b',
  crisis: '#dc2626',
};

const FACTOR_COLORS: Record<string, string> = {
  MTUM: '#3b82f6',
  QUAL: '#8b5cf6',
  USMV: '#10b981',
  VLUE: '#f59e0b',
};

function FactorPie({ allocation }: { allocation: FactorAllocation }) {
  const segments = [
    { label: 'MTUM', pct: allocation.mtum_pct },
    { label: 'QUAL', pct: allocation.qual_pct },
    { label: 'USMV', pct: allocation.usmv_pct },
    { label: 'VLUE', pct: allocation.vlue_pct },
  ];

  let cumulative = 0;
  const paths = segments.map((seg) => {
    const startAngle = (cumulative / 100) * 360;
    cumulative += seg.pct;
    const endAngle = (cumulative / 100) * 360;

    const r = 40;
    const cx = 50;
    const cy = 50;
    const startRad = ((startAngle - 90) * Math.PI) / 180;
    const endRad = ((endAngle - 90) * Math.PI) / 180;

    const x1 = cx + r * Math.cos(startRad);
    const y1 = cy + r * Math.sin(startRad);
    const x2 = cx + r * Math.cos(endRad);
    const y2 = cy + r * Math.sin(endRad);

    const largeArc = endAngle - startAngle > 180 ? 1 : 0;

    const d = [
      `M ${cx} ${cy}`,
      `L ${x1.toFixed(2)} ${y1.toFixed(2)}`,
      `A ${r} ${r} 0 ${largeArc} 1 ${x2.toFixed(2)} ${y2.toFixed(2)}`,
      'Z',
    ].join(' ');

    return (
      <path
        key={seg.label}
        d={d}
        fill={FACTOR_COLORS[seg.label]}
        stroke="#fff"
        strokeWidth="1"
      >
        <title>{seg.label}: {seg.pct.toFixed(0)}%</title>
      </path>
    );
  });

  return (
    <svg viewBox="0 0 100 100" className="factor-pie" width="120" height="120">
      {paths}
    </svg>
  );
}

function RegimeBadge({ regime }: { regime: string }) {
  const color = REGIME_COLORS[regime] || '#6b7280';
  return (
    <span className="badge" style={{ backgroundColor: color, color: '#fff' }}>
      {regime.replace(/_/g, ' ').toUpperCase()}
    </span>
  );
}

export function FactorRotationPanel({ data }: FactorRotationPanelProps) {
  if (!data || !data.active) {
    return (
      <div className="panel">
        <h3>Factor Rotation (v3.00)</h3>
        <p className="muted">
          {data?.regime
            ? `Inactive — regime: ${data.regime}`
            : 'No factor rotation data available'}
        </p>
      </div>
    );
  }

  const adjustColor = data.equity_adjustment > 0
    ? '#22c55e'
    : data.equity_adjustment < 0
      ? '#ef4444'
      : '#6b7280';

  return (
    <div className="panel">
      <h3>Factor Rotation (v3.00)</h3>

      {/* Regime and score */}
      <div className="panel-section">
        <div className="metric-row">
          <span className="label">Regime</span>
          <RegimeBadge regime={data.regime} />
        </div>
        <div className="metric-row">
          <span className="label">Q+M Score</span>
          <span className="value" style={{
            color: data.quality_momentum_score > 0
              ? '#22c55e'
              : data.quality_momentum_score < 0
                ? '#ef4444'
                : '#6b7280',
          }}>
            {data.quality_momentum_score >= 0 ? '+' : ''}
            {data.quality_momentum_score.toFixed(2)}
          </span>
        </div>
        <div className="metric-row">
          <span className="label">Confidence</span>
          <span className="value">{(data.confidence * 100).toFixed(0)}%</span>
        </div>
      </div>

      {/* Factor allocation pie + legend */}
      <div className="panel-section">
        <h4>Factor Allocation</h4>
        <div className="pie-row">
          <FactorPie allocation={data.factor_allocations} />
          <div className="pie-legend">
            {(['MTUM', 'QUAL', 'USMV', 'VLUE'] as const).map((factor) => {
              const pct = data.factor_allocations[`${factor.toLowerCase()}_pct` as keyof FactorAllocation] as number;
              return (
                <div key={factor} className="legend-item">
                  <span
                    className="legend-dot"
                    style={{ backgroundColor: FACTOR_COLORS[factor] }}
                  />
                  <span className="legend-label">{factor}</span>
                  <span className="legend-value">{pct.toFixed(0)}%</span>
                </div>
              );
            })}
          </div>
        </div>
      </div>

      {/* Equity adjustment */}
      <div className="panel-grid">
        <div className="metric">
          <span className="label">Equity Adjustment</span>
          <span className="value" style={{ color: adjustColor }}>
            {data.equity_adjustment >= 0 ? '+' : ''}
            {data.equity_adjustment.toFixed(1)}%
          </span>
        </div>
        {data.mtum_momentum_6m !== undefined && (
          <div className="metric">
            <span className="label">MTUM 6m Mom</span>
            <span className="value" style={{
              color: data.mtum_momentum_6m > 0 ? '#22c55e' : '#ef4444',
            }}>
              {data.mtum_momentum_6m >= 0 ? '+' : ''}
              {data.mtum_momentum_6m.toFixed(1)}%
            </span>
          </div>
        )}
        {data.qual_roe !== undefined && (
          <div className="metric">
            <span className="label">QUAL ROE</span>
            <span className="value">{data.qual_roe.toFixed(1)}%</span>
          </div>
        )}
      </div>

      {/* Factor metrics */}
      <div className="panel-grid">
        {data.usmv_beta !== undefined && (
          <div className="metric">
            <span className="label">USMV Beta</span>
            <span className="value">{data.usmv_beta.toFixed(2)}</span>
          </div>
        )}
        {data.vlue_pe_ratio !== undefined && (
          <div className="metric">
            <span className="label">VLUE P/E</span>
            <span className="value">{data.vlue_pe_ratio.toFixed(1)}</span>
          </div>
        )}
      </div>

      {/* Backtest finding */}
      {data.backtest_finding && (
        <div className="panel-section">
          <h4>Backtest (Phase 3)</h4>
          <p className="muted small">{data.backtest_finding}</p>
        </div>
      )}
    </div>
  );
}
