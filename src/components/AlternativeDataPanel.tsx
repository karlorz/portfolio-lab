
interface ComponentData {
  score: number | null;
  confidence: number | null;
  weight: number | null;
}

export interface AlternativeData {
  regime?: string;
  probability?: number;
  confidence?: number;
  timestamp?: string;
  components?: Record<string, ComponentData>;
  composite_score?: number | null;
  z_score?: number | null;
  sources_count?: number | null;
  data_freshness_hours?: number | null;
}

interface AlternativeDataProps {
  data: AlternativeData | null;
}

const COMPONENT_LABELS: Record<string, string> = {
  // Current seven-component producer
  treasury_curve: 'Treasury Curve',
  sector_rotation: 'Sector Rotation',
  credit_spread: 'Credit Spread',
  tail_risk: 'Tail Risk',
  broad_momentum: 'Broad Momentum',
  crypto_sentiment: 'Crypto Sentiment',
  crypto_fg: 'Crypto F&G',
  // Legacy flat keys (fallback projection)
  earnings: 'Earnings',
  news: 'News',
  jobs: 'Jobs',
  social: 'Social',
};

function ScoreBar({ value, label, color }: { value: number; label: string; color: string }) {
  const pct = Math.min(Math.abs(value) * 100, 100);
  const isPositive = value >= 0;
  return (
    <div style={{ display: 'flex', alignItems: 'center', gap: 6, marginBottom: 4 }}>
      <span style={{ minWidth: 60, fontSize: 11, color: '#94a3b8' }}>{label}</span>
      <div style={{ flex: 1, height: 6, background: '#1e293b', borderRadius: 3, position: 'relative', overflow: 'hidden' }}>
        <div style={{
          position: 'absolute',
          left: isPositive ? '50%' : `${50 - pct / 2}%`,
          width: `${pct / 2}%`,
          height: '100%',
          background: color,
          borderRadius: 3,
        }} />
      </div>
      <span style={{ fontSize: 11, color, minWidth: 36, textAlign: 'right' }}>
        {value >= 0 ? '+' : ''}{(value * 100).toFixed(1)}%
      </span>
    </div>
  );
}

export function AlternativeDataPanel({ data }: AlternativeDataProps) {
  if (!data) {
    return (
      <div className="panel">
        <h3>Alternative Data</h3>
        <p className="muted">No alternative data available</p>
      </div>
    );
  }

  const zColor = data.z_score !== null && data.z_score !== undefined
    ? Math.abs(data.z_score) > 2 ? '#ef4444' : Math.abs(data.z_score) > 1 ? '#f59e0b' : '#10b981'
    : '#94a3b8';

  return (
    <div className="panel">
      <h3>Alternative Data</h3>
      <div className="panel-grid">
        <div className="metric">
          <span className="label">Composite</span>
          <span className="value">
            {data.composite_score !== null && data.composite_score !== undefined
              ? (data.composite_score * 100).toFixed(1) + '%'
              : '--'}
          </span>
        </div>
        <div className="metric">
          <span className="label">Z-Score</span>
          <span className="value" style={{ color: zColor }}>
            {data.z_score !== null && data.z_score !== undefined
              ? data.z_score.toFixed(2)
              : '--'}
          </span>
        </div>
        <div className="metric">
          <span className="label">Confidence</span>
          <span className="value">
            {data.confidence !== null && data.confidence !== undefined
              ? (data.confidence * 100).toFixed(0) + '%'
              : '--'}
          </span>
        </div>
        <div className="metric">
          <span className="label">Sources</span>
          <span className="value">{data.sources_count ?? '--'}</span>
        </div>
        <div className="metric">
          <span className="label">Regime</span>
          <span className="value">{data.regime || '--'}</span>
        </div>
        <div className="metric">
          <span className="label">Freshness</span>
          <span className="value" style={{
            color: (data.data_freshness_hours ?? 999) > 24 ? '#ef4444' :
              (data.data_freshness_hours ?? 0) > 6 ? '#f59e0b' : '#10b981'
          }}>
            {data.data_freshness_hours !== null && data.data_freshness_hours !== undefined
              ? data.data_freshness_hours.toFixed(1) + 'h'
              : '--'}
          </span>
        </div>
      </div>

      {/* Component Breakdown */}
      {data.components && Object.keys(data.components).length > 0 && (
        <div style={{ marginTop: 12 }}>
          <span className="label" style={{ marginBottom: 6, display: 'block' }}>Signal Components</span>
          {Object.entries(data.components).map(([key, comp]) => {
            const score = comp.score ?? 0;
            const color = score > 0.1 ? '#10b981' : score < -0.1 ? '#ef4444' : '#f59e0b';
            return (
              <div key={key} style={{ marginBottom: 6 }}>
                <ScoreBar value={score} label={COMPONENT_LABELS[key] || key} color={color} />
                <div style={{ display: 'flex', gap: 12, fontSize: 10, color: '#64748b', paddingLeft: 66 }}>
                  <span>Conf: {comp.confidence !== null ? (comp.confidence * 100).toFixed(0) + '%' : '--'}</span>
                  <span>Wt: {comp.weight !== null ? (comp.weight * 100).toFixed(0) + '%' : '--'}</span>
                </div>
              </div>
            );
          })}
        </div>
      )}

      {data.timestamp && (
        <div style={{ marginTop: 8, fontSize: 10, color: '#475569' }}>
          Updated: {data.timestamp.slice(0, 19)}
        </div>
      )}
    </div>
  );
}
