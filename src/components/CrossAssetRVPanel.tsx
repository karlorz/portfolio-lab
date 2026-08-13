
export interface AssetPairRV {
  pair: string;           // "SPY/GLD", "SPY/TLT", "GLD/TLT"
  z_score: number;        // current z-score
  percentile_1y: number;  // 1-year percentile
  direction: 'long_first' | 'long_second' | 'neutral';
  strength: number;       // 0-1
}

export interface CrossAssetRVData {
  signal_value: number;         // -1 to +1
  pairs: AssetPairRV[];
  current_regime: string;
  is_gated_off: boolean;        // true in HIGH_VOL/CRISIS (v961: -8.68 Sharpe)
  regime_note: string;          // "Mean-reversion fails in volatile regimes"
  weight_in_ensemble: number;   // 0.13 (13%)
  generated_at: string;
}

interface CrossAssetRVPanelProps {
  data: CrossAssetRVData | null;
}

function signalValueColor(value: number): string {
  if (value > 0.3) return '#22c55e';
  if (value > 0.1) return '#86efac';
  if (value < -0.3) return '#ef4444';
  if (value < -0.1) return '#fca5a5';
  return '#6b7280';
}

function zScoreColor(z: number): string {
  if (z < -1) return '#3b82f6';
  if (z > 1) return '#f59e0b';
  return '#94a3b8';
}

function DirectionBadge({ direction }: { direction: AssetPairRV['direction'] }) {
  const config: Record<string, { label: string; color: string; arrow: string }> = {
    long_first: { label: 'Long First', color: '#22c55e', arrow: '\u2191' },
    long_second: { label: 'Long Second', color: '#22c55e', arrow: '\u2191' },
    neutral: { label: 'Neutral', color: '#6b7280', arrow: '\u2194' },
  };
  const c = config[direction] ?? config.neutral;
  return (
    <span className="badge" style={{ backgroundColor: c.color, color: '#fff', fontSize: 11 }}>
      {c.arrow} {c.label}
    </span>
  );
}

function StrengthBar({ value }: { value: number }) {
  const pct = Math.max(0, Math.min(100, value * 100));
  const color = value > 0.7 ? '#22c55e' : value > 0.4 ? '#f59e0b' : '#6b7280';
  return (
    <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
      <div style={{ flex: 1, height: 6, background: '#1e293b', borderRadius: 3, overflow: 'hidden' }}>
        <div style={{ width: `${pct}%`, height: '100%', background: color, borderRadius: 3 }} />
      </div>
      <span style={{ fontSize: 10, color, minWidth: 28, textAlign: 'right' }}>
        {(value * 100).toFixed(0)}%
      </span>
    </div>
  );
}

function PercentileMiniBar({ value }: { value: number | null | undefined }) {
  const safe = typeof value === 'number' && Number.isFinite(value) ? value : 0;
  const pct = Math.max(0, Math.min(100, safe));
  return (
    <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
      <div style={{ flex: 1, height: 4, background: '#1e293b', borderRadius: 2, overflow: 'hidden' }}>
        <div style={{ width: `${pct}%`, height: '100%', background: '#8b5cf6', borderRadius: 2 }} />
      </div>
      <span style={{ fontSize: 10, color: '#94a3b8', minWidth: 30, textAlign: 'right' }}>
        {safe.toFixed(0)}%
      </span>
    </div>
  );
}

function StrengthBarSafe({ value }: { value: number | null | undefined }) {
  const safe = typeof value === 'number' && Number.isFinite(value) ? value : 0;
  return <StrengthBar value={safe} />;
}

/** Normalize producer pair rows (pair_name/conviction/…) into panel AssetPairRV. */
export function normalizeCrossAssetRVPair(raw: unknown): AssetPairRV | null {
  if (typeof raw !== 'object' || raw === null || Array.isArray(raw)) return null;
  const row = raw as Record<string, unknown>;
  const pair =
    typeof row.pair === 'string' && row.pair.length > 0
      ? row.pair
      : typeof row.pair_name === 'string'
        ? String(row.pair_name).replace(/_/g, '/').toUpperCase()
        : typeof row.symbol_a === 'string' && typeof row.symbol_b === 'string'
          ? `${row.symbol_a}/${row.symbol_b}`
          : null;
  if (!pair) return null;
  const z = typeof row.z_score === 'number' && Number.isFinite(row.z_score) ? row.z_score : 0;
  const percentile =
    typeof row.percentile_1y === 'number' && Number.isFinite(row.percentile_1y)
      ? row.percentile_1y
      : typeof row.percentile === 'number' && Number.isFinite(row.percentile)
        ? row.percentile
        : Math.max(0, Math.min(100, ((z + 3) / 6) * 100));
  const directionRaw = row.direction;
  const direction: AssetPairRV['direction'] =
    directionRaw === 'long_first' || directionRaw === 'long_second' || directionRaw === 'neutral'
      ? directionRaw
      : z < -0.5
        ? 'long_first'
        : z > 0.5
          ? 'long_second'
          : 'neutral';
  const strength =
    typeof row.strength === 'number' && Number.isFinite(row.strength)
      ? row.strength
      : typeof row.conviction === 'number' && Number.isFinite(row.conviction)
        ? row.conviction
        : Math.min(1, Math.abs(z) / 2);
  return { pair, z_score: z, percentile_1y: percentile, direction, strength };
}

export function normalizeCrossAssetRVData(value: unknown): CrossAssetRVData | null {
  if (typeof value !== 'object' || value === null || Array.isArray(value)) return null;
  const raw = value as Record<string, unknown>;
  if (typeof raw.signal_value !== 'number' || !Number.isFinite(raw.signal_value)) return null;
  const pairsRaw = Array.isArray(raw.pairs) ? raw.pairs : [];
  const pairs = pairsRaw
    .map(normalizeCrossAssetRVPair)
    .filter((p): p is AssetPairRV => p !== null);
  return {
    signal_value: raw.signal_value,
    pairs,
    current_regime: typeof raw.current_regime === 'string' ? raw.current_regime : 'unknown',
    is_gated_off: Boolean(raw.is_gated_off),
    regime_note: typeof raw.regime_note === 'string' ? raw.regime_note : '',
    weight_in_ensemble:
      typeof raw.weight_in_ensemble === 'number' && Number.isFinite(raw.weight_in_ensemble)
        ? raw.weight_in_ensemble
        : 0,
    generated_at: typeof raw.generated_at === 'string' ? raw.generated_at : '',
  };
}

export function CrossAssetRVPanel({ data }: CrossAssetRVPanelProps) {
  const normalized = normalizeCrossAssetRVData(data);
  if (!normalized) {
    return (
      <div className="panel">
        <h3>Cross-Asset Relative Value</h3>
        <p className="muted">No cross-asset RV data available</p>
      </div>
    );
  }
  const dataSafe = normalized;

  const signalPct = ((dataSafe.signal_value + 1) / 2) * 100;
  const signalColor = signalValueColor(dataSafe.signal_value);
  const weightPct = Math.max(0, Math.min(100, dataSafe.weight_in_ensemble * 100));

  return (
    <div className="panel">
      <h3>Cross-Asset Relative Value</h3>

      {/* Signal Overview */}
      <div className="panel-section">
        <h4>Signal Overview</h4>

        {/* Signal value gauge */}
        <div style={{ marginBottom: 10 }}>
          <div style={{
            position: 'relative', height: 14, borderRadius: 7, overflow: 'hidden',
            background: 'linear-gradient(90deg, #ef4444 0%, #6b7280 50%, #22c55e 100%)',
          }}>
            <div style={{
              position: 'absolute', top: -3, left: `${signalPct}%`, width: 4, height: 20,
              background: '#fff', borderRadius: 2, transform: 'translateX(-50%)',
              boxShadow: '0 0 4px rgba(255,255,255,0.6)',
            }} />
          </div>
          <div style={{ display: 'flex', justifyContent: 'space-between', marginTop: 2 }}>
            <span style={{ fontSize: 9, color: '#ef4444' }}>-1.0</span>
            <span style={{ fontSize: 9, color: '#94a3b8' }}>0.0</span>
            <span style={{ fontSize: 9, color: '#22c55e' }}>+1.0</span>
          </div>
          <div style={{ textAlign: 'center', marginTop: 4 }}>
            <span className="value large" style={{ color: signalColor }}>
              {dataSafe.signal_value >= 0 ? '+' : ''}{dataSafe.signal_value.toFixed(3)}
            </span>
          </div>
        </div>

        {/* Regime badge + gate status */}
        <div className="panel-grid">
          <div className="metric">
            <span className="label">Current Regime</span>
            <span className="badge" style={{
              backgroundColor: dataSafe.is_gated_off ? '#dc2626' : '#1e3a5f',
              color: '#fff',
            }}>
              {dataSafe.current_regime}
            </span>
          </div>
          <div className="metric">
            <span className="label">Gate Status</span>
            {dataSafe.is_gated_off ? (
              <span className="badge" style={{ backgroundColor: '#dc2626', color: '#fff' }}>
                GATED OFF
              </span>
            ) : (
              <span className="badge" style={{ backgroundColor: '#065f46', color: '#6ee7b7' }}>
                ACTIVE
              </span>
            )}
          </div>
        </div>

        {/* Ensemble weight bar */}
        <div style={{ marginTop: 8 }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 2 }}>
            <span className="label">Ensemble Weight</span>
            <span style={{ fontSize: 12, color: '#94a3b8' }}>
              {(dataSafe.weight_in_ensemble * 100).toFixed(0)}%
            </span>
          </div>
          <div style={{ height: 8, background: '#1e293b', borderRadius: 4, overflow: 'hidden' }}>
            <div style={{
              width: `${weightPct}%`, height: '100%',
              background: 'linear-gradient(90deg, #3b82f6, #8b5cf6)',
              borderRadius: 4,
            }} />
          </div>
        </div>
      </div>

      {/* Asset Pair Z-Scores */}
      <div className="panel-section">
        <h4>Asset Pair Z-Scores</h4>
        {dataSafe.pairs.length === 0 ? (
          <p className="muted small">No asset pair data available</p>
        ) : (
          <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
            {dataSafe.pairs.map((pair) => (
              <div key={pair.pair} style={{
                background: 'rgba(31, 41, 55, 0.4)',
                borderRadius: 6,
                padding: '8px 10px',
                border: '1px solid rgba(55, 65, 81, 0.5)',
              }}>
                {/* Pair name + direction badge */}
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 6 }}>
                  <span style={{ fontSize: 13, fontWeight: 600, color: '#e2e8f0' }}>
                    {pair.pair}
                  </span>
                  <DirectionBadge direction={pair.direction} />
                </div>

                {/* Z-score */}
                <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 4 }}>
                  <span style={{ fontSize: 11, color: '#94a3b8' }}>Z-Score</span>
                  <span style={{ fontSize: 12, fontWeight: 600, color: zScoreColor(pair.z_score) }}>
                    {pair.z_score >= 0 ? '+' : ''}{pair.z_score.toFixed(2)}
                  </span>
                </div>

                {/* 1Y Percentile */}
                <div style={{ marginBottom: 4 }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 2 }}>
                    <span style={{ fontSize: 11, color: '#94a3b8' }}>1Y Percentile</span>
                  </div>
                  <PercentileMiniBar value={pair.percentile_1y} />
                </div>

                {/* Strength */}
                <div>
                  <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 2 }}>
                    <span style={{ fontSize: 11, color: '#94a3b8' }}>Strength</span>
                  </div>
                  <StrengthBarSafe value={pair.strength} />
                </div>
              </div>
            ))}
          </div>
        )}
      </div>

      {/* Regime Warning */}
      {dataSafe.is_gated_off && (
        <div className="panel-section">
          <div style={{
            padding: '8px 10px', borderRadius: 6,
            background: 'rgba(239, 68, 68, 0.12)',
            border: '1px solid rgba(239, 68, 68, 0.3)',
          }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: 6, marginBottom: 4 }}>
              <span style={{ color: '#ef4444', fontSize: 14, fontWeight: 700 }}>&#9888;</span>
              <span style={{ color: '#fca5a5', fontSize: 12, fontWeight: 600 }}>
                Cross-Asset RV Gated OFF
              </span>
            </div>
            <p style={{ color: '#fca5a5', fontSize: 11, margin: 0, lineHeight: 1.4 }}>
              {dataSafe.regime_note} &mdash; mean-reversion signals fail when volatility clusters
              (v961: -8.68 Sharpe in {dataSafe.current_regime} regime)
            </p>
          </div>
        </div>
      )}

      {/* Timestamp */}
      {dataSafe.generated_at && (
        <div style={{ marginTop: 6, fontSize: 10, color: '#475569' }}>
          Updated: {dataSafe.generated_at.slice(0, 19)}
        </div>
      )}
    </div>
  );
}
