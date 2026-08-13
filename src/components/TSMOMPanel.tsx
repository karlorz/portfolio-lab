
// ── Types ──────────────────────────────────────────────────────────────────

export interface TSMOMSpeed {
  label: string;
  weight: number;
  signal: number;
  asset_signals: Record<string, number>;
}

export interface TSMOMData {
  composite_signal: number;
  speed_breakdown: TSMOMSpeed[];
  position_recommendation: 'long' | 'short' | 'neutral';
  confidence: number;
  standalone_sharpe: number;
  overlay_sharpe: number;
  health_score: number;
  is_gated_off: boolean;
  generated_at: string;
}

interface TSMOMPanelProps {
  data: TSMOMData | null;
}

// ── Constants ──────────────────────────────────────────────────────────────

const ASSET_COLORS: Record<string, string> = {
  SPY: '#3b82f6',
  GLD: '#f59e0b',
  TLT: '#8b5cf6',
  IEF: '#06b6d4',
  EFA: '#10b981',
  VXUS: '#ec4899',
  DBC: '#f97316',
};

const DEFAULT_ASSET_COLOR = '#6b7280';

const BASELINE_SHARPE = 0.79; // Champion portfolio Sharpe

// ── Helpers ────────────────────────────────────────────────────────────────

function signalColor(signal: number): string {
  if (signal > 0.3) return '#10b981';
  if (signal > 0) return '#34d399';
  if (signal === 0) return '#94a3b8';
  if (signal > -0.3) return '#fbbf24';
  return '#ef4444';
}

function signalArrow(signal: number): string {
  if (signal > 0) return '\u2191';
  if (signal < 0) return '\u2193';
  return '\u2192';
}

function healthColor(score: number): string {
  if (score >= 0.80) return '#10b981';
  if (score >= 0.60) return '#f59e0b';
  return '#ef4444';
}

function positionBadge(
  recommendation: 'long' | 'short' | 'neutral',
  compositeSignal: number,
) {
  const styles: Record<string, { tone: string; label: string }> = {
    long: { tone: 'alc-chip-success', label: 'LONG' },
    short: { tone: 'alc-chip-danger', label: 'SHORT' },
    neutral: { tone: 'alc-chip-neutral', label: 'NEUTRAL' },
  };
  const s = styles[recommendation];
  return (
    <span
      className={`alc-chip ${s.tone}`}
    >
      {signalArrow(compositeSignal)} {s.label}
    </span>
  );
}

// ── Sub-Components ─────────────────────────────────────────────────────────

function SignalGauge({ signal }: { signal: number }) {
  // Normalise -1..+1 to 0..100 for the bar position
  const pct = ((signal + 1) / 2) * 100;

  return (
    <div className="alc-card">
      <div className="alc-row">
        <span className="alc-label">Composite Signal</span>
        <span
          className="alc-value-xl"
          style={{ color: signalColor(signal) }}
        >
          {signal >= 0 ? '+' : ''}
          {signal.toFixed(2)}
        </span>
      </div>

      {/* Gauge track */}
      <div className="alc-progress-tall">
        {/* Red zone (negative) */}
        <div
          className="alc-progress-segment"
          style={{ width: '50%', background: 'linear-gradient(to right, #ef4444, #fbbf24)' }}
        />
        {/* Green zone (positive) */}
        <div
          className="alc-progress-segment"
          style={{ left: '50%', width: '50%', background: 'linear-gradient(to left, #10b981, #34d399)' }}
        />
        {/* Center divider */}
        <div className="alc-progress-marker" style={{ left: '50%', backgroundColor: '#475569' }} />
        {/* Signal marker */}
        <div
          className="alc-progress-marker"
          style={{ left: `calc(${pct}% - 1px)` }}
        />
      </div>

      {/* Scale labels */}
      <div className="alc-scale">
        <span>-1.0</span>
        <span className="alc-strong">0.0</span>
        <span>+1.0</span>
      </div>
    </div>
  );
}

function SpeedCard({ speed }: { speed: TSMOMSpeed }) {
  const direction = speed.signal > 0 ? 'alc-text-success' : speed.signal < 0 ? 'alc-text-danger' : 'alc-text-muted';
  const weightPct = `${(speed.weight * 100).toFixed(0)}%`;
  const assetEntries = Object.entries(speed.asset_signals).slice(0, 6);

  return (
    <div className="alc-card alc-stack-sm">
      {/* Header: label + weight */}
      <div className="alc-row">
        <span className="alc-strong">{speed.label}</span>
        <span className="alc-chip-small alc-chip-neutral alc-mono">
          {weightPct} weight
        </span>
      </div>

      {/* Weight bar */}
      <div>
        <div className="alc-row">
          <span>Weight in composite</span>
          <span>{weightPct}</span>
        </div>
        <div className="alc-progress">
          <div
            className="alc-progress-fill"
            style={{ width: weightPct, backgroundColor: '#3b82f6' }}
          />
        </div>
      </div>

      {/* Signal value with arrow */}
      <div className="alc-cluster">
        <span className="alc-label">Signal</span>
        <span className={`alc-value-lg ${direction}`}>
          {signalArrow(speed.signal)} {speed.signal >= 0 ? '+' : ''}
          {speed.signal.toFixed(2)}
        </span>
      </div>

      {/* Per-asset signals */}
      {assetEntries.length > 0 && (
        <div>
          <span className="alc-label">
            Per-Asset Signals
          </span>
          <div className="alc-stack-xs">
            {assetEntries.map(([asset, sig]) => {
              const assetColor = ASSET_COLORS[asset] || DEFAULT_ASSET_COLOR;
              // Normalise -1..+1 to 0..100 for bar width, centre at 50%
              const barPct = ((sig + 1) / 2) * 100;
              return (
                <div key={asset} className="alc-row">
                  <span
                    className="alc-small alc-strong alc-align-right alc-fixed-sm"
                    style={{ color: assetColor }}
                  >
                    {asset}
                  </span>
                  <div className="alc-progress-tall alc-grow">
                    {/* Negative fill (left half) */}
                    {sig < 0 && (
                      <div
                        className="alc-progress-segment"
                        style={{
                          left: `${barPct}%`,
                          width: `${50 - barPct}%`,
                          backgroundColor: '#ef4444',
                        }}
                      />
                    )}
                    {/* Positive fill (right half) */}
                    {sig > 0 && (
                      <div
                        className="alc-progress-segment"
                        style={{
                          left: '50%',
                          width: `${barPct - 50}%`,
                          backgroundColor: '#10b981',
                        }}
                      />
                    )}
                    {/* Center line */}
                    <div className="alc-progress-marker" style={{ left: '50%', backgroundColor: '#475569' }} />
                  </div>
                  <span className="alc-small alc-mono alc-align-right alc-fixed-sm">
                    {sig >= 0 ? '+' : ''}
                    {sig.toFixed(2)}
                  </span>
                </div>
              );
            })}
          </div>
        </div>
      )}
    </div>
  );
}

function HealthGatingCard({
  healthScore,
  isGatedOff,
}: {
  healthScore: number;
  isGatedOff: boolean;
}) {
  const color = healthColor(healthScore);
  const pct = Math.min(healthScore * 100, 100);
  const viabilityFloor = 0.60;
  const belowFloor = healthScore < viabilityFloor;

  return (
    <div className="alc-card alc-stack">
      <h4 className="alc-section-title">
        Health &amp; Gating
      </h4>

      {/* Health score bar */}
      <div>
        <div className="alc-row">
          <span className="alc-label">Health Score</span>
          <span className="alc-value" style={{ color }}>
            {healthScore.toFixed(2)}
          </span>
        </div>
        <div className="alc-progress-tall">
          {/* Viability floor line */}
          <div
            className="alc-progress-marker alc-progress-marker-danger"
            style={{
              left: `${viabilityFloor * 100}%`,
              opacity: 0.8,
            }}
            title={`Viability floor: ${(viabilityFloor * 100).toFixed(0)}%`}
          />
          {/* Score bar */}
          <div
            className="alc-progress-fill"
            style={{ width: `${pct}%`, backgroundColor: color }}
          />
        </div>
        <span className="alc-small">
          Viability floor: {(viabilityFloor * 100).toFixed(0)}%
          {belowFloor && (
            <span className="alc-text-danger alc-strong">
              &mdash; Below floor
            </span>
          )}
        </span>
      </div>

      {/* Gate status */}
      <div>
        <span className="alc-label">Gate Status</span>
        {isGatedOff ? (
          <div className="alc-note alc-tone-danger">
            <span className="alc-strong">
              GATED OFF in HIGH_VOL / CRISIS
            </span>
            <p className="alc-small">
              TSMOM disabled in high volatility and crisis regimes via RegimeGate.
            </p>
          </div>
        ) : (
          <div className="alc-note alc-tone-success">
            <span className="alc-strong">
              ACTIVE
            </span>
            <p className="alc-small">
              Signal operational under current regime conditions.
            </p>
          </div>
        )}
      </div>
    </div>
  );
}

function PerformanceRow({
  standaloneSharpe,
  overlaySharpe,
}: {
  standaloneSharpe: number;
  overlaySharpe: number;
}) {
  const overlayColor =
    overlaySharpe >= standaloneSharpe
      ? '#10b981'
      : overlaySharpe >= standaloneSharpe * 0.8
        ? '#f59e0b'
        : '#ef4444';

  const isNetNegative = overlaySharpe < BASELINE_SHARPE;
  const netImpactLabel = isNetNegative
    ? 'Net-negative as overlay'
    : overlaySharpe >= standaloneSharpe
      ? 'Neutral or additive'
      : 'Slightly degraded';

  return (
    <div className="alc-card">
      <h4 className="alc-section-title">
        Performance Comparison
      </h4>

      <div className="alc-grid">
        <div className="alc-card alc-card-compact alc-card-accent">
          <span className="alc-label">
            Standalone Sharpe
          </span>
          <span className="alc-value-lg alc-text-success">
            {standaloneSharpe.toFixed(2)}
          </span>
        </div>
        <div className="alc-card alc-card-compact alc-card-accent">
          <span className="alc-label">
            Overlay Sharpe
          </span>
          <span className="alc-value-lg" style={{ color: overlayColor }}>
            {overlaySharpe.toFixed(2)}
          </span>
        </div>
      </div>

      {/* Net impact badge */}
      <div
        className={`alc-note ${
          isNetNegative
            ? 'alc-tone-danger'
            : overlaySharpe >= standaloneSharpe
              ? 'alc-tone-success'
              : 'alc-tone-warning'
        }`}
      >
        <div className="alc-row">
          <span className="alc-label">Net Impact vs Baseline ({BASELINE_SHARPE.toFixed(2)})</span>
          <span
            className={`alc-strong ${
              isNetNegative ? 'alc-text-danger' : overlaySharpe >= standaloneSharpe ? 'alc-text-success' : 'alc-text-warning'
            }`}
          >
            {netImpactLabel}
          </span>
        </div>
        <p className="alc-small">
          {isNetNegative
            ? 'Signal conflicts erode alpha when used as overlay — standalone use preferred.'
            : 'Overlay integration does not materially degrade standalone performance.'}
        </p>
      </div>
    </div>
  );
}

// ── Main Component ─────────────────────────────────────────────────────────

export function TSMOMPanel({ data }: TSMOMPanelProps) {
  if (!data) {
    return (
      <div className="alc-panel alc-panel-muted">
        <h3 className="alc-title">TSMOM Overlay</h3>
        <p className="alc-muted">No TSMOM data available</p>
      </div>
    );
  }

  const {
    composite_signal,
    speed_breakdown,
    position_recommendation,
    confidence,
    standalone_sharpe,
    overlay_sharpe,
    health_score,
    is_gated_off,
    generated_at,
  } = data;

  return (
    <div className="alc-panel alc-panel-muted">
      {/* ── Header ─────────────────────────────────────────── */}
      <div className="alc-header">
        <div>
          <h3 className="alc-title">TSMOM Overlay</h3>
          <p className="alc-subtitle">
            Time-Series Momentum &middot; Generated {generated_at}
          </p>
        </div>
        <div className="alc-header-actions">
          {positionBadge(position_recommendation, composite_signal)}
          <span className="alc-chip-small alc-chip-neutral alc-mono">
            {(confidence * 100).toFixed(0)}% conf
          </span>
        </div>
      </div>

      {/* ── Section 1: Signal Strength Gauge ──────────────── */}
      <div>
        <SignalGauge signal={composite_signal} />
      </div>

      {/* ── Section 2: Speed Breakdown ─────────────────────── */}
      <div className="alc-section">
        <h4 className="alc-section-title">
          Speed Breakdown ({speed_breakdown.length})
        </h4>
        {speed_breakdown.length === 0 ? (
          <p className="alc-muted">No speed breakdown available</p>
        ) : (
          <div className="alc-grid alc-grid-three">
            {speed_breakdown.map((speed) => (
              <SpeedCard key={speed.label} speed={speed} />
            ))}
          </div>
        )}
      </div>

      {/* ── Section 3: Health & Gating ─────────────────────── */}
      <HealthGatingCard healthScore={health_score} isGatedOff={is_gated_off} />

      {/* ── Section 4: Performance Comparison ──────────────── */}
      <PerformanceRow standaloneSharpe={standalone_sharpe} overlaySharpe={overlay_sharpe} />
    </div>
  );
}
