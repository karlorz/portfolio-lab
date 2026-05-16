import React from 'react';

interface OptionsSentiment {
  skew_index: number;
  vix: number;
  vix9d: number;
  vix9d_ratio: number;
  put_call_ratio: number;
  fear_greed_score: number;
}

interface RetailFlow {
  retail_call_put_ratio: number;
  retail_buy_sell_imbalance: number;
}

interface SocialIntensity {
  mention_velocity_7d: number;
  sentiment_divergence: number;
}

interface BehavioralSentimentData {
  active: boolean;
  composite_score: number;        // -3 (extreme fear) to +3 (extreme greed)
  signal_type: string;            // contrarian_buy, contrarian_sell, moderate_buy, moderate_sell, neutral
  confidence: number;             // 0-1
  equity_shift_pct: number;       // recommended allocation change
  z_score: number;
  vix: number;
  regime_suppressed: boolean;
  options?: OptionsSentiment;
  retail?: RetailFlow;
  social?: SocialIntensity;
  signal_count_5d: number;
  backtest_finding?: string;      // Phase 4 finding summary
}

interface BehavioralSentimentPanelProps {
  data: BehavioralSentimentData | null;
}

const GAUGE_SEGMENTS = [
  { min: -3.0, max: -2.0, label: 'Extreme Fear', color: '#dc2626' },
  { min: -2.0, max: -1.0, label: 'Fear', color: '#f97316' },
  { min: -1.0, max: 1.0, label: 'Neutral', color: '#6b7280' },
  { min: 1.0, max: 2.0, label: 'Greed', color: '#22c55e' },
  { min: 2.0, max: 3.0, label: 'Extreme Greed', color: '#16a34a' },
];

const SIGNAL_COLORS: Record<string, string> = {
  contrarian_buy: '#22c55e',
  contrarian_sell: '#ef4444',
  moderate_buy: '#86efac',
  moderate_sell: '#fca5a5',
  neutral: '#6b7280',
};

function Gauge({ value }: { value: number }) {
  // Map -3..+3 → 0..100%
  const clamped = Math.max(-3, Math.min(3, value));
  const pct = ((clamped + 3) / 6) * 100;

  return (
    <div className="fear-greed-gauge">
      <div className="gauge-bar">
        {GAUGE_SEGMENTS.map((seg) => {
          const left = ((seg.min + 3) / 6) * 100;
          const width = ((seg.max - seg.min) / 6) * 100;
          return (
            <div
              key={seg.label}
              className="gauge-segment"
              style={{
                left: `${left}%`,
                width: `${width}%`,
                backgroundColor: seg.color,
              }}
              title={seg.label}
            />
          );
        })}
        <div
          className="gauge-needle"
          style={{ left: `${pct}%` }}
        />
      </div>
      <div className="gauge-labels">
        <span className="gauge-label" style={{ color: '#dc2626' }}>Fear</span>
        <span className="gauge-label" style={{ color: '#6b7280' }}>Neutral</span>
        <span className="gauge-label" style={{ color: '#16a34a' }}>Greed</span>
      </div>
    </div>
  );
}

function SignalBadge({ signalType, regimeSuppressed }: {
  signalType: string;
  regimeSuppressed: boolean;
}) {
  if (regimeSuppressed) {
    return <span className="badge badge-suppressed">SUPPRESSED</span>;
  }

  const color = SIGNAL_COLORS[signalType] || '#6b7280';
  const label = signalType.replace(/_/g, ' ').toUpperCase();

  return (
    <span className="badge" style={{ backgroundColor: color, color: '#fff' }}>
      {label}
    </span>
  );
}

export function BehavioralSentimentPanel({ data }: BehavioralSentimentPanelProps) {
  if (!data) {
    return (
      <div className="panel">
        <h3>Behavioral Sentiment (v2.70)</h3>
        <p className="muted">No behavioral sentiment data available</p>
      </div>
    );
  }

  const signalLabel = data.signal_type.replace(/_/g, ' ');
  const shiftDirection = data.equity_shift_pct > 0 ? '+' : '';
  const shiftColor = data.equity_shift_pct > 0
    ? '#22c55e'
    : data.equity_shift_pct < 0
      ? '#ef4444'
      : '#6b7280';

  return (
    <div className="panel">
      <h3>Behavioral Sentiment (v2.70)</h3>

      {/* Fear / Greed gauge */}
      <div className="panel-section">
        <Gauge value={data.composite_score} />
        <div className="metric-row center">
          <span className="label">Composite Score</span>
          <span className="value large">
            {data.composite_score >= 0 ? '+' : ''}{data.composite_score.toFixed(2)}
          </span>
          <span className="subtext">({signalLabel})</span>
        </div>
      </div>

      {/* Signal status */}
      <div className="panel-grid">
        <div className="metric">
          <span className="label">Signal</span>
          <SignalBadge
            signalType={data.signal_type}
            regimeSuppressed={data.regime_suppressed}
          />
        </div>
        <div className="metric">
          <span className="label">Confidence</span>
          <span className="value">{(data.confidence * 100).toFixed(0)}%</span>
        </div>
        <div className="metric">
          <span className="label">Equity Shift</span>
          <span className="value" style={{ color: shiftColor }}>
            {shiftDirection}{data.equity_shift_pct.toFixed(1)}%
          </span>
        </div>
        <div className="metric">
          <span className="label">Z-Score</span>
          <span className="value">
            {data.z_score >= 0 ? '+' : ''}{data.z_score.toFixed(2)}
          </span>
        </div>
        <div className="metric">
          <span className="label">VIX</span>
          <span className="value">{data.vix.toFixed(1)}</span>
        </div>
        <div className="metric">
          <span className="label">5d Signal Count</span>
          <span className="value" style={{
            color: data.signal_count_5d >= 2 ? '#ef4444' : '#6b7280',
          }}>
            {data.signal_count_5d}
          </span>
        </div>
      </div>

      {/* Options market sentiment */}
      {data.options && (
        <div className="panel-section">
          <h4>Options Market</h4>
          <div className="panel-grid">
            <div className="metric">
              <span className="label">SKEW Index</span>
              <span className="value">{data.options.skew_index.toFixed(1)}</span>
            </div>
            <div className="metric">
              <span className="label">VIX9D/VIX</span>
              <span className="value">{data.options.vix9d_ratio.toFixed(2)}</span>
            </div>
            <div className="metric">
              <span className="label">P/C Ratio</span>
              <span className="value">{data.options.put_call_ratio.toFixed(2)}</span>
            </div>
            <div className="metric">
              <span className="label">F/G Score</span>
              <span className="value">
                {data.options.fear_greed_score >= 0 ? '+' : ''}
                {data.options.fear_greed_score.toFixed(1)}
              </span>
            </div>
          </div>
        </div>
      )}

      {/* Retail flow */}
      {data.retail && (
        <div className="panel-section">
          <h4>Retail Flow</h4>
          <div className="panel-grid">
            <div className="metric">
              <span className="label">Call/Put Ratio</span>
              <span className="value">{data.retail.retail_call_put_ratio.toFixed(2)}</span>
            </div>
            <div className="metric">
              <span className="label">Buy/Sell Imbalance</span>
              <span className="value" style={{
                color: data.retail.retail_buy_sell_imbalance > 0 ? '#22c55e' : '#ef4444',
              }}>
                {data.retail.retail_buy_sell_imbalance >= 0 ? '+' : ''}
                {data.retail.retail_buy_sell_imbalance.toFixed(2)}
              </span>
            </div>
          </div>
        </div>
      )}

      {/* Social sentiment */}
      {data.social && (
        <div className="panel-section">
          <h4>Social Sentiment</h4>
          <div className="panel-grid">
            <div className="metric">
              <span className="label">Mention Velocity (7d)</span>
              <span className="value">{data.social.mention_velocity_7d.toFixed(2)}x</span>
            </div>
            <div className="metric">
              <span className="label">Sentiment Divergence</span>
              <span className="value" style={{
                color: Math.abs(data.social.sentiment_divergence) > 0.3 ? '#f59e0b' : '#6b7280',
              }}>
                {data.social.sentiment_divergence >= 0 ? '+' : ''}
                {data.social.sentiment_divergence.toFixed(3)}
              </span>
            </div>
          </div>
        </div>
      )}

      {/* Backtest finding */}
      {data.backtest_finding && (
        <div className="panel-section">
          <h4>Backtest Finding (Phase 4)</h4>
          <p className="muted small">{data.backtest_finding}</p>
        </div>
      )}
    </div>
  );
}
