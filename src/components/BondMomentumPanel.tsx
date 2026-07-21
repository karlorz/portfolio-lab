import React from 'react';
import type {
  BondMomentumEnsemble,
  BondMomentumSignal,
  BondMomentumSummaryData,
} from '../types/live';

type UnknownRecord = Record<string, unknown>;

/** Legacy overlay rows (TSMOM-style per-ETF signals). */
export interface BondMomentumLegacyData {
  signals: BondMomentumSignal[];
  timestamp?: string;
  ensemble?: BondMomentumEnsemble;
}

export type BondMomentumPanelInput =
  | BondMomentumSummaryData
  | BondMomentumLegacyData
  | UnknownRecord
  | null
  | undefined;

interface BondMomentumPanelProps {
  /** Producer summary, legacy overlay rows, or null. Prefer `data` over split props. */
  data?: BondMomentumPanelInput;
  /** @deprecated Prefer `data` — kept for transitional call sites. */
  signals?: BondMomentumSignal[];
  timestamp?: string;
  ensembleRecommendation?: BondMomentumEnsemble;
}

const ETF_CONFIG: Record<string, {
  name: string;
  duration: string;
  color: string;
  formationMonths: number;
  description: string;
}> = {
  SHY: {
    name: 'SHY',
    duration: '1-3 Year Treasury',
    color: '#10b981',
    formationMonths: 12,
    description: 'Short duration - strong momentum effect',
  },
  IEF: {
    name: 'IEF',
    duration: '7-10 Year Treasury',
    color: '#8b5cf6',
    formationMonths: 12,
    description: 'Intermediate - moderate effectiveness',
  },
  TLT: {
    name: 'TLT',
    duration: '20+ Year Treasury',
    color: '#3b82f6',
    formationMonths: 18,
    description: 'Long duration - crisis detection focus',
  },
  BIL: {
    name: 'BIL',
    duration: '1-3 Month T-Bill',
    color: '#f59e0b',
    formationMonths: 12,
    description: 'Ultra short - conservative',
  },
};

type BondTone = 'success' | 'info' | 'warning' | 'danger' | 'muted';

function isRecord(value: unknown): value is UnknownRecord {
  return typeof value === 'object' && value !== null && !Array.isArray(value);
}

function asFiniteNumber(value: unknown, fallback = 0): number {
  const n = typeof value === 'number' ? value : Number(value);
  return Number.isFinite(n) ? n : fallback;
}

function asString(value: unknown, fallback = ''): string {
  return typeof value === 'string' ? value : fallback;
}

function asBool(value: unknown, fallback = false): boolean {
  return typeof value === 'boolean' ? value : fallback;
}

function actionTone(action: string): BondTone {
  switch (action) {
    case 'increase':
      return 'success';
    case 'hold':
      return 'info';
    case 'reduce':
      return 'warning';
    case 'avoid':
      return 'danger';
    default:
      return 'muted';
  }
}

function confidenceTone(confidence: string): { tone: BondTone; label: string } {
  switch (confidence) {
    case 'strong':
      return { tone: 'success', label: 'Strong' };
    case 'moderate':
      return { tone: 'info', label: 'Moderate' };
    case 'weak':
      return { tone: 'warning', label: 'Weak' };
    default:
      return { tone: 'muted', label: confidence };
  }
}

function signalTone(signal: number): BondTone {
  if (signal > 1.5) return 'success';
  if (signal > 0.5) return 'info';
  if (signal > 0) return 'warning';
  return 'muted';
}

function positionTone(position: string): BondTone {
  switch (position.toLowerCase()) {
    case 'long':
    case 'extended':
      return 'success';
    case 'intermediate':
      return 'info';
    case 'short':
    case 'defensive':
      return 'warning';
    case 'cash':
    case 'avoid':
      return 'danger';
    default:
      return 'muted';
  }
}

function isLegacyShape(raw: UnknownRecord): boolean {
  return Array.isArray(raw.signals);
}

function isSummaryShape(raw: UnknownRecord): boolean {
  return (
    raw.position !== undefined
    || raw.status_text !== undefined
    || raw.curve_regime !== undefined
    || raw.effective_duration !== undefined
    || raw.tlt_weight !== undefined
    || raw.yield_10y !== undefined
  );
}

/**
 * Map producer-shaped `signals.json.bond_momentum` (bond-duration summary)
 * and legacy overlay `{signals, timestamp, ensemble}` into a view-model.
 */
export function normalizeBondMomentumData(
  raw: unknown,
):
  | { kind: 'summary'; data: BondMomentumSummaryData }
  | { kind: 'legacy'; data: BondMomentumLegacyData }
  | null {
  if (!isRecord(raw)) return null;

  if (isLegacyShape(raw)) {
    const signals = raw.signals as BondMomentumSignal[];
    if (!Array.isArray(signals) || signals.length === 0) return null;
    return {
      kind: 'legacy',
      data: {
        signals,
        timestamp: typeof raw.timestamp === 'string' ? raw.timestamp : undefined,
        ensemble: isRecord(raw.ensemble)
          // Cast via unknown: UnknownRecord is not structurally assignable to
          // BondMomentumEnsemble under strict TS (2025 unknown-first pattern).
          ? (raw.ensemble as unknown as BondMomentumEnsemble)
          : undefined,
      },
    };
  }

  if (!isSummaryShape(raw)) return null;

  const position = asString(raw.position, 'unknown');
  const status_text = asString(raw.status_text, '');
  // Active recommendation present even without status_text.
  const active = asBool(raw.active, true);

  return {
    kind: 'summary',
    data: {
      active,
      yield_10y: asFiniteNumber(raw.yield_10y),
      yield_2y: asFiniteNumber(raw.yield_2y),
      spread: asFiniteNumber(raw.spread),
      curve_regime: asString(raw.curve_regime, 'unknown'),
      rate_direction: asString(raw.rate_direction, 'unknown'),
      tlt_weight: asFiniteNumber(raw.tlt_weight),
      ief_weight: asFiniteNumber(raw.ief_weight),
      shy_weight: asFiniteNumber(raw.shy_weight),
      effective_duration: asFiniteNumber(raw.effective_duration),
      position,
      confidence: asFiniteNumber(raw.confidence),
      status_text: status_text || `Bonds: ${position}`,
      generated_at: typeof raw.generated_at === 'string' ? raw.generated_at : undefined,
      timestamp: typeof raw.timestamp === 'string' ? raw.timestamp : undefined,
    },
  };
}

function WeightBar({
  label,
  weight,
  color,
}: {
  label: string;
  weight: number;
  color: string;
}) {
  // Producer weights are fractions (0.2 = 20%).
  const pct = Math.abs(weight) <= 1.5 ? weight * 100 : weight;
  const width = Math.min(Math.max(pct, 0), 100);
  return (
    <div className="bond-weight-row">
      <div className="bond-weight-header">
        <span className="bond-swatch" style={{ backgroundColor: color }} />
        <strong>{label}</strong>
        <span>{pct.toFixed(0)}%</span>
      </div>
      <div className="risk-gauge-track">
        <div
          className="risk-gauge-fill risk-gauge-fill-info"
          style={{ width: `${width}%`, backgroundColor: color }}
        />
      </div>
    </div>
  );
}

function SummaryView({ data }: { data: BondMomentumSummaryData }) {
  const posTone = positionTone(data.position);
  const confLabel =
    data.confidence >= 70 ? 'High' : data.confidence >= 40 ? 'Moderate' : 'Low';
  const confTone: BondTone =
    data.confidence >= 70 ? 'success' : data.confidence >= 40 ? 'info' : 'warning';

  return (
    <div className="risk-card bond-momentum-card">
      <div className="risk-card-header">
        <div>
          <h3>Bond Duration Allocation</h3>
          <p className="risk-subtitle">Yield-curve / rate-direction summary</p>
        </div>
        <span className={`risk-badge risk-badge-${data.active ? 'info' : 'muted'}`}>
          {data.active ? 'Active' : 'Inactive'}
        </span>
      </div>

      <div className="risk-metric-grid risk-metric-grid-three">
        <div className="risk-metric">
          <span className="risk-label">Position</span>
          <span className={`risk-value compact risk-text-${posTone}`}>
            {data.position}
          </span>
        </div>
        <div className="risk-metric">
          <span className="risk-label">Eff. Duration</span>
          <span className="risk-value compact">{data.effective_duration.toFixed(1)} yr</span>
        </div>
        <div className="risk-metric">
          <span className="risk-label">Confidence</span>
          <span className={`risk-value compact risk-text-${confTone}`}>
            {confLabel} ({data.confidence.toFixed(0)})
          </span>
        </div>
      </div>

      <div className="risk-metric-grid risk-metric-grid-three">
        <div className="risk-metric">
          <span className="risk-label">10Y Yield</span>
          <span className="risk-value compact">{data.yield_10y.toFixed(2)}%</span>
        </div>
        <div className="risk-metric">
          <span className="risk-label">2Y Yield</span>
          <span className="risk-value compact">{data.yield_2y.toFixed(2)}%</span>
        </div>
        <div className="risk-metric">
          <span className="risk-label">2s10s Spread</span>
          <span className="risk-value compact">{data.spread.toFixed(2)}</span>
        </div>
      </div>

      <div className="risk-metric-grid risk-metric-grid-three">
        <div className="risk-metric">
          <span className="risk-label">Curve Regime</span>
          <span className="risk-value small">{data.curve_regime}</span>
        </div>
        <div className="risk-metric">
          <span className="risk-label">Rate Direction</span>
          <span className="risk-value small">{data.rate_direction}</span>
        </div>
        <div className="risk-metric">
          <span className="risk-label">Recommendation</span>
          <span className="risk-value small">{data.status_text}</span>
        </div>
      </div>

      <div className="bond-signal-list">
        <h4>ETF Weights</h4>
        <WeightBar label="TLT" weight={data.tlt_weight} color={ETF_CONFIG.TLT.color} />
        <WeightBar label="IEF" weight={data.ief_weight} color={ETF_CONFIG.IEF.color} />
        <WeightBar label="SHY" weight={data.shy_weight} color={ETF_CONFIG.SHY.color} />
      </div>

      <div className="risk-footnote">
        <strong>Bond duration summary:</strong> Public{' '}
        <code>signals.json.bond_momentum</code> publishes curve regime, rate
        direction, and duration ETF weights (not the retired TSMOM per-ETF
        overlay rows).
      </div>
    </div>
  );
}

function LegacyView({
  signals,
  timestamp,
  ensembleRecommendation,
}: {
  signals: BondMomentumSignal[];
  timestamp?: string;
  ensembleRecommendation?: BondMomentumEnsemble;
}) {
  const avgSignal = signals.reduce((sum, signal) => sum + signal.signal, 0) / signals.length;
  const activeSignals = signals.filter((signal) => signal.signal > 0);
  const activePct = (activeSignals.length / signals.length) * 100;
  const avgTone = avgSignal > 1 ? 'success' : avgSignal > 0.5 ? 'info' : 'muted';

  return (
    <div className="risk-card bond-momentum-card">
      <div className="risk-card-header">
        <div>
          <h3>Bond Momentum Overlay</h3>
          <p className="risk-subtitle">v3.30 - TSMOM-Style Fixed Income Signals</p>
        </div>
        {ensembleRecommendation && (
          <span className={`risk-badge risk-badge-${ensembleRecommendation.weight > 0 ? 'info' : 'muted'}`}>
            Ensemble: {ensembleRecommendation.weight}% weight
          </span>
        )}
      </div>

      <div className="risk-metric-grid risk-metric-grid-three">
        <div className="risk-metric">
          <span className="risk-label">Avg Signal</span>
          <span className={`risk-value compact risk-text-${avgTone}`}>{avgSignal.toFixed(2)}x</span>
        </div>
        <div className="risk-metric">
          <span className="risk-label">Active Signals</span>
          <span className="risk-value compact">{activePct.toFixed(0)}%</span>
          <small>{activeSignals.length}/{signals.length} ETFs</small>
        </div>
        <div className="risk-metric">
          <span className="risk-label">Last Update</span>
          <span className="risk-value small">{timestamp ? new Date(timestamp).toLocaleTimeString() : 'N/A'}</span>
        </div>
      </div>

      <div className="bond-signal-list">
        <h4>Individual ETF Signals</h4>
        {signals.map((signal) => {
          const config = ETF_CONFIG[signal.etf] || ETF_CONFIG.SHY;
          const confidence = confidenceTone(signal.confidence);
          const strengthTone = signalTone(signal.signal);

          return (
            <div key={signal.etf} className="bond-signal-card">
              <div className="bond-signal-header">
                <div className="bond-symbol-row">
                  <span className="bond-swatch" style={{ backgroundColor: config.color }} />
                  <strong>{config.name}</strong>
                  <span>{config.duration}</span>
                </div>
                <div className="risk-badge-row">
                  <span className={`risk-badge risk-badge-${confidence.tone}`}>
                    {confidence.label}
                  </span>
                  <span className={`risk-badge risk-badge-${actionTone(signal.action)}`}>
                    {signal.action.toUpperCase()}
                  </span>
                </div>
              </div>

              <div className="risk-gauge">
                <div className="risk-gauge-header">
                  <span>Signal Strength</span>
                  <strong>{signal.signal.toFixed(3)}x</strong>
                </div>
                <div className="risk-gauge-track">
                  <div
                    className={`risk-gauge-fill risk-gauge-fill-${strengthTone}`}
                    style={{ width: `${Math.min((signal.signal / 2) * 100, 100)}%` }}
                  />
                </div>
              </div>

              <div className="bond-metrics-grid">
                <div>
                  <span>Formation: </span>
                  <strong className={signal.formation_return > 0 ? 'positive' : 'negative'}>
                    {signal.formation_return > 0 ? '+' : ''}{signal.formation_return.toFixed(2)}%
                  </strong>
                </div>
                <div>
                  <span>Vol: </span>
                  <strong>{(signal.realized_vol * 100).toFixed(1)}%</strong>
                </div>
                <div>
                  <span>Position: </span>
                  <strong>{signal.position_size.toFixed(2)}x</strong>
                </div>
              </div>

              <p className="bond-signal-meta">
                {config.formationMonths}-month formation, {config.description.toLowerCase()}
              </p>
            </div>
          );
        })}
      </div>

      {ensembleRecommendation && (
        <div className="risk-note risk-note-info">
          <h4>Ensemble Integration</h4>
          <div className="bond-metrics-grid">
            <div>
              <span>Weight: </span>
              <strong>{ensembleRecommendation.weight}%</strong>
            </div>
            <div>
              <span>Confidence: </span>
              <strong>{ensembleRecommendation.confidence}</strong>
            </div>
          </div>
          <p>
            Recommended action: <strong>{ensembleRecommendation.action.toUpperCase()}</strong>
          </p>
        </div>
      )}

      <div className="risk-footnote">
        <strong>TSMOM-Style Bond Momentum:</strong> Uses time-series momentum with
        volatility-scaled position sizing. Long-only constraint appropriate for fixed
        income. Research shows stronger momentum effects in short-duration bonds
        (SHY) vs long-duration (TLT).
      </div>
    </div>
  );
}

export function BondMomentumPanel({
  data,
  signals,
  timestamp,
  ensembleRecommendation,
}: BondMomentumPanelProps) {
  const raw: unknown =
    data !== undefined && data !== null
      ? data
      : signals && signals.length > 0
        ? { signals, timestamp, ensemble: ensembleRecommendation }
        : data;

  const view = normalizeBondMomentumData(raw);

  if (!view) {
    // Present but unusable payloads must not look like "Loading…".
    if (isRecord(data) || (data === undefined && signals && signals.length === 0)) {
      const hasSomething =
        isRecord(data)
        || (Array.isArray(signals) && signals.length === 0 && data === undefined);
      if (hasSomething && isRecord(data) && Object.keys(data).length > 0) {
        return (
          <div className="risk-card risk-card-empty">
            <h3>Bond Duration Allocation</h3>
            <p>Bond momentum payload present but not in a supported shape.</p>
          </div>
        );
      }
    }
    return (
      <div className="risk-card risk-card-empty">
        <h3>Bond Duration Allocation</h3>
        <p>No bond duration recommendation available</p>
      </div>
    );
  }

  if (view.kind === 'summary') {
    return <SummaryView data={view.data} />;
  }

  return (
    <LegacyView
      signals={view.data.signals}
      timestamp={view.data.timestamp}
      ensembleRecommendation={view.data.ensemble}
    />
  );
}
