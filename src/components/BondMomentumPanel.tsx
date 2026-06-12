import React from 'react';
import type { BondMomentumSignal } from '../types/live';

interface BondMomentumPanelProps {
  signals: BondMomentumSignal[];
  timestamp?: string;
  ensembleRecommendation?: {
    weight: number;
    confidence: string;
    action: string;
  };
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

export function BondMomentumPanel({
  signals,
  timestamp,
  ensembleRecommendation,
}: BondMomentumPanelProps) {
  if (!signals || signals.length === 0) {
    return (
      <div className="risk-card risk-card-empty">
        <h3>Bond Momentum Overlay (v3.30)</h3>
        <p>Loading bond momentum signals...</p>
      </div>
    );
  }

  const avgSignal = signals.reduce((sum, signal) => sum + signal.signal, 0) / signals.length;
  const activeSignals = signals.filter(signal => signal.signal > 0);
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
