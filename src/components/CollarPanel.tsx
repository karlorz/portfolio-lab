import React from 'react';

interface CollarData {
  active: boolean;
  regime: string;
  call_strike: number;
  put_strike: number;
  net_premium: number;
  is_cashless: boolean;
  max_upside_pct: number;
  max_downside_pct: number;
  vix_level: number;
  confidence: number;
}

interface CollarPanelProps {
  data: CollarData | null;
  spyPrice?: number;
}

const REGIME_COLORS: Record<string, string> = {
  normal: '#10b981',
  elevated: '#f59e0b',
  stress: '#ef4444',
  crisis: '#dc2626',
};

export function CollarPanel({ data, spyPrice = 550 }: CollarPanelProps) {
  if (!data || !data.active) {
    return (
      <div className="panel">
        <h3>Cashless Collar (v4.60)</h3>
        <p className="muted">Collar inactive — {data?.regime || 'no data'}</p>
      </div>
    );
  }

  const callOtmPct = ((data.call_strike / spyPrice) - 1) * 100;
  const putOtmPct = ((spyPrice - data.put_strike) / spyPrice) * 100;
  const regimeColor = REGIME_COLORS[data.regime] || '#6b7280';

  return (
    <div className="panel">
      <h3>Cashless Collar (v4.60)</h3>
      <div className="panel-grid">
        <div className="metric">
          <span className="label">Regime</span>
          <span className="value" style={{ color: regimeColor }}>{data.regime.toUpperCase()}</span>
        </div>
        <div className="metric">
          <span className="label">VIX</span>
          <span className="value">{data.vix_level.toFixed(1)}</span>
        </div>
        <div className="metric">
          <span className="label">Call Strike</span>
          <span className="value">${data.call_strike.toFixed(0)} (+{callOtmPct.toFixed(1)}%)</span>
        </div>
        <div className="metric">
          <span className="label">Put Strike</span>
          <span className="value">${data.put_strike.toFixed(0)} (-{putOtmPct.toFixed(1)}%)</span>
        </div>
        <div className="metric">
          <span className="label">Net Premium</span>
          <span className="value" style={{ color: data.is_cashless ? '#10b981' : '#f59e0b' }}>
            ${data.net_premium.toFixed(2)} {data.is_cashless ? '✓ cashless' : ''}
          </span>
        </div>
        <div className="metric">
          <span className="label">Confidence</span>
          <span className="value">{data.confidence.toFixed(0)}%</span>
        </div>
      </div>
      <div className="collar-viz">
        <div className="bar">
          <div className="put-zone" style={{ width: `${putOtmPct * 3}%` }} />
          <div className="neutral-zone" style={{ width: `${(callOtmPct + putOtmPct) * 2}%` }} />
          <div className="call-zone" style={{ width: `${callOtmPct * 3}%` }} />
        </div>
        <div className="bar-labels">
          <span>${data.put_strike.toFixed(0)}</span>
          <span>${spyPrice.toFixed(0)}</span>
          <span>${data.call_strike.toFixed(0)}</span>
        </div>
      </div>
    </div>
  );
}
