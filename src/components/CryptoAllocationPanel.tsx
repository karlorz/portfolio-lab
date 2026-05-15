import React from 'react';

interface CryptoData {
  active: boolean;
  btc_weight: number;
  eth_weight: number;
  total_crypto: number;
  btc_momentum_6m: number;
  eth_momentum_6m: number;
  btc_vol_regime: string;
  eth_vol_regime: string;
  confidence: number;
}

interface CryptoAllocationPanelProps {
  data: CryptoData | null;
  portfolioValue?: number;
}

const VOL_COLORS: Record<string, string> = {
  low: '#10b981',
  normal: '#3b82f6',
  high: '#f59e0b',
  extreme: '#ef4444',
};

export function CryptoAllocationPanel({ data, portfolioValue = 100000 }: CryptoAllocationPanelProps) {
  if (!data || !data.active) {
    return (
      <div className="panel">
        <h3>Crypto Tactical (v4.70)</h3>
        <p className="muted">Crypto inactive — {data?.btc_vol_regime || 'no signal'}</p>
      </div>
    );
  }

  const btcValue = portfolioValue * data.btc_weight;
  const ethValue = portfolioValue * data.eth_weight;
  const totalValue = portfolioValue * data.total_crypto;

  return (
    <div className="panel">
      <h3>Crypto Tactical (v4.70)</h3>
      <div className="panel-grid">
        <div className="metric">
          <span className="label">Total Crypto</span>
          <span className="value">{data.total_crypto.toFixed(2)}% (${totalValue.toFixed(0)})</span>
        </div>
        <div className="metric">
          <span className="label">BTC</span>
          <span className="value">
            {data.btc_weight.toFixed(2)}% — ${btcValue.toFixed(0)}
          </span>
        </div>
        <div className="metric">
          <span className="label">ETH</span>
          <span className="value">
            {data.eth_weight.toFixed(2)}% — ${ethValue.toFixed(0)}
          </span>
        </div>
        <div className="metric">
          <span className="label">BTC 6m Mom</span>
          <span className="value" style={{ color: data.btc_momentum_6m > 0 ? '#10b981' : '#ef4444' }}>
            {data.btc_momentum_6m >= 0 ? '+' : ''}{data.btc_momentum_6m.toFixed(1)}%
          </span>
        </div>
        <div className="metric">
          <span className="label">ETH 6m Mom</span>
          <span className="value" style={{ color: data.eth_momentum_6m > 0 ? '#10b981' : '#ef4444' }}>
            {data.eth_momentum_6m >= 0 ? '+' : ''}{data.eth_momentum_6m.toFixed(1)}%
          </span>
        </div>
        <div className="metric">
          <span className="label">BTC Vol</span>
          <span className="value" style={{ color: VOL_COLORS[data.btc_vol_regime] || '#6b7280' }}>
            {data.btc_vol_regime.toUpperCase()}
          </span>
        </div>
        <div className="metric">
          <span className="label">ETH Vol</span>
          <span className="value" style={{ color: VOL_COLORS[data.eth_vol_regime] || '#6b7280' }}>
            {data.eth_vol_regime.toUpperCase()}
          </span>
        </div>
        <div className="metric">
          <span className="label">Confidence</span>
          <span className="value">{data.confidence.toFixed(0)}%</span>
        </div>
      </div>
    </div>
  );
}
