import React from 'react';

export interface CryptoData {
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

interface CryptoAssetPresentation {
  sleeveShare: number;
  portfolioWeight: number;
  value: number;
}

export function buildCryptoPresentation(data: CryptoData, portfolioValue: number) {
  const sleeveTotal = data.btc_weight + data.eth_weight;
  const btcSleeveShare = sleeveTotal > 0 ? data.btc_weight / sleeveTotal : 0;
  const ethSleeveShare = sleeveTotal > 0 ? data.eth_weight / sleeveTotal : 0;
  const btcPortfolioWeight = data.total_crypto * btcSleeveShare;
  const ethPortfolioWeight = data.total_crypto * ethSleeveShare;

  return {
    totalValue: portfolioValue * data.total_crypto,
    btc: {
      sleeveShare: btcSleeveShare,
      portfolioWeight: btcPortfolioWeight,
      value: portfolioValue * btcPortfolioWeight,
    } satisfies CryptoAssetPresentation,
    eth: {
      sleeveShare: ethSleeveShare,
      portfolioWeight: ethPortfolioWeight,
      value: portfolioValue * ethPortfolioWeight,
    } satisfies CryptoAssetPresentation,
  };
}

function formatWeight(weight: number): string {
  return `${(weight * 100).toFixed(2)}%`;
}

function formatSleeveShare(weight: number): string {
  return `${(weight * 100).toFixed(1)}%`;
}

function formatDollars(value: number): string {
  return `$${value.toFixed(0)}`;
}

function formatMomentum(momentum: number): string {
  return `${momentum >= 0 ? '+' : ''}${(momentum * 100).toFixed(1)}%`;
}

export function CryptoAllocationPanel({ data, portfolioValue = 100000 }: CryptoAllocationPanelProps) {
  if (!data || !data.active) {
    return (
      <div className="panel signal-card crypto-allocation-card">
        <h3>Crypto Tactical (v4.70)</h3>
        <p className="muted">Crypto inactive — {data?.btc_vol_regime || 'no signal'}</p>
      </div>
    );
  }

  const presentation = buildCryptoPresentation(data, portfolioValue);

  return (
    <div className="panel signal-card crypto-allocation-card">
      <div className="signal-card-header">
        <h3>Crypto Tactical (v4.70)</h3>
        <span className="signal-status-pill signal-status-warning">Max 5%</span>
      </div>

      <div className="signal-card-hero compact">
        <div className="signal-hero-summary">
          <span className="label">Portfolio Allocation</span>
          <span className="value hero-value">{formatWeight(data.total_crypto)}</span>
          <span className="subtext">{formatDollars(presentation.totalValue)} funded from GLD</span>
        </div>
        <div className="signal-asset-stack">
          <div className="signal-asset-row">
            <div>
              <span className="asset-symbol">BTC</span>
              <span className="subtext">BTC sleeve {formatSleeveShare(presentation.btc.sleeveShare)}</span>
            </div>
            <div className="asset-values">
              <span className="value">{formatWeight(presentation.btc.portfolioWeight)}</span>
              <span className="subtext">{formatDollars(presentation.btc.value)}</span>
            </div>
          </div>
          <div className="signal-asset-row">
            <div>
              <span className="asset-symbol">ETH</span>
              <span className="subtext">ETH sleeve {formatSleeveShare(presentation.eth.sleeveShare)}</span>
            </div>
            <div className="asset-values">
              <span className="value">{formatWeight(presentation.eth.portfolioWeight)}</span>
              <span className="subtext">{formatDollars(presentation.eth.value)}</span>
            </div>
          </div>
        </div>
      </div>

      <div className="panel-grid signal-kpi-grid">
        <div className="metric">
          <span className="label">BTC 6m Mom</span>
          <span className="value" style={{ color: data.btc_momentum_6m > 0 ? '#10b981' : '#ef4444' }}>
            {formatMomentum(data.btc_momentum_6m)}
          </span>
        </div>
        <div className="metric">
          <span className="label">ETH 6m Mom</span>
          <span className="value" style={{ color: data.eth_momentum_6m > 0 ? '#10b981' : '#ef4444' }}>
            {formatMomentum(data.eth_momentum_6m)}
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
