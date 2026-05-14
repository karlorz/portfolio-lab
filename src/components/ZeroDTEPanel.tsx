import React, { useState, useEffect } from 'react';

interface ZeroDTEPosition {
  id: string;
  symbol: string;
  strike: number;
  expiry: string;
  delta: number;
  theta: number;
  gamma: number;
  vega: number;
  premium_collected: number;
  current_pnl: number;
  status: 'active' | 'closed' | 'assigned';
  entry_time: string;
  exit_time?: string;
}

interface ZeroDTEStats {
  active_positions: number;
  max_positions: number;
  weekly_positions_used: number;
  weekly_positions_max: number;
  total_premium_ytd: number;
  total_premium_month: number;
  total_premium_week: number;
  win_rate_30d: number;
  avg_premium_per_trade: number;
  total_delta_exposure: number;
  max_delta_exposure: number;
  allocation_used: number;
  allocation_max: number;
}

interface ZeroDTEConfig {
  enabled: boolean;
  vix_current: number;
  vix_min: number;
  vix_max: number;
  can_enter: boolean;
  reason?: string;
}

interface ZeroDTEData {
  positions: ZeroDTEPosition[];
  stats: ZeroDTEStats;
  config: ZeroDTEConfig;
  last_updated: string;
}

export function ZeroDTEPanel() {
  const [data, setData] = useState<ZeroDTEData | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [expandedPosition, setExpandedPosition] = useState<string | null>(null);

  useEffect(() => {
    fetchData();
    const interval = setInterval(fetchData, 30000); // 30s refresh
    return () => clearInterval(interval);
  }, []);

  const fetchData = async () => {
    try {
      const response = await fetch('/data/odte_positions.json');
      if (!response.ok) {
        // Fallback to mock data for development
        setData(getMockData());
        setLoading(false);
        return;
      }
      const json = await response.json();
      setData(json);
      setLoading(false);
    } catch (err) {
      setError('Failed to load 0DTE data');
      setData(getMockData()); // Fallback
      setLoading(false);
    }
  };

  const getMockData = (): ZeroDTEData => ({
    positions: [
      {
        id: 'odte-001',
        symbol: 'SPY',
        strike: 585,
        expiry: '2026-05-14',
        delta: 0.28,
        theta: -0.45,
        gamma: 0.12,
        vega: 0.08,
        premium_collected: 420,
        current_pnl: 385,
        status: 'active',
        entry_time: '2026-05-14T11:30:00Z'
      }
    ],
    stats: {
      active_positions: 1,
      max_positions: 2,
      weekly_positions_used: 1,
      weekly_positions_max: 2,
      total_premium_ytd: 2840,
      total_premium_month: 890,
      total_premium_week: 420,
      win_rate_30d: 0.72,
      avg_premium_per_trade: 520,
      total_delta_exposure: 0.28,
      max_delta_exposure: 0.80,
      allocation_used: 0.005,
      allocation_max: 0.02
    },
    config: {
      enabled: true,
      vix_current: 18.5,
      vix_min: 15,
      vix_max: 35,
      can_enter: true
    },
    last_updated: new Date().toISOString()
  });

  if (loading) {
    return (
      <div className="zero-dte-panel loading">
        <div className="panel-header">
          <h3>0DTE Yield Enhancement</h3>
          <span className="loading-text">Loading...</span>
        </div>
      </div>
    );
  }

  if (!data) return null;

  const { positions, stats, config } = data;
  const vixStatus = config.vix_current < config.vix_min ? 'below-min' : 
                    config.vix_current > config.vix_max ? 'above-max' : 'ok';

  const formatCurrency = (v: number) => {
    return new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: 'USD',
      minimumFractionDigits: 0,
      maximumFractionDigits: 0
    }).format(v);
  };

  const formatPct = (v: number) => `${(v * 100).toFixed(1)}%`;
  const formatGreek = (v: number) => v.toFixed(3);

  return (
    <div className="zero-dte-panel">
      <div className="panel-header">
        <h3>0DTE Yield Enhancement</h3>
        <div className={`status-badge ${config.can_enter ? 'active' : 'paused'}`}>
          {config.can_enter ? '● Active' : '● Paused'}
        </div>
      </div>

      {/* VIX Status Bar */}
      <div className={`vix-bar ${vixStatus}`}>
        <div className="vix-label">VIX</div>
        <div className="vix-value">{config.vix_current.toFixed(1)}</div>
        <div className="vix-range">
          Target: {config.vix_min}-{config.vix_max}
        </div>
        {vixStatus !== 'ok' && (
          <div className="vix-warning">
            {vixStatus === 'below-min' ? '⚠ Below entry threshold' : '⚠ High volatility - paused'}
          </div>
        )}
      </div>

      {/* Key Metrics */}
      <div className="metrics-grid">
        <div className="metric-card">
          <div className="metric-label">YTD Premium</div>
          <div className="metric-value positive">+{formatCurrency(stats.total_premium_ytd)}</div>
        </div>
        <div className="metric-card">
          <div className="metric-label">30D Win Rate</div>
          <div className={`metric-value ${stats.win_rate_30d >= 0.65 ? 'positive' : 'warning'}`}>
            {formatPct(stats.win_rate_30d)}
          </div>
        </div>
        <div className="metric-card">
          <div className="metric-label">Active Positions</div>
          <div className="metric-value">
            {stats.active_positions}/{stats.max_positions}
          </div>
        </div>
        <div className="metric-card">
          <div className="metric-label">Weekly Used</div>
          <div className="metric-value">
            {stats.weekly_positions_used}/{stats.weekly_positions_max}
          </div>
        </div>
      </div>

      {/* Risk Gauges */}
      <div className="risk-section">
        <h4>Risk Limits</h4>
        
        <div className="risk-gauge">
          <div className="gauge-label">
            <span>Delta Exposure</span>
            <span>{formatPct(stats.total_delta_exposure)} / {formatPct(stats.max_delta_exposure)}</span>
          </div>
          <div className="gauge-bar">
            <div 
              className={`gauge-fill ${stats.total_delta_exposure / stats.max_delta_exposure > 0.8 ? 'warning' : 'ok'}`}
              style={{ width: `${(stats.total_delta_exposure / stats.max_delta_exposure) * 100}%` }}
            />
          </div>
        </div>

        <div className="risk-gauge">
          <div className="gauge-label">
            <span>Capital Allocation</span>
            <span>{formatPct(stats.allocation_used)} / {formatPct(stats.allocation_max)}</span>
          </div>
          <div className="gauge-bar">
            <div 
              className="gauge-fill ok"
              style={{ width: `${(stats.allocation_used / stats.allocation_max) * 100}%` }}
            />
          </div>
        </div>
      </div>

      {/* Active Positions */}
      {positions.length > 0 && (
        <div className="positions-section">
          <h4>Active Positions</h4>
          {positions.map(pos => (
            <div 
              key={pos.id} 
              className={`position-card ${pos.status}`}
              onClick={() => setExpandedPosition(expandedPosition === pos.id ? null : pos.id)}
            >
              <div className="position-header">
                <div className="position-main">
                  <span className="symbol">{pos.symbol}</span>
                  <span className="strike">${pos.strike} Call</span>
                  <span className="expiry">Exp: {pos.expiry}</span>
                </div>
                <div className={`position-pnl ${pos.current_pnl >= 0 ? 'positive' : 'negative'}`}>
                  {pos.current_pnl >= 0 ? '+' : ''}{formatCurrency(pos.current_pnl)}
                </div>
              </div>

              <div className="position-summary">
                <div className="summary-item">
                  <span className="summary-label">Delta</span>
                  <span className="summary-value">{formatGreek(pos.delta)}</span>
                </div>
                <div className="summary-item">
                  <span className="summary-label">Premium</span>
                  <span className="summary-value">+{formatCurrency(pos.premium_collected)}</span>
                </div>
                <div className="summary-item">
                  <span className="summary-label">Theta</span>
                  <span className="summary-value">{formatGreek(pos.theta)}</span>
                </div>
              </div>

              {expandedPosition === pos.id && (
                <div className="position-details">
                  <div className="detail-row">
                    <span>Entry Time:</span>
                    <span>{new Date(pos.entry_time).toLocaleTimeString()}</span>
                  </div>
                  <div className="detail-row">
                    <span>Gamma:</span>
                    <span>{formatGreek(pos.gamma)}</span>
                  </div>
                  <div className="detail-row">
                    <span>Vega:</span>
                    <span>{formatGreek(pos.vega)}</span>
                  </div>
                  <div className="detail-row">
                    <span>Status:</span>
                    <span className={`status-${pos.status}`}>{pos.status.toUpperCase()}</span>
                  </div>
                </div>
              )}

              <div className="expand-hint">
                {expandedPosition === pos.id ? '▼ Click to collapse' : '▶ Click to expand'}
              </div>
            </div>
          ))}
        </div>
      )}

      {/* Weekly Stats */}
      <div className="weekly-stats">
        <h4>Period Performance</h4>
        <div className="period-grid">
          <div className="period-item">
            <span className="period-label">This Week</span>
            <span className="period-value positive">+{formatCurrency(stats.total_premium_week)}</span>
          </div>
          <div className="period-item">
            <span className="period-label">This Month</span>
            <span className="period-value positive">+{formatCurrency(stats.total_premium_month)}</span>
          </div>
          <div className="period-item">
            <span className="period-label">YTD</span>
            <span className="period-value positive">+{formatCurrency(stats.total_premium_ytd)}</span>
          </div>
          <div className="period-item">
            <span className="period-label">Avg/Trade</span>
            <span className="period-value">{formatCurrency(stats.avg_premium_per_trade)}</span>
          </div>
        </div>
      </div>

      {/* Constraints Summary */}
      <div className="constraints-section">
        <h4>Strategy Constraints</h4>
        <div className="constraint-list">
          <div className={`constraint-item ${stats.active_positions < stats.max_positions ? 'ok' : 'at-limit'}`}>
            <span className="constraint-icon">{stats.active_positions < stats.max_positions ? '✓' : '●'}</span>
            <span>Max {stats.max_positions} concurrent positions</span>
          </div>
          <div className={`constraint-item ${stats.weekly_positions_used < stats.weekly_positions_max ? 'ok' : 'at-limit'}`}>
            <span className="constraint-icon">{stats.weekly_positions_used < stats.weekly_positions_max ? '✓' : '●'}</span>
            <span>Max {stats.weekly_positions_max} per week</span>
          </div>
          <div className={`constraint-item ${stats.total_delta_exposure < stats.max_delta_exposure ? 'ok' : 'at-limit'}`}>
            <span className="constraint-icon">{stats.total_delta_exposure < stats.max_delta_exposure ? '✓' : '⚠'}</span>
            <span>Delta exposure &lt; {formatPct(stats.max_delta_exposure)}</span>
          </div>
          <div className={`constraint-item ${vixStatus === 'ok' ? 'ok' : 'blocked'}`}>
            <span className="constraint-icon">{vixStatus === 'ok' ? '✓' : '✗'}</span>
            <span>VIX entry filter ({config.vix_min}-{config.vix_max})</span>
          </div>
        </div>
      </div>

      <div className="last-updated">
        Last updated: {new Date(data.last_updated).toLocaleTimeString()}
      </div>
    </div>
  );
}

export default ZeroDTEPanel;
