import React from 'react';
import type { ZeroDTEPosition, ZeroDTEConfig, ZeroDTETrade } from '../types/live';

interface ZeroDTEPanelProps {
  positions: ZeroDTEPosition[];
  config: ZeroDTEConfig | null;
  portfolioValue: number;
  vix: number | null;
  weeklyLimitRemaining: number;
}

export function ZeroDTEPanel({ 
  positions, 
  config, 
  portfolioValue, 
  vix,
  weeklyLimitRemaining 
}: ZeroDTEPanelProps) {
  const formatCurrency = (v: number) => {
    return new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: 'USD',
      minimumFractionDigits: 0,
      maximumFractionDigits: 0
    }).format(v);
  };

  const formatPct = (v: number) => `${(v * 100).toFixed(2)}%`;
  const formatGreek = (v: number) => v.toFixed(3);

  // Calculate metrics
  const totalDeltaExposure = positions.reduce((sum, pos) => 
    sum + (pos.delta_exposure || 0), 0);
  const totalPremiumCollected = positions.reduce((sum, pos) => 
    sum + (pos.premium_collected || 0), 0);
  const totalUnrealizedPnl = positions.reduce((sum, pos) => 
    sum + (pos.unrealized_pnl || 0), 0);
  const totalAllocated = positions.reduce((sum, pos) => 
    sum + (pos.notional_value || 0), 0);

  const maxAllocation = portfolioValue * (config?.max_portfolio_allocation || 0.02);
  const allocationPct = totalAllocated / portfolioValue;
  const maxDeltaExposure = portfolioValue * (config?.max_delta_exposure || 0.08);

  // Entry status
  const canEnterNew = (
    (vix === null || (vix >= (config?.min_vix || 15) && vix <= (config?.max_vix || 35))) &&
    weeklyLimitRemaining > 0 &&
    totalDeltaExposure < maxDeltaExposure &&
    totalAllocated < maxAllocation
  );

  const entryStatus = canEnterNew ? 'ready' : 
    (vix !== null && (vix < (config?.min_vix || 15) || vix > (config?.max_vix || 35))) ? 'vix-blocked' :
    weeklyLimitRemaining === 0 ? 'limit-reached' :
    totalDeltaExposure >= maxDeltaExposure ? 'delta-limit' : 'allocation-limit';

  return (
    <div className="zero-dte-panel">
      <div className="panel-header">
        <h3>0DTE Yield Enhancement</h3>
        <span className={`entry-badge ${entryStatus}`}>
          {entryStatus === 'ready' ? '● Ready' :
           entryStatus === 'vix-blocked' ? `VIX ${vix?.toFixed(1)}` :
           entryStatus === 'limit-reached' ? 'Weekly Limit' :
           entryStatus === 'delta-limit' ? 'Delta Limit' : 'Allocation Limit'}
        </span>
      </div>

      {/* Risk Summary */}
      <div className="risk-summary">
        <div className="risk-gauge">
          <div className="gauge-row">
            <label>Delta Exposure</label>
            <div className="gauge-bar">
              <div 
                className={`gauge-fill ${totalDeltaExposure / maxDeltaExposure > 0.8 ? 'warning' : ''}`}
                style={{ width: `${Math.min((totalDeltaExposure / maxDeltaExposure) * 100, 100)}%` }}
              />
            </div>
            <span className="gauge-value">
              {formatPct(totalDeltaExposure / portfolioValue)} / {formatPct(config?.max_delta_exposure || 0.08)}
            </span>
          </div>
          
          <div className="gauge-row">
            <label>Allocation</label>
            <div className="gauge-bar">
              <div 
                className={`gauge-fill ${allocationPct > 0.015 ? 'warning' : ''}`}
                style={{ width: `${Math.min((allocationPct / (config?.max_portfolio_allocation || 0.02)) * 100, 100)}%` }}
              />
            </div>
            <span className="gauge-value">
              {formatPct(allocationPct)} / {formatPct(config?.max_portfolio_allocation || 0.02)}
            </span>
          </div>

          <div className="gauge-row">
            <label>Weekly Trades</label>
            <div className="gauge-bar">
              <div 
                className="gauge-fill"
                style={{ width: `${((config?.max_weekly_positions || 2) - weeklyLimitRemaining) / (config?.max_weekly_positions || 2) * 100}%` }}
              />
            </div>
            <span className="gauge-value">
              {(config?.max_weekly_positions || 2) - weeklyLimitRemaining} / {config?.max_weekly_positions || 2}
            </span>
          </div>
        </div>

        {/* Premium Summary */}
        <div className="premium-summary">
          <div className="premium-stat">
            <label>Premium Collected (MTD)</label>
            <span className="positive">+{formatCurrency(totalPremiumCollected)}</span>
          </div>
          <div className={`premium-stat ${totalUnrealizedPnl >= 0 ? 'positive' : 'negative'}`}>
            <label>Unrealized P&L</label>
            <span>{totalUnrealizedPnl >= 0 ? '+' : ''}{formatCurrency(totalUnrealizedPnl)}</span>
          </div>
        </div>
      </div>

      {/* Active Positions */}
      {positions.length > 0 ? (
        <div className="positions-table-container">
          <h4>Active Positions ({positions.length})</h4>
          <table className="positions-table detailed">
            <thead>
              <tr>
                <th>Symbol</th>
                <th>Type</th>
                <th>Strike</th>
                <th>Expiry</th>
                <th>Delta</th>
                <th>Theta</th>
                <th>Premium</th>
                <th>P&L</th>
                <th>Status</th>
              </tr>
            </thead>
            <tbody>
              {positions.map((pos, i) => (
                <tr key={i} className={pos.status}>
                  <td><strong>{pos.underlying}</strong></td>
                  <td>{pos.option_type === 'call' ? 'C' : 'P'}</td>
                  <td>{pos.strike.toFixed(1)}</td>
                  <td>{new Date(pos.expiration).toLocaleDateString()}</td>
                  <td>{formatGreek(pos.current_delta || 0)}</td>
                  <td>{formatGreek(pos.current_theta || 0)}</td>
                  <td className="positive">+{formatCurrency(pos.premium_collected || 0)}</td>
                  <td className={pos.unrealized_pnl && pos.unrealized_pnl >= 0 ? 'positive' : 'negative'}>
                    {pos.unrealized_pnl && pos.unrealized_pnl >= 0 ? '+' : ''}
                    {formatCurrency(pos.unrealized_pnl || 0)}
                  </td>
                  <td>
                    <span className={`status-badge ${pos.status}`}>
                      {pos.status === 'open' ? 'Open' :
                       pos.status === 'pending' ? 'Pending' :
                       pos.status === 'stopped' ? 'Stopped' : pos.status}
                    </span>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      ) : (
        <div className="no-positions">
          <p>No active 0DTE positions</p>
          {entryStatus === 'ready' && (
            <small>VIX in range, ready for next entry opportunity</small>
          )}
        </div>
      )}

      {/* Risk Controls */}
      <div className="risk-controls">
        <h4>Risk Controls</h4>
        <div className="control-grid">
          <div className="control-item">
            <label>Max Loss/Trade</label>
            <span>{formatPct(config?.max_loss_pct || 0.015)}</span>
          </div>
          <div className="control-item">
            <label>Delta Target</label>
            <span>{formatPct(config?.delta_target || 0.30)}</span>
          </div>
          <div className="control-item">
            <label>Emergency Delta</label>
            <span>{formatPct(config?.emergency_close_delta || 0.50)}</span>
          </div>
          <div className="control-item">
            <label>Min Premium</label>
            <span>{formatPct(config?.min_premium_pct || 0.004)}</span>
          </div>
        </div>
      </div>

      {/* VIX Indicator */}
      {vix !== null && (
        <div className="vix-indicator">
          <div className={`vix-value ${vix < 15 ? 'low' : vix > 35 ? 'high' : 'normal'}`}>
            <label>VIX</label>
            <span>{vix.toFixed(1)}</span>
          </div>
          <div className="vix-range">
            <span className={vix < 15 ? 'active' : ''}>Below 15: No Entry</span>
            <span className={vix >= 15 && vix <= 35 ? 'active' : ''}>15-35: Active</span>
            <span className={vix > 35 ? 'active' : ''}>Above 35: Pause</span>
          </div>
        </div>
      )}
    </div>
  );
}
