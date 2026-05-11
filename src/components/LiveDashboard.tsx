import React, { useState, useEffect, useMemo } from 'react';
import { HealthPanel } from './HealthPanel';
import { RegimeTimeline } from './RegimeTimeline';
import { SPYComparisonChart } from './SPYComparisonChart';
import { RebalancePanel } from './RebalancePanel';
import { UnderwaterChart, RollingMetricsChart, CrisisOverlay } from './AnalyticsCharts';
import type { SignalsData, PerformanceEntry, Alert, AssetStat, DashboardData, HealthData, StatsData, AnalyticsData } from '../types/live';

interface LiveDashboardProps {
  refreshInterval?: number; // seconds
}

type TabType = 'overview' | 'health' | 'history' | 'performance' | 'rebalance' | 'analytics';

export function LiveDashboard({ refreshInterval = 60 }: LiveDashboardProps) {
  const [activeTab, setActiveTab] = useState<TabType>('overview');
  const [signals, setSignals] = useState<SignalsData | null>(null);
  const [performance, setPerformance] = useState<PerformanceEntry[]>([]);
  const [alerts, setAlerts] = useState<Alert[]>([]);
  const [stats, setStats] = useState<StatsData | null>(null);
  const [dashboard, setDashboard] = useState<DashboardData | null>(null);
  const [health, setHealth] = useState<HealthData | null>(null);
  const [analytics, setAnalytics] = useState<AnalyticsData | null>(null);
  const [lastUpdate, setLastUpdate] = useState<string>('');
  const [error, setError] = useState<string | null>(null);
  const [expandedHealth, setExpandedHealth] = useState(false);

  const fetchData = async () => {
    try {
      const [signalsRes, perfRes, alertsRes, statsRes, dashboardRes, healthRes, analyticsRes] = await Promise.all([
        fetch('/data/signals.json'),
        fetch('/data/dashboard.json'),
        fetch('/data/alerts.json'),
        fetch('/data/stats.json'),
        fetch('/data/dashboard.json'),
        fetch('/data/health.json'),
        fetch('/data/analytics.json')
      ]);

      if (signalsRes.ok) {
        const s = await signalsRes.json();
        setSignals(s);
        setLastUpdate(new Date(s.generated_at).toLocaleTimeString());
      }
      if (perfRes.ok) {
        const p = await perfRes.json();
        setPerformance(p.paper_portfolio || []);
      }
      if (alertsRes.ok) {
        const a = await alertsRes.json();
        setAlerts(a.alerts || []);
      }
      if (statsRes.ok) {
        const st = await statsRes.json();
        setStats(st);
      }
      if (dashboardRes.ok) {
        const d = await dashboardRes.json();
        setDashboard(d);
      }
      if (healthRes.ok) {
        const h = await healthRes.json();
        setHealth(h);
      }
      if (analyticsRes.ok) {
        const an = await analyticsRes.json();
        setAnalytics(an);
      }

      setError(null);
    } catch (err) {
      setError('Failed to load live data');
    }
  };

  useEffect(() => {
    fetchData();
    const interval = setInterval(fetchData, refreshInterval * 1000);
    return () => clearInterval(interval);
  }, [refreshInterval]);

  const portfolioValue = useMemo(() => {
    return signals?.portfolio?.total_value || 100000;
  }, [signals]);

  const regimeColor = useMemo(() => {
    const r = signals?.regime?.regime;
    switch (r) {
      case 'crisis': return '#ef4444';
      case 'vol_spike': return '#f59e0b';
      case 'low_vol': return '#10b981';
      default: return '#3b82f6';
    }
  }, [signals]);

  const formatCurrency = (v: number) => {
    return new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: 'USD',
      minimumFractionDigits: 0,
      maximumFractionDigits: 0
    }).format(v);
  };

  const formatPct = (v: number) => `${(v * 100).toFixed(1)}%`;

  const criticalAlerts = alerts.filter(a => a.level === 'error' || a.requires_action);
  const warningAlerts = alerts.filter(a => a.level === 'warning' && !a.requires_action);

  const tabs: { id: TabType; label: string; badge?: number }[] = [
    { id: 'overview', label: 'Overview', badge: criticalAlerts.length || undefined },
    { id: 'health', label: 'Health', badge: health?.system_status === 'critical' ? 1 : undefined },
    { id: 'history', label: 'History' },
    { id: 'performance', label: 'Performance' },
    { id: 'rebalance', label: 'Rebalance' },
    { id: 'analytics', label: 'Analytics' }
  ];

  return (
    <div className="live-dashboard">
      {/* Header */}
      <div className="dashboard-header">
        <div className="header-main">
          <h2>Live Paper Trading</h2>
          <div className="status-bar">
            <span
              className="regime-badge"
              style={{ backgroundColor: regimeColor }}
            >
              {signals?.regime?.regime?.toUpperCase() || 'LOADING'}
            </span>
            <span className="last-update">Updated: {lastUpdate || 'Never'}</span>
            {error && <span className="error">{error}</span>}
          </div>
        </div>

        {/* Health Summary (always visible) */}
        {health && (
          <div 
            className={`health-summary-bar status-${health.system_status}`}
            onClick={() => setActiveTab('health')}
          >
            <span className="health-indicator"></span>
            <span className="health-text">
              System: {health.system_status}
              {health.cron_jobs.length > 0 && ` • ${health.cron_jobs.length} jobs`}
            </span>
          </div>
        )}
      </div>

      {/* Tabs */}
      <div className="dashboard-tabs">
        {tabs.map(tab => (
          <button
            key={tab.id}
            className={`tab ${activeTab === tab.id ? 'active' : ''}`}
            onClick={() => setActiveTab(tab.id)}
          >
            {tab.label}
            {tab.badge !== undefined && tab.badge > 0 && (
              <span className="tab-badge">{tab.badge}</span>
            )}
          </button>
        ))}
      </div>

      {/* Tab Content */}
      <div className="tab-content">
        {/* Overview Tab */}
        {activeTab === 'overview' && (
          <div className="tab-panel overview-panel">
            {/* Critical Alerts */}
            {criticalAlerts.length > 0 && (
              <div className="alerts-section critical">
                {criticalAlerts.slice(0, 3).map((alert, i) => (
                  <div key={i} className={`alert alert-${alert.level}`}>
                    <strong>{alert.title}</strong>
                    <span>{alert.message}</span>
                    {alert.requires_action && (
                      <span className="action-required">ACTION REQUIRED</span>
                    )}
                  </div>
                ))}
              </div>
            )}

            {/* Warning Alerts */}
            {warningAlerts.length > 0 && (
              <div className="alerts-section warnings">
                <details>
                  <summary>{warningAlerts.length} warnings</summary>
                  {warningAlerts.slice(0, 5).map((alert, i) => (
                    <div key={i} className={`alert alert-${alert.level}`}>
                      <strong>{alert.title}</strong>
                      <span>{alert.message}</span>
                    </div>
                  ))}
                </details>
              </div>
            )}

            {/* Portfolio Summary */}
            <div className="metrics-grid">
              <div className="metric-card primary">
                <label>Portfolio Value</label>
                <span className="value-display">{formatCurrency(portfolioValue)}</span>
                {signals?.portfolio?.cash && (
                  <small>Cash: {formatCurrency(signals.portfolio.cash)}</small>
                )}
              </div>

              <div className="metric-card">
                <label>Regime</label>
                <span className="value-display" style={{ color: regimeColor }}>
                  {signals?.regime?.regime?.toUpperCase()}
                </span>
                {signals?.regime?.vix && (
                  <small>VIX: {signals.regime.vix.toFixed(1)}</small>
                )}
              </div>

              <div className="metric-card">
                <label>Target Allocation</label>
                <div className="alloc-preview">
                  {signals?.target_allocation && Object.entries(signals.target_allocation)
                    .map(([sym, weight]) => (
                      <span key={sym} className="alloc-tag">
                        {sym}: {formatPct(weight as number)}
                      </span>
                    ))
                  }
                </div>
              </div>
            </div>

            {/* Positions & Orders */}
            <div className="positions-orders-row">
              {signals?.portfolio?.positions && signals.portfolio.positions.length > 0 && (
                <div className="positions-section">
                  <h3>Current Positions</h3>
                  <table className="positions-table">
                    <thead>
                      <tr>
                        <th>Symbol</th>
                        <th>Shares</th>
                        <th>Value</th>
                        <th>Weight</th>
                        <th>P&L</th>
                      </tr>
                    </thead>
                    <tbody>
                      {signals.portfolio.positions.map((pos) => (
                        <tr key={pos.symbol}>
                          <td><strong>{pos.symbol}</strong></td>
                          <td>{pos.shares.toFixed(2)}</td>
                          <td>{formatCurrency(pos.value)}</td>
                          <td>{formatPct(pos.weight)}</td>
                          <td className={pos.unrealized >= 0 ? 'positive' : 'negative'}>
                            {formatCurrency(pos.unrealized)}
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              )}

              {signals?.recent_orders && signals.recent_orders.length > 0 && (
                <div className="orders-section">
                  <h3>Recent Orders</h3>
                  <table className="orders-table">
                    <thead>
                      <tr>
                        <th>Symbol</th>
                        <th>Side</th>
                        <th>Shares</th>
                        <th>Value</th>
                      </tr>
                    </thead>
                    <tbody>
                      {signals.recent_orders.map((order, i) => (
                        <tr key={i}>
                          <td><strong>{order.sym}</strong></td>
                          <td className={order.side === 'buy' ? 'positive' : 'negative'}>
                            {order.side.toUpperCase()}
                          </td>
                          <td>{order.shares.toFixed(2)}</td>
                          <td>{formatCurrency(order.value)}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              )}
            </div>
          </div>
        )}

        {/* Health Tab */}
        {activeTab === 'health' && (
          <div className="tab-panel health-panel-container">
            <HealthPanel 
              health={health}
              expanded={expandedHealth}
              onToggleExpand={() => setExpandedHealth(!expandedHealth)}
            />
          </div>
        )}

        {/* History Tab */}
        {activeTab === 'history' && (
          <div className="tab-panel history-panel">
            <RegimeTimeline history={dashboard?.regimes || []} />
          </div>
        )}

        {/* Performance Tab */}
        {activeTab === 'performance' && (
          <div className="tab-panel performance-panel">
            {/* SPY Comparison Chart */}
            <SPYComparisonChart stats={stats} performance={performance} />

            {/* Paper Portfolio Stats */}
            {performance.length > 0 && (
              <div className="performance-summary">
                <div className="perf-card">
                  <label>Current Value</label>
                  <span className="value-display">
                    {formatCurrency(performance[performance.length - 1]?.v || 100000)}
                  </span>
                </div>
                <div className="perf-card">
                  <label>Start Value</label>
                  <span className="value-display">
                    {formatCurrency(performance[0]?.v || 100000)}
                  </span>
                </div>
                <div className="perf-card">
                  <label>Days Tracked</label>
                  <span className="value-display">{performance.length}</span>
                </div>
                <div className="perf-card">
                  <label>Total Return</label>
                  <span className={`value-display ${
                    ((performance[performance.length - 1]?.v || 100000) - 100000) >= 0 ? 'positive' : 'negative'
                  }`}>
                    {formatCurrency((performance[performance.length - 1]?.v || 100000) - 100000)}
                  </span>
                </div>
              </div>
            )}

            {/* Asset Stats */}
            {stats?.asset_stats && Object.keys(stats.asset_stats).length > 0 && (
              <div className="stats-section">
                <h3>Market Overview (30d)</h3>
                <div className="stats-grid">
                  {Object.entries(stats.asset_stats).map(([sym, stat]) => (
                    <div key={sym} className="stat-card">
                      <h4>{sym}</h4>
                      <div className="stat-value">
                        <span className={stat['30d_return'] >= 0 ? 'positive' : 'negative'}>
                          {stat['30d_return'] >= 0 ? '+' : ''}{stat['30d_return'].toFixed(1)}%
                        </span>
                        <small>30d return</small>
                      </div>
                      <div className="stat-vol">
                        <span>Vol: {stat.volatility.toFixed(1)}%</span>
                        <small>${stat.current.toFixed(2)}</small>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        )}

        {/* Rebalance Tab */}
        {activeTab === 'rebalance' && (
          <div className="tab-panel rebalance-panel-container">
            <RebalancePanel 
              signals={signals}
              readOnly={true}
              onRebalanceRequest={() => {
                // In paper mode, rebalancing is automatic via cron
                // This would trigger manual rebalance in future live mode
                console.log('Manual rebalance requested');
              }}
            />
          </div>
        )}

        {/* Analytics Tab */}
        {activeTab === 'analytics' && (
          <div className="tab-panel analytics-panel">
            {analytics?.status === 'success' ? (
              <>
                {/* Underwater Chart - Drawdown */}
                {analytics.drawdown?.series?.length > 0 && (
                  <UnderwaterChart 
                    series={analytics.drawdown.series}
                    maxDrawdown={analytics.drawdown.max_drawdown}
                  />
                )}

                {/* Rolling Metrics */}
                {(analytics.rolling_metrics?.sharpe_63d?.length > 0 || 
                  analytics.rolling_metrics?.sharpe_126d?.length > 0 ||
                  analytics.rolling_metrics?.sharpe_252d?.length > 0) && (
                  <RollingMetricsChart
                    sharpe63d={analytics.rolling_metrics.sharpe_63d}
                    sharpe126d={analytics.rolling_metrics.sharpe_126d}
                    sharpe252d={analytics.rolling_metrics.sharpe_252d}
                  />
                )}

                {/* Crisis Periods */}
                {analytics.crisis_periods?.length > 0 && (
                  <CrisisOverlay periods={analytics.crisis_periods} />
                )}

                {/* Data Summary */}
                <div className="analytics-summary">
                  <div className="analytics-card">
                    <label>Data Points</label>
                    <span>{analytics.data_points}</span>
                  </div>
                  <div className="analytics-card">
                    <label>Date Range</label>
                    <span>{analytics.date_range.start} to {analytics.date_range.end}</span>
                  </div>
                  <div className="analytics-card">
                    <label>Max Drawdown</label>
                    <span className={analytics.drawdown?.max_drawdown?.max_drawdown < -15 ? 'negative' : ''}>
                      {analytics.drawdown?.max_drawdown?.max_drawdown?.toFixed(2)}%
                    </span>
                  </div>
                  {analytics.drawdown?.max_drawdown?.recovery_date ? (
                    <div className="analytics-card">
                      <label>Recovered</label>
                      <span className="positive">Yes</span>
                    </div>
                  ) : (
                    <div className="analytics-card">
                      <label>Underwater Days</label>
                      <span className="warning">{analytics.drawdown?.max_drawdown?.underwater_days}</span>
                    </div>
                  )}
                </div>
              </>
            ) : (
              <div className="analytics-empty">
                <p>{analytics?.message || 'Analytics data not available'}</p>
                <small>Data points: {analytics?.data_points || 0}</small>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}
