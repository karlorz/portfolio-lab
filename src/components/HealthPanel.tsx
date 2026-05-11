import React from 'react';
import type { HealthData, CronJobStatus, DataFreshness } from '../types/live';

interface HealthPanelProps {
  health: HealthData | null;
  expanded?: boolean;
  onToggleExpand?: () => void;
}

export function HealthPanel({ health, expanded = false, onToggleExpand }: HealthPanelProps) {
  if (!health) {
    return (
      <div className="health-panel loading">
        <div className="panel-header">
          <h3>System Health</h3>
          <span className="loading-text">Loading...</span>
        </div>
      </div>
    );
  }

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'healthy': return '#10b981';
      case 'fresh': return '#10b981';
      case 'ok': return '#10b981';
      case 'scheduled': return '#3b82f6';
      case 'warning': return '#f59e0b';
      case 'stale': return '#f59e0b';
      case 'critical': return '#ef4444';
      case 'error': return '#ef4444';
      default: return '#6b7280';
    }
  };

  const formatTime = (isoString: string | null) => {
    if (!isoString) return 'Never';
    const date = new Date(isoString);
    const now = new Date();
    const diffMs = now.getTime() - date.getTime();
    const diffMins = Math.floor(diffMs / 60000);
    const diffHours = Math.floor(diffMs / 3600000);
    const diffDays = Math.floor(diffMs / 86400000);
    
    if (diffMins < 1) return 'Just now';
    if (diffMins < 60) return `${diffMins}m ago`;
    if (diffHours < 24) return `${diffHours}h ago`;
    return `${diffDays}d ago`;
  };

  const freshAssets = Object.entries(health.data_freshness || {}).filter(
    ([_, d]) => d.status === 'fresh'
  );
  const staleAssets = Object.entries(health.data_freshness || {}).filter(
    ([_, d]) => d.status === 'stale'
  );
  const criticalAssets = Object.entries(health.data_freshness || {}).filter(
    ([_, d]) => d.status === 'critical'
  );

  return (
    <div className="health-panel">
      <div className="panel-header" onClick={onToggleExpand}>
        <h3>
          System Health
          <span 
            className="status-badge"
            style={{ backgroundColor: getStatusColor(health.system_status) }}
          >
            {health.system_status?.toUpperCase()}
          </span>
        </h3>
        <button className="expand-btn">
          {expanded ? '▼' : '▶'}
        </button>
      </div>

      {!expanded && (
        <div className="health-summary">
          <div className="summary-item">
            <span className="dot" style={{ backgroundColor: '#10b981' }}></span>
            <span>{freshAssets.length} fresh</span>
          </div>
          <div className="summary-item">
            <span className="dot" style={{ backgroundColor: '#f59e0b' }}></span>
            <span>{staleAssets.length} stale</span>
          </div>
          <div className="summary-item">
            <span className="dot" style={{ backgroundColor: '#ef4444' }}></span>
            <span>{criticalAssets.length} critical</span>
          </div>
          {health.cron_jobs && health.cron_jobs.length > 0 && (
            <div className="summary-item">
              <span className="dot" style={{ backgroundColor: '#3b82f6' }}></span>
              <span>{health.cron_jobs.length} jobs</span>
            </div>
          )}
        </div>
      )}

      {expanded && (
        <div className="health-details">
          {/* Cron Jobs Section */}
          {health.cron_jobs && health.cron_jobs.length > 0 && (
            <div className="section">
              <h4>Cron Jobs</h4>
              <div className="cron-jobs-grid">
                {health.cron_jobs.map((job) => (
                  <div key={job.id} className={`cron-job-card status-${job.status}`}>
                    <div className="job-name">{job.name.replace('portfolio-lab-', '')}</div>
                    <div className="job-schedule">{job.schedule}</div>
                    <div className="job-last-run">
                      Last: {formatTime(job.last_run)}
                    </div>
                    <div className="job-next-run">
                      Next: {formatTime(job.next_run)}
                    </div>
                    <span 
                      className="job-status"
                      style={{ backgroundColor: getStatusColor(job.status) }}
                    >
                      {job.status}
                    </span>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Data Freshness Section */}
          <div className="section">
            <h4>Data Freshness</h4>
            <div className="freshness-grid">
              {Object.entries(health.data_freshness || {})
                .sort(([, a], [, b]) => (a.days_stale || 0) - (b.days_stale || 0))
                .map(([symbol, data]) => (
                  <div 
                    key={symbol} 
                    className={`freshness-item status-${data.status}`}
                  >
                    <span className="symbol">{symbol}</span>
                    <span 
                      className="status-dot"
                      style={{ backgroundColor: getStatusColor(data.status) }}
                    ></span>
                    <span className="days">{data.days_stale}d</span>
                    <span className="date">{data.last_update}</span>
                  </div>
                ))
              }
            </div>
          </div>

          {/* Last Updated */}
          <div className="last-updated">
            Health data: {formatTime(health.generated_at)}
          </div>
        </div>
      )}
    </div>
  );
}
