import React from 'react';
import { AlertTriangle, CheckCircle2, Info } from 'lucide-react';
import type { DashboardIncident, IncidentSeverity, IncidentTab } from './dashboardIncidents';

interface IncidentSummaryProps {
  title: string;
  incidents: DashboardIncident[];
  showTab?: boolean;
  onIncidentSelect?: (incident: DashboardIncident) => void;
}

const severityLabels: Record<IncidentSeverity, string> = {
  critical: 'Critical',
  warning: 'Warning',
  info: 'Attention',
  success: 'Success',
};

const tabLabels: Record<IncidentTab, string> = {
  overview: 'Overview',
  health: 'Health',
  history: 'History',
  performance: 'Performance',
  rebalance: 'Rebalance',
  analytics: 'Analytics',
  options: '0DTE',
  auction: 'Auction',
  risk: 'Risk',
  labs: 'Labs',
  decisions: 'Decisions',
  tasks: 'Tasks',
  chat: 'Chat',
};

function IncidentIcon({ severity }: { severity: IncidentSeverity }) {
  if (severity === 'success') {
    return <CheckCircle2 size={18} aria-hidden="true" />;
  }
  if (severity === 'info') {
    return <Info size={18} aria-hidden="true" />;
  }
  return <AlertTriangle size={18} aria-hidden="true" />;
}

function formatTimestamp(timestamp?: string): string | null {
  if (!timestamp) return null;
  const date = new Date(timestamp);
  if (Number.isNaN(date.getTime())) return timestamp;
  return date.toLocaleString();
}

export function IncidentSummary({
  title,
  incidents,
  showTab = false,
  onIncidentSelect,
}: IncidentSummaryProps) {
  if (incidents.length === 0) return null;

  return (
    <section className="incident-summary" aria-label={title}>
      <div className="incident-summary-header">
        <h3>{title}</h3>
        <span>{incidents.length} active</span>
      </div>

      <div className="incident-list">
        {incidents.map((incident) => {
          const updated = formatTimestamp(incident.timestamp);
          const rowContent = (
            <>
              <span className={`incident-icon incident-icon-${incident.severity}`}>
                <IncidentIcon severity={incident.severity} />
              </span>
              <span className="incident-body">
                <span className="incident-title-line">
                  <span className={`incident-severity incident-severity-${incident.severity}`}>
                    {severityLabels[incident.severity]}
                  </span>
                  <strong>{incident.title}</strong>
                </span>
                <span className="incident-meta">
                  <span>Source: {incident.source}</span>
                  {showTab && <span>Tab: {tabLabels[incident.tab]}</span>}
                  {incident.currentValue && <span>Current: {incident.currentValue}</span>}
                  {incident.threshold && <span>Threshold: {incident.threshold}</span>}
                  {updated && <span>Updated: {updated}</span>}
                </span>
                <span className="incident-message">{incident.message}</span>
                {incident.nextAction && (
                  <span className="incident-next-action">Next: {incident.nextAction}</span>
                )}
              </span>
            </>
          );

          if (onIncidentSelect) {
            return (
              <button
                key={incident.id}
                type="button"
                className={`incident-row incident-row-${incident.severity}`}
                onClick={() => onIncidentSelect(incident)}
              >
                {rowContent}
              </button>
            );
          }

          return (
            <div key={incident.id} className={`incident-row incident-row-${incident.severity}`}>
              {rowContent}
            </div>
          );
        })}
      </div>
    </section>
  );
}
