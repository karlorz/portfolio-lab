import React, { useId } from 'react';
import type { DashboardIncident } from '../dashboardIncidents';
import { StatusBadge, type StatusTone } from './StatusBadge';

interface ActionCenterProps {
  incidents: DashboardIncident[];
  onSelect?: (incident: DashboardIncident) => void;
  limit?: number;
}

const severityTone: Record<DashboardIncident['severity'], StatusTone> = {
  critical: 'critical',
  warning: 'warning',
  info: 'info',
  success: 'success',
};

export function ActionCenter({ incidents, onSelect, limit }: ActionCenterProps) {
  const headingId = useId();
  const visible = limit ? incidents.slice(0, limit) : incidents;

  return (
    <section className="action-center" aria-labelledby={headingId}>
      <div className="control-section-heading">
        <div>
          <p className="control-eyebrow">Exception queue</p>
          <h2 id={headingId}>Action Center</h2>
        </div>
        <span className="control-count">{incidents.length}</span>
      </div>
      {visible.length === 0 ? (
        <p className="control-empty">No open operator actions.</p>
      ) : (
        <ol className="action-list">
          {visible.map((incident) => (
            <li key={incident.id}>
              <button type="button" onClick={() => onSelect?.(incident)}>
                <span className="action-list-heading">
                  <StatusBadge label={incident.severity} tone={severityTone[incident.severity]} />
                  <strong>{incident.title}</strong>
                </span>
                <span className="action-list-message">{incident.message}</span>
                <span className="action-list-next">{incident.nextAction || 'Review the source before acting.'}</span>
              </button>
            </li>
          ))}
        </ol>
      )}
      {limit && incidents.length > limit && (
        <p className="action-list-overflow">Showing {limit} of {incidents.length} actions.</p>
      )}
    </section>
  );
}
