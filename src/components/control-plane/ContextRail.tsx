import React from 'react';
import type { DashboardIncident } from '../dashboardIncidents';
import { ActionCenter } from './ActionCenter';
import { AuthorityBadge } from './AuthorityBadge';

interface ContextRailProps {
  incidents: DashboardIncident[];
  routed?: boolean;
  freshness?: string;
  openIncidentCount?: number;
  onIncidentSelect?: (incident: DashboardIncident) => void;
}

export function ContextRail({
  incidents,
  routed,
  freshness = 'Awaiting refresh',
  openIncidentCount = 0,
  onIncidentSelect,
}: ContextRailProps) {
  return (
    <aside className="context-rail" aria-label="Operator context">
      <ActionCenter incidents={incidents} onSelect={onIncidentSelect} limit={3} />
      <details open>
        <summary>Authority proof</summary>
        <AuthorityBadge routed={routed} />
      </details>
      <details open>
        <summary>Freshness</summary>
        <p>{freshness}</p>
      </details>
      <details open>
        <summary>Open incidents</summary>
        <p className="context-rail-stat">{openIncidentCount}</p>
      </details>
    </aside>
  );
}
