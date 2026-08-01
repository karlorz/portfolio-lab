import React from 'react';
import type { DashboardIncident } from '../dashboardIncidents';
import { ActionCenter } from './ActionCenter';
import { AuthorityBadge } from './AuthorityBadge';

interface ContextRailProps {
  incidents: DashboardIncident[];
  routed?: boolean;
  freshness?: string;
  openIncidentCount?: number;
  runtimeProvenance?: RuntimeProvenanceDisclosure;
  onIncidentSelect?: (incident: DashboardIncident) => void;
}

export interface RuntimeProvenanceDisclosure {
  staticRelease?: string;
  runtimeArtifact?: string;
  runtimeStatus?: string;
  orderAuthority?: string;
}

export function ContextRail({
  incidents,
  routed,
  freshness = 'Awaiting refresh',
  openIncidentCount = 0,
  runtimeProvenance,
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
        <summary>Persisted incidents</summary>
        <p className="context-rail-stat" aria-label="Persisted open incident count">{openIncidentCount}</p>
      </details>
      <details open>
        <summary>Runtime provenance</summary>
        <dl className="context-rail-provenance">
          <div className="context-rail-provenance-row">
            <dt>Static release</dt>
            <dd>{runtimeProvenance?.staticRelease || 'Unavailable'}</dd>
          </div>
          <div className="context-rail-provenance-row">
            <dt>Runtime artifact</dt>
            <dd>{runtimeProvenance?.runtimeArtifact || 'Unavailable'}</dd>
          </div>
          <div className="context-rail-provenance-row">
            <dt>Runtime status</dt>
            <dd>{runtimeProvenance?.runtimeStatus || 'Unavailable'}</dd>
          </div>
          <div className="context-rail-provenance-row">
            <dt>Order authority</dt>
            <dd>{runtimeProvenance?.orderAuthority || 'Unavailable'}</dd>
          </div>
        </dl>
      </details>
    </aside>
  );
}
