import React from 'react';
import { StatusBadge } from './StatusBadge';

interface AuthorityBadgeProps {
  routed?: boolean;
  source?: string;
  blocked?: boolean;
}

function authorityStatus(routed: boolean | undefined, blocked: boolean) {
  if (blocked) return { label: 'Routing blocked', tone: 'critical' as const };
  if (routed === undefined) return { label: 'Authority unavailable', tone: 'warning' as const };
  if (routed) return { label: 'Order-routed authority', tone: 'info' as const };
  return { label: 'Advisory · not routed', tone: 'neutral' as const };
}

export function AuthorityBadge({ routed, source = 'signals.json.target_allocations', blocked = false }: AuthorityBadgeProps) {
  const status = authorityStatus(routed, blocked);
  return (
    <span className="authority-badge">
      <StatusBadge label={status.label} tone={status.tone} />
      <code translate="no">{source}</code>
    </span>
  );
}
