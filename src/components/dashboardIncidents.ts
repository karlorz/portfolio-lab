import type { Alert, HealthData, SignalsData } from '../types/live';

export type IncidentSeverity = 'critical' | 'warning' | 'info' | 'success';

export type IncidentTab =
  | 'overview'
  | 'health'
  | 'history'
  | 'performance'
  | 'rebalance'
  | 'analytics'
  | 'options'
  | 'auction'
  | 'risk'
  | 'labs'
  | 'tasks'
  | 'chat';

export interface DashboardIncident {
  id: string;
  tab: IncidentTab;
  severity: IncidentSeverity;
  title: string;
  source: string;
  currentValue?: string;
  threshold?: string;
  message: string;
  nextAction?: string;
  timestamp?: string;
}

export interface DashboardIncidentInputs {
  alerts: Alert[];
  signals: SignalsData | null;
  health: HealthData | null;
}

export interface TabIncidentBadge {
  count: number;
  severity: IncidentSeverity;
}

const SEVERITY_RANK: Record<IncidentSeverity, number> = {
  critical: 4,
  warning: 3,
  info: 2,
  success: 1,
};

const RISK_WARNING_CVAR_RATIO = 1.5;
const RISK_SEVERE_CVAR_RATIO = 1.8;

function slugify(value: string): string {
  const slug = value
    .trim()
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, '-')
    .replace(/^-+|-+$/g, '');
  return slug || 'incident';
}

function formatAlertSource(type: string): string {
  if (type === 'graduation_candidate') return 'Graduation checklist';
  if (type === 'kill_switch') return 'Kill switch';
  if (type === 'portfolio_drift') return 'Portfolio drift';
  if (type === 'ic_decay') return 'IC decay monitor';
  if (type === 'staleness') return 'Signal staleness';

  return type
    .split(/[_-]+/)
    .filter(Boolean)
    .map((part) => `${part.charAt(0).toUpperCase()}${part.slice(1)}`)
    .join(' ') || 'Dashboard alert';
}

function extractAlertCurrentValue(alert: Alert): string | undefined {
  if (alert.type === 'graduation_candidate') {
    const sharpe = alert.message.match(/Sharpe:\s*([0-9.]+)/i);
    if (sharpe?.[1]) return `Sharpe ${sharpe[1]}`;
  }
  return undefined;
}

function alertNextAction(alert: Alert, severity: IncidentSeverity): string | undefined {
  if (alert.type === 'graduation_candidate') {
    return 'Review graduation checklist before live approval.';
  }
  if (alert.type === 'kill_switch') {
    return 'Review kill-switch state before placing new orders.';
  }
  if (alert.type === 'portfolio_drift') {
    return 'Review drift details and rebalance constraints.';
  }
  if (alert.type === 'ic_decay') {
    return 'Review decaying signal weights before trusting new ensemble votes.';
  }
  if (alert.type === 'staleness') {
    return 'Refresh stale data sources before acting on the signal.';
  }
  if (severity === 'critical') {
    return 'Inspect the latest dashboard data and resolve the blocking condition.';
  }
  if (alert.requires_action) {
    return 'Review this item before the next paper-trading decision.';
  }
  return undefined;
}

function alertSeverity(alert: Alert): IncidentSeverity | null {
  if (alert.type === 'kill_switch') return 'critical';
  if (alert.level === 'error') return 'critical';
  if (alert.level === 'warning') return 'warning';
  if (alert.level === 'info' && alert.requires_action) return 'info';
  if (alert.level === 'success' && alert.requires_action) return 'info';
  return null;
}

function mapAlertToIncident(alert: Alert): DashboardIncident | null {
  const severity = alertSeverity(alert);
  if (!severity) return null;

  return {
    id: `alert:${alert.type}:${slugify(alert.title)}`,
    tab: 'overview',
    severity,
    title: alert.title,
    source: formatAlertSource(alert.type),
    currentValue: extractAlertCurrentValue(alert),
    message: alert.message,
    nextAction: alertNextAction(alert, severity),
    timestamp: alert.timestamp,
  };
}

export function buildRiskIncidents(signals: SignalsData | null): DashboardIncident[] {
  const cvarRatio = signals?.garch_cvar?.cvar_ratio;
  if (cvarRatio === undefined || cvarRatio < RISK_WARNING_CVAR_RATIO) {
    return [];
  }

  const isSevere = cvarRatio >= RISK_SEVERE_CVAR_RATIO;
  return [
    {
      id: isSevere ? 'risk:garch-cvar:severe-tail-risk' : 'risk:garch-cvar:elevated-tail-risk',
      tab: 'risk',
      severity: isSevere ? 'critical' : 'warning',
      title: isSevere ? 'Severe Tail Risk' : 'Elevated Tail Risk',
      source: 'GARCH CVaR',
      currentValue: `${cvarRatio.toFixed(2)}x CVaR/VaR`,
      threshold: '>= 1.80x severe, >= 1.50x warning',
      message: isSevere
        ? 'Tail loss severity is above the dashboard severe risk threshold.'
        : 'Tail loss severity is above the dashboard warning threshold.',
      nextAction: isSevere
        ? 'Review equity exposure, hedge state, and kill-switch status.'
        : 'Monitor volatility clustering and confirm hedge posture.',
      timestamp: signals?.timestamp,
    },
  ];
}

function buildHealthIncidents(health: HealthData | null): DashboardIncident[] {
  if (!health || health.system_status === 'healthy') return [];

  const isCritical = health.system_status === 'critical';
  return [
    {
      id: `health:system:${health.system_status}`,
      tab: 'health',
      severity: isCritical ? 'critical' : 'warning',
      title: isCritical ? 'Critical System Health' : 'Degraded System Health',
      source: 'Health check',
      currentValue: `System ${health.system_status}`,
      threshold: 'Expected healthy',
      message: 'One or more live paper-trading health checks need attention.',
      nextAction: 'Open Health and inspect failing jobs, stale data, and scheduler state.',
      timestamp: health.generated_at,
    },
  ];
}

function sortIncidents(incidents: DashboardIncident[]): DashboardIncident[] {
  return [...incidents].sort((a, b) => {
    const severityDelta = SEVERITY_RANK[b.severity] - SEVERITY_RANK[a.severity];
    if (severityDelta !== 0) return severityDelta;
    return a.id.localeCompare(b.id);
  });
}

export function buildDashboardIncidents(inputs: DashboardIncidentInputs): DashboardIncident[] {
  return sortIncidents([
    ...inputs.alerts.map(mapAlertToIncident).filter((incident): incident is DashboardIncident => incident !== null),
    ...buildRiskIncidents(inputs.signals),
    ...buildHealthIncidents(inputs.health),
  ]);
}

export function getIncidentsForTab(incidents: DashboardIncident[], tab: IncidentTab): DashboardIncident[] {
  if (tab === 'overview') {
    return sortIncidents(incidents);
  }
  return sortIncidents(incidents.filter((incident) => incident.tab === tab));
}

export function getTabIncidentBadge(
  incidents: DashboardIncident[],
  tab: IncidentTab,
): TabIncidentBadge | undefined {
  const tabIncidents = getIncidentsForTab(incidents, tab);
  if (tabIncidents.length === 0) return undefined;

  const severity = tabIncidents.reduce<IncidentSeverity>((highest, incident) => {
    return SEVERITY_RANK[incident.severity] > SEVERITY_RANK[highest] ? incident.severity : highest;
  }, 'success');

  return {
    count: tabIncidents.length,
    severity,
  };
}
