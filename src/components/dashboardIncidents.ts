import type {
  Alert,
  HealthData,
  IcDecaySummary,
  IncidentLifecycleIncident,
  IncidentLifecycleSummary,
  SignalsData,
} from '../types/live';

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
  | 'decisions'
  | 'tasks'
  | 'chat';

export interface DashboardIncident {
  id: string;
  tab: IncidentTab;
  severity: IncidentSeverity;
  attention: 'action' | 'advisory';
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
  incidentSummary?: IncidentLifecycleSummary | null;
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

function safeText(value: unknown, fallback: string): string {
  return typeof value === 'string' && value.trim() ? value.trim() : fallback;
}

function slugify(value: unknown): string {
  const slug = safeText(value, '')
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, '-')
    .replace(/^-+|-+$/g, '');
  return slug || 'incident';
}

function humanizeAlertType(type: unknown): string {
  return safeText(type, 'dashboard_alert')
    .split(/[_-]+/)
    .filter(Boolean)
    .map((part) => `${part.charAt(0).toUpperCase()}${part.slice(1)}`)
    .join(' ');
}

function formatAlertSource(type: unknown): string {
  const normalizedType = safeText(type, 'dashboard_alert');
  if (normalizedType === 'graduation_candidate') return 'Graduation checklist';
  if (normalizedType === 'kill_switch') return 'Kill switch';
  if (normalizedType === 'portfolio_drift') return 'Portfolio drift';
  if (normalizedType === 'ic_decay') return 'IC decay monitor';
  if (normalizedType === 'staleness') return 'Signal staleness';

  return humanizeAlertType(normalizedType);
}

function extractAlertCurrentValue(alert: Alert): string | undefined {
  if (alert.type === 'graduation_candidate' && typeof alert.message === 'string') {
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
  if (alert.level === 'critical') return 'critical';
  if (alert.level === 'error') return 'critical';
  if (alert.level === 'warning') return 'warning';
  if (alert.level === 'info' && alert.requires_action) return 'info';
  if (alert.level === 'success' && alert.requires_action) return 'info';
  return null;
}

function mapAlertToIncident(alert: Alert): DashboardIncident | null {
  const severity = alertSeverity(alert);
  if (!severity) return null;
  const type = safeText(alert.type, 'dashboard_alert');
  const title = typeof alert.title === 'string' && alert.title.trim()
    ? alert.title.trim()
    : humanizeAlertType(type);
  const message = safeText(alert.message, 'Alert details are unavailable.');
  const stableId = safeText(
    alert.stable_id,
    safeText(alert.incident_id, `${type}:${slugify(title)}`),
  );

  return {
    id: `alert:${stableId}`,
    tab: 'overview',
    severity,
    attention: alert.requires_action === false && type !== 'kill_switch' ? 'advisory' : 'action',
    title,
    source: formatAlertSource(type),
    currentValue: extractAlertCurrentValue(alert),
    message,
    nextAction: alertNextAction(alert, severity),
    timestamp: alert.timestamp,
  };
}

function incidentSeverity(severity: string): IncidentSeverity {
  if (severity === 'p0') return 'critical';
  if (severity === 'p1' || severity === 'p2') return 'warning';
  return 'info';
}

function incidentTitle(channel: string): string {
  return `${formatAlertSource(channel)} Incident`;
}

function persistedIncidentNextAction(incident: IncidentLifecycleIncident): string {
  if (
    incident.channel === 'cron_failure'
    || incident.channel === 'evaluator_error'
    || incident.channel === 'signal_staleness'
  ) {
    return 'Open Health and inspect scheduler status, data freshness, and incident details.';
  }
  if (incident.channel === 'portfolio_drift') {
    return 'Open Rebalance and inspect drift details before placing new orders.';
  }
  if (incident.channel === 'ic_decay') {
    return 'Open Analytics and inspect signal quality before trusting new ensemble votes.';
  }
  return 'Review the incident lifecycle record before the next paper-trading decision.';
}

function qualitySignalValue(
  signals: SignalsData | null,
  name: string,
  minimum: number,
): string {
  const row = signals?.ic_decay?.signals?.[name];
  if (!row) return name;
  const value = typeof row.ic_rolling === 'number' ? row.ic_rolling.toFixed(4) : 'n/a';
  return `${name} IC ${value} (${row.observations}/${minimum} observations)`;
}

function buildIcDecayIncident(
  summary: IcDecaySummary | undefined,
  signals: SignalsData | null,
): DashboardIncident | null {
  if (!summary || !['critical', 'warning'].includes(summary.status)) return null;
  const severity: IncidentSeverity = summary.status === 'critical' ? 'critical' : 'warning';
  const critical = summary.critical_signals ?? [];
  const warning = summary.warning_signals ?? [];
  const attentionSignals = [...critical, ...warning];
  const currentValue = attentionSignals.length > 0
    ? attentionSignals.map((name) => qualitySignalValue(signals, name, summary.min_observations)).join('; ')
    : `${severity} quality status`;
  const pending = `${summary.staged_pending_predictions} staged pending labels in ${summary.staged_pending_scope}`;
  const historical = `${summary.historical_unlabeled_rows} historical unlabeled rows in ${summary.historical_unlabeled_scope}`;

  return {
    id: 'quality:ic-decay',
    tab: 'health',
    severity,
    attention: 'action',
    title: 'Signal quality: IC decay',
    source: 'IC decay monitor',
    currentValue,
    threshold: `Minimum observations: ${summary.min_observations}`,
    message: `${critical.length} critical IC signal(s): ${critical.join(', ') || 'none'}; ${pending}; ${historical}.`,
    nextAction: 'Review IC evidence and the post-fix cohort before trusting new votes; do not auto-invert or change allocations.',
    timestamp: summary.evidence_generated_at ?? undefined,
  };
}

function mapPersistedIncidentToDashboard(
  incident: IncidentLifecycleIncident,
  icSummary?: IcDecaySummary,
  signals: SignalsData | null = null,
  forceCritical = false,
): DashboardIncident | null {
  if (incident.state === 'resolved') return null;
  const qualityIncident = incident.channel === 'ic_decay'
    ? buildIcDecayIncident(icSummary, signals)
    : null;
  const severity = forceCritical && incident.channel === 'ic_decay'
    ? 'critical'
    : incidentSeverity(incident.severity);

  return {
    id: `persisted:${incident.channel}:${incident.incident_id}`,
    tab: 'health',
    severity,
    attention: 'action',
    title: incidentTitle(incident.channel),
    source: 'Incident lifecycle',
    currentValue: qualityIncident?.currentValue ?? `State: ${incident.state}`,
    threshold: qualityIncident?.threshold ?? `Severity: ${incident.severity}`,
    message: qualityIncident
      ? `${incident.message} ${qualityIncident.message}`
      : incident.message,
    nextAction: qualityIncident?.nextAction ?? persistedIncidentNextAction(incident),
    timestamp: qualityIncident?.timestamp ?? incident.updated_at ?? incident.created_at,
  };
}

function buildPersistedIncidents(
  summary: IncidentLifecycleSummary | null | undefined,
  icSummary?: IcDecaySummary,
  signals: SignalsData | null = null,
  forceCriticalIcIncident = false,
): DashboardIncident[] {
  if (!summary) return [];
  return summary.incidents
    .map((incident) => mapPersistedIncidentToDashboard(
      incident,
      icSummary,
      signals,
      forceCriticalIcIncident,
    ))
    .filter((incident): incident is DashboardIncident => incident !== null);
}

function isDuplicateIcAlert(
  alert: Alert,
  summary: IncidentLifecycleSummary | null | undefined,
): boolean {
  if (alert.type !== 'ic_decay' && alert.type !== 'kill_switch') return false;
  const openIcIncidents = summary?.incidents.filter(
    (incident) => incident.channel === 'ic_decay' && incident.state !== 'resolved',
  ) ?? [];
  if (openIcIncidents.length === 0) return false;

  const identityMatches = typeof alert.incident_id === 'string'
    && openIcIncidents.some((incident) => incident.incident_id === alert.incident_id);
  if (identityMatches || alert.type === 'ic_decay') return true;

  return alert.type === 'kill_switch'
    && (alert.reason === 'unresolved_incident:ic_decay'
      || /\bic\s+decay\b/i.test(alert.message));
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
      attention: 'action',
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
      attention: 'action',
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

export function buildDecisionIncidents(signals: SignalsData | null): DashboardIncident[] {
  if (!signals) return [];

  const incidents: DashboardIncident[] = [];
  const staleness = signals.staleness as
    | { stale_signals?: string[]; stale?: string[] }
    | undefined;
  const staleList = staleness?.stale_signals ?? staleness?.stale;
  if (Array.isArray(staleList) && staleList.length > 0) {
    incidents.push({
      id: 'decisions:staleness',
      tab: 'decisions',
      severity: 'warning',
      attention: 'action',
      title: 'Stale signals in last cycle',
      source: 'Decision replay',
      currentValue: `${staleList.length} stale`,
      message: `Signals past TTL may have degraded ensemble weights: ${staleList.slice(0, 5).join(', ')}${
        staleList.length > 5 ? '…' : ''
      }`,
      nextAction: 'Open Decisions and compare gates on the latest recorded decision.',
      timestamp: signals.timestamp,
    });
  }

  const smart = signals.smart_rebalance;
  if (
    smart &&
    smart.should_execute === false &&
    smart.decision &&
    !['observe', 'no_positions'].includes(String(smart.decision))
  ) {
    incidents.push({
      id: 'decisions:smart-rebalance-hold',
      tab: 'decisions',
      severity: 'info',
      attention: 'advisory',
      title: 'Rebalance held',
      source: 'Smart rebalance',
      currentValue: String(smart.decision),
      message: smart.reason ? String(smart.reason) : 'Smart rebalance did not execute this cycle.',
      nextAction: 'Review replay detail for gates and weight deltas.',
      timestamp: signals.timestamp,
    });
  }

  return incidents;
}

export function buildDashboardIncidents(inputs: DashboardIncidentInputs): DashboardIncident[] {
  const duplicateIcAlerts = inputs.alerts.filter((alert) => isDuplicateIcAlert(
    alert,
    inputs.incidentSummary,
  ));
  const persisted = buildPersistedIncidents(
    inputs.incidentSummary,
    inputs.health?.ic_decay_summary,
    inputs.signals,
    duplicateIcAlerts.some((alert) => alertSeverity(alert) === 'critical'),
  );
  const hasPersistedIcIncident = persisted.some((incident) => incident.id.startsWith('persisted:ic_decay:'));
  const derivedQuality = hasPersistedIcIncident
    ? []
    : [buildIcDecayIncident(inputs.health?.ic_decay_summary, inputs.signals)].filter(
      (incident): incident is DashboardIncident => incident !== null,
    );
  return sortIncidents([
    ...persisted,
    ...inputs.alerts
      .filter((alert) => !isDuplicateIcAlert(alert, inputs.incidentSummary))
      .map(mapAlertToIncident)
      .filter((incident): incident is DashboardIncident => incident !== null),
    ...buildRiskIncidents(inputs.signals),
    ...buildHealthIncidents(inputs.health),
    ...derivedQuality,
    ...buildDecisionIncidents(inputs.signals),
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
