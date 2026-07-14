import type { DataPipelineRunbookAction, HealthData } from '../types/live';

export interface HealthOperationsRunbookAction {
  dimension: string;
  code: string;
  severity: 'ok' | 'warning' | 'critical' | 'unknown';
  action: string;
  label: string;
  artifact?: string;
  provider?: string;
  reason?: string;
}

export interface HealthOperationsSummary {
  headline: string;
  headerText: string;
  scheduler: {
    status: string;
    totalJobs: number;
    failedJobs: number;
    label: string;
  };
  dataFreshness: {
    status: 'fresh' | 'stale' | 'critical' | 'unknown';
    fresh: number;
    stale: number;
    critical: number;
    label: string;
  };
  dataPipelineSlo: {
    status: string;
    topDimension: string | null;
    label: string;
    runbook: {
      status: string;
      topCause: HealthOperationsRunbookAction | null;
      actions: HealthOperationsRunbookAction[];
    } | null;
  } | null;
  topCauses: string[];
}

const normalizeSystemStatus = (status: string | undefined): string => (
  status === 'healthy' ? 'healthy' : status ?? 'unknown'
);

const runbookScopeLabel = (action: DataPipelineRunbookAction): string => {
  if (action.artifact) return `${action.dimension}/${action.artifact}`;
  if (action.provider) return `${action.dimension}/${action.provider}`;
  return action.dimension;
};

const normalizeRunbookAction = (
  action: DataPipelineRunbookAction | null | undefined,
): HealthOperationsRunbookAction | null => {
  if (!action) return null;
  const scope = runbookScopeLabel(action);
  return {
    dimension: action.dimension,
    code: action.code,
    severity: action.severity,
    action: action.action,
    label: `${scope}: ${action.action}`,
    artifact: action.artifact,
    provider: action.provider,
    reason: action.reason,
  };
};

export interface HealthOperationsKillContext {
  /** alerts.json rows — kill_switch type elevates ops summary when health omits kill block */
  alerts?: Array<{ type?: string; kill_switch_level?: string | null; level?: string; title?: string; message?: string | null }>;
  /** signals.broker kill fields when health.kill_switch absent */
  broker?: {
    kill_switch?: boolean;
    kill_switch_level?: string | null;
    kill_switch_incident_id?: string | null;
    kill_switch_reason?: string | null;
  } | null;
}

export function summarizeHealthOperations(
  health: HealthData,
  context?: HealthOperationsKillContext,
): HealthOperationsSummary {
  const freshnessEntries = Object.entries(health.data_freshness || {});
  const fresh = freshnessEntries.filter(([, d]) => d.status === 'fresh').length;
  const stale = freshnessEntries.filter(([, d]) => d.status === 'stale').length;
  const critical = freshnessEntries.filter(([, d]) => d.status === 'critical').length;
  const dataStatus: HealthOperationsSummary['dataFreshness']['status'] =
    critical > 0 ? 'critical' : stale > 0 ? 'stale' : freshnessEntries.length > 0 ? 'fresh' : 'unknown';

  const schedulerBackends = Object.values(health.scheduler_status?.backends ?? {});
  const totalJobs = schedulerBackends.length > 0
    ? schedulerBackends.reduce((sum, backend) => sum + backend.total_jobs, 0)
    : (health.cron_jobs ?? []).length;
  const failedJobs = schedulerBackends.length > 0
    ? schedulerBackends.reduce((sum, backend) => sum + backend.failed_jobs, 0)
    : (health.cron_jobs ?? []).filter((job) => job.status === 'error').length;
  const schedulerStatus = health.scheduler_status?.status ?? (failedJobs > 0 ? 'warning' : 'unknown');

  const schedulerLabel = `Scheduler ${schedulerStatus}: ${totalJobs} scheduled jobs, ${failedJobs} failed`;
  const dataLabel = `Data freshness ${dataStatus}: ${critical} critical, ${stale} stale, ${fresh} fresh`;
  const slo = health.data_pipeline_slo;
  const sloStatus = slo?.status ?? null;
  const sloTopDimension = slo?.top_dimension ?? null;
  const runbookActions = (slo?.runbook?.actions ?? [])
    .map(normalizeRunbookAction)
    .filter((action): action is HealthOperationsRunbookAction => action !== null)
    .slice(0, 6);
  const runbookTopCause = normalizeRunbookAction(slo?.runbook?.top_cause) ?? runbookActions[0] ?? null;
  const runbook = slo?.runbook
    ? {
      status: slo.runbook.status,
      topCause: runbookTopCause,
      actions: runbookActions,
    }
    : null;
  const sloFailingDimensions = Object.entries(slo?.dimensions ?? {})
    .filter(([, dimension]) => dimension.status !== 'ok')
    .map(([name, dimension]) => `${name}: ${dimension.message ?? `status ${dimension.status}`}`);
  const sloLabel = slo
    ? `Data pipeline SLO ${slo.status}${slo.top_dimension ? `: ${slo.top_dimension}` : ''}`
    : '';
  // Kill/halt may appear on health.kill_switch, alerts kill_switch rows, or broker.
  // Prefer health SSOT projection; fall back so ops bar still discloses HALT.
  const healthKill = health.kill_switch;
  const alertKill = (context?.alerts ?? []).find((a) => a?.type === 'kill_switch');
  const broker = context?.broker;
  const brokerKillEnabled = Boolean(broker?.kill_switch);
  const killEnabled = Boolean(healthKill?.enabled) || Boolean(alertKill) || brokerKillEnabled;
  const killLevel = (
    healthKill?.level
    ?? alertKill?.kill_switch_level
    ?? broker?.kill_switch_level
    ?? ''
  ).toString().toLowerCase();
  const killStatus = (healthKill?.status ?? '').toString().toLowerCase();
  const killIncidentId = healthKill?.incident_id
    ?? broker?.kill_switch_incident_id
    ?? undefined;
  const openIncidentsCritical = (health.open_incidents?.status ?? '').toString().toLowerCase() === 'critical';
  const titleLooksHalt = Boolean(alertKill?.title && /halt|kill/i.test(alertKill.title));
  const killHalt =
    (killEnabled && (killLevel === 'halt' || killStatus === 'critical'))
    || openIncidentsCritical
    || (Boolean(alertKill) && (killLevel === 'halt' || titleLooksHalt || !killLevel));
  const killActive = killEnabled || openIncidentsCritical;

  const primaryCause = killHalt
    ? 'kill/halt active'
    : killActive
    ? 'kill switch active'
    : sloStatus && sloStatus !== 'ok'
    ? `data pipeline ${sloTopDimension ?? sloStatus}`
    : dataStatus === 'critical'
    ? 'data freshness critical'
    : dataStatus === 'stale'
      ? 'data freshness stale'
      : failedJobs > 0
        ? 'scheduler failures'
        : 'all tracked subsystems nominal';
  const headline = `System ${normalizeSystemStatus(health.system_status)}: ${primaryCause}; scheduler ${schedulerStatus}`;
  const headerText = `${headline} (${totalJobs} scheduled jobs, ${failedJobs} failed)`;

  const freshnessCauses = freshnessEntries
    .filter(([, data]) => data.status !== 'fresh')
    .sort(([, a], [, b]) => (
      (b.market_lag_days ?? b.days_stale ?? 0) - (a.market_lag_days ?? a.days_stale ?? 0)
    ))
    .slice(0, 5)
    .map(([symbol, data]) => {
      const lagDays = data.market_lag_days ?? data.days_stale;
      const lagLabel = data.market_lag_days === undefined ? 'stale' : 'market lag';
      return `${symbol} ${lagLabel} ${lagDays}d (last update ${data.last_update})`;
    });
  const killCause = killHalt
    ? [`kill/halt${killLevel ? ` level=${killLevel}` : ''}${killIncidentId ? ` incident=${killIncidentId}` : ''}`]
    : killActive
      ? [`kill switch active${killLevel ? ` level=${killLevel}` : ''}`]
      : [];
  const topCauses = killCause.length > 0
    ? [...killCause, ...(runbookTopCause ? [runbookTopCause.label] : sloFailingDimensions.length > 0 ? sloFailingDimensions : freshnessCauses)].slice(0, 5)
    : runbookTopCause
    ? [runbookTopCause.label]
    : sloFailingDimensions.length > 0
      ? sloFailingDimensions
      : freshnessCauses;

  return {
    headline,
    headerText,
    scheduler: {
      status: schedulerStatus,
      totalJobs,
      failedJobs,
      label: schedulerLabel,
    },
    dataFreshness: {
      status: dataStatus,
      fresh,
      stale,
      critical,
      label: dataLabel,
    },
    dataPipelineSlo: slo ? {
      status: slo.status,
      topDimension: slo.top_dimension,
      label: sloLabel,
      runbook,
    } : null,
    topCauses,
  };
}
