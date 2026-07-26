import type {
  DataPipelineRunbookAction,
  HealthData,
  ProvenanceCompleteness,
} from '../types/live';

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

/** Dual-write / split-brain badge for operator health bar (M11). */
export interface DualWriteProvenanceSummary {
  /** absent = no provenance block; ok/warning/critical for operator scan */
  status: 'ok' | 'warning' | 'critical' | 'unknown' | 'absent';
  label: string;
  detail: string | null;
  dualWriteOk: boolean | null;
  dualWriteLagSeconds: number | null;
  dualWriteLagStale: boolean;
  dualWriteAttempted: boolean;
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
  /** Dual-write lag / ok badge (provenance_completeness). */
  dualWrite: DualWriteProvenanceSummary;
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
  /**
   * Dual-write provenance from health_ops (or other dual-write artifact) when
   * health.json itself lacks provenance_completeness.
   */
  dualWriteProvenance?: ProvenanceCompleteness | null;
}

/**
 * Map provenance_completeness → operator badge (decision-first health bar).
 *
 * Severity (advisory, not live authority):
 * - critical: dual_write_attempted && dual_write_ok === false (split-brain write fail)
 * - warning: dual_write_lag_stale (public mtime behind private beyond threshold)
 * - ok: attempted + ok, or paths identical / not attempted
 * - absent: no block
 */
export function summarizeDualWriteProvenance(
  pc: ProvenanceCompleteness | null | undefined,
): DualWriteProvenanceSummary {
  if (!pc || typeof pc !== 'object') {
    return {
      status: 'absent',
      label: 'Dual-write: n/a',
      detail: null,
      dualWriteOk: null,
      dualWriteLagSeconds: null,
      dualWriteLagStale: false,
      dualWriteAttempted: false,
    };
  }
  const attempted = Boolean(pc.dual_write_attempted);
  const ok = pc.dual_write_ok;
  const lag =
    typeof pc.dual_write_lag_seconds === 'number' && Number.isFinite(pc.dual_write_lag_seconds)
      ? pc.dual_write_lag_seconds
      : null;
  const lagStale = Boolean(pc.dual_write_lag_stale);
  const thr =
    typeof pc.dual_write_lag_threshold_seconds === 'number'
      ? pc.dual_write_lag_threshold_seconds
      : 120;

  if (attempted && ok === false) {
    return {
      status: 'critical',
      label: 'Dual-write: FAIL',
      detail: pc.note
        ? String(pc.note)
        : 'dual_write_attempted but dual_write_ok=false (check private vs public SSOT)',
      dualWriteOk: false,
      dualWriteLagSeconds: lag,
      dualWriteLagStale: lagStale,
      dualWriteAttempted: true,
    };
  }
  if (lagStale) {
    const lagTxt = lag === null ? '?' : `${lag.toFixed(0)}s`;
    return {
      status: 'warning',
      label: `Dual-write lag: ${lagTxt}`,
      detail:
        `Public mtime behind private (threshold ${thr}s). ` +
        'Advisory forensics only — private DATA_DIR is producer SSOT when paths differ.',
      dualWriteOk: ok === undefined ? null : Boolean(ok),
      dualWriteLagSeconds: lag,
      dualWriteLagStale: true,
      dualWriteAttempted: attempted,
    };
  }
  if (attempted && ok === true) {
    const lagTxt = lag === null ? '' : ` lag ${lag.toFixed(0)}s`;
    return {
      status: 'ok',
      label: `Dual-write: OK${lagTxt}`,
      detail: pc.disclosure ? String(pc.disclosure) : null,
      dualWriteOk: true,
      dualWriteLagSeconds: lag,
      dualWriteLagStale: false,
      dualWriteAttempted: true,
    };
  }
  if (pc.paths_identical === true) {
    return {
      status: 'ok',
      label: 'Dual-write: same path',
      detail: 'Private and public paths resolve identical — no dual-write lag.',
      dualWriteOk: true,
      dualWriteLagSeconds: lag,
      dualWriteLagStale: false,
      dualWriteAttempted: attempted,
    };
  }
  return {
    status: 'unknown',
    label: 'Dual-write: unknown',
    detail: pc.note ? String(pc.note) : 'provenance_completeness present but incomplete',
    dualWriteOk: ok === undefined || ok === null ? null : Boolean(ok),
    dualWriteLagSeconds: lag,
    dualWriteLagStale: lagStale,
    dualWriteAttempted: attempted,
  };
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
  const dualWrite = summarizeDualWriteProvenance(
    health.provenance_completeness ?? context?.dualWriteProvenance ?? null,
  );

  const dualWriteCause =
    dualWrite.status === 'critical'
      ? [dualWrite.label]
      : dualWrite.status === 'warning'
        ? [dualWrite.label]
        : [];

  const topCauses = killCause.length > 0
    ? [...killCause, ...(runbookTopCause ? [runbookTopCause.label] : sloFailingDimensions.length > 0 ? sloFailingDimensions : freshnessCauses)].slice(0, 5)
    : runbookTopCause
    ? [runbookTopCause.label]
    : sloFailingDimensions.length > 0
      ? sloFailingDimensions
      : freshnessCauses;

  // Surface dual-write failures in topCauses (after kill; before pure freshness)
  const mergedCauses = dualWriteCause.length > 0
    ? [...dualWriteCause, ...topCauses.filter((c) => !dualWriteCause.includes(c))].slice(0, 5)
    : topCauses;

  // Elevate headline primary cause when dual-write is critical (after kill)
  const dualWritePrimary =
    !killHalt && !killActive && dualWrite.status === 'critical'
      ? 'dual-write fail'
      : !killHalt && !killActive && dualWrite.status === 'warning' && primaryCause === 'all tracked subsystems nominal'
        ? 'dual-write lag stale'
        : null;
  const headlineFinal = dualWritePrimary
    ? `System ${normalizeSystemStatus(health.system_status)}: ${dualWritePrimary}; scheduler ${schedulerStatus}`
    : headline;
  const headerTextFinal = `${headlineFinal} (${totalJobs} scheduled jobs, ${failedJobs} failed)`;

  return {
    headline: headlineFinal,
    headerText: headerTextFinal,
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
    dualWrite,
    topCauses: mergedCauses,
  };
}
