import type { HealthData } from '../types/live';

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
  } | null;
  topCauses: string[];
}

const normalizeSystemStatus = (status: string | undefined): string => (
  status === 'healthy' ? 'healthy' : status ?? 'unknown'
);

export function summarizeHealthOperations(health: HealthData): HealthOperationsSummary {
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
  const sloFailingDimensions = Object.entries(slo?.dimensions ?? {})
    .filter(([, dimension]) => dimension.status !== 'ok')
    .map(([name, dimension]) => `${name}: ${dimension.message ?? `status ${dimension.status}`}`);
  const sloLabel = slo
    ? `Data pipeline SLO ${slo.status}${slo.top_dimension ? `: ${slo.top_dimension}` : ''}`
    : '';
  const primaryCause = sloStatus && sloStatus !== 'ok'
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
  const topCauses = sloFailingDimensions.length > 0 ? sloFailingDimensions : freshnessCauses;

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
    } : null,
    topCauses,
  };
}
