import { describe, expect, it } from 'bun:test';
import { summarizeHealthOperations } from '../../src/components/healthOperations';
import type { HealthData } from '../../src/types/live';

describe('HealthPanel operations summary', () => {
  it('separates critical data freshness from healthy scheduler job counts', () => {
    const summary = summarizeHealthOperations({
      system_status: 'critical',
      generated_at: '2026-06-11T11:13:40Z',
      cron_jobs: [
        { id: 'portfolio-lab-data', name: 'portfolio-lab-data', schedule: '5 * * * *', last_run: null, next_run: null, status: 'ok', state: 'scheduled' },
        { id: 'portfolio-lab-dashboard', name: 'portfolio-lab-dashboard', schedule: '15 * * * *', last_run: null, next_run: null, status: 'ok', state: 'scheduled' },
      ],
      scheduler_status: {
        status: 'ok',
        backends: {
          local: { backend: 'tasker', status: 'ok', source: 'data/cron_status.json', total_jobs: 15, failed_jobs: 0 },
        },
      },
      data_freshness: {
        SPY: { last_update: '2026-05-21', days_stale: 21, status: 'critical' },
        GLD: { last_update: '2026-05-21', days_stale: 21, status: 'critical' },
        TLT: { last_update: '2026-06-10', days_stale: 1, status: 'fresh' },
      },
    } satisfies HealthData);

    expect(summary.headline).toBe('System critical: data freshness critical; scheduler ok');
    expect(summary.headerText).toBe('System critical: data freshness critical; scheduler ok (15 scheduled jobs, 0 failed)');
    expect(summary.scheduler.label).toBe('Scheduler ok: 15 scheduled jobs, 0 failed');
    expect(summary.dataFreshness.label).toBe('Data freshness critical: 2 critical, 0 stale, 1 fresh');
    expect(summary.topCauses).toEqual([
      'SPY stale 21d (last update 2026-05-21)',
      'GLD stale 21d (last update 2026-05-21)',
    ]);
  });

  it('uses market lag in data freshness causes when provided', () => {
    const summary = summarizeHealthOperations({
      system_status: 'warning',
      generated_at: '2026-06-11T11:13:40Z',
      cron_jobs: [],
      scheduler_status: {
        status: 'ok',
        backends: {
          local: { backend: 'tasker', status: 'ok', source: 'data/cron_status.json', total_jobs: 15, failed_jobs: 0 },
        },
      },
      data_freshness: {
        SPY: {
          last_update: '2026-06-09',
          days_stale: 2,
          market_lag_days: 0,
          latest_available_market_date: '2026-06-09',
          status: 'fresh',
        },
        GLD: {
          last_update: '2026-06-04',
          days_stale: 7,
          market_lag_days: 5,
          latest_available_market_date: '2026-06-09',
          status: 'critical',
        },
      },
    } satisfies HealthData);

    expect(summary.dataFreshness.label).toBe('Data freshness critical: 1 critical, 0 stale, 1 fresh');
    expect(summary.topCauses).toEqual([
      'GLD market lag 5d (last update 2026-06-04)',
    ]);
  });

  it('prefers explicit data pipeline SLO causes when present', () => {
    const summary = summarizeHealthOperations({
      system_status: 'warning',
      generated_at: '2026-06-11T11:13:40Z',
      cron_jobs: [],
      scheduler_status: {
        status: 'ok',
        backends: {
          local: { backend: 'tasker', status: 'ok', source: 'data/cron_status.json', total_jobs: 15, failed_jobs: 0 },
        },
      },
      data_freshness: {
        SPY: { last_update: '2026-06-11', days_stale: 0, status: 'fresh' },
      },
      data_pipeline_slo: {
        schema_version: 'data-pipeline-slo/v1',
        status: 'warning',
        top_dimension: 'provider',
        dimensions: {
          provider: { status: 'warning', message: 'provider degraded for prices.json' },
          scheduler: { status: 'ok', message: 'scheduler ok' },
          artifact: { status: 'ok', message: 'artifacts fresh' },
          signal: { status: 'ok', message: 'required signals fresh' },
        },
      },
    } satisfies HealthData);

    expect(summary.headline).toBe('System warning: data pipeline provider; scheduler ok');
    expect(summary.dataPipelineSlo?.label).toBe('Data pipeline SLO warning: provider');
    expect(summary.topCauses).toEqual(['provider: provider degraded for prices.json']);
  });
});
