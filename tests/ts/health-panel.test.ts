import { describe, expect, it } from 'bun:test';
import { summarizeHealthOperations } from '../../src/components/healthOperations';
import {
  summarizeRebalanceLiveDiagnostics,
  type RebalanceHealthData,
} from '../../src/components/RebalanceHealthPanel';
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
    expect(summary.dataPipelineSlo?.runbook).toBeNull();
  });

  it('surfaces data pipeline runbook top cause and remediation actions', () => {
    const summary = summarizeHealthOperations({
      system_status: 'critical',
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
        status: 'critical',
        top_dimension: 'artifact',
        dimensions: {
          artifact: { status: 'critical', message: 'index.json is older than source_manifest.json' },
        },
        runbook: {
          status: 'critical',
          top_cause: {
            dimension: 'artifact',
            code: 'stale_public_data_index',
            severity: 'critical',
            artifact: 'index.json',
            reason: 'stale_index',
            action: 'Regenerate public/data/index.json after source_manifest.json changes.',
          },
          actions: [
            {
              dimension: 'artifact',
              code: 'stale_public_data_index',
              severity: 'critical',
              artifact: 'index.json',
              reason: 'stale_index',
              action: 'Regenerate public/data/index.json after source_manifest.json changes.',
            },
            {
              dimension: 'fred_readiness',
              code: 'fred_missing_api_key',
              severity: 'warning',
              provider: 'FRED',
              action: 'Set FRED_API_KEY in the deployment environment.',
            },
          ],
        },
      },
    } satisfies HealthData);

    expect(summary.dataPipelineSlo?.runbook?.topCause).toMatchObject({
      code: 'stale_public_data_index',
      severity: 'critical',
      label: 'artifact/index.json: Regenerate public/data/index.json after source_manifest.json changes.',
    });
    expect(summary.dataPipelineSlo?.runbook?.actions.map((action) => action.code)).toEqual([
      'stale_public_data_index',
      'fred_missing_api_key',
    ]);
    expect(summary.topCauses[0]).toBe(
      'artifact/index.json: Regenerate public/data/index.json after source_manifest.json changes.',
    );
  });
});

describe('RebalanceHealthPanel live diagnostics summary', () => {
  const baseHealth = (): RebalanceHealthData => ({
    generated: '2026-06-12T16:43:07.176691',
    next_rebalance: {
      date: '2026-06-11',
      days_until: -2,
      frequency: 'monthly (~30 days)',
    },
    schedule_compliance: {
      on_time: 0,
      delayed: 1,
      total: 1,
      compliance_pct: 0,
    },
    execution_history: [],
    total_executions: 37,
  });

  it('summarizes rejected feed entitlement as not live-acceptable', () => {
    const summary = summarizeRebalanceLiveDiagnostics({
      ...baseHealth(),
      alpaca_feed_entitlement: {
        configured_feed: 'iex',
        effective_feed: 'iex',
        entitlement: 'unknown',
        delayed: false,
        acceptable_for_live: false,
        policy_decision: 'reject',
        reason: 'missing_entitlement',
      },
    });

    expect(summary.hasDiagnostics).toBe(true);
    expect(summary.feedEntitlement).toEqual({
      status: 'reject',
      label: 'Feed reject: iex / unknown',
      detail: 'missing_entitlement',
      acceptableForLive: false,
    });
  });

  it('summarizes accepted feed entitlement as live-acceptable', () => {
    const summary = summarizeRebalanceLiveDiagnostics({
      ...baseHealth(),
      alpaca_feed_entitlement: {
        configured_feed: 'sip',
        effective_feed: 'sip',
        entitlement: 'sip',
        delayed: false,
        acceptable_for_live: true,
        policy_decision: 'accept',
      },
    });

    expect(summary.feedEntitlement?.status).toBe('accept');
    expect(summary.feedEntitlement?.detail).toBe('live acceptable');
    expect(summary.feedEntitlement?.acceptableForLive).toBe(true);
  });

  it('summarizes unavailable market-data consistency separately from feed policy', () => {
    const summary = summarizeRebalanceLiveDiagnostics({
      ...baseHealth(),
      market_data_consistency: {
        status: 'unavailable',
        reason: 'alpaca_not_configured',
        checked_at: '2026-06-12T08:43:07.177011+00:00',
        rows: [],
        warnings: [],
      },
    });

    expect(summary.marketDataConsistency).toEqual({
      status: 'unavailable',
      label: 'Market data unavailable',
      detail: 'alpaca_not_configured',
    });
  });

  it('returns no diagnostics for missing live-data sections', () => {
    const summary = summarizeRebalanceLiveDiagnostics(baseHealth());

    expect(summary).toEqual({
      hasDiagnostics: false,
      feedEntitlement: null,
      marketDataConsistency: null,
    });
  });
});
