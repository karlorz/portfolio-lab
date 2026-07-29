import { describe, expect, it } from 'bun:test';
import * as LabsSchemas from '../../src/schemas/labs';
import {
  LABS_DASHBOARD_ENDPOINTS,
  MAX_INDEXED_DIFF_FETCHES,
  PUBLIC_DATA_INDEX_ENDPOINT,
  buildEmptyLabsDashboardData,
  fetchLabsDashboardData,
  fetchLabsDashboardDataFromIndex,
} from '../../src/data/labs';
import {
  LabsDashboardDataSchema,
  type LabsDashboardData,
  type LabsEndpointKey,
  type LabsEndpointStatus,
  type LabsRegistryRow,
  LabsProvenanceSchema,
  LabsRegistrySchema,
  LabsReplaySchema,
  LabsScorecardSchema,
  LabsValidationReportSchema,
  parseLabsJson,
} from '../../src/schemas/labs';
import { buildLabsPanelViewModel } from '../../src/components/LabsPanel';
import { loadLabsFixture } from './labs-fixtures';

describe('Labs artifact schemas', () => {
  it('validates strict registry, provenance, scorecard, replay, and validation fixtures', () => {
    expect(LabsRegistrySchema.safeParse(loadLabsFixture('valid_registry')).success).toBe(true);
    expect(LabsProvenanceSchema.safeParse(loadLabsFixture('valid_provenance')).success).toBe(true);
    expect(LabsScorecardSchema.safeParse(loadLabsFixture('valid_scorecard')).success).toBe(true);
    expect(LabsReplaySchema.safeParse(loadLabsFixture('valid_replay_pass')).success).toBe(true);
    expect(LabsValidationReportSchema.safeParse(loadLabsFixture('validation_report')).success).toBe(true);
    expect(LabsValidationReportSchema.parse(loadLabsFixture('validation_report')).schema_version).toBe(
      'labs-validation/v1',
    );
  });

  it('validates static experiment diff artifacts for Labs dashboard consumption', () => {
    const schemas = LabsSchemas as Record<string, { safeParse: (value: unknown) => { success: boolean } }>;

    expect(schemas.LabsExperimentDiffSchema).toBeDefined();
    expect(schemas.LabsExperimentDiffSchema.safeParse(loadLabsFixture('valid_experiment_diff')).success).toBe(true);
  });

  it('accepts optional public data source metadata for market artifacts', () => {
    const result = LabsSchemas.PublicDataIndexSchema.safeParse({
      schema_version: 'public-data-index/v1',
      generated_at: '2026-06-11T00:00:00Z',
      entries: [
        {
          filename: 'prices.json',
          path: 'prices.json',
          category: 'market_data',
          schema_version: 'prices/compact-v1',
          status: 'present',
          validation_status: 'not_applicable',
          validation_errors: [],
          size_bytes: 123,
          size_budget: { render_strategy: 'direct' },
          sha256: 'a'.repeat(64),
          generated_at: '2026-06-11T00:00:00Z',
          source_manifest_path: 'source_manifest.json',
          source_metadata: {
            provider: 'Yahoo Finance',
            feed: 'chart/v8',
            source_mode: 'live',
            status: 'success',
            fetched_at: '2026-06-11T00:00:00Z',
            latest_observation: '2026-06-10',
            row_count: 1,
            data_quality: {
              artifact: 'data_quality.json',
              schema_version: 'price-data-quality/v1',
              generated_at: '2026-06-11T00:00:00Z',
              status: 'ok',
              issue_counts: {
                duplicate_dates: 0,
                empty_symbols: 0,
                extreme_returns: 0,
                internal_gaps: 0,
                invalid_dates: 0,
                invalid_prices: 0,
                missing_required_keys: 0,
                non_monotonic_rows: 0,
                non_object_records: 0,
                split_like_returns: 0,
                stale_latest_dates: 0,
                stale_latest_dates_within_tolerance: 1,
                total: 1,
              },
            },
          },
        },
        {
          filename: 'yields.json',
          path: 'yields.json',
          category: 'market_data',
          schema_version: 'yields/v1',
          status: 'present',
          validation_status: 'not_applicable',
          validation_errors: [],
          size_bytes: 456,
          size_budget: { render_strategy: 'direct' },
          sha256: 'b'.repeat(64),
          generated_at: '2026-06-11T00:00:00Z',
          source_manifest_path: 'source_manifest.json',
          source_metadata: {
            provider: 'FRED',
            feed: 'series/observations',
            source_mode: 'stale_cached',
            status: 'degraded',
            fetched_at: '2026-06-11T00:00:00Z',
            latest_observation: '2026-06-10',
            row_count: 1,
            failure_reason: 'cache_stale',
            fallback_reason: 'rate_limited',
          },
        },
      ],
    });

    expect(result.success).toBe(true);
  });

  it('rejects malformed public data quality source metadata', () => {
    const result = LabsSchemas.PublicDataSourceMetadataSchema.safeParse({
      provider: 'Yahoo Finance',
      feed: 'chart/v8',
      source_mode: 'live',
      status: 'success',
      data_quality: {
        artifact: 'data_quality.json',
        schema_version: 'price-data-quality/v1',
        generated_at: '2026-06-11T00:00:00Z',
        status: 'unknown',
        issue_counts: { total: 0 },
      },
    });

    expect(result.success).toBe(false);
  });

  it('validates optional top-level source manifest identity metadata', () => {
    const valid = LabsSchemas.PublicDataIndexSchema.safeParse({
      schema_version: 'public-data-index/v1',
      generated_at: '2026-06-11T00:00:00Z',
      source_manifest: {
        path: 'source_manifest.json',
        schema_version: 'market-data-source-manifest/v1',
        generated_at: '2026-06-11T00:00:00Z',
        sha256: 'a'.repeat(64),
      },
      entries: [],
    });
    const invalidHash = LabsSchemas.PublicDataIndexSchema.safeParse({
      schema_version: 'public-data-index/v1',
      generated_at: '2026-06-11T00:00:00Z',
      source_manifest: {
        path: 'source_manifest.json',
        schema_version: 'market-data-source-manifest/v1',
        generated_at: '2026-06-11T00:00:00Z',
        sha256: 'not-a-sha',
      },
      entries: [],
    });

    expect(valid.success).toBe(true);
    expect(invalidHash.success).toBe(false);
  });

  it('accepts generated registry metadata without allowing unrelated fields', () => {
    const registry = {
      ...loadLabsFixture('valid_registry'),
      sources: ['data/backtest_results/bad_result.json'],
      warnings: [
        {
          artifact_path: 'data/backtest_results/bad_result.json',
          error: 'invalid JSON: Expecting property name enclosed in double quotes',
        },
      ],
    };

    const result = LabsRegistrySchema.safeParse(registry);
    const unrelatedResult = LabsRegistrySchema.safeParse({
      ...registry,
      unexpected_metadata: true,
    });

    expect(result.success).toBe(true);
    expect(unrelatedResult.success).toBe(false);
  });

  it('accepts Labs governance disclosure fields for missing-provenance rows', () => {
    const registry = {
      ...loadLabsFixture('valid_registry'),
      experiments: [
        {
          ...loadLabsFixture('valid_registry').experiments[0],
          status: 'candidate',
          provenance_status: 'missing',
          governance_state: 'governance_blocked',
          governance_reasons: ['provenance_missing'],
          promotion_governance: {
            recommended_status: 'candidate',
            pass: false,
            failures: ['provenance_missing'],
            metric_gate_status: 'promoted',
            metric_gate_pass: true,
            governance_status: 'candidate',
            provenance_status: 'missing',
          },
        },
      ],
    };
    const scorecard = {
      ...loadLabsFixture('valid_scorecard'),
      status: 'watch',
      provenance_status: 'missing',
      governance_state: 'governance_blocked',
      governance_reasons: ['provenance_missing'],
      promotion_governance: {
        recommended_status: 'candidate',
        pass: false,
        failures: ['provenance_missing'],
        metric_gate_status: 'promoted',
        metric_gate_pass: true,
        governance_status: 'candidate',
        provenance_status: 'missing',
      },
    };

    expect(LabsRegistrySchema.safeParse(registry).success).toBe(true);
    expect(LabsScorecardSchema.safeParse(scorecard).success).toBe(true);
  });

  it('rejects malformed Labs fixture JSON with clear schema diagnostics', () => {
    const malformed = loadLabsFixture('invalid_missing_metrics');

    const result = LabsRegistrySchema.safeParse(malformed);

    expect(result.success).toBe(false);
    if (!result.success) {
      const paths = result.error.issues.map((issue) => issue.path.join('.'));
      expect(paths).toContain('experiments.0.metrics');
    }
  });

  it('accepts replay smoke status fields used by the experiment replay helper', () => {
    const replay = {
      ...loadLabsFixture('valid_replay_pass'),
      passed: true,
      command: 'python -m tests.fixtures.labs.fixture_experiment',
      duration_seconds: 0.12,
      metric_deltas: {
        sharpe: 0.006,
        cagr_pct: 0.05,
      },
    };

    expect(LabsReplaySchema.safeParse(replay).success).toBe(true);
  });

  it('accepts optional replay failure diagnostic fields', () => {
    const replay = {
      ...loadLabsFixture('valid_replay_pass'),
      status: 'failed',
      passed: false,
      failure_reason: 'timeout',
      error_type: 'TimeoutExpired',
      error_message: 'replay command timed out after 0.01 seconds',
    };

    const result = LabsReplaySchema.safeParse(replay);

    expect(result.success).toBe(true);
  });

  it('accepts optional validation report truncation metadata', () => {
    const report = {
      ...loadLabsFixture('validation_report'),
      truncation: {
        max_results: 2,
        max_errors_per_result: 2,
        total_result_count: 4,
        returned_result_count: 2,
        omitted_result_count: 2,
        omitted_error_count: 18,
      },
      results: [
        {
          ...loadLabsFixture('validation_report').results[0],
          errors: ['$.status: unsupported status'],
          omitted_error_count: 3,
        },
      ],
    };

    const result = LabsValidationReportSchema.safeParse(report);

    expect(result.success).toBe(true);
  });

  it('accepts optional validation row identity keys', () => {
    const report = {
      ...loadLabsFixture('validation_report'),
      results: [
        {
          ...loadLabsFixture('validation_report').results[0],
          experiment_id: 'gold-sweep',
          artifact_path: 'data/gold_allocation_sweep.json',
        },
      ],
    };

    const result = LabsValidationReportSchema.safeParse(report);

    expect(result.success).toBe(true);
  });

  it('builds an empty disabled Labs state when artifacts are absent', () => {
    const empty = buildEmptyLabsDashboardData(['registry', 'scorecards']);

    expect(empty.available).toBe(false);
    expect(empty.registry).toBeNull();
    expect(empty.scorecards).toEqual([]);
    expect(empty.replays).toEqual([]);
    expect(empty.validation).toBeNull();
    expect(empty.missing).toEqual(['registry', 'scorecards']);
    expect(LabsDashboardDataSchema.safeParse(empty).success).toBe(true);
  });

  it('parses typed Labs JSON and returns diagnostics instead of passthrough data', () => {
    const parsed = parseLabsJson(loadLabsFixture('valid_scorecard'), LabsScorecardSchema, 'scorecard');
    const malformed = parseLabsJson(loadLabsFixture('stale_schema'), LabsRegistrySchema, 'registry');

    expect(parsed.data?.experiment_id).toBe('gold-sweep');
    expect(parsed.errors).toEqual([]);
    expect(malformed.data).toBeNull();
    expect(malformed.errors[0]).toContain('registry.schema_version');
  });
});

describe('Labs dashboard data fetch helper', () => {
  const indexEntry = (filename: string, overrides: Record<string, unknown> = {}) => ({
    filename,
    path: filename,
    category: 'labs',
    schema_version: 'labs-registry/v1',
    status: 'present',
    validation_status: 'valid',
    validation_errors: [],
    size_bytes: 1000,
    size_budget: {
      schema_version: 'public-data-size-budget/v1',
      status: 'within_budget',
      size_bytes: 1000,
      row_count: 1,
      estimated_parse_ms: 0.02,
      warning_bytes: 524288,
      max_bytes: 1048576,
      warning_rows: 500,
      max_rows: 1000,
      requires_downsampling: false,
      requires_pagination: false,
      render_strategy: 'direct',
    },
    sha256: 'a'.repeat(64),
    generated_at: '2026-06-08T12:00:00+00:00',
    ...overrides,
  });

  const publicIndex = (entries: unknown[]) => ({
    schema_version: 'public-data-index/v1',
    files: entries
      .filter((entry) => typeof entry === 'object' && entry !== null && (entry as { status?: string }).status === 'present')
      .map((entry) => (entry as { filename: string }).filename),
    entries,
    generated_at: '2026-06-08T12:00:00+00:00',
  });

  it('returns a disabled empty state when Labs artifacts are missing', async () => {
    const fetcher = async () => new Response('', { status: 404 });

    const data = await fetchLabsDashboardData(fetcher);

    expect(data.available).toBe(false);
    expect(data.missing).toEqual([
      'registry',
      'scorecards',
      'replays',
      'validation',
    ]);
    expect(data.errors).toEqual([]);
  });

  it('treats static HTML fallback responses as missing optional Labs endpoints', async () => {
    const fetcher = async () => new Response('<!doctype html><title>App</title>', {
      status: 200,
      headers: { 'content-type': 'text/html; charset=utf-8' },
    });

    const data = await fetchLabsDashboardData(fetcher);

    expect(data.available).toBe(false);
    expect(data.missing).toEqual([
      'registry',
      'scorecards',
      'replays',
      'validation',
    ]);
    expect(data.errors).toEqual([]);
  });

  it('keeps malformed JSON diagnostics when Labs endpoints claim JSON content', async () => {
    const fetcher = async (url: string) => {
      if (url === LABS_DASHBOARD_ENDPOINTS.registry) {
        return new Response('{not-json', {
          status: 200,
          headers: { 'content-type': 'application/json' },
        });
      }
      return new Response('', { status: 404 });
    };

    const data = await fetchLabsDashboardData(fetcher);

    expect(data.missing).toEqual(['scorecards', 'replays', 'validation']);
    expect(data.errors).toHaveLength(1);
    expect(data.errors[0]).toContain('registry: invalid JSON');
  });

  it('fetches and validates the endpoint group without ad hoc dashboard state', async () => {
    const payloads: Record<string, unknown> = {
      [LABS_DASHBOARD_ENDPOINTS.registry]: loadLabsFixture('valid_registry'),
      [LABS_DASHBOARD_ENDPOINTS.scorecards]: [loadLabsFixture('valid_scorecard')],
      [LABS_DASHBOARD_ENDPOINTS.replays]: [loadLabsFixture('valid_replay_pass')],
      [LABS_DASHBOARD_ENDPOINTS.validation]: loadLabsFixture('validation_report'),
    };
    const fetcher = async (url: string) => new Response(JSON.stringify(payloads[url]), { status: 200 });

    const data = await fetchLabsDashboardData(fetcher);

    expect(data.available).toBe(true);
    expect(data.registry?.experiments[0].experiment_id).toBe('gold-sweep');
    expect(data.scorecards[0].status).toBe('promote');
    expect(data.replays[0].status).toBe('passed');
    expect(data.validation?.results[0].valid).toBe(false);
    expect(data.missing).toEqual([]);
    expect(data.errors).toEqual([]);
  });

  it('caps oversized Labs rows before dashboard rendering', async () => {
    const registryFixture = loadLabsFixture('valid_registry');
    const registryRow = registryFixture.experiments[0];
    const scorecardFixture = loadLabsFixture('valid_scorecard');
    const replayFixture = loadLabsFixture('valid_replay_pass');
    const payloads: Record<string, unknown> = {
      [LABS_DASHBOARD_ENDPOINTS.registry]: {
        ...registryFixture,
        experiments: Array.from({ length: 150 }, (_, idx) => ({
          ...registryRow,
          experiment_id: `experiment-${idx}`,
        })),
      },
      [LABS_DASHBOARD_ENDPOINTS.scorecards]: Array.from({ length: 150 }, (_, idx) => ({
        ...scorecardFixture,
        experiment_id: `experiment-${idx}`,
      })),
      [LABS_DASHBOARD_ENDPOINTS.replays]: Array.from({ length: 150 }, (_, idx) => ({
        ...replayFixture,
        experiment_id: `experiment-${idx}`,
      })),
      [LABS_DASHBOARD_ENDPOINTS.validation]: loadLabsFixture('validation_report'),
    };
    const fetcher = async (url: string) => new Response(JSON.stringify(payloads[url]), { status: 200 });

    const data = await fetchLabsDashboardData(fetcher);

    expect(data.registry?.experiments).toHaveLength(100);
    expect(data.scorecards).toHaveLength(100);
    expect(data.replays).toHaveLength(100);
    expect(data.available).toBe(true);
  });

  it('caps Labs rows by retained registry ids before filtering scorecards and replays', async () => {
    const registryFixture = loadLabsFixture('valid_registry');
    const registryRow = registryFixture.experiments[0];
    const scorecardFixture = loadLabsFixture('valid_scorecard');
    const replayFixture = loadLabsFixture('valid_replay_pass');
    const registryIds = Array.from({ length: 150 }, (_, idx) => `experiment-${idx}`);
    const reversedIds = [...registryIds].reverse();
    const payloads: Record<string, unknown> = {
      [LABS_DASHBOARD_ENDPOINTS.registry]: {
        ...registryFixture,
        experiments: registryIds.map((experiment_id) => ({
          ...registryRow,
          experiment_id,
          artifact_path: `data/${experiment_id}.json`,
        })),
      },
      [LABS_DASHBOARD_ENDPOINTS.scorecards]: reversedIds.map((experiment_id) => ({
        ...scorecardFixture,
        experiment_id,
      })),
      [LABS_DASHBOARD_ENDPOINTS.replays]: reversedIds.map((experiment_id) => ({
        ...replayFixture,
        experiment_id,
        artifact_path: `data/${experiment_id}.json`,
      })),
      [LABS_DASHBOARD_ENDPOINTS.validation]: loadLabsFixture('validation_report'),
    };
    const fetcher = async (url: string) => new Response(JSON.stringify(payloads[url]), { status: 200 });

    const data = await fetchLabsDashboardData(fetcher);
    const retainedRegistryIds = new Set(data.registry?.experiments.map((row) => row.experiment_id));
    const view = buildLabsPanelViewModel(data, {
      sortBy: 'status',
      sortDirection: 'asc',
    });

    expect(data.registry?.experiments).toHaveLength(100);
    expect(data.scorecards).toHaveLength(100);
    expect(data.replays).toHaveLength(100);
    expect([...retainedRegistryIds]).toEqual(registryIds.slice(0, 100));
    expect(data.scorecards.every((row) => retainedRegistryIds.has(row.experiment_id))).toBe(true);
    expect(data.replays.every((row) => retainedRegistryIds.has(row.experiment_id))).toBe(true);
    expect(view.rows.every((row) => row.scorecardStatus !== 'missing')).toBe(true);
    expect(view.rows.every((row) => row.replayStatus !== 'missing')).toBe(true);
  });

  it('uses public index metadata to fetch present valid Labs endpoints', async () => {
    const payloads: Record<string, unknown> = {
      [PUBLIC_DATA_INDEX_ENDPOINT]: publicIndex([
        indexEntry('labs_registry.json'),
        indexEntry('labs_scorecards.json', { schema_version: 'labs-scorecard/v1' }),
        indexEntry('labs_replays.json', { schema_version: 'labs-replay/v1' }),
        indexEntry('labs_validation.json', { schema_version: 'labs-validation/v1' }),
      ]),
      [LABS_DASHBOARD_ENDPOINTS.registry]: loadLabsFixture('valid_registry'),
      [LABS_DASHBOARD_ENDPOINTS.scorecards]: [loadLabsFixture('valid_scorecard')],
      [LABS_DASHBOARD_ENDPOINTS.replays]: [loadLabsFixture('valid_replay_pass')],
      [LABS_DASHBOARD_ENDPOINTS.validation]: loadLabsFixture('validation_report'),
    };
    const requested: string[] = [];
    const fetcher = async (url: string) => {
      requested.push(url);
      return new Response(JSON.stringify(payloads[url]), { status: 200 });
    };

    const data = await fetchLabsDashboardDataFromIndex(fetcher);

    expect(requested).toEqual([
      PUBLIC_DATA_INDEX_ENDPOINT,
      LABS_DASHBOARD_ENDPOINTS.registry,
      LABS_DASHBOARD_ENDPOINTS.scorecards,
      LABS_DASHBOARD_ENDPOINTS.replays,
      LABS_DASHBOARD_ENDPOINTS.validation,
    ]);
    expect(data.available).toBe(true);
    expect(data.endpoint_status?.map((entry) => entry.render_strategy)).toEqual([
      'direct',
      'direct',
      'direct',
      'direct',
    ]);
    expect(data.registry?.experiments[0].experiment_id).toBe('gold-sweep');
  });

  it('does not fetch Labs endpoints marked missing in the public index', async () => {
    const payloads: Record<string, unknown> = {
      [PUBLIC_DATA_INDEX_ENDPOINT]: publicIndex([
        indexEntry('labs_registry.json', {
          status: 'missing',
          validation_status: 'missing',
          size_bytes: null,
          sha256: null,
          size_budget: {
            schema_version: 'public-data-size-budget/v1',
            status: 'missing',
            size_bytes: null,
            row_count: null,
            estimated_parse_ms: null,
            warning_bytes: 524288,
            max_bytes: 1048576,
            warning_rows: 500,
            max_rows: 1000,
            requires_downsampling: false,
            requires_pagination: false,
            render_strategy: 'missing',
          },
        }),
        indexEntry('labs_scorecards.json', { schema_version: 'labs-scorecard/v1' }),
      ]),
      [LABS_DASHBOARD_ENDPOINTS.scorecards]: [loadLabsFixture('valid_scorecard')],
    };
    const requested: string[] = [];
    const fetcher = async (url: string) => {
      requested.push(url);
      if (url === LABS_DASHBOARD_ENDPOINTS.registry) {
        throw new Error('missing registry endpoint should not be fetched');
      }
      return new Response(JSON.stringify(payloads[url]), { status: 200 });
    };

    const data = await fetchLabsDashboardDataFromIndex(fetcher);

    expect(requested).toEqual([PUBLIC_DATA_INDEX_ENDPOINT, LABS_DASHBOARD_ENDPOINTS.scorecards]);
    expect(data.missing).toContain('registry');
    expect(data.endpoint_status?.find((entry) => entry.endpoint === 'registry')?.status).toBe('missing');
    expect(data.scorecards[0].experiment_id).toBe('gold-sweep');
  });

  it('surfaces invalid Labs index entries without fetching invalid endpoints', async () => {
    const payloads: Record<string, unknown> = {
      [PUBLIC_DATA_INDEX_ENDPOINT]: publicIndex([
        indexEntry('labs_registry.json', {
          validation_status: 'invalid',
          validation_errors: ['$.experiments[0].metrics: missing required field'],
        }),
      ]),
    };
    const requested: string[] = [];
    const fetcher = async (url: string) => {
      requested.push(url);
      if (url === LABS_DASHBOARD_ENDPOINTS.registry) {
        throw new Error('invalid registry endpoint should not be fetched');
      }
      return new Response(JSON.stringify(payloads[url]), { status: 200 });
    };

    const data = await fetchLabsDashboardDataFromIndex(fetcher);

    expect(requested).toEqual([PUBLIC_DATA_INDEX_ENDPOINT]);
    expect(data.available).toBe(false);
    expect(data.errors).toEqual(['registry: $.experiments[0].metrics: missing required field']);
    expect(data.endpoint_status?.[0].validation_status).toBe('invalid');
  });

  it('uses public index summary metadata for summarized Labs endpoints without fetching full artifacts', async () => {
    const payloads: Record<string, unknown> = {
      [PUBLIC_DATA_INDEX_ENDPOINT]: publicIndex([
        indexEntry('labs_registry.json', {
          size_bytes: 750_000,
          size_budget: {
            schema_version: 'public-data-size-budget/v1',
            status: 'warning',
            size_bytes: 750_000,
            row_count: 1500,
            estimated_parse_ms: 18,
            warning_bytes: 524288,
            max_bytes: 1048576,
            warning_rows: 500,
            max_rows: 2000,
            requires_downsampling: true,
            requires_pagination: false,
            render_strategy: 'summarize',
          },
        }),
      ]),
    };
    const requested: string[] = [];
    const fetcher = async (url: string) => {
      requested.push(url);
      if (url === LABS_DASHBOARD_ENDPOINTS.registry) {
        throw new Error('summarized registry endpoint should not be fetched');
      }
      return new Response(JSON.stringify(payloads[url]), { status: 200 });
    };

    const data = await fetchLabsDashboardDataFromIndex(fetcher);
    const registryStatus = data.endpoint_status?.find((entry) => entry.endpoint === 'registry');

    expect(requested).toEqual([PUBLIC_DATA_INDEX_ENDPOINT]);
    expect(data.available).toBe(true);
    expect(data.registry).toBeNull();
    expect(data.missing).not.toContain('registry');
    expect(registryStatus).toMatchObject({
      endpoint: 'registry',
      render_strategy: 'summarize',
      summary_limited: true,
      row_count: 1500,
      size_bytes: 750_000,
      requires_downsampling: true,
      validation_status: 'valid',
    });
    expect(data.errors).toEqual([]);
  });

  it('exposes oversized Labs render strategy metadata without blindly fetching paginated payloads', async () => {
    const payloads: Record<string, unknown> = {
      [PUBLIC_DATA_INDEX_ENDPOINT]: publicIndex([
        indexEntry('labs_registry.json', {
          size_bytes: 2_000_000,
          size_budget: {
            schema_version: 'public-data-size-budget/v1',
            status: 'oversized',
            size_bytes: 2_000_000,
            row_count: 2000,
            estimated_parse_ms: 40,
            warning_bytes: 524288,
            max_bytes: 1048576,
            warning_rows: 500,
            max_rows: 1000,
            requires_downsampling: true,
            requires_pagination: true,
            render_strategy: 'paginate',
          },
        }),
      ]),
    };
    const requested: string[] = [];
    const fetcher = async (url: string) => {
      requested.push(url);
      if (url === LABS_DASHBOARD_ENDPOINTS.registry) {
        throw new Error('paginated registry endpoint should not be fetched');
      }
      return new Response(JSON.stringify(payloads[url]), { status: 200 });
    };

    const data = await fetchLabsDashboardDataFromIndex(fetcher);

    expect(requested).toEqual([PUBLIC_DATA_INDEX_ENDPOINT]);
    expect(data.available).toBe(false);
    expect(data.endpoint_status?.[0]).toMatchObject({
      endpoint: 'registry',
      render_strategy: 'paginate',
      size_bytes: 2_000_000,
    });
    expect(data.errors).toEqual(['registry: render strategy paginate requires paginated Labs artifact access']);
  });

  it('fetches the first static page for paginated Labs endpoints with complete metadata', async () => {
    const registryFixture = loadLabsFixture('valid_registry');
    const registryRow = registryFixture.experiments[0];
    const pagePath = 'labs_registry.page-1.json';
    const pagePayload = {
      ...registryFixture,
      experiments: [
        registryRow,
        {
          ...registryRow,
          experiment_id: 'gold-sweep-page-2',
          artifact_path: 'data/gold_sweep_page_2.json',
        },
      ],
    };
    const payloads: Record<string, unknown> = {
      [PUBLIC_DATA_INDEX_ENDPOINT]: publicIndex([
        indexEntry('labs_registry.json', {
          size_bytes: 2_000_000,
          size_budget: {
            schema_version: 'public-data-size-budget/v1',
            status: 'oversized',
            size_bytes: 2_000_000,
            row_count: 2000,
            estimated_parse_ms: 40,
            warning_bytes: 524288,
            max_bytes: 1048576,
            warning_rows: 500,
            max_rows: 1000,
            requires_downsampling: true,
            requires_pagination: true,
            render_strategy: 'paginate',
          },
          pagination: {
            total_rows: 2000,
            page_size: 2,
            page_count: 1000,
            pages: [{ page: 1, path: pagePath, row_count: 2 }],
          },
        }),
      ]),
      [`/data/${pagePath}`]: pagePayload,
    };
    const requested: string[] = [];
    const fetcher = async (url: string) => {
      requested.push(url);
      if (url === LABS_DASHBOARD_ENDPOINTS.registry) {
        throw new Error('paginated registry endpoint should not be fetched directly');
      }
      return new Response(JSON.stringify(payloads[url]), { status: 200 });
    };

    const data = await fetchLabsDashboardDataFromIndex(fetcher);
    const registryStatus = data.endpoint_status?.find((entry) => entry.endpoint === 'registry');

    expect(requested).toEqual([PUBLIC_DATA_INDEX_ENDPOINT, `/data/${pagePath}`]);
    expect(data.errors).toEqual([]);
    expect(data.registry?.experiments).toHaveLength(2);
    expect(registryStatus?.render_strategy).toBe('paginate');
    expect(registryStatus?.pagination?.total_rows).toBe(2000);
    expect(registryStatus?.pagination?.pages[0].path).toBe(pagePath);
  });

  it('fetches the selected static page for paginated Labs endpoints without reloading every page', async () => {
    const registryFixture = loadLabsFixture('valid_registry');
    const registryRow = registryFixture.experiments[0];
    const firstPagePath = 'labs_registry.page-1.json';
    const secondPagePath = 'labs_registry.page-2.json';
    const secondPagePayload = {
      ...registryFixture,
      experiments: [
        {
          ...registryRow,
          experiment_id: 'second-page-experiment',
          artifact_path: 'data/second_page_experiment.json',
        },
      ],
    };
    const payloads: Record<string, unknown> = {
      [PUBLIC_DATA_INDEX_ENDPOINT]: publicIndex([
        indexEntry('labs_registry.json', {
          size_bytes: 2_000_000,
          size_budget: {
            schema_version: 'public-data-size-budget/v1',
            status: 'oversized',
            size_bytes: 2_000_000,
            row_count: 2000,
            estimated_parse_ms: 40,
            warning_bytes: 524288,
            max_bytes: 1048576,
            warning_rows: 500,
            max_rows: 1000,
            requires_downsampling: true,
            requires_pagination: true,
            render_strategy: 'paginate',
          },
          pagination: {
            total_rows: 2000,
            page_size: 1,
            page_count: 2,
            pages: [
              { page: 1, path: firstPagePath, row_count: 1 },
              { page: 2, path: secondPagePath, row_count: 1 },
            ],
          },
        }),
      ]),
      [`/data/${secondPagePath}`]: secondPagePayload,
    };
    const requested: string[] = [];
    const fetcher = async (url: string) => {
      requested.push(url);
      if (url === `/data/${firstPagePath}`) {
        throw new Error('selected page fetch should not request page 1');
      }
      return new Response(JSON.stringify(payloads[url]), { status: 200 });
    };

    const data = await fetchLabsDashboardDataFromIndex(fetcher, {
      selectedPages: { registry: 2 },
    });

    expect(requested).toEqual([PUBLIC_DATA_INDEX_ENDPOINT, `/data/${secondPagePath}`]);
    expect(data.registry?.experiments.map((row) => row.experiment_id)).toEqual(['second-page-experiment']);
    expect(data.endpoint_status?.find((entry) => entry.endpoint === 'registry')?.selected_page).toBe(2);
    expect(data.errors).toEqual([]);
  });

  it('passes an abort signal to indexed Labs index and selected page fetches', async () => {
    const registryFixture = loadLabsFixture('valid_registry');
    const registryRow = registryFixture.experiments[0];
    const firstPagePath = 'labs_registry.page-1.json';
    const secondPagePath = 'labs_registry.page-2.json';
    const controller = new AbortController();
    const secondPagePayload = {
      ...registryFixture,
      experiments: [
        {
          ...registryRow,
          experiment_id: 'second-page-experiment',
          artifact_path: 'data/second_page_experiment.json',
        },
      ],
    };
    const payloads: Record<string, unknown> = {
      [PUBLIC_DATA_INDEX_ENDPOINT]: publicIndex([
        indexEntry('labs_registry.json', {
          size_bytes: 2_000_000,
          size_budget: {
            schema_version: 'public-data-size-budget/v1',
            status: 'oversized',
            size_bytes: 2_000_000,
            row_count: 2000,
            estimated_parse_ms: 40,
            warning_bytes: 524288,
            max_bytes: 1048576,
            warning_rows: 500,
            max_rows: 1000,
            requires_downsampling: true,
            requires_pagination: true,
            render_strategy: 'paginate',
          },
          pagination: {
            total_rows: 2000,
            page_size: 1,
            page_count: 2,
            pages: [
              { page: 1, path: firstPagePath, row_count: 1 },
              { page: 2, path: secondPagePath, row_count: 1 },
            ],
          },
        }),
      ]),
      [`/data/${secondPagePath}`]: secondPagePayload,
    };
    const observed: Array<{ url: string; signal: AbortSignal | null }> = [];
    const fetcher = async (url: string, init?: RequestInit) => {
      observed.push({ url, signal: init?.signal ?? null });
      return new Response(JSON.stringify(payloads[url]), { status: 200 });
    };

    await fetchLabsDashboardDataFromIndex(fetcher, {
      selectedPages: { registry: 2 },
      signal: controller.signal,
    });

    expect(observed).toEqual([
      { url: PUBLIC_DATA_INDEX_ENDPOINT, signal: controller.signal },
      { url: `/data/${secondPagePath}`, signal: controller.signal },
    ]);
  });

  it('propagates AbortError from indexed Labs fetches as cancellation', async () => {
    const abortError = Object.assign(new Error('request aborted'), { name: 'AbortError' });
    const controller = new AbortController();
    const fetcher = async () => {
      throw abortError;
    };
    let caught: unknown;

    try {
      await fetchLabsDashboardDataFromIndex(fetcher, { signal: controller.signal });
    } catch (error) {
      caught = error;
    }

    expect(caught).toBe(abortError);
  });

  it('surfaces missing paginated Labs page shards as endpoint diagnostics', async () => {
    const pagePath = 'labs_registry.page-1.json';
    const payloads: Record<string, unknown> = {
      [PUBLIC_DATA_INDEX_ENDPOINT]: publicIndex([
        indexEntry('labs_registry.json', {
          size_bytes: 2_000_000,
          size_budget: {
            schema_version: 'public-data-size-budget/v1',
            status: 'oversized',
            size_bytes: 2_000_000,
            row_count: 2000,
            estimated_parse_ms: 40,
            warning_bytes: 524288,
            max_bytes: 1048576,
            warning_rows: 500,
            max_rows: 1000,
            requires_downsampling: true,
            requires_pagination: true,
            render_strategy: 'paginate',
          },
          pagination: {
            total_rows: 2000,
            page_size: 100,
            page_count: 20,
            pages: [{ page: 1, path: pagePath, row_count: 100 }],
          },
        }),
      ]),
    };
    const requested: string[] = [];
    const fetcher = async (url: string) => {
      requested.push(url);
      if (url === `/data/${pagePath}`) {
        return new Response('', { status: 404 });
      }
      return new Response(JSON.stringify(payloads[url]), { status: 200 });
    };

    const data = await fetchLabsDashboardDataFromIndex(fetcher);

    expect(requested).toEqual([PUBLIC_DATA_INDEX_ENDPOINT, `/data/${pagePath}`]);
    expect(data.available).toBe(false);
    expect(data.errors).toEqual([`registry: paginated shard missing (${pagePath})`]);
  });

  it('fetches static experiment diff artifacts from the public index', async () => {
    const diffPath = 'experiment_diff.json';
    const payloads: Record<string, unknown> = {
      [PUBLIC_DATA_INDEX_ENDPOINT]: publicIndex([
        indexEntry(diffPath, {
          category: 'labs',
          schema_version: 'experiment-diff/v1',
        }),
      ]),
      [`/data/${diffPath}`]: loadLabsFixture('valid_experiment_diff'),
    };
    const requested: string[] = [];
    const fetcher = async (url: string) => {
      requested.push(url);
      return new Response(JSON.stringify(payloads[url]), { status: 200 });
    };

    const data = await fetchLabsDashboardDataFromIndex(fetcher);

    expect(requested).toEqual([PUBLIC_DATA_INDEX_ENDPOINT, `/data/${diffPath}`]);
    expect(data.diffs?.[0].metric_deltas.sharpe.delta).toBe(0.04);
    expect(data.errors).toEqual([]);
  });

  it('caps indexed experiment-diff fetches at maxIndexedDiffFetches and surfaces skipped count', async () => {
    const diffPathA = 'experiment_diff_a.json';
    const diffPathB = 'experiment_diff_b.json';
    const payloads: Record<string, unknown> = {
      [PUBLIC_DATA_INDEX_ENDPOINT]: publicIndex([
        indexEntry(diffPathA, {
          category: 'labs',
          schema_version: 'experiment-diff/v1',
        }),
        indexEntry(diffPathB, {
          category: 'labs',
          schema_version: 'experiment-diff/v1',
        }),
      ]),
      [`/data/${diffPathA}`]: loadLabsFixture('valid_experiment_diff'),
      [`/data/${diffPathB}`]: loadLabsFixture('valid_experiment_diff'),
    };
    const requested: string[] = [];
    const fetcher = async (url: string) => {
      requested.push(url);
      return new Response(JSON.stringify(payloads[url]), { status: 200 });
    };

    const data = await fetchLabsDashboardDataFromIndex(fetcher, { maxIndexedDiffFetches: 1 });

    expect(requested).toEqual([PUBLIC_DATA_INDEX_ENDPOINT, `/data/${diffPathA}`]);
    expect(data.diffs?.length).toBe(1);
    expect(data.errors).toEqual([
      'diff:budget: skipped 1 of 2 experiment-diff artifacts (limit 1)',
    ]);
  });

  it('exposes MAX_INDEXED_DIFF_FETCHES as a stable default', () => {
    expect(MAX_INDEXED_DIFF_FETCHES).toBeGreaterThan(0);
    expect(typeof MAX_INDEXED_DIFF_FETCHES).toBe('number');
  });

  it('falls back to direct Labs endpoints when the public index is missing or legacy-only', async () => {
    const payloads: Record<string, unknown> = {
      [LABS_DASHBOARD_ENDPOINTS.registry]: loadLabsFixture('valid_registry'),
      [LABS_DASHBOARD_ENDPOINTS.scorecards]: [loadLabsFixture('valid_scorecard')],
      [LABS_DASHBOARD_ENDPOINTS.replays]: [loadLabsFixture('valid_replay_pass')],
      [LABS_DASHBOARD_ENDPOINTS.validation]: loadLabsFixture('validation_report'),
    };
    const requested: string[] = [];
    const fetcher = async (url: string) => {
      requested.push(url);
      if (url === PUBLIC_DATA_INDEX_ENDPOINT) {
        return new Response('', { status: 404 });
      }
      return new Response(JSON.stringify(payloads[url]), { status: 200 });
    };

    const data = await fetchLabsDashboardDataFromIndex(fetcher);

    expect(requested).toEqual([
      PUBLIC_DATA_INDEX_ENDPOINT,
      LABS_DASHBOARD_ENDPOINTS.registry,
      LABS_DASHBOARD_ENDPOINTS.scorecards,
      LABS_DASHBOARD_ENDPOINTS.replays,
      LABS_DASHBOARD_ENDPOINTS.validation,
    ]);
    expect(data.available).toBe(true);
    expect(data.endpoint_status).toEqual([]);
  });
});

describe('Labs panel view model', () => {
  const endpointStatus = (
    endpoint: LabsEndpointKey,
    overrides: Partial<LabsEndpointStatus> = {},
  ): LabsEndpointStatus => ({
    endpoint,
    filename: `labs_${endpoint}.json`,
    path: `labs_${endpoint}.json`,
    status: 'present',
    validation_status: 'valid',
    validation_errors: [],
    render_strategy: 'direct',
    size_bytes: 1000,
    generated_at: '2026-06-08T12:00:00+00:00',
    ...overrides,
  });

  const registryRow = (
    experiment_id: string,
    overrides: Partial<LabsRegistryRow> = {},
  ): LabsRegistryRow => ({
    experiment_id,
    artifact_path: `data/${experiment_id}.json`,
    status: 'validated',
    provenance_status: 'present',
    metrics: {
      sharpe: 0.95,
      cagr_pct: 10.4,
      max_drawdown_pct: -25,
    },
    baseline_deltas: {
      sharpe: 0.04,
    },
    ...overrides,
  });

  const labsData = (): LabsDashboardData => ({
    available: true,
    registry: {
      schema_version: 'labs-registry/v1',
      generated_at: '2026-06-08T12:00:00+00:00',
      experiments: [
        registryRow('gold-sweep'),
        registryRow('risk-sweep', {
          status: 'warning',
          provenance_status: 'missing',
          metrics: {
            sharpe: 0.72,
            cagr_pct: 8.1,
            max_drawdown_pct: -18,
          },
        }),
      ],
    },
    scorecards: [
      {
        schema_version: 'labs-scorecard/v1',
        experiment_id: 'gold-sweep',
        generated_at: '2026-06-08T12:00:00+00:00',
        status: 'promote',
        provenance_status: 'present',
        metrics: { sharpe: 0.95 },
        baseline_deltas: { sharpe: 0.04 },
      },
      {
        schema_version: 'labs-scorecard/v1',
        experiment_id: 'risk-sweep',
        generated_at: '2026-06-08T12:00:00+00:00',
        status: 'watch',
        provenance_status: 'missing',
        metrics: { sharpe: 0.72 },
        baseline_deltas: { sharpe: -0.02 },
      },
    ],
    replays: [
      {
        schema_version: 'labs-replay/v1',
        experiment_id: 'gold-sweep',
        generated_at: '2026-06-08T12:00:00+00:00',
        artifact_path: 'data/gold-sweep.json',
        status: 'passed',
        provenance_status: 'present',
        metrics: { rows_replayed: 109 },
        baseline_deltas: { sharpe: 0 },
      },
      {
        schema_version: 'labs-replay/v1',
        experiment_id: 'risk-sweep',
        generated_at: '2026-06-08T12:00:00+00:00',
        artifact_path: 'data/risk-sweep.json',
        status: 'failed',
        provenance_status: 'missing',
        metrics: { rows_replayed: 3 },
        baseline_deltas: { sharpe: -0.1 },
      },
    ],
    validation: {
      results: [
        {
          path: 'data/gold-sweep.json',
          artifact_type: 'registry',
          schema_version: 'labs-registry/v1',
          valid: false,
          errors: ['fixture validation warning'],
        },
        {
          path: 'data/risk-sweep.json',
          artifact_type: 'registry',
          schema_version: 'labs-registry/v1',
          valid: true,
          errors: [],
        },
      ],
    },
    missing: [],
    errors: [],
  });

  it('builds a disabled missing-artifact state without throwing', () => {
    const view = buildLabsPanelViewModel(buildEmptyLabsDashboardData(['registry', 'scorecards']));

    expect(view.disabled).toBe(true);
    expect(view.emptyMessage).toBe('Labs artifacts are not published yet');
    expect(view.missingEndpoints).toEqual(['registry', 'scorecards']);
    expect(view.rows).toEqual([]);
    expect(view.summary.available).toBe(false);
  });

  it('surfaces summary-limited endpoint counts without disabling Labs state', () => {
    const data: LabsDashboardData = {
      available: true,
      registry: null,
      scorecards: [],
      replays: [],
      validation: null,
      diffs: [],
      missing: [],
      errors: [],
      endpoint_status: [
        {
          endpoint: 'registry',
          filename: 'labs_registry.json',
          path: 'labs_registry.json',
          status: 'present',
          validation_status: 'valid',
          validation_errors: [],
          render_strategy: 'summarize',
          size_bytes: 750_000,
          row_count: 1500,
          requires_downsampling: true,
          requires_pagination: false,
          summary_limited: true,
          generated_at: '2026-06-08T12:00:00+00:00',
        },
      ],
    };

    const view = buildLabsPanelViewModel(data);

    expect(view.disabled).toBe(false);
    expect(view.emptyMessage).toContain('summary metadata');
    expect(view.summary.summaryLimitedEndpoints).toBe(1);
    expect(view.summary.indexedRows).toBe(1500);
    expect(view.summaryLimitedEndpoints[0]).toMatchObject({
      endpoint: 'registry',
      row_count: 1500,
      validation_status: 'valid',
    });
  });

  it('exposes supported Labs sort and filter dimensions', () => {
    const view = buildLabsPanelViewModel(labsData());

    expect(view.sortOptions.map((option) => option.value)).toEqual([
      'sharpe',
      'max_drawdown',
      'wfe',
      'dsr',
      'status',
      'provenance_status',
      'replay_status',
      'validation_status',
    ]);
    expect(view.filterOptions.status).toContain('validated');
    expect(view.filterOptions.provenanceStatus).toContain('missing');
    expect(view.filterOptions.replayStatus).toContain('failed');
    expect(view.filterOptions.validationStatus).toEqual(['all', 'valid', 'invalid', 'missing']);
  });

  it('summarizes missing-provenance candidate/watch rows with governance reasons', () => {
    const data = labsData();
    data.registry!.experiments = [
      registryRow('metric-only', {
        status: 'candidate',
        provenance_status: 'missing',
        metrics: {
          sharpe: 1.21,
          cagr_pct: 12.4,
          max_drawdown_pct: -14,
        },
        baseline_deltas: {
          sharpe: 0.08,
        },
        governance_state: 'governance_blocked',
        governance_reasons: ['provenance_missing'],
      }),
      registryRow('clean-watch', {
        status: 'candidate',
        provenance_status: 'present',
      }),
    ];
    data.scorecards = [
      {
        schema_version: 'labs-scorecard/v1',
        experiment_id: 'metric-only',
        generated_at: '2026-06-08T12:00:00+00:00',
        status: 'watch',
        provenance_status: 'missing',
        metrics: { sharpe: 1.21 },
        baseline_deltas: { sharpe: 0.08 },
        governance_state: 'governance_blocked',
        governance_reasons: ['provenance_missing'],
      },
    ];

    const view = buildLabsPanelViewModel(data);

    expect(view.summary.missingProvenanceCandidates).toBe(1);
    expect(view.rows.find((row) => row.experimentId === 'metric-only')).toMatchObject({
      governanceState: 'governance_blocked',
      governanceReasons: ['provenance_missing'],
    });
  });

  it('filters Labs rows by status, provenance, replay, and validation state', () => {
    const view = buildLabsPanelViewModel(labsData(), {
      status: 'validated',
      provenanceStatus: 'present',
      replayStatus: 'passed',
      validationStatus: 'invalid',
      sortBy: 'sharpe',
      sortDirection: 'desc',
    });

    expect(view.disabled).toBe(false);
    expect(view.rows.map((row) => row.experimentId)).toEqual(['gold-sweep']);
    expect(view.rows[0].scorecardStatus).toBe('promote');
    expect(view.rows[0].replayStatus).toBe('passed');
    expect(view.rows[0].validationStatus).toBe('invalid');
  });

  it('filters Labs rows by text search and numeric metric thresholds without matching missing metrics', () => {
    const data = labsData();
    data.registry?.experiments.push(
      registryRow('duration-lab', {
        artifact_path: 'data/tlt_duration_lab.json',
        metrics: {
          cagr_pct: 6.2,
          max_drawdown_pct: -12,
        },
      }),
    );

    const searched = buildLabsPanelViewModel(data, {
      searchText: 'duration',
      sortBy: 'status',
      sortDirection: 'asc',
    });
    const thresholded = buildLabsPanelViewModel(data, {
      minSharpe: 0.8,
      maxDrawdownPct: -20,
      sortBy: 'status',
      sortDirection: 'asc',
    });

    expect(searched.rows.map((row) => row.experimentId)).toEqual(['duration-lab']);
    expect(thresholded.rows.map((row) => row.experimentId)).toEqual([]);
  });

  it('exposes optional robustness metrics without coercing missing values to zero', () => {
    const data = labsData();
    data.registry!.experiments = [
      registryRow('robust-champion', {
        metrics: {
          sharpe: 0.95,
          cagr_pct: 10.4,
          max_drawdown_pct: -25,
          wfe: 1.37,
          dsr: 0.979,
          positive_oos_ratio: 0.733,
          regime_coverage: 0.91,
        },
      }),
      registryRow('legacy-row', {
        metrics: {
          sharpe: 0.82,
          cagr_pct: 8.8,
          max_drawdown_pct: -17,
        },
      }),
    ];

    const view = buildLabsPanelViewModel(data, {
      sortBy: 'status',
      sortDirection: 'asc',
    });

    const champion = view.rows.find((row) => row.experimentId === 'robust-champion');
    const legacy = view.rows.find((row) => row.experimentId === 'legacy-row');

    expect(champion?.wfe).toBe(1.37);
    expect(champion?.dsr).toBe(0.979);
    expect(champion?.positiveOosRatio).toBe(0.733);
    expect(champion?.regimeCoverage).toBe(0.91);
    expect(legacy?.wfe).toBeNull();
    expect(legacy?.dsr).toBeNull();
    expect(legacy?.positiveOosRatio).toBeNull();
    expect(legacy?.regimeCoverage).toBeNull();
  });

  it('sorts Labs rows by robustness metrics while leaving missing metrics unavailable', () => {
    const data = labsData();
    data.registry!.experiments = [
      registryRow('missing-robustness', {
        metrics: {
          sharpe: 0.8,
          cagr_pct: 8.4,
          max_drawdown_pct: -15,
        },
      }),
      registryRow('weak-robustness', {
        metrics: {
          sharpe: 0.9,
          cagr_pct: 9.1,
          max_drawdown_pct: -20,
          wfe: 0.74,
          dsr: 0.61,
        },
      }),
      registryRow('strong-robustness', {
        metrics: {
          sharpe: 0.88,
          cagr_pct: 8.9,
          max_drawdown_pct: -18,
          wfe: 1.39,
          dsr: 0.979,
        },
      }),
    ];

    const byWfe = buildLabsPanelViewModel(data, {
      sortBy: 'wfe',
      sortDirection: 'desc',
    });
    const byDsr = buildLabsPanelViewModel(data, {
      sortBy: 'dsr',
      sortDirection: 'desc',
    });

    expect(byWfe.rows.map((row) => row.experimentId)).toEqual([
      'strong-robustness',
      'weak-robustness',
      'missing-robustness',
    ]);
    expect(byDsr.rows.map((row) => row.experimentId)).toEqual([
      'strong-robustness',
      'weak-robustness',
      'missing-robustness',
    ]);
    expect(byWfe.rows[2].wfe).toBeNull();
    expect(byDsr.rows[2].dsr).toBeNull();
  });

  it('exposes bounded replay diagnostics for failed and warning rows', () => {
    const data = labsData();
    data.replays = [
      {
        ...data.replays[0],
        duration_seconds: 0.42,
      },
      {
        ...data.replays[1],
        status: 'failed',
        failure_reason: 'timeout',
        error_type: 'TimeoutExpired',
        error_message: 'replay command timed out after 30 seconds',
        duration_seconds: 30.12,
      },
    ];

    const view = buildLabsPanelViewModel(data, {
      sortBy: 'status',
      sortDirection: 'asc',
    });
    const passed = view.rows.find((row) => row.experimentId === 'gold-sweep');
    const failed = view.rows.find((row) => row.experimentId === 'risk-sweep');

    expect(passed?.replayDiagnostics).toBeNull();
    expect(failed?.replayDiagnostics).toEqual({
      failureReason: 'timeout',
      errorType: 'TimeoutExpired',
      errorMessage: 'replay command timed out after 30 seconds',
      durationSeconds: 30.12,
      details: [
        'reason: timeout',
        'error: TimeoutExpired',
        'message: replay command timed out after 30 seconds',
        'duration: 30.12s',
      ],
    });
  });

  it('keeps raw replay commands and secret-like diagnostics out of the drilldown', () => {
    const data = labsData();
    data.replays = [
      {
        ...data.replays[1],
        experiment_id: 'risk-sweep',
        status: 'warning',
        command: 'ALPACA_API_SECRET=super-secret python -m unsafe.replay --token abc123',
        failure_reason: 'command_failure',
        error_type: 'RuntimeError',
        error_message: 'failed with ALPACA_API_SECRET=super-secret and token=abc123',
        duration_seconds: 2.5,
      },
    ];

    const view = buildLabsPanelViewModel(data, {
      replayStatus: 'warning',
    });
    const diagnostics = view.rows[0].replayDiagnostics;
    const serialized = JSON.stringify(diagnostics);

    expect(diagnostics).not.toBeNull();
    expect(serialized).toContain('[redacted]');
    expect(serialized).not.toContain('ALPACA_API_SECRET');
    expect(serialized).not.toContain('super-secret');
    expect(serialized).not.toContain('token=abc123');
    expect(serialized).not.toContain('unsafe.replay');
  });

  it('exposes bounded validation drilldown metadata and report truncation totals', () => {
    const data = labsData();
    data.validation = {
      schema_version: 'labs-validation/v1',
      generated_at: '2026-06-08T12:00:00+00:00',
      truncation: {
        max_results: 2,
        max_errors_per_result: 2,
        total_result_count: 4,
        returned_result_count: 2,
        omitted_result_count: 2,
        omitted_error_count: 5,
      },
      results: [
        {
          path: 'public/data/labs_registry.json[0]',
          artifact_type: 'registry',
          schema_version: 'labs-registry/v1',
          experiment_id: 'gold-sweep',
          valid: false,
          errors: ['missing metric', 'bad status'],
          omitted_error_count: 3,
        },
      ],
    };

    const view = buildLabsPanelViewModel(data);
    const row = view.rows.find((entry) => entry.experimentId === 'gold-sweep');

    expect(row?.validationErrors).toEqual(['missing metric', 'bad status']);
    expect(row?.validationErrorCount).toBe(5);
    expect(row?.omittedValidationErrorCount).toBe(3);
    expect(view.validationTruncation).toEqual({
      totalResultCount: 4,
      returnedResultCount: 2,
      omittedResultCount: 2,
      omittedErrorCount: 5,
      maxErrorsPerResult: 2,
    });
  });

  it('normalizes compact endpoint badges for missing, invalid, summarized, and paginated endpoints', () => {
    const directView = buildLabsPanelViewModel({
      ...labsData(),
      endpoint_status: [endpointStatus('registry')],
    });
    const data = labsData();
    data.endpoint_status = [
      endpointStatus('registry', {
        status: 'missing',
        validation_status: 'missing',
        render_strategy: 'missing',
        size_bytes: null,
      }),
      endpointStatus('scorecards', {
        validation_status: 'invalid',
        validation_errors: ['scorecard schema mismatch'],
      }),
      endpointStatus('replays', {
        render_strategy: 'summarize',
        summary_limited: true,
        row_count: 1200,
        requires_downsampling: true,
      }),
      endpointStatus('validation', {
        render_strategy: 'paginate',
        requires_pagination: true,
        pagination: {
          total_rows: 2500,
          page_size: 100,
          page_count: 25,
          pages: [
            { page: 1, path: 'labs_validation.page-1.json', row_count: 100 },
            { page: 2, path: 'labs_validation.page-2.json', row_count: 100 },
          ],
        },
        selected_page: 2,
      }),
    ];

    const view = buildLabsPanelViewModel(data);

    expect(directView.endpointBadges.map((badge) => [badge.endpoint, badge.label, badge.tone])).toEqual([
      ['registry', 'direct', 'success'],
    ]);
    expect(view.endpointBadges.map((badge) => [badge.endpoint, badge.label, badge.tone])).toEqual([
      ['registry', 'missing', 'warning'],
      ['scorecards', 'invalid', 'danger'],
      ['replays', 'summarized', 'warning'],
      ['validation', 'paginated', 'info'],
    ]);
    expect(view.endpointBadges.find((badge) => badge.endpoint === 'scorecards')?.details).toContain(
      'scorecard schema mismatch',
    );
    expect(view.paginationControls).toEqual([
      {
        endpoint: 'validation',
        selectedPage: 2,
        pages: [
          { page: 1, path: 'labs_validation.page-1.json', rowCount: 100 },
          { page: 2, path: 'labs_validation.page-2.json', rowCount: 100 },
        ],
      },
    ]);
  });

  it('joins validation rows by experiment id before collection paths', () => {
    const data = labsData();
    data.validation = {
      schema_version: 'labs-validation/v1',
      generated_at: '2026-06-08T12:00:00+00:00',
      results: [
        {
          path: 'public/data/labs_replays.json[0]',
          artifact_type: 'replay',
          schema_version: 'labs-replay/v1',
          experiment_id: 'risk-sweep',
          valid: false,
          errors: ['risk replay failed'],
        },
        {
          path: 'public/data/labs_scorecards.json[0]',
          artifact_type: 'scorecard',
          schema_version: 'labs-scorecard/v1',
          experiment_id: 'gold-sweep',
          valid: true,
          errors: [],
        },
      ],
    };

    const view = buildLabsPanelViewModel(data, {
      sortBy: 'sharpe',
      sortDirection: 'desc',
    });

    expect(view.rows.find((row) => row.experimentId === 'gold-sweep')?.validationStatus).toBe('valid');
    expect(view.rows.find((row) => row.experimentId === 'risk-sweep')?.validationStatus).toBe('invalid');
    expect(view.rows.find((row) => row.experimentId === 'risk-sweep')?.validationErrors).toEqual([
      'risk replay failed',
    ]);
  });

  it('joins validation rows by artifact path before legacy path heuristics', () => {
    const data = labsData();
    data.validation = {
      schema_version: 'labs-validation/v1',
      generated_at: '2026-06-08T12:00:00+00:00',
      results: [
        {
          path: 'public/data/labs_registry.json[0]',
          artifact_type: 'registry',
          schema_version: 'labs-registry/v1',
          artifact_path: 'data/risk-sweep.json',
          valid: false,
          errors: ['risk registry row failed'],
        },
      ],
    };

    const view = buildLabsPanelViewModel(data, {
      sortBy: 'sharpe',
      sortDirection: 'desc',
    });

    expect(view.rows.find((row) => row.experimentId === 'gold-sweep')?.validationStatus).toBe('missing');
    expect(view.rows.find((row) => row.experimentId === 'risk-sweep')?.validationStatus).toBe('invalid');
    expect(view.rows.find((row) => row.experimentId === 'risk-sweep')?.validationErrors).toEqual([
      'risk registry row failed',
    ]);
  });

  it('exposes static experiment diff summaries for metric and provenance changes', () => {
    const data = labsData();
    data.diffs = [loadLabsFixture('valid_experiment_diff')];

    const view = buildLabsPanelViewModel(data);

    expect(view.diffs).toHaveLength(1);
    expect(view.diffs[0].title).toBe('champion -> challenger');
    expect(view.diffs[0].metricDeltas).toEqual([
      { metric: 'sharpe', left: 0.95, right: 0.99, delta: 0.04 },
      { metric: 'max_drawdown_pct', left: -27.6, right: -20.1, delta: 7.5 },
    ]);
    expect(view.diffs[0].missingMetrics).toEqual([{ metric: 'dsr', missingFrom: ['right'] }]);
    expect(view.diffs[0].configChanges).toEqual([{ key: 'target_vol', left: 0.09, right: 0.11 }]);
    expect(view.diffs[0].provenanceChange).toEqual({ left: 'present', right: 'stale', changed: true });
  });

  it('uses the first replay row when duplicate replay ids are present', () => {
    const data = labsData();
    data.replays = [
      data.replays[0],
      {
        ...data.replays[0],
        status: 'failed',
        metrics: { duplicate_count: 1 },
      },
      data.replays[1],
    ];

    const view = buildLabsPanelViewModel(data, {
      status: 'validated',
      provenanceStatus: 'present',
      sortBy: 'sharpe',
      sortDirection: 'desc',
    });

    expect(view.rows[0].experimentId).toBe('gold-sweep');
    expect(view.rows[0].replayStatus).toBe('passed');
  });

  it('sorts Labs rows by Sharpe and max drawdown without unit conversion', () => {
    const bySharpe = buildLabsPanelViewModel(labsData(), {
      sortBy: 'sharpe',
      sortDirection: 'asc',
    });
    const byDrawdown = buildLabsPanelViewModel(labsData(), {
      sortBy: 'max_drawdown',
      sortDirection: 'desc',
    });

    expect(bySharpe.rows.map((row) => row.experimentId)).toEqual(['risk-sweep', 'gold-sweep']);
    expect(byDrawdown.rows.map((row) => row.experimentId)).toEqual(['risk-sweep', 'gold-sweep']);
    expect(byDrawdown.rows[0].maxDrawdownPct).toBe(-18);
  });
});
