import { describe, expect, it } from 'bun:test';
import {
  LABS_DASHBOARD_ENDPOINTS,
  PUBLIC_DATA_INDEX_ENDPOINT,
  fetchLabsDashboardData,
  fetchLabsDashboardDataFromIndex,
} from '../../src/data/labs';
import {
  LABS_EXPERIMENT_DIFF_SCHEMA_VERSION,
  type PublicDataIndexEntry,
} from '../../src/schemas/labs';
import { loadLabsFixture } from './labs-fixtures';

// Direct tests for `fetchLabsDashboardDataFromIndex` (src/data/labs.ts:429) —
// the live LabsPanel production fetch path (LabsPanel.tsx:651). Covers the
// index-fallback, status-missing skip, status surfacing, diff diagnostics +
// cap, and abort propagation branches.

const sizeBudget = (overrides: Record<string, unknown> = {}) => ({
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
  ...overrides,
});

// Builder pattern mirrors labs-schemas.test.ts:346-386.
function indexEntry(filename: string, overrides: Record<string, unknown> = {}): PublicDataIndexEntry {
  return {
    filename,
    path: filename,
    category: 'labs',
    schema_version: 'labs-registry/v1',
    status: 'present',
    validation_status: 'valid',
    validation_errors: [],
    size_bytes: 1000,
    size_budget: sizeBudget(),
    sha256: 'a'.repeat(64),
    generated_at: '2026-06-08T12:00:00+00:00',
    ...overrides,
  } as PublicDataIndexEntry;
}

function publicIndex(entries: unknown[]) {
  return {
    schema_version: 'public-data-index/v1',
    files: entries
      .filter(
        (entry) =>
          typeof entry === 'object' && entry !== null && (entry as { status?: string }).status === 'present',
      )
      .map((entry) => (entry as { filename: string }).filename),
    entries,
    generated_at: '2026-06-08T12:00:00+00:00',
  };
}

const json = (body: unknown) =>
  new Response(JSON.stringify(body), { status: 200, headers: { 'content-type': 'application/json' } });

const notFound = () => new Response('', { status: 404 });

/** URL-route stub fetcher that records every URL it is asked for (DI — no global fetch mock). */
function routesFetcher(routes: Record<string, Response>) {
  const calls: string[] = [];
  const fetcher = async (url: string) => {
    calls.push(url);
    // Clone per call: Response bodies are single-use (the index-path and
    // direct-path fetches may consume the same route twice).
    return (routes[url] ?? notFound()).clone();
  };
  return { fetcher, calls };
}

const directEndpoints = () => ({
  [LABS_DASHBOARD_ENDPOINTS.registry]: json(loadLabsFixture('valid_registry')),
  [LABS_DASHBOARD_ENDPOINTS.scorecards]: json([loadLabsFixture('valid_scorecard')]),
  [LABS_DASHBOARD_ENDPOINTS.replays]: json([loadLabsFixture('valid_replay_pass')]),
  [LABS_DASHBOARD_ENDPOINTS.validation]: json(loadLabsFixture('validation_report')),
});

const presentEntries = () => [
  indexEntry('labs_registry.json'),
  indexEntry('labs_scorecards.json'),
  indexEntry('labs_replays.json'),
  indexEntry('labs_validation.json'),
];

describe('Labs dashboard data fetch helper (index-driven path)', () => {
  it('falls back to the direct-endpoint path when the public data index is unavailable', async () => {
    const { fetcher, calls } = routesFetcher(directEndpoints());

    const data = await fetchLabsDashboardDataFromIndex(fetcher);
    const direct = await fetchLabsDashboardData(fetcher);

    // Data equals the direct-endpoint path (endpoint_status is index-only
    // metadata the direct path does not produce — compare per-key).
    expect(data.registry).toEqual(direct.registry);
    expect(data.scorecards).toEqual(direct.scorecards);
    expect(data.replays).toEqual(direct.replays);
    expect(data.validation).toEqual(direct.validation);
    expect(data.diffs).toEqual(direct.diffs);
    expect(data.missing).toEqual(direct.missing);
    expect(data.errors).toEqual(direct.errors);
    expect(data.available).toBe(direct.available);
    // And the index was fetched exactly once (404 -> null -> fallback).
    expect(calls.filter((url) => url === PUBLIC_DATA_INDEX_ENDPOINT)).toHaveLength(1);
    expect(calls[0]).toBe(PUBLIC_DATA_INDEX_ENDPOINT);
    expect(data.available).toBe(true);
    expect(data.missing).toEqual([]);
    expect(data.errors).toEqual([]);
    expect(data.registry).not.toBeNull();
  });

  it('skips endpoints the index marks missing without fetching them (live replays shape)', async () => {
    const entries = [
      indexEntry('labs_registry.json'),
      indexEntry('labs_scorecards.json'),
      indexEntry('labs_replays.json', { status: 'missing', validation_status: 'missing', size_bytes: null }),
      indexEntry('labs_validation.json'),
    ];
    const { fetcher, calls } = routesFetcher({
      [PUBLIC_DATA_INDEX_ENDPOINT]: json(publicIndex(entries)),
      ...directEndpoints(),
    });

    const data = await fetchLabsDashboardDataFromIndex(fetcher);

    expect(data.missing).toEqual(['replays']);
    expect(data.replays).toEqual([]);
    // Skip path must not touch the network for the missing endpoint (labs.ts:371).
    expect(calls).not.toContain(LABS_DASHBOARD_ENDPOINTS.replays);
    // One missing endpoint is not fatal: availability only drops when all four
    // are missing or errors exist (labs.ts:59-67 availability flag).
    expect(data.available).toBe(true);
    const replaysStatus = data.endpoint_status.find((status) => status.endpoint === 'replays');
    expect(replaysStatus?.status).toBe('missing');
    expect(replaysStatus?.validation_status).toBe('missing');
  });

  it('parses all endpoints and surfaces index statuses when every entry is present and valid', async () => {
    const { fetcher } = routesFetcher({
      [PUBLIC_DATA_INDEX_ENDPOINT]: json(publicIndex(presentEntries())),
      ...directEndpoints(),
    });

    const data = await fetchLabsDashboardDataFromIndex(fetcher);

    expect(data.available).toBe(true);
    expect(data.missing).toEqual([]);
    expect(data.errors).toEqual([]);
    expect(data.registry).not.toBeNull();
    expect(data.scorecards.length).toBeGreaterThan(0);
    expect(data.replays.length).toBeGreaterThan(0);
    expect(data.validation).not.toBeNull();
    expect(data.endpoint_status).toHaveLength(4);
    const registryStatus = data.endpoint_status.find((status) => status.endpoint === 'registry');
    expect(registryStatus?.filename).toBe('labs_registry.json');
    expect(registryStatus?.status).toBe('present');
    expect(registryStatus?.validation_status).toBe('valid');
    expect(registryStatus?.render_strategy).toBe('direct');
    expect(registryStatus?.requires_pagination).toBe(false);
  });

  it('reports missing experiment-diff artifacts with diagnostics and respects the fetch cap', async () => {
    const diffEntries = [1, 2].map((i) =>
      indexEntry(`labs_experiment_diffs_${i}.json`, {
        schema_version: LABS_EXPERIMENT_DIFF_SCHEMA_VERSION,
        path: `/data/labs_experiment_diffs_${i}.json`,
      }),
    );
    const { fetcher, calls } = routesFetcher({
      [PUBLIC_DATA_INDEX_ENDPOINT]: json(publicIndex([...presentEntries(), ...diffEntries])),
      ...directEndpoints(),
      '/data/labs_experiment_diffs_1.json': notFound(),
      '/data/labs_experiment_diffs_2.json': notFound(),
    });

    const data = await fetchLabsDashboardDataFromIndex(fetcher, { maxIndexedDiffFetches: 1 });

    expect(data.errors).toContain(
      'diff:labs_experiment_diffs_1.json: missing static experiment diff artifact',
    );
    expect(data.errors).toContain('diff:budget: skipped 1 of 2 experiment-diff artifacts (limit 1)');
    expect(calls.filter((url) => url.includes('labs_experiment_diffs'))).toEqual([
      '/data/labs_experiment_diffs_1.json',
    ]);
    expect(data.diffs).toEqual([]);
    // Diagnostics present => not fully available.
    expect(data.available).toBe(false);
  });

  it('propagates abort errors instead of swallowing them', async () => {
    const abortError = Object.assign(new Error('The operation was aborted'), { name: 'AbortError' });
    const fetcher = async () => {
      throw abortError;
    };

    await expect(
      fetchLabsDashboardDataFromIndex(fetcher, { signal: new AbortController().signal }),
    ).rejects.toThrow('The operation was aborted');
  });
});
