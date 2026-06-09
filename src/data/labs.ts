import { z } from 'zod';
import {
  LABS_EXPERIMENT_DIFF_SCHEMA_VERSION,
  LabsDashboardDataSchema,
  LabsExperimentDiffSchema,
  LabsRegistrySchema,
  LabsReplaySchema,
  LabsScorecardSchema,
  LabsValidationReportSchema,
  PublicDataIndexSchema,
  parseLabsJson,
  type LabsDashboardData,
  type LabsEndpointKey,
  type LabsEndpointStatus,
  type LabsExperimentDiffData,
  type LabsRegistryData,
  type LabsReplayData,
  type LabsScorecardData,
  type LabsValidationReport,
  type PublicDataIndexData,
  type PublicDataIndexEntry,
} from '../schemas/labs';

export const LABS_DASHBOARD_ENDPOINTS: Record<LabsEndpointKey, string> = {
  registry: '/data/labs_registry.json',
  scorecards: '/data/labs_scorecards.json',
  replays: '/data/labs_replays.json',
  validation: '/data/labs_validation.json',
};

export const PUBLIC_DATA_INDEX_ENDPOINT = '/data/index.json';
export const LABS_RENDER_ROW_LIMIT = 100;

const LABS_ENDPOINT_KEYS: LabsEndpointKey[] = ['registry', 'scorecards', 'replays', 'validation'];
const LABS_ENDPOINT_FILENAMES: Record<LabsEndpointKey, string> = {
  registry: 'labs_registry.json',
  scorecards: 'labs_scorecards.json',
  replays: 'labs_replays.json',
  validation: 'labs_validation.json',
};

type LabsFetcher = (url: string) => Promise<Response>;

export interface LabsDashboardFetchOptions {
  selectedPages?: Partial<Record<LabsEndpointKey, number>>;
}

interface EndpointResult<T> {
  data: T | null;
  missing: boolean;
  errors: string[];
}

export function buildEmptyLabsDashboardData(missing: LabsEndpointKey[] = LABS_ENDPOINT_KEYS): LabsDashboardData {
  return {
    available: false,
    registry: null,
    scorecards: [],
    replays: [],
    validation: null,
    diffs: [],
    missing,
    errors: [],
    endpoint_status: [],
  };
}

function capLabsRows<T>(rows: T[]): T[] {
  return rows.length > LABS_RENDER_ROW_LIMIT ? rows.slice(0, LABS_RENDER_ROW_LIMIT) : rows;
}

function capRegistryRows(registry: LabsRegistryData | null): LabsRegistryData | null {
  if (registry === null) {
    return null;
  }
  return {
    ...registry,
    experiments: capLabsRows(registry.experiments),
  };
}

function retainedRegistryIds(registry: LabsRegistryData | null): Set<string> | null {
  if (registry === null) {
    return null;
  }
  return new Set(registry.experiments.map((row) => row.experiment_id));
}

function capDependentLabsRows<T extends { experiment_id: string }>(
  rows: T[],
  retainedIds: Set<string> | null,
): T[] {
  const filteredRows = retainedIds === null ? rows : rows.filter((row) => retainedIds.has(row.experiment_id));
  return capLabsRows(filteredRows);
}

function hasJsonContentType(response: Response): boolean {
  const contentType = response.headers.get('content-type')?.toLowerCase() ?? '';
  if (!contentType) {
    return true;
  }
  return contentType.includes('application/json') || contentType.includes('+json');
}

async function fetchLabsEndpoint<T>(
  fetcher: LabsFetcher,
  key: LabsEndpointKey | string,
  schema: z.ZodType<T>,
  url: string = LABS_DASHBOARD_ENDPOINTS[key as LabsEndpointKey],
): Promise<EndpointResult<T>> {
  let response: Response;
  try {
    response = await fetcher(url);
  } catch (error) {
    return {
      data: null,
      missing: true,
      errors: [`${key}: fetch failed (${String(error)})`],
    };
  }

  if (!response.ok) {
    return { data: null, missing: true, errors: [] };
  }

  if (!hasJsonContentType(response)) {
    return { data: null, missing: true, errors: [] };
  }

  let raw: unknown;
  try {
    raw = await response.json();
  } catch (error) {
    return {
      data: null,
      missing: false,
      errors: [`${key}: invalid JSON (${String(error)})`],
    };
  }

  const parsed = parseLabsJson(raw, schema, key);
  return {
    data: parsed.data,
    missing: false,
    errors: parsed.errors,
  };
}

function buildLabsDashboardData(
  registry: EndpointResult<LabsRegistryData>,
  scorecards: EndpointResult<LabsScorecardData[]>,
  replays: EndpointResult<LabsReplayData[]>,
  validation: EndpointResult<LabsValidationReport>,
  endpointStatus: LabsEndpointStatus[] = [],
  diffs: LabsExperimentDiffData[] = [],
  extraErrors: string[] = [],
): LabsDashboardData {
  const missing = LABS_ENDPOINT_KEYS.filter((key) => {
    const result = { registry, scorecards, replays, validation }[key];
    return result.missing;
  });
  const errors = [registry, scorecards, replays, validation].flatMap((result) => result.errors).concat(extraErrors);

  const cappedRegistry = capRegistryRows(registry.data);
  const retainedIds = retainedRegistryIds(cappedRegistry);
  const data: LabsDashboardData = {
    available: missing.length < LABS_ENDPOINT_KEYS.length && errors.length === 0,
    registry: cappedRegistry,
    scorecards: capDependentLabsRows<LabsScorecardData>(scorecards.data ?? [], retainedIds),
    replays: capDependentLabsRows<LabsReplayData>(replays.data ?? [], retainedIds),
    validation: validation.data,
    diffs,
    missing,
    errors,
    endpoint_status: endpointStatus,
  };

  const validated = LabsDashboardDataSchema.safeParse(data);
  return validated.success ? validated.data : { ...buildEmptyLabsDashboardData(missing), errors, endpoint_status: endpointStatus };
}

export async function fetchLabsDashboardData(fetcher: LabsFetcher = fetch): Promise<LabsDashboardData> {
  const [registry, scorecards, replays, validation] = await Promise.all([
    fetchLabsEndpoint(fetcher, 'registry', LabsRegistrySchema),
    fetchLabsEndpoint(fetcher, 'scorecards', z.array(LabsScorecardSchema)),
    fetchLabsEndpoint(fetcher, 'replays', z.array(LabsReplaySchema)),
    fetchLabsEndpoint(fetcher, 'validation', LabsValidationReportSchema),
  ]);

  return buildLabsDashboardData(registry, scorecards, replays, validation);
}

async function fetchPublicDataIndex(fetcher: LabsFetcher): Promise<PublicDataIndexData | null> {
  let response: Response;
  try {
    response = await fetcher(PUBLIC_DATA_INDEX_ENDPOINT);
  } catch {
    return null;
  }
  if (!response.ok) {
    return null;
  }

  let raw: unknown;
  try {
    raw = await response.json();
  } catch {
    return null;
  }

  const parsed = PublicDataIndexSchema.safeParse(raw);
  return parsed.success ? parsed.data : null;
}

function endpointStatusFromIndexEntry(
  endpoint: LabsEndpointKey,
  entry: PublicDataIndexEntry,
  selectedPage?: number,
): LabsEndpointStatus {
  return {
    endpoint,
    filename: entry.filename,
    path: entry.path,
    status: entry.status,
    validation_status: entry.validation_status,
    validation_errors: entry.validation_errors,
    render_strategy: entry.size_budget.render_strategy,
    size_bytes: entry.size_bytes,
    row_count: entry.size_budget.row_count ?? null,
    requires_downsampling: entry.size_budget.requires_downsampling,
    requires_pagination: entry.size_budget.requires_pagination,
    summary_limited: entry.size_budget.render_strategy === 'summarize',
    size_budget_status: entry.size_budget.status,
    pagination: entry.pagination,
    selected_page: selectedPage,
    generated_at: entry.generated_at ?? null,
  };
}

function dataUrlForPath(path: string): string {
  if (path.startsWith('/')) {
    return path;
  }
  if (path.startsWith('data/')) {
    return `/${path}`;
  }
  return `/data/${path}`;
}

function dataUrlForIndexEntry(entry: PublicDataIndexEntry): string {
  return dataUrlForPath(entry.path);
}

function skippedEndpoint<T>(missing: boolean, errors: string[] = []): EndpointResult<T> {
  return { data: null, missing, errors };
}

function isExperimentDiffEntry(entry: PublicDataIndexEntry): boolean {
  return (
    entry.status === 'present' &&
    entry.validation_status === 'valid' &&
    entry.schema_version === LABS_EXPERIMENT_DIFF_SCHEMA_VERSION
  );
}

async function fetchIndexedExperimentDiffs(
  fetcher: LabsFetcher,
  index: PublicDataIndexData,
): Promise<EndpointResult<LabsExperimentDiffData[]>> {
  const entries = index.entries.filter(isExperimentDiffEntry);
  if (entries.length === 0) {
    return { data: [], missing: false, errors: [] };
  }

  const results = await Promise.all(entries.map(async (entry) => ({
    entry,
    result: await fetchLabsEndpoint(
      fetcher,
      `diff:${entry.filename}`,
      LabsExperimentDiffSchema,
      dataUrlForIndexEntry(entry),
    ),
  })));

  return {
    data: results.flatMap(({ result }) => (result.data ? [result.data] : [])),
    missing: false,
    errors: results.flatMap(({ entry, result }) => {
      if (!result.missing) {
        return result.errors;
      }
      return result.errors.length > 0
        ? result.errors
        : [`diff:${entry.filename}: missing static experiment diff artifact`];
    }),
  };
}

async function fetchIndexedLabsEndpoint<T>(
  fetcher: LabsFetcher,
  key: LabsEndpointKey,
  schema: z.ZodType<T>,
  entriesByFilename: Map<string, PublicDataIndexEntry>,
  options: LabsDashboardFetchOptions = {},
): Promise<{ result: EndpointResult<T>; status: LabsEndpointStatus | null }> {
  const entry = entriesByFilename.get(LABS_ENDPOINT_FILENAMES[key]);
  if (!entry) {
    return {
      result: skippedEndpoint<T>(true),
      status: {
        endpoint: key,
        filename: LABS_ENDPOINT_FILENAMES[key],
        path: LABS_ENDPOINT_FILENAMES[key],
        status: 'missing',
        validation_status: 'missing',
        validation_errors: [],
        render_strategy: 'missing',
        size_bytes: null,
        generated_at: null,
      },
    };
  }

  const selectedPage = options.selectedPages?.[key];
  const status = endpointStatusFromIndexEntry(key, entry, selectedPage);
  if (entry.status === 'missing') {
    return { result: skippedEndpoint<T>(true), status };
  }

  if (entry.validation_status === 'invalid') {
    return {
      result: skippedEndpoint<T>(
        false,
        entry.validation_errors.map((error) => `${key}: ${error}`),
      ),
      status,
    };
  }

  if (entry.size_budget.render_strategy === 'summarize') {
    return {
      result: skippedEndpoint<T>(false),
      status,
    };
  }

  if (entry.size_budget.render_strategy === 'paginate') {
    const page = selectedPage === undefined
      ? entry.pagination?.pages[0]
      : entry.pagination?.pages.find((candidate) => candidate.page === selectedPage);
    status.selected_page = page?.page ?? selectedPage;
    if (page) {
      const pageResult = await fetchLabsEndpoint(fetcher, key, schema, dataUrlForPath(page.path));
      return {
        result: pageResult.missing
          ? skippedEndpoint<T>(
            false,
            pageResult.errors.length > 0
              ? pageResult.errors
              : [`${key}: paginated shard missing (${page.path})`],
          )
          : pageResult,
        status,
      };
    }

    return {
      result: skippedEndpoint<T>(
        false,
        selectedPage === undefined
          ? [`${key}: render strategy paginate requires paginated Labs artifact access`]
          : [`${key}: paginated page ${selectedPage} is not listed in public data index`],
      ),
      status,
    };
  }

  return {
    result: await fetchLabsEndpoint(fetcher, key, schema, dataUrlForIndexEntry(entry)),
    status,
  };
}

export async function fetchLabsDashboardDataFromIndex(
  fetcher: LabsFetcher = fetch,
  options: LabsDashboardFetchOptions = {},
): Promise<LabsDashboardData> {
  const index = await fetchPublicDataIndex(fetcher);
  if (!index) {
    return fetchLabsDashboardData(fetcher);
  }

  const entriesByFilename = new Map(index.entries.map((entry) => [entry.filename, entry]));
  const [registry, scorecards, replays, validation, diffs] = await Promise.all([
    fetchIndexedLabsEndpoint(fetcher, 'registry', LabsRegistrySchema, entriesByFilename, options),
    fetchIndexedLabsEndpoint(fetcher, 'scorecards', z.array(LabsScorecardSchema), entriesByFilename, options),
    fetchIndexedLabsEndpoint(fetcher, 'replays', z.array(LabsReplaySchema), entriesByFilename, options),
    fetchIndexedLabsEndpoint(fetcher, 'validation', LabsValidationReportSchema, entriesByFilename, options),
    fetchIndexedExperimentDiffs(fetcher, index),
  ]);

  const endpointStatus = [registry, scorecards, replays, validation]
    .map((endpoint) => endpoint.status)
    .filter((status): status is LabsEndpointStatus => status !== null);

  return buildLabsDashboardData(
    registry.result,
    scorecards.result,
    replays.result,
    validation.result,
    endpointStatus,
    diffs.data ?? [],
    diffs.errors,
  );
}
