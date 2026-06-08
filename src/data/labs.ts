import { z } from 'zod';
import {
  LabsDashboardDataSchema,
  LabsRegistrySchema,
  LabsReplaySchema,
  LabsScorecardSchema,
  LabsValidationReportSchema,
  parseLabsJson,
  type LabsDashboardData,
  type LabsEndpointKey,
} from '../schemas/labs';

export const LABS_DASHBOARD_ENDPOINTS: Record<LabsEndpointKey, string> = {
  registry: '/data/labs_registry.json',
  scorecards: '/data/labs_scorecards.json',
  replays: '/data/labs_replays.json',
  validation: '/data/labs_validation.json',
};

const LABS_ENDPOINT_KEYS: LabsEndpointKey[] = ['registry', 'scorecards', 'replays', 'validation'];

type LabsFetcher = (url: string) => Promise<Response>;

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
    missing,
    errors: [],
  };
}

async function fetchLabsEndpoint<T>(
  fetcher: LabsFetcher,
  key: LabsEndpointKey,
  schema: z.ZodType<T>,
): Promise<EndpointResult<T>> {
  const url = LABS_DASHBOARD_ENDPOINTS[key];
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

export async function fetchLabsDashboardData(fetcher: LabsFetcher = fetch): Promise<LabsDashboardData> {
  const [registry, scorecards, replays, validation] = await Promise.all([
    fetchLabsEndpoint(fetcher, 'registry', LabsRegistrySchema),
    fetchLabsEndpoint(fetcher, 'scorecards', z.array(LabsScorecardSchema)),
    fetchLabsEndpoint(fetcher, 'replays', z.array(LabsReplaySchema)),
    fetchLabsEndpoint(fetcher, 'validation', LabsValidationReportSchema),
  ]);

  const missing = LABS_ENDPOINT_KEYS.filter((key) => {
    const result = { registry, scorecards, replays, validation }[key];
    return result.missing;
  });
  const errors = [registry, scorecards, replays, validation].flatMap((result) => result.errors);

  const data: LabsDashboardData = {
    available: missing.length < LABS_ENDPOINT_KEYS.length && errors.length === 0,
    registry: registry.data,
    scorecards: scorecards.data ?? [],
    replays: replays.data ?? [],
    validation: validation.data,
    missing,
    errors,
  };

  const validated = LabsDashboardDataSchema.safeParse(data);
  return validated.success ? validated.data : { ...buildEmptyLabsDashboardData(missing), errors };
}
