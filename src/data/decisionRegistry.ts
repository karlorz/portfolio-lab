import {
  DecisionRegistrySchema,
  parseDecisionRegistryJson,
  type DecisionRegistryData,
} from '../schemas/decision_registry';

export const DECISION_REGISTRY_ENDPOINT = '/data/decision_registry.json';

export async function fetchDecisionRegistry(
  fetcher: typeof fetch = fetch,
  init?: RequestInit,
): Promise<{ data: DecisionRegistryData | null; error: string | null }> {
  try {
    const response = await fetcher(DECISION_REGISTRY_ENDPOINT, init);
    if (!response.ok) {
      if (response.status === 404) {
        return { data: null, error: null };
      }
      return { data: null, error: `HTTP ${response.status}` };
    }
    const raw: unknown = await response.json();
    const parsed = parseDecisionRegistryJson(raw);
    if (!parsed) {
      const strict = DecisionRegistrySchema.safeParse(raw);
      const detail = strict.success ? '' : strict.error.issues[0]?.message ?? 'invalid shape';
      return { data: null, error: detail || 'validation failed' };
    }
    return { data: parsed, error: null };
  } catch (err) {
    return { data: null, error: err instanceof Error ? err.message : 'fetch failed' };
  }
}