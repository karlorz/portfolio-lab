export const MARKET_DATA_SOURCE_MANIFEST_SCHEMA_VERSION = 'market-data-source-manifest/v1';
export const MARKET_DATA_SOURCE_MANIFEST_FILENAME = 'source_manifest.json';

export type MarketDataSourceMode = 'live' | 'last_good' | 'cached' | 'synthetic';
export type MarketDataSourceStatus = 'success' | 'degraded' | 'failed' | 'skipped';

export interface MarketDataSourceRow {
  artifact: string;
  provider: string;
  feed: string;
  source_mode: MarketDataSourceMode;
  status: MarketDataSourceStatus;
  fetched_at: string;
  latest_observation: string | null;
  row_count: number;
  symbols?: string[];
  failure_reason?: string | null;
  fallback_reason?: string | null;
  notes?: string[];
}

export interface MarketDataSourceManifest {
  schema_version: typeof MARKET_DATA_SOURCE_MANIFEST_SCHEMA_VERSION;
  generated_at: string;
  artifacts: MarketDataSourceRow[];
}

export function buildMarketDataSourceManifest(
  artifacts: MarketDataSourceRow[],
  generatedAt: string = new Date().toISOString(),
): MarketDataSourceManifest {
  return {
    schema_version: MARKET_DATA_SOURCE_MANIFEST_SCHEMA_VERSION,
    generated_at: generatedAt,
    artifacts,
  };
}

export function latestObservationFromRows(rows: Array<{ date?: string }>): string | null {
  let latest: string | null = null;
  for (const row of rows) {
    if (typeof row.date !== 'string') continue;
    if (latest === null || row.date > latest) {
      latest = row.date;
    }
  }
  return latest;
}
