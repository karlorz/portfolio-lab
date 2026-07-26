import { SYMBOL_UNIVERSE_METADATA } from './symbol_universe';
import type {
  PriceDataQualityStatus,
  PriceIssueCounts,
} from './price_quality';
import { generatorGitShaShort } from './price_quality';

export const MARKET_DATA_SOURCE_MANIFEST_SCHEMA_VERSION = 'market-data-source-manifest/v1';
export const MARKET_DATA_SOURCE_MANIFEST_FILENAME = 'source_manifest.json';

export type MarketDataSourceMode = 'live' | 'last_good' | 'cached' | 'stale_cached' | 'synthetic';
export type MarketDataSourceStatus = 'success' | 'degraded' | 'failed' | 'skipped';
export type MarketDataQualityStatus = PriceDataQualityStatus | 'unavailable';

export interface MarketDataQualitySummary {
  artifact: string;
  schema_version?: string;
  generated_at?: string;
  status: MarketDataQualityStatus;
  issue_counts?: PriceIssueCounts;
}

export interface MarketDataSourceRow {
  artifact: string;
  provider: string;
  feed: string;
  provider_chain?: string[];
  primary_provider?: string | null;
  fallback_provider?: string | null;
  source_mode: MarketDataSourceMode;
  status: MarketDataSourceStatus;
  fetched_at: string;
  latest_observation: string | null;
  row_count: number;
  symbols?: string[];
  failure_reason?: string | null;
  fallback_reason?: string | null;
  data_quality?: MarketDataQualitySummary;
  notes?: string[];
}

export interface MarketDataSourceManifest {
  schema_version: typeof MARKET_DATA_SOURCE_MANIFEST_SCHEMA_VERSION;
  generated_at: string;
  symbol_universe: typeof SYMBOL_UNIVERSE_METADATA;
  artifacts: MarketDataSourceRow[];
  /** Short HEAD when producer could resolve git (operator lag detection). */
  generator_git_sha?: string | null;
  generator_git_sha_status?: string;
}

export function buildMarketDataSourceManifest(
  artifacts: MarketDataSourceRow[],
  generatedAt: string = new Date().toISOString(),
): MarketDataSourceManifest {
  const manifest: MarketDataSourceManifest = {
    schema_version: MARKET_DATA_SOURCE_MANIFEST_SCHEMA_VERSION,
    generated_at: generatedAt,
    symbol_universe: SYMBOL_UNIVERSE_METADATA,
    artifacts,
  };
  const sha = generatorGitShaShort();
  if (sha) {
    manifest.generator_git_sha = sha;
    manifest.generator_git_sha_status = 'full_generate';
  }
  return manifest;
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
