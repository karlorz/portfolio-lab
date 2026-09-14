import { describe, expect, it } from 'bun:test';
import { existsSync, readFileSync } from 'node:fs';
import { join } from 'node:path';
import { z } from 'zod';
import {
  AdaptiveSizingSchema,
  AlertsWireDataSchema,
  AnalyticsDataSchema,
  BlackLittermanSchema,
  CrossAssetRVSchema,
  DashboardDataSchema,
  ExplainabilitySchema,
  GraduationDataSchema,
  HealthDataSchema,
  IncidentLifecycleSummarySchema,
  RebalanceHealthSchema,
  RegimeGateSchema,
  SignalsDataSchema,
  StatsDataSchema,
  TSMOMSchema,
  TurnoverValidatorSchema,
  VixyHedgeSchema,
} from '../../src/schemas/signals';
import {
  LabsRegistrySchema,
  LabsScorecardSchema,
  LabsValidationReportSchema,
  PublicDataIndexSchema,
} from '../../src/schemas/labs';
import { TaskerStatusSchema } from '../../src/schemas/tasker';
import { DecisionRegistrySchema } from '../../src/schemas/decision_registry';

// Repo-local data <-> schema parity regression gate (TS-SCHEMA-LIVE-PARITY s3).
// Every published file with a zod contract must parse its public/data seed
// with the exact frontend schema — schema growth must land with seed
// regeneration in one commit. Failure mode this gate prevents: live payloads
// rejected by strict/outdated schemas (e.g. labs projection envelope
// 2026-08-13, tasker "blocked" run status 2026-08-13).
// Excluded by design (no zod contract): health_ops.json (inline type guard in
// LiveDashboard.tsx:300-308), prices.json / prices_compact.json (fetched
// price snapshots without a frontend schema). File inventory source:
// src/dashboard/public_data_index.py — extend this table when new published
// files gain schemas.
//
// CI-vs-live contract: public/data/ seeds are generated payloads, untracked
// by design since fda0020 — a fresh CI clone has zero seed files, so the
// parse gate cannot run there. Skip the whole test when the published seed
// SET is absent (no signals.json). A leftover file such as only
// tasker_status.json is not a published set — treating `readdirSync.length > 0`
// as present caused ENOENT on side-dev. On hosts WITH signals.json the gate
// stays strict: any contracted file missing mid-set is a drift failure
// (`${file}: missing`), never a silent per-file skip, and payloads still
// parse with the exact frontend schema.
const SEED_DIR = new URL('../../public/data/', import.meta.url).pathname;
const PUBLISHED_SEED_SENTINEL = 'signals.json';

export function publishedSeedSetPresent(dir: string, sentinel = PUBLISHED_SEED_SENTINEL): boolean {
  return existsSync(join(dir, sentinel));
}

const SEED_SET_PRESENT = publishedSeedSetPresent(SEED_DIR);

const checks: [string, z.ZodType][] = [
  ['signals.json', SignalsDataSchema],
  ['dashboard.json', DashboardDataSchema],
  ['alerts.json', AlertsWireDataSchema],
  ['stats.json', StatsDataSchema],
  ['health.json', HealthDataSchema],
  ['incidents.json', IncidentLifecycleSummarySchema],
  ['decision_registry.json', DecisionRegistrySchema],
  ['index.json', PublicDataIndexSchema],
  ['labs_registry.json', LabsRegistrySchema],
  ['labs_scorecards.json', z.array(LabsScorecardSchema)],
  ['labs_validation.json', LabsValidationReportSchema],
  ['tasker_status.json', TaskerStatusSchema],
  ['adaptive_sizing.json', AdaptiveSizingSchema],
  ['black_litterman.json', BlackLittermanSchema],
  ['analytics.json', AnalyticsDataSchema],
  ['rebalance_health.json', RebalanceHealthSchema],
  ['graduation.json', GraduationDataSchema],
  ['regime_gate.json', RegimeGateSchema],
  ['tsmom.json', TSMOMSchema],
  ['explainability/explainability_latest.json', ExplainabilitySchema],
  ['cross_asset_rv.json', CrossAssetRVSchema],
  ['vixy_hedge.json', VixyHedgeSchema],
  ['turnover_validator.json', TurnoverValidatorSchema],
];

describe('published data seed ↔ frontend schema parity', () => {
  it('does not treat a leftover partial public/data dir as a published seed set', () => {
    // Side-dev / CI: tasker_status.json alone must skip, not ENOENT signals.json.
    expect(SEED_SET_PRESENT).toBe(existsSync(join(SEED_DIR, PUBLISHED_SEED_SENTINEL)));
    expect(publishedSeedSetPresent('/tmp/data-schema-parity-missing-dir')).toBe(false);
  });

  it(`parses all ${checks.length} seeded payloads with their exact frontend schemas`, () => {
    if (!SEED_SET_PRESENT) {
      // Fresh CI clone or leftover-only dir: public/data/ seeds are generated
      // payloads, untracked by design since fda0020. Skip cleanly — the
      // live-parity gate activates on hosts where signals.json is published.
      // Do NOT convert to per-file `if exists` parsing on those hosts, which
      // would let partial seed sets drift silently.
      console.warn(
        `[data-schema-parity] skip: no published ${PUBLISHED_SEED_SENTINEL} at ${SEED_DIR} — live-parity gate activates where payloads are published`,
      );
      return;
    }
    const failures: string[] = [];
    for (const [file, schema] of checks) {
      const path = join(SEED_DIR, file);
      if (!existsSync(path)) {
        failures.push(`${file}: missing`);
        continue;
      }
      const raw = JSON.parse(readFileSync(path, 'utf8')) as unknown;
      const parsed = schema.safeParse(raw);
      if (!parsed.success) {
        failures.push(`${file}: ${JSON.stringify(parsed.error.issues?.[0] ?? parsed.error).slice(0, 180)}`);
      }
    }
    expect(failures).toEqual([]);
  });
});
