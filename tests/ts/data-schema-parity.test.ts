import { describe, expect, it } from 'bun:test';
import { existsSync, readFileSync, readdirSync } from 'node:fs';
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
// parse gate cannot run there. Skip the whole test when the seed SET is
// absent; on hosts WITH seeds the gate stays strict — any file missing
// mid-set is still a drift failure, never a silent per-file skip.
const SEED_DIR = new URL('../../public/data/', import.meta.url).pathname;
const SEED_SET_PRESENT = existsSync(SEED_DIR) && readdirSync(SEED_DIR).length > 0;

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
  it(`parses all ${checks.length} seeded payloads with their exact frontend schemas`, () => {
    if (!SEED_SET_PRESENT) {
      // Fresh CI clone: public/data/ seeds are generated payloads,
      // untracked by design since fda0020. Skip cleanly — the live-parity
      // gate activates on hosts where payloads are published (dev host,
      // production). Do NOT convert to per-file `if exists` parsing, which
      // would let partial seed sets drift silently on hosts WITH seeds.
      console.warn(
        `[data-schema-parity] skip: no public/data/ seeds at ${SEED_DIR} — live-parity gate activates where payloads are published`,
      );
      return;
    }
    const failures: string[] = [];
    for (const [file, schema] of checks) {
      const raw = JSON.parse(readFileSync(join(SEED_DIR, file), 'utf8')) as unknown;
      const parsed = schema.safeParse(raw);
      if (!parsed.success) {
        failures.push(`${file}: ${JSON.stringify(parsed.error.issues?.[0] ?? parsed.error).slice(0, 180)}`);
      }
    }
    expect(failures).toEqual([]);
  });
});
