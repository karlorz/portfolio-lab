import { readFileSync } from 'node:fs';

export const LABS_FIXTURE_NAMES = [
  'valid_registry',
  'valid_provenance',
  'valid_scorecard',
  'valid_replay_pass',
  'valid_replay_fail',
  'validation_report',
  'invalid_missing_metrics',
  'invalid_mixed_units',
  'stale_schema',
  'dirty_provenance',
] as const;

export type LabsFixtureName = typeof LABS_FIXTURE_NAMES[number];
export type LabsFixtureKind = 'registry' | 'provenance' | 'scorecard' | 'replay';
export type LabsFixtureVariant =
  | 'valid'
  | 'missing_metrics'
  | 'stale_schema'
  | 'dirty'
  | 'mixed_units'
  | 'pass'
  | 'drift_fail';

const BUILD_VARIANTS: Record<string, LabsFixtureName> = {
  'registry/valid': 'valid_registry',
  'registry/missing_metrics': 'invalid_missing_metrics',
  'registry/stale_schema': 'stale_schema',
  'provenance/valid': 'valid_provenance',
  'provenance/dirty': 'dirty_provenance',
  'scorecard/valid': 'valid_scorecard',
  'scorecard/mixed_units': 'invalid_mixed_units',
  'replay/pass': 'valid_replay_pass',
  'replay/drift_fail': 'valid_replay_fail',
};

export function labsFixturePath(name: LabsFixtureName): string {
  return new URL(`../fixtures/labs/${name}.json`, import.meta.url).pathname;
}

export function loadLabsFixture<T = Record<string, unknown>>(name: LabsFixtureName): T {
  return JSON.parse(readFileSync(labsFixturePath(name), 'utf8')) as T;
}

export function buildLabsFixture<T = Record<string, any>>(
  kind: LabsFixtureKind,
  variant: LabsFixtureVariant = 'valid',
): T {
  const fixtureName = BUILD_VARIANTS[`${kind}/${variant}`];
  if (!fixtureName) {
    throw new Error(`Unknown Labs fixture variant: ${kind}/${variant}`);
  }
  return structuredClone(loadLabsFixture<T>(fixtureName));
}
