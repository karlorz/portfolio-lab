import { describe, expect, it } from 'bun:test';
import {
  normalizeTurnoverValidatorData,
} from '../../src/components/TurnoverValidatorPanel';

// Live payload captured from https://lab.karldigi.dev/data/turnover_validator.json
// (23:43Z 2026-08-11): turnover-validator/v1 nested shape with provenance keys.
const livePayload = {
  schema_version: 'turnover-validator/v1',
  signals: {
    multi_speed_momentum: {
      periods: 20,
      mean: 0.24666666666666667,
      std: 0.28539497612547565,
      sign_flip_rate: 0.10526315789473684,
      mag_vol: 0.13333333333333333,
      turnover_penalty: 0.11649122807017542,
      stability_score: 0.6581182535542723,
      marginal_score: 0.15101707562622474,
    },
    cross_asset_rv: {
      periods: 20,
      mean: 0.5,
      std: 0.12,
      sign_flip_rate: 0.05,
      mag_vol: 0.2,
      turnover_penalty: 0.08,
      stability_score: 0.81,
      marginal_score: 0.22,
    },
  },
  synthetic_baselines: {
    stable: {
      metadata: { source_type: 'synthetic_or_fixture' },
      diagnostics: {
        periods: 20,
        mean: 0.5,
        std: 0.0,
        sign_flip_rate: 0.0,
        mag_vol: 0.0,
        turnover_penalty: 0.0,
        stability_score: 1,
        marginal_score: 0,
      },
    },
  },
  generated_at: '2026-08-11T23:34:27.927843+00:00',
  generator_git_sha: '879ffa69d28f36ffd5d4e715d220e4936b435293',
  artifact_id: 'turnover-validator-20260811',
  plane: 'research',
  runtime_provenance: { source: 'get_state_diagnostics' },
};

describe('TurnoverValidatorPanel data normalization', () => {
  it('accepts the live nested turnover-validator/v1 payload with per-source diagnostics', () => {
    const data = normalizeTurnoverValidatorData(livePayload);

    expect(data).not.toBeNull();
    expect(data?.schema_version).toBe('turnover-validator/v1');
    expect(Object.keys(data?.signals ?? {}).sort()).toEqual([
      'cross_asset_rv',
      'multi_speed_momentum',
    ]);
    expect(data?.signals.multi_speed_momentum.stability_score).toBeCloseTo(0.6581, 3);
    expect(data?.signals.multi_speed_momentum.turnover_penalty).toBeCloseTo(0.1165, 3);
    expect(data?.signals.cross_asset_rv.marginal_score).toBeCloseTo(0.22, 3);
    // Synthetic baselines disclosed separately
    expect(data?.synthetic_baselines?.stable.metadata?.source_type).toBe('synthetic_or_fixture');
    expect(data?.synthetic_baselines?.stable.diagnostics?.stability_score).toBe(1);
    // Extra provenance keys tolerated
    expect(data?.generated_at).toBe('2026-08-11T23:34:27.927843+00:00');
  });

  it('returns null only for genuinely empty or non-signals payloads', () => {
    expect(normalizeTurnoverValidatorData(null)).toBeNull();
    expect(normalizeTurnoverValidatorData({})).toBeNull();
    // Legacy/non-matching shapes carry no usable signals block
    expect(
      normalizeTurnoverValidatorData({
        stable: { periods: 20, mean: 0.5, stability_score: 1 },
        generated_at: '2026-06-11T10:15:18',
      }),
    ).toBeNull();
    // signals block with no usable diagnostics → null
    expect(
      normalizeTurnoverValidatorData({
        signals: { source_a: { periods: 'nope' } },
      }),
    ).toBeNull();
  });
});
