import { describe, expect, it } from 'bun:test';
import {
  formatSourceDirection,
  normalizeSourceVote,
} from '../../src/components/EnsembleVotingPanel';

describe('EnsembleVotingPanel source vote normalization', () => {
  it('maps string directions from signals.json into numeric panel values', () => {
    expect(normalizeSourceVote({
      source: 'MULTI_SPEED_MOM',
      direction: 'bullish',
      strength: 0.4,
      confidence: 0.7,
      weight: 0.12,
    })).toEqual({
      source: 'MULTI_SPEED_MOM',
      direction: 1,
      strength: 0.4,
      confidence: 0.7,
      weight: 0.12,
    });

    expect(formatSourceDirection('bearish')).toBe('-1');
    expect(formatSourceDirection('neutral')).toBe('0');
  });

  it('uses safe numeric fallbacks for malformed source vote fields', () => {
    expect(normalizeSourceVote({
      source: '',
      direction: 'not-a-direction',
      strength: 'bad',
      confidence: null,
      weight: undefined,
    })).toEqual({
      source: 'unknown',
      direction: 0,
      strength: 0,
      confidence: 0,
      weight: 0,
    });
  });
});
