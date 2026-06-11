import { describe, expect, it } from 'bun:test';
import {
  formatSourceLabel,
  normalizeContributionRows,
} from '../../src/components/PortfolioExplainabilityPanel';

describe('PortfolioExplainabilityPanel contribution normalization', () => {
  it('normalizes legacy string driver rows without losing the source label', () => {
    const rows = normalizeContributionRows(['Factor Rotation'], 'driver');

    expect(rows).toEqual([
      {
        source: 'Factor Rotation',
        contribution: null,
        direction: 'unknown',
      },
    ]);
  });

  it('preserves object driver rows while filling nullable fields safely', () => {
    const rows = normalizeContributionRows(
      [
        { source: 'cross_asset_rv', contribution: 0.24, direction: 'bullish' },
        { source: undefined, contribution: undefined, direction: undefined },
      ],
      'driver',
    );

    expect(rows).toEqual([
      { source: 'cross_asset_rv', contribution: 0.24, direction: 'bullish' },
      { source: 'Unknown signal', contribution: null, direction: 'unknown' },
    ]);
  });

  it('formats missing and snake_case source labels without throwing', () => {
    expect(formatSourceLabel(undefined)).toBe('Unknown signal');
    expect(formatSourceLabel('cross_asset_rv')).toBe('cross asset rv');
  });
});
