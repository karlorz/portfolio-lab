import { describe, expect, it } from 'bun:test';
import {
  formatSourceLabel,
  getExplainabilityEmptyState,
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

  it('distinguishes stale unavailable payloads from missing data', () => {
    const state = getExplainabilityEmptyState({
      timestamp: '2026-07-06T12:00:00',
      analysis_date: '2026-07-06',
      latest_decision: null,
      recent_decisions: [],
      signal_deep_dives: {},
      top_sources_today: [],
      decision_quality: {
        status: 'unavailable_current_signals',
        reason: 'Current signals.json was not available.',
      },
      freshness: {
        status: 'unavailable',
        stale_source_file: 'explainability_2026-05-18.json',
        stale_analysis_date: '2026-05-18',
      },
    });

    expect(state.title).toBe('No current explainability available');
    expect(state.detail).toContain('last historical report was explainability_2026-05-18.json');
    expect(state.detail).toContain('2026-05-18');
  });
});
