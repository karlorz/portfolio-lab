import { describe, expect, it } from 'bun:test';
import React from 'react';
import { renderToStaticMarkup } from 'react-dom/server';
import { StackingEnsemblePanel } from '../../src/components/StackingEnsemblePanel';

describe('StackingEnsemblePanel fallback disclosure', () => {
  it('renders no-model fallback as an operator-visible disclosure', () => {
    const html = renderToStaticMarkup(
      React.createElement(StackingEnsemblePanel, {
        data: {
          active: true,
          stacking_available: false,
          prediction_direction: 'neutral',
          confidence: 0,
          probability_bullish: 0,
          probability_bearish: 0,
          probability_neutral: 1,
          fallback_used: true,
          model_version: 'fallback_v2.81',
          voting_accuracy: 0.65,
          stacking_accuracy: 0.76,
          feature_count: null,
          feature_count_metadata_available: false,
          feature_count_source: 'unavailable_no_model',
          runtime_mode: 'fallback_no_model',
          model_backed: false,
          operator_disclosure: 'No stacking model loaded; panel is showing weighted-voting fallback.',
          latency_ms: 0.12,
        },
      }),
    );

    expect(html).toContain('No stacking model loaded');
    expect(html).toContain('Not model-backed');
    expect(html).toContain('stacking-runtime-disclosure');
  });
});
