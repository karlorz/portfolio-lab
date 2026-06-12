import { describe, expect, it } from 'bun:test';
import React from 'react';
import { renderToStaticMarkup } from 'react-dom/server';
import {
  BehavioralSentimentPanel,
  type BehavioralSentimentData,
} from '../../src/components/BehavioralSentimentPanel';
import {
  CryptoAllocationPanel,
  type CryptoData,
} from '../../src/components/CryptoAllocationPanel';
import {
  CalendarSeasonalityPanel,
  type CalendarData,
} from '../../src/components/CalendarSeasonalityPanel';

function render(element: React.ReactElement): string {
  return renderToStaticMarkup(element);
}

describe('signal card presentation', () => {
  it('frames behavioral sentiment as a diagnostic signal with decision hierarchy', () => {
    const data: BehavioralSentimentData = {
      active: true,
      composite_score: 0.18,
      signal_type: 'neutral',
      confidence: 0.7,
      equity_shift_pct: 0,
      z_score: 3.11,
      vix: 21.5,
      regime_suppressed: false,
      signal_count_5d: 0,
      options: {
        skew_index: 143.1,
        vix: 21.5,
        vix9d: 23.2,
        vix9d_ratio: 1.08,
        put_call_ratio: 0.65,
        fear_greed_score: 0.4,
      },
      retail: {
        retail_call_put_ratio: 1.54,
        retail_buy_sell_imbalance: 0,
      },
      social: {
        mention_velocity_7d: 1,
        sentiment_divergence: 0.076,
      },
      backtest_finding: 'VIX-proxy contrarian signals degrade Sharpe by -0.216 (2021-2026). Real-time SKEW/PCR data needed for behavioral alpha.',
    };

    const html = render(React.createElement(BehavioralSentimentPanel, { data }));

    expect(html).toContain('Diagnostic only');
    expect(html).toContain('Decision');
    expect(html).toContain('NEUTRAL');
    expect(html).toContain('Composite Score');
    expect(html).toContain('+0.18');
    expect(html).toContain('Backtest Finding (Phase 4)');
  });

  it('shows crypto portfolio allocation separately from BTC and ETH sleeve split', () => {
    const data: CryptoData = {
      active: true,
      btc_weight: 0.6,
      eth_weight: 0.4,
      total_crypto: 0.0119,
      btc_momentum_6m: 0.007,
      eth_momentum_6m: 0.017,
      btc_vol_regime: 'high',
      eth_vol_regime: 'normal',
      confidence: 63,
    };

    const html = render(React.createElement(CryptoAllocationPanel, { data, portfolioValue: 100000 }));

    expect(html).toContain('Portfolio Allocation');
    expect(html).toContain('1.19%');
    expect(html).toContain('$1190');
    expect(html).toContain('BTC sleeve 60.0%');
    expect(html).toContain('0.71%');
    expect(html).toContain('$714');
    expect(html).toContain('ETH sleeve 40.0%');
    expect(html).toContain('0.48%');
    expect(html).toContain('$476');
    expect(html).not.toContain('$60000');
    expect(html).not.toContain('$40000');
  });

  it('presents calendar seasonality as an execution timing decision', () => {
    const data: CalendarData = {
      active: true,
      modifier: 1,
      active_windows: [],
      next_window: 'pre_fomc',
      days_to_next: 4,
      recommendation: 'proceed',
      effect: 'neutral',
    };

    const html = render(React.createElement(CalendarSeasonalityPanel, { data }));

    expect(html).toContain('Execution Timing');
    expect(html).toContain('Recommendation');
    expect(html).toContain('PROCEED');
    expect(html).toContain('Urgency');
    expect(html).toContain('1.00x');
    expect(html).toContain('Execution Window');
    expect(html).toContain('Pre-FOMC in 4d');
  });
});
