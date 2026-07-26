import { describe, expect, it } from 'bun:test';
import {
  buildLastNBarsCompact,
  resolvePricesCompactNBars,
  PRICES_COMPACT_DEFAULT_N_BARS,
} from '../../scripts/fetch-data.ts';
import { unwrapCompactPricePayload } from '../../src/backtest/price-loader.ts';

describe('Batch BK prices_compact last-N contract', () => {
  it('slices each symbol to last n bars and stamps meta', () => {
    const full = {
      SPY: Array.from({ length: 1000 }, (_, i) => ({
        d: `2020-01-${String((i % 28) + 1).padStart(2, '0')}`,
        p: 100 + i,
      })),
      GLD: Array.from({ length: 100 }, (_, i) => ({
        d: `2024-01-${String((i % 28) + 1).padStart(2, '0')}`,
        p: 180 + i,
      })),
    };
    // Fix dates to be unique ascending for SPY
    full.SPY = Array.from({ length: 1000 }, (_, i) => ({
      d: `2000-01-01`,
      p: 100 + i,
    }));
    // Use sortable day index in price only; slice is by array tail
    const payload = buildLastNBarsCompact(full, 504);
    expect(payload.meta.schema).toBe('prices/compact-v1');
    expect(payload.meta.n_bars).toBe(504);
    expect(payload.meta.full_artifact).toBe('prices.json');
    expect(payload.symbols.SPY).toHaveLength(504);
    expect(payload.symbols.GLD).toHaveLength(100); // shorter than n
    expect(payload.symbols.SPY[0].p).toBe(100 + (1000 - 504));
    expect(payload.symbols.SPY[503].p).toBe(100 + 999);
    expect(payload.meta.bar_count).toBe(504 + 100);
  });

  it('defaults n_bars to 504 and resolves env override', () => {
    expect(PRICES_COMPACT_DEFAULT_N_BARS).toBe(504);
    expect(resolvePricesCompactNBars({})).toBe(504);
    expect(resolvePricesCompactNBars({ PRICES_COMPACT_N_BARS: '252' })).toBe(252);
    expect(resolvePricesCompactNBars({ PRICES_COMPACT_N_BARS: 'nope' })).toBe(504);
  });

  it('unwraps wrapped compact payload and accepts legacy flat map', () => {
    const wrapped = buildLastNBarsCompact(
      {
        SPY: [
          { d: '2026-01-01', p: 1 },
          { d: '2026-01-02', p: 2 },
        ],
      },
      10,
    );
    const unwrapped = unwrapCompactPricePayload(wrapped);
    expect(unwrapped?.SPY).toHaveLength(2);
    expect(unwrapped?.SPY[1].p).toBe(2);

    const legacy = unwrapCompactPricePayload({
      SPY: [{ d: '2026-01-01', p: 10 }],
    });
    expect(legacy?.SPY[0].p).toBe(10);
  });

  it('compact bar count is much smaller than full for long series', () => {
    const full: Record<string, { d: string; p: number }[]> = {};
    for (const sym of ['SPY', 'GLD', 'TLT']) {
      full[sym] = Array.from({ length: 5000 }, (_, i) => ({ d: `d${i}`, p: i }));
    }
    const compact = buildLastNBarsCompact(full, 504);
    const fullBars = 5000 * 3;
    expect(compact.meta.bar_count).toBe(504 * 3);
    expect(compact.meta.bar_count / fullBars).toBeLessThan(0.2);
  });
});
