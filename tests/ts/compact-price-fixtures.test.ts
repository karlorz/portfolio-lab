import { describe, expect, it } from 'bun:test';
import {
  COMPACT_PRICE_FIXTURE_NAMES,
  adjustedCloseProxyCompactPrices,
  cleanCompactPrices,
  duplicateDateCompactPrices,
  extremeReturnCompactPrices,
  internalGapCompactPrices,
  negativePriceCompactPrices,
  nonMonotonicCompactPrices,
  splitLikeReturnCompactPrices,
  staleLatestCompactPrices,
  zeroPriceCompactPrices,
} from './compact-price-fixtures';

describe('shared compact price fixture pack', () => {
  it('exposes named clean and dirty compact-price payload builders', () => {
    expect(new Set(COMPACT_PRICE_FIXTURE_NAMES)).toEqual(new Set([
      'clean',
      'adjusted_close_proxy',
      'duplicate_date',
      'internal_gap',
      'stale_latest',
      'non_monotonic',
      'zero_price',
      'negative_price',
      'split_like_return',
      'extreme_return',
    ]));

    expect(cleanCompactPrices()).toEqual({
      SPY: [
        { d: '2026-06-10', p: 612.34 },
        { d: '2026-06-11', p: 614.25 },
      ],
      GLD: [{ d: '2026-06-11', p: 318.12 }],
    });
    expect(adjustedCloseProxyCompactPrices()).toEqual({
      SPY: [{ d: '2026-06-10', p: 612.34 }],
      TLT: [{ d: '2026-06-10', p: 88.75 }],
    });
    expect(duplicateDateCompactPrices().SPY).toEqual([
      { d: '2026-06-10', p: 612.34 },
      { d: '2026-06-10', p: 612.35 },
    ]);
    expect(internalGapCompactPrices().TLT).toEqual([
      { d: '2026-06-10', p: 88.75 },
      { d: '2026-06-12', p: 89.01 },
    ]);
    expect(staleLatestCompactPrices().GLD.at(-1)?.d).toBe('2026-06-11');
    expect(nonMonotonicCompactPrices().SPY[1].d).toBe('2026-06-10');
    expect(zeroPriceCompactPrices().SPY[0].p).toBe(0);
    expect(negativePriceCompactPrices().SPY[0].p).toBe(-1);
    expect(splitLikeReturnCompactPrices().SPY).toEqual([
      { d: '2026-06-10', p: 100 },
      { d: '2026-06-11', p: 45 },
    ]);
    expect(extremeReturnCompactPrices().SPY).toEqual([
      { d: '2026-06-10', p: 100 },
      { d: '2026-06-11', p: 250 },
    ]);
  });

  it('returns fresh compact payload instances', () => {
    const first = cleanCompactPrices();
    const second = cleanCompactPrices();

    first.SPY[0].p = 1;

    expect(second.SPY[0].p).toBe(612.34);
  });
});
