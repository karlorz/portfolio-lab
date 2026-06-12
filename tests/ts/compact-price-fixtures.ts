export type CompactPriceRow = { d: string; p: number };
export type CompactPricePayload = Record<string, CompactPriceRow[]>;

export const COMPACT_PRICE_FIXTURE_NAMES = [
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
] as const;

export type CompactPriceFixtureName = typeof COMPACT_PRICE_FIXTURE_NAMES[number];

const FIXTURES: Record<CompactPriceFixtureName, CompactPricePayload> = {
  clean: {
    SPY: [
      { d: '2026-06-10', p: 612.34 },
      { d: '2026-06-11', p: 614.25 },
    ],
    GLD: [{ d: '2026-06-11', p: 318.12 }],
  },
  adjusted_close_proxy: {
    SPY: [{ d: '2026-06-10', p: 612.34 }],
    TLT: [{ d: '2026-06-10', p: 88.75 }],
  },
  duplicate_date: {
    SPY: [
      { d: '2026-06-10', p: 612.34 },
      { d: '2026-06-10', p: 612.35 },
    ],
  },
  internal_gap: {
    SPY: [
      { d: '2026-06-10', p: 612.34 },
      { d: '2026-06-11', p: 614.25 },
      { d: '2026-06-12', p: 615.10 },
    ],
    TLT: [
      { d: '2026-06-10', p: 88.75 },
      { d: '2026-06-12', p: 89.01 },
    ],
  },
  stale_latest: {
    SPY: [
      { d: '2026-06-10', p: 612.34 },
      { d: '2026-06-11', p: 614.25 },
      { d: '2026-06-12', p: 615.10 },
    ],
    GLD: [
      { d: '2026-06-10', p: 318.12 },
      { d: '2026-06-11', p: 319.20 },
    ],
  },
  non_monotonic: {
    SPY: [
      { d: '2026-06-11', p: 612.34 },
      { d: '2026-06-10', p: 613.50 },
    ],
  },
  zero_price: {
    SPY: [{ d: '2026-06-10', p: 0 }],
  },
  negative_price: {
    SPY: [{ d: '2026-06-10', p: -1 }],
  },
  split_like_return: {
    SPY: [
      { d: '2026-06-10', p: 100 },
      { d: '2026-06-11', p: 45 },
    ],
  },
  extreme_return: {
    SPY: [
      { d: '2026-06-10', p: 100 },
      { d: '2026-06-11', p: 250 },
    ],
  },
};

export function buildCompactPriceFixture(name: CompactPriceFixtureName): CompactPricePayload {
  return structuredClone(FIXTURES[name]);
}

export function cleanCompactPrices(): CompactPricePayload {
  return buildCompactPriceFixture('clean');
}

export function adjustedCloseProxyCompactPrices(): CompactPricePayload {
  return buildCompactPriceFixture('adjusted_close_proxy');
}

export function duplicateDateCompactPrices(): CompactPricePayload {
  return buildCompactPriceFixture('duplicate_date');
}

export function internalGapCompactPrices(): CompactPricePayload {
  return buildCompactPriceFixture('internal_gap');
}

export function staleLatestCompactPrices(): CompactPricePayload {
  return buildCompactPriceFixture('stale_latest');
}

export function nonMonotonicCompactPrices(): CompactPricePayload {
  return buildCompactPriceFixture('non_monotonic');
}

export function zeroPriceCompactPrices(): CompactPricePayload {
  return buildCompactPriceFixture('zero_price');
}

export function negativePriceCompactPrices(): CompactPricePayload {
  return buildCompactPriceFixture('negative_price');
}

export function splitLikeReturnCompactPrices(): CompactPricePayload {
  return buildCompactPriceFixture('split_like_return');
}

export function extremeReturnCompactPrices(): CompactPricePayload {
  return buildCompactPriceFixture('extreme_return');
}
