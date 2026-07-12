import { describe, expect, it } from 'bun:test';
import React from 'react';
import { renderToStaticMarkup } from 'react-dom/server';
import {
  CrossAssetRVPanel,
  normalizeCrossAssetRVData,
} from '../../src/components/CrossAssetRVPanel';
import producer from '../../public/data/cross_asset_rv.json';

describe('CrossAssetRVPanel producer shape', () => {
  it('normalizes public producer pairs without percentile_1y', () => {
    const data = normalizeCrossAssetRVData(producer);
    expect(data).not.toBeNull();
    expect(data!.pairs.length).toBeGreaterThan(0);
    expect(data!.pairs[0].pair).toContain('/');
    expect(Number.isFinite(data!.pairs[0].percentile_1y)).toBe(true);
  });

  it('renders producer payload without throwing', () => {
    const html = renderToStaticMarkup(React.createElement(CrossAssetRVPanel, { data: producer as never }));
    expect(html).toContain('Cross-Asset Relative Value');
    expect(html).not.toContain('No cross-asset RV data available');
  });
});
