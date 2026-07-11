import { describe, expect, it } from 'bun:test';
import {
  makeSignalListKey,
  uniqueSignalNames,
} from '../../src/components/RegimeGatePanel';

describe('RegimeGatePanel list keys', () => {
  it('keeps duplicate signal names unique in active and inactive lists', () => {
    const signals = ['cross_asset_rv', 'alt_data', 'cross_asset_rv'];
    const keys = signals.map(makeSignalListKey);

    expect(new Set(keys).size).toBe(signals.length);
    expect(keys).toEqual(['cross_asset_rv-0', 'alt_data-1', 'cross_asset_rv-2']);
  });

  it('counts active signals by unique identifier order', () => {
    const signals = ['cross_asset_rv', 'alt_data', 'cross_asset_rv', 'unified_overlay'];

    expect(uniqueSignalNames(signals)).toEqual(['cross_asset_rv', 'alt_data', 'unified_overlay']);
  });
});
