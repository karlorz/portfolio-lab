import { describe, expect, it } from 'bun:test';
import {
  CRYPTO_SYMBOLS,
  DEFAULT_SYMBOL_UNIVERSE,
  MARKET_DATA_SYMBOLS,
  SYMBOL_UNIVERSE_HASH,
  SYMBOL_UNIVERSE_METADATA,
  SYMBOL_UNIVERSE_VERSION,
  resolveProviderSymbol,
  symbolsForProvider,
  validateSymbolUniverse,
  type SymbolUniverseEntry,
} from '../../src/data/symbol_universe';

describe('market data symbol universe contract', () => {
  it('preserves the current Yahoo market-data symbol coverage', () => {
    expect(() => validateSymbolUniverse(DEFAULT_SYMBOL_UNIVERSE)).not.toThrow();
    expect(MARKET_DATA_SYMBOLS).toEqual([
      'SPY', 'QQQ', 'VTI', 'VBR', 'TLT', 'IEF', 'SHY', 'GLD', 'AGG', 'DBC', 'EFA', 'VXUS', '^VIX3M', '^VIX',
      'XLK', 'XLV', 'XLF', 'XLY', 'XLI', 'XLE', 'XLP', 'XLU', 'XLB', 'XLRE', 'XLC',
      'UBT', 'TMF',
      'UUP', 'UDN', 'FXE', 'FXY', 'FXB', 'FXA', 'FXC', 'FXF',
      'MTUM', 'VLUE', 'USMV', 'QUAL',
      'BTC-USD', 'ETH-USD',
    ]);
    expect(DEFAULT_SYMBOL_UNIVERSE.map((entry) => entry.symbol)).toContain('SPY');
    expect(DEFAULT_SYMBOL_UNIVERSE.map((entry) => entry.symbol)).toContain('VIX3M');
    expect(DEFAULT_SYMBOL_UNIVERSE.map((entry) => entry.symbol)).toContain('VIX');
    expect(DEFAULT_SYMBOL_UNIVERSE.map((entry) => entry.symbol)).toContain('BTC-USD');
    expect(DEFAULT_SYMBOL_UNIVERSE.map((entry) => entry.symbol)).toContain('ETH-USD');
    expect(CRYPTO_SYMBOLS).toEqual(['BTC-USD', 'ETH-USD']);
    expect(SYMBOL_UNIVERSE_VERSION).toBe('symbol-universe/v1');
    expect(SYMBOL_UNIVERSE_HASH).toMatch(/^[a-f0-9]{64}$/);
    expect(SYMBOL_UNIVERSE_METADATA).toEqual({
      version: SYMBOL_UNIVERSE_VERSION,
      hash: SYMBOL_UNIVERSE_HASH,
    });
  });

  it('resolves provider-specific aliases for Yahoo, licensed EOD, broker, and FRED providers', () => {
    expect(resolveProviderSymbol('VIX3M', 'yahoo')).toBe('^VIX3M');
    expect(resolveProviderSymbol('VIX3M', 'licensed_eod')).toBe('VIX3M');
    expect(resolveProviderSymbol('VIX', 'yahoo')).toBe('^VIX');
    expect(resolveProviderSymbol('VIX', 'licensed_eod')).toBe('VIX');
    expect(resolveProviderSymbol('SPY', 'broker')).toBe('SPY');
    expect(resolveProviderSymbol('DGS10', 'fred')).toBe('DGS10');
    expect(symbolsForProvider('yahoo')).toContain('^VIX3M');
    expect(symbolsForProvider('yahoo')).toContain('^VIX');
    expect(symbolsForProvider('licensed_eod')).toContain('VIX3M');
    expect(symbolsForProvider('licensed_eod')).toContain('VIX');
  });

  it('fails clearly when a required provider alias is missing', () => {
    const universe: SymbolUniverseEntry[] = [
      {
        symbol: 'SPY',
        category: 'core',
        required: true,
        aliases: { yahoo: 'SPY' },
      },
    ];

    expect(() => validateSymbolUniverse(universe)).toThrow('Missing licensed_eod alias for required symbol SPY');
    expect(() => resolveProviderSymbol('SPY', 'licensed_eod', universe)).toThrow(
      'Missing licensed_eod alias for SPY',
    );
  });

  it('rejects unknown canonical symbols during alias resolution', () => {
    expect(() => resolveProviderSymbol('NOTREAL', 'yahoo')).toThrow('Unknown symbol NOTREAL');
  });
});
