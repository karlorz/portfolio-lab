export const SYMBOL_UNIVERSE_VERSION = 'symbol-universe/v1';

export type SymbolProvider = 'yahoo' | 'licensed_eod' | 'broker' | 'fred';
export type SymbolCategory = 'core' | 'sector' | 'leveraged_treasury' | 'fx' | 'factor' | 'macro';

export interface SymbolUniverseEntry {
  symbol: string;
  category: SymbolCategory;
  required: boolean;
  aliases: Partial<Record<SymbolProvider, string>>;
}

const MARKET_PROVIDER_ALIASES: SymbolProvider[] = ['yahoo', 'licensed_eod', 'broker'];

function marketSymbol(symbol: string, category: Exclude<SymbolCategory, 'macro'>, yahooAlias = symbol): SymbolUniverseEntry {
  const providerNeutralAlias = yahooAlias.startsWith('^') ? yahooAlias.slice(1) : yahooAlias;
  return {
    symbol,
    category,
    required: true,
    aliases: {
      yahoo: yahooAlias,
      licensed_eod: providerNeutralAlias,
      broker: providerNeutralAlias,
    },
  };
}

function macroSymbol(symbol: string): SymbolUniverseEntry {
  return {
    symbol,
    category: 'macro',
    required: true,
    aliases: { fred: symbol },
  };
}

export const DEFAULT_SYMBOL_UNIVERSE: SymbolUniverseEntry[] = [
  marketSymbol('SPY', 'core'),
  marketSymbol('QQQ', 'core'),
  marketSymbol('VTI', 'core'),
  marketSymbol('VBR', 'core'),
  marketSymbol('TLT', 'core'),
  marketSymbol('IEF', 'core'),
  marketSymbol('SHY', 'core'),
  marketSymbol('GLD', 'core'),
  marketSymbol('AGG', 'core'),
  marketSymbol('DBC', 'core'),
  marketSymbol('EFA', 'core'),
  marketSymbol('VXUS', 'core'),
  marketSymbol('VIX3M', 'core', '^VIX3M'),
  marketSymbol('XLK', 'sector'),
  marketSymbol('XLV', 'sector'),
  marketSymbol('XLF', 'sector'),
  marketSymbol('XLY', 'sector'),
  marketSymbol('XLI', 'sector'),
  marketSymbol('XLE', 'sector'),
  marketSymbol('XLP', 'sector'),
  marketSymbol('XLU', 'sector'),
  marketSymbol('XLB', 'sector'),
  marketSymbol('XLRE', 'sector'),
  marketSymbol('XLC', 'sector'),
  marketSymbol('UBT', 'leveraged_treasury'),
  marketSymbol('TMF', 'leveraged_treasury'),
  marketSymbol('UUP', 'fx'),
  marketSymbol('UDN', 'fx'),
  marketSymbol('FXE', 'fx'),
  marketSymbol('FXY', 'fx'),
  marketSymbol('FXB', 'fx'),
  marketSymbol('FXA', 'fx'),
  marketSymbol('FXC', 'fx'),
  marketSymbol('FXF', 'fx'),
  marketSymbol('MTUM', 'factor'),
  marketSymbol('VLUE', 'factor'),
  marketSymbol('USMV', 'factor'),
  marketSymbol('QUAL', 'factor'),
  macroSymbol('DGS2'),
  macroSymbol('DGS10'),
  macroSymbol('DGS30'),
];

function requiredAliases(entry: SymbolUniverseEntry): SymbolProvider[] {
  return entry.category === 'macro' ? ['fred'] : MARKET_PROVIDER_ALIASES;
}

function stableUniverseHash(entries: SymbolUniverseEntry[]): string {
  const payload = JSON.stringify(entries.map((entry) => ({
    symbol: entry.symbol,
    category: entry.category,
    required: entry.required,
    aliases: Object.fromEntries(Object.entries(entry.aliases).sort()),
  })));
  const seeds = [
    0x811c9dc5, 0x9e3779b9, 0x85ebca6b, 0xc2b2ae35,
    0x27d4eb2f, 0x165667b1, 0xd3a2646c, 0xfd7046c5,
  ];
  const chunks = seeds.map((seed) => {
    let hash = seed;
    for (let index = 0; index < payload.length; index += 1) {
      hash ^= payload.charCodeAt(index);
      hash = Math.imul(hash, 16777619);
    }
    return (hash >>> 0).toString(16).padStart(8, '0');
  });
  return chunks.join('');
}

export function validateSymbolUniverse(entries: SymbolUniverseEntry[] = DEFAULT_SYMBOL_UNIVERSE): SymbolUniverseEntry[] {
  const seen = new Set<string>();
  for (const entry of entries) {
    if (!entry.symbol) {
      throw new Error('Symbol universe entries require a symbol');
    }
    if (seen.has(entry.symbol)) {
      throw new Error(`Duplicate symbol ${entry.symbol}`);
    }
    seen.add(entry.symbol);
    if (!entry.required) continue;
    for (const provider of requiredAliases(entry)) {
      if (!entry.aliases[provider]) {
        throw new Error(`Missing ${provider} alias for required symbol ${entry.symbol}`);
      }
    }
  }
  return entries;
}

export function resolveProviderSymbol(
  symbol: string,
  provider: SymbolProvider,
  entries: SymbolUniverseEntry[] = DEFAULT_SYMBOL_UNIVERSE,
): string {
  const entry = entries.find((candidate) => candidate.symbol === symbol);
  if (!entry) {
    throw new Error(`Unknown symbol ${symbol}`);
  }
  const alias = entry.aliases[provider];
  if (!alias) {
    throw new Error(`Missing ${provider} alias for ${symbol}`);
  }
  return alias;
}

export function symbolsForProvider(
  provider: SymbolProvider,
  entries: SymbolUniverseEntry[] = DEFAULT_SYMBOL_UNIVERSE,
): string[] {
  validateSymbolUniverse(entries);
  return entries
    .filter((entry) => entry.aliases[provider])
    .map((entry) => resolveProviderSymbol(entry.symbol, provider, entries));
}

function marketSymbolsForCategory(category: Exclude<SymbolCategory, 'macro'>): string[] {
  return DEFAULT_SYMBOL_UNIVERSE
    .filter((entry) => entry.category === category)
    .map((entry) => resolveProviderSymbol(entry.symbol, 'yahoo'));
}

export const CORE_SYMBOLS = marketSymbolsForCategory('core');
export const SECTOR_ETFS = marketSymbolsForCategory('sector');
export const LEVERAGED_TREASURY_ETFS = marketSymbolsForCategory('leveraged_treasury');
export const FX_SYMBOLS = marketSymbolsForCategory('fx');
export const FACTOR_ETFS = marketSymbolsForCategory('factor');
export const MARKET_DATA_SYMBOLS = [
  ...CORE_SYMBOLS,
  ...SECTOR_ETFS,
  ...LEVERAGED_TREASURY_ETFS,
  ...FX_SYMBOLS,
  ...FACTOR_ETFS,
];
export const FRED_SERIES = {
  dgs2: resolveProviderSymbol('DGS2', 'fred'),
  dgs10: resolveProviderSymbol('DGS10', 'fred'),
  dgs30: resolveProviderSymbol('DGS30', 'fred'),
} as const;
export const SYMBOL_UNIVERSE_HASH = stableUniverseHash(DEFAULT_SYMBOL_UNIVERSE);
export const SYMBOL_UNIVERSE_METADATA = {
  version: SYMBOL_UNIVERSE_VERSION,
  hash: SYMBOL_UNIVERSE_HASH,
};
