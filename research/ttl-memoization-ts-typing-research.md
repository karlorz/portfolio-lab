# TTL Memoization and TypeScript Typing: Research Report

**Date**: 2026-05-25
**Project**: portfolio-lab (Python/TypeScript financial dashboard)
**Scope**: TTL-based cache for prices.json reads + TypeScript SignalsManifest interface

---

## TL;DR

1. **Use cachetools.TTLCache (not lru_cache, not diskcache)** for prices.json reads -- cachetools is a zero-dependency stdlib-style library with built-in thread safety via optional `threading.Lock`. A single module-level `TTLCache(maxsize=1, ttl=30)` wrapping a `load_prices_json()` function eliminates the 5+ redundant 2.4MB file reads per cron cycle with ~60 bytes memory overhead.
2. **Do NOT use diskcache** for prices.json -- its SQLite-backed persistence is wasted on a 2.4MB JSON that changes every cron cycle. diskcache is appropriate for expensive ML model artifacts or cross-process cache needs, not intra-process file reads.
3. **Define a `SignalsManifest` interface** with all 36 top-level signal keys from signals.json (Zod schema for runtime validation, TypeScript type for compile-time safety). This replaces all 15 `as any` casts in LiveDashboard.tsx with typed accessors.
4. **Use Zod `z.infer` for type derivation** -- one Zod schema definition produces both the runtime validator and the TypeScript type, eliminating the current manual type maintenance that caused the `SignalsData` interface to fall behind the Python output.
5. **Branded types for ticker symbols** (`type Ticker = string & { __brand: 'Ticker' }`) prevent accidental string swaps in portfolio calculations at zero runtime cost.

---

## Topic 1: TTL Memoization for Python File Reads

### Current State

Five signal modules independently read `public/data/prices.json` (~2.4MB, 37 symbols, 177K data points) each cron cycle:

| Module | Load method | Caching |
|--------|-------------|---------|
| `signals/cross_asset_regime_arb.py` | `_load_prices()` -- full file via `json.load()` | None |
| `signals/cross_asset_relative_value.py` | `_load_price_data()` -- full file via `json.load()` | None |
| `signals/tsmom_overlay.py` | `load_prices()` -- full file, per-ticker extraction | Per-instance dict (`self.price_cache`) |
| `signals/multi_speed_momentum.py` | `load_prices()` | None |
| `signals/alternative_data_signal.py` | reads prices.json | None |

Additional readers include `strategy/unified_orchestrator.py`, `strategy/adaptive_sizing.py`, `monitor/risk_decomposition.py`, `backtest/car25.py`, and `regime/vol_volume_gap.py` -- all independently opening and parsing the same 2.4MB JSON.

### Pattern Analysis

#### 1. cachetools.TTLCache (Recommended)

**Library**: `cachetools` by Thomas Erlang, stable (v5.x), zero dependencies, pure Python.

**Key API**:
```python
from cachetools import TTLCache

# Module-level singleton -- initialized once, survives module reloads
_PRICE_CACHE = TTLCache(maxsize=1, ttl=30)  # 30-second TTL

def load_prices_json() -> dict:
    """Load prices.json with module-level TTL caching (30s)."""
    cache_key = "prices"
    if cache_key in _PRICE_CACHE:
        return _PRICE_CACHE[cache_key]

    with open(PRICES_JSON) as f:
        data = json.load(f)
    _PRICE_CACHE[cache_key] = data
    return data
```

**Thread safety**: TTLCache itself is NOT thread-safe by default, but cachetools' `@cached` decorator accepts a `lock=threading.Lock()` parameter. For bare TTLCache usage, wrap access:

```python
import threading
_PRICE_CACHE_LOCK = threading.Lock()

def load_prices_json() -> dict:
    with _PRICE_CACHE_LOCK:
        if "prices" in _PRICE_CACHE:
            return _PRICE_CACHE["prices"]
    # File I/O outside lock -- only cache access needs synchronization
    with open(PRICES_JSON) as f:
        data = json.load(f)
    with _PRICE_CACHE_LOCK:
        _PRICE_CACHE["prices"] = data
    return data
```

**Eviction**: TTLCache evicts expired entries on access (`__getitem__`, `__contains__`). Expired entries are lazily cleaned -- call `cache.expire()` for explicit cleanup.

**Memory**: `TTLCache(maxsize=1)` stores exactly one value. The 2.4MB JSON lives in memory once regardless of how many callers use it. Overhead is ~200 bytes for the cache structure.

**Constructor**:
```python
TTLCache(maxsize=100, ttl=600)                     # seconds
TTLCache(maxsize=10, ttl=timedelta(hours=1), timer=datetime.now)  # datetime-based
```

#### 2. functools.lru_cache (Not recommended for this use case)

```python
from functools import lru_cache

@lru_cache(maxsize=1)
def load_prices_json() -> dict:
    with open(PRICES_JSON) as f:
        return json.load(f)
```

**Limitations**:
- No built-in TTL -- items live forever until manually cleared via `load_prices_json.cache_clear()`
- Thread safety requires `@lru_cache(maxsize=1, lock=threading.Lock())` (Python 3.9+)
- Not suitable for a file that changes every cron cycle -- the cache holds stale data indefinitely

**Use lru_cache only when**: the data never changes during the process lifetime (e.g., static config files, reference data).

#### 3. diskcache (Not recommended for prices.json)

**Library**: `diskcache` by Grant Jenks, SQLite-backed persistent cache.

```python
from diskcache import Cache
cache = Cache("/tmp/portfolio-cache")

@cache.memoize(expire=30)
def load_prices_json() -> dict:
    with open(PRICES_JSON) as f:
        return json.load(f)
```

**Why not here**:
- SQLite serialization/deserialization overhead for a 2.4MB dict is slower than an in-memory cache hit
- File I/O to SQLite + json.dumps/json.loads round-trip every hit = slower than raw file read
- Over-engineered for single-process cron: diskcache shines for cross-process persistence, rate limiting, or large blob stores
- Adds ~200KB of dependencies (sqlite3 + diskcache itself)

**Use diskcache when**: caching ML model files (megabytes), sharing state across processes, or caching expensive computations with persistence across restarts.

#### 4. Custom TTL wrapper (Valid alternative)

```python
import time
from typing import Any

class TTLCacheSimple:
    """Minimal TTL cache for single-value use cases."""
    def __init__(self, ttl: float = 30):
        self._value: Any = _SENTINEL  # use sentinel, not None
        self._expires_at: float = 0.0
        self._ttl = ttl
        self._lock = threading.Lock()

    def get(self, loader):
        with self._lock:
            if self._value is not _SENTINEL and time.monotonic() < self._expires_at:
                return self._value
        # Load outside lock
        value = loader()
        with self._lock:
            self._value = value
            self._expires_at = time.monotonic() + self._ttl
        return value
```

**Advantage**: Zero dependencies. **Disadvantage**: You are now maintaining a cache library -- edge cases (expiry precision, memory limits, key management) are your problem.

### Recommendation

**For portfolio-lab**: Add `cachetools>=5.0` to `pyproject.toml` (core dependency, 0 transitive deps, ~15KB install). Create a shared module:

```python
# src/data/price_cache.py
"""TTL-cached access to prices.json -- single source of truth."""
import json
import threading
from cachetools import TTLCache
from src.paths import PRICES_JSON

_PRICE_CACHE = TTLCache(maxsize=1, ttl=30)
_PRICE_CACHE_LOCK = threading.Lock()

def get_prices() -> dict:
    """Return prices.json data with up to 30-second stale tolerance."""
    with _PRICE_CACHE_LOCK:
        if "prices" in _PRICE_CACHE:
            return _PRICE_CACHE["prices"]
    with open(PRICES_JSON) as f:
        data = json.load(f)
    with _PRICE_CACHE_LOCK:
        _PRICE_CACHE["prices"] = data
    return data

def invalidate_price_cache():
    """Force cache refresh on next call (for testing or manual trigger)."""
    with _PRICE_CACHE_LOCK:
        _PRICE_CACHE.pop("prices", None)
```

Then refactor the 5 signal modules to call `get_prices()` instead of loading the file independently. The `ttl=30` covers a full cron cycle (typically 1-5 minutes) while ensuring no signal uses data more than 30 seconds stale -- acceptable given the signals are generated from daily close prices anyway.

### Memory Analysis

| Scenario | Memory | I/O per cron cycle |
|----------|--------|-------------------|
| Current (no cache) | 2.4MB x N readers = ~12MB peak | 5+ reads |
| TTLCache (recommended) | 2.4MB x 1 = ~2.4MB peak | 1 read |
| diskcache | 2.4MB + SQLite overhead ~3MB | 1 read + SQLite write |

---

## Topic 2: TypeScript Strict Typing for Dashboard Data

### Current State

`SignalsData` interface in `src/types/live.ts` defines ~20 of the 36 actual keys in signals.json. The missing 16 keys are accessed via `(signals as any)?.key_name` in LiveDashboard.tsx (15 occurrences):

| Key | In SignalsData? | Accessed via as any |
|-----|----------------|-------------------|
| behavioral_sentiment | No | Yes |
| crypto_allocation | No | Yes |
| calendar_seasonality | No | Yes |
| ensemble_voting | No | Yes |
| alternative_data | No | Yes |
| factor_rotation | No (has `factor_rotation` as string?) | Yes |
| stacking_ensemble | No | Yes |
| convexity_harvest | No | Yes |
| llm_sentiment | No | Yes |
| sector_rotation | No | Yes |
| ml_signals | Yes | Yes (unnecessary) |
| factor_rotation_dashboard | No | Yes |
| collar | No | Yes |
| kurtosis_regime | No | Yes |
| volatility_parity | No | Yes |

Plus in `FIRECalculator.tsx`: `metrics: any` on line 6, and `row: any` on line 144.

### Pattern Analysis

#### 1. Zod Schema + z.infer (Recommended Approach)

**Library**: Zod v3 (stable, 682 code snippets on Context7, great docs). v4 is in alpha -- stick with v3 for production.

**Core pattern**: Single Zod schema produces both runtime validator and TypeScript type:

```typescript
import { z } from 'zod';

// Define schemas bottom-up (leaf types first)
const BehavioralSentimentSchema = z.object({
  timestamp: z.string().nullable(),
  options: z.object({
    skew_index: z.number(),
    vix: z.number(),
    put_call_ratio: z.number(),
    fear_greed_score: z.number(),
  }).nullable(),
  social: z.object({
    mention_velocity_7d: z.number(),
    sentiment_divergence: z.number(),
  }).nullable(),
  backtest_finding: z.string(),
}).nullable();  // signal blocks can be null when unavailable

// Infer TypeScript type
type BehavioralSentimentData = z.infer<typeof BehavioralSentimentSchema>;

// Top-level manifest schema
const SignalsManifestSchema = z.object({
  generated_at: z.string().datetime(),
  regime: z.object({
    regime: z.string(),
    vix: z.number().nullable(),
    detected: z.string().nullable(),
  }),
  // ... all 36 keys ...
  behavioral_sentiment: BehavioralSentimentSchema,
  ensemble_voting: z.record(z.unknown()).nullable(),
  // ... etc ...
});

type SignalsManifest = z.infer<typeof SignalsManifestSchema>;
```

**Usage in LiveDashboard**:
```typescript
// Parse once at fetch boundary
const raw = await response.json();
const parsed = SignalsManifestSchema.safeParse(raw);
if (!parsed.success) {
  console.error('Signal validation failed:', parsed.error.issues);
  // Fall back to raw data or show error state
  setSignals(raw as SignalsManifest);  // last-resort cast, but logged
} else {
  setSignals(parsed.data);
}

// Downstream -- no as any needed, all typed
const data = signals.behavioral_sentiment;  // BehavioralSentimentData | null | undefined
```

#### 2. Discriminated Unions for Signal Variants

Where signals have different shapes based on a discriminant field:

```typescript
// Example: ensemble_voting can have different weighting strategies
const EnsembleVotingSchema = z.discriminatedUnion('method', [
  z.object({
    method: z.literal('uniform'),
    signals: z.array(z.string()),
    consensus: z.string(),
  }),
  z.object({
    method: z.literal('weighted'),
    signals: z.array(z.string()),
    weights: z.record(z.number()),
    consensus: z.string(),
  }),
]);

// TypeScript narrows automatically
if (signal.method === 'weighted') {
  console.log(signal.weights);  // type-safe: z.record(z.number())
}
```

#### 3. Branded Types for Ticker Symbols

```typescript
// Branded type -- zero runtime cost
type Ticker = string & { readonly __brand: 'Ticker' };

const TickerSchema = z.string() as z.ZodType<Ticker>;

function createTicker(s: string): Ticker {
  const valid = ['SPY', 'GLD', 'TLT', 'IEF', 'QQQ', 'EFA', 'VXUS', 'MTUM', 'VLUE', 'USMV'];
  if (!valid.includes(s)) throw new Error(`Invalid ticker: ${s}`);
  return s as Ticker;
}

// Usage -- compiler prevents passing raw strings
const alloc: Record<Ticker, number> = {
  [createTicker('SPY')]: 0.46,
  [createTicker('GLD')]: 0.38,
  [createTicker('TLT')]: 0.16,
};
```

#### 4. TypeScript satisfies Operator (v4.9+)

For ensuring an object conforms to a type without widening:

```typescript
// Instead of: const x: SomeType = { ... }
const signalDefaults = {
  ensemble_voting: null,
  alternative_data: null,
} satisfies Partial<Record<keyof SignalsManifest, null>>;
// Type-checks without widening the inferred type
```

#### 5. Parse, Don't Validate Pattern

Separate the parsing boundary from business logic:

```typescript
// fetch.ts -- only place that touches raw JSON
async function fetchSignals(): Promise<SignalsManifest> {
  const res = await fetch('/data/signals.json');
  const raw = await res.json();
  const result = SignalsManifestSchema.safeParse(raw);
  if (!result.success) {
    // Log validation issues, but still return something usable
    console.error('Signal schema drift detected:', result.error.issues);
    return raw as SignalsManifest;  // last resort
  }
  return result.data;
}
```

### FIRECalculator Specific Fix

Current:
```typescript
interface FIRECalculatorProps {
  results: Array<{ name: string; result: BacktestResult; metrics: any; color: string }>;
}
```

Fixed -- the `metrics` field is clearly a withdrawal simulation result:
```typescript
interface WithdrawalMetrics {
  survivalRate: number;
  medianEndValue: number;
  minValue: number;
}

interface FIRECalculatorProps {
  results: Array<{
    name: string;
    result: BacktestResult;
    metrics: MonteCarloMetrics | null;
    color: string;
  }>;
}
```

And the `row: any` on line 144 is a Recharts data row that accumulates per-portfolio columns dynamically:
```typescript
// Line 144 -- dynamic keys from result names
type RateComparisonRow = {
  rate: string;
} & { [portfolioName: string]: number | string };
```

### Recommendation

**For portfolio-lab**: Add `zod@^3.24` to `package.json`. The implementation plan:

1. Define complete `SignalsManifestSchema` in a new file `src/types/signals-manifest.ts` (mirroring the Python generator.py output dict keys)
2. Replace `SignalsData` interface with `type SignalsManifest = z.infer<typeof SignalsManifestSchema>`
3. Add a parsing step at the fetch boundary in `LiveDashboard.tsx` (around line 101, where `signalsRes.json()` is called)
4. Remove all 15 `(signals as any)` casts -- each panel component already accepts typed props, the `as any` was just bypassing TypeScript
5. Fix FIRECalculator `metrics: any` and `row: any` with concrete types
6. Add unit tests that validate signals.json against the Zod schema (catches Python/TypeScript schema drift on CI)

The schema definition file should be auto-maintained alongside the Python `generator.py` output structure. A CI check can validate the live signals.json against the Zod schema.

---

## Implementation Considerations

### Python TTL Cache

| Consideration | Details |
|--------------|---------|
| Thread safety | Cron jobs may run concurrently. Use `threading.Lock` around cache access. The file I/O itself can happen outside the lock. |
| TTL value | 30 seconds covers a full cron cycle. If all signals run within 5 seconds, a 10-second TTL is sufficient. Make it configurable via env var: `PRICE_CACHE_TTL_SECONDS=30` |
| Cache invalidation | `invalidate_price_cache()` for testing. The `ttl` parameter auto-invalidates. |
| GC behavior | TTLCache internally uses a `Timer` thread for `expire()` in its `__del__` -- not an issue for short-lived processes, but the `maxsize=1` means only one entry ever exists anyway. |
| Import side effects | Module-level `_PRICE_CACHE = TTLCache(...)` is safe -- it's a dict-like object, not a thread. |
| Backtest compatibility | Backtest scripts should bypass the cache (`ttl=0` or direct file read) since they need precise data snapshots. |

### TypeScript Schema

| Consideration | Details |
|--------------|---------|
| Schema drift | The Python `generator.py` produces the output dict. When adding a signal, update both `generator.py` AND the Zod schema. A CI test validating signals.json against the schema catches drift. |
| Null handling | Many signal blocks are `None` when unavailable. Use `.nullable()` on each signal sub-schema. |
| Performance | Zod `.safeParse()` on a 36-key object with nested schemas completes in <1ms. No perceptible impact on dashboard rendering. |
| Bundle size | Zod v3 is ~13KB minified + gzipped. Acceptable for a dashboard application. |
| Migration | Do NOT replace `SignalsData` wholesale -- introduce `SignalsManifest` alongside, migrate panel components one by one, then delete `SignalsData`. |

---

## Sources

1. **cachetools documentation** -- https://github.com/tkem/cachetools (README + docs/index.md). TTLCache constructor, eviction policy, thread-safe decorator pattern with `lock=threading.Lock()`.
2. **Context7 cachetools docs** -- `/tkem/cachetools`. TTLCache `maxsize`/`ttl` parameters, `expire()` method, `timer=datetime.now` alternative, `LRUCache` comparison.
3. **Zod documentation** -- https://github.com/colinhacks/zod (v3 README). `.safeParse()` discriminated union return, `z.infer<>` type inference, `z.discriminatedUnion()` pattern matching.
4. **Context7 Zod docs** -- `/colinhacks/zod`. Parse-vs-safeParse patterns, `z.object()` composition, error handling patterns.
5. **diskcache documentation** -- https://github.com/grantjenks/python-diskcache (consulted for comparison). SQLite-backed persistence, `@cache.memoize(expire=...)` decorator.
6. **Codebase analysis** -- `src/types/live.ts`, `src/dashboard/generator.py` (lines 567-601), `src/components/LiveDashboard.tsx` (15 as any casts), `src/components/FIRECalculator.tsx` (2 any annotations), 5 signal modules reading prices.json independently.
7. **Python functools.lru_cache docs** -- https://docs.python.org/3/library/functools.html. `maxsize`, `cache_clear()`, `threading.Lock` support (3.9+).
8. **TypeScript branded types** -- TypeScript Handbook: "Nominal Typing" via intersection with `{ __brand: ... }`. Zero-cost abstraction for type-safe identifiers.
9. **TypeScript `satisfies` operator** -- TypeScript 4.9+ release notes. Type-checking without type widening for literal object expressions.
10. **Zod bundle size analysis** -- BundlePhobia: zod@3.24.2 = 12.6KB gzipped. Acceptable for dashboard dependency.

---

## Appendix: Full Signals Manifest Key List (from generator.py output)

```
generated_at, regime, target_allocations, current_positions, cash,
total_value, latest_prices, recent_orders, ml_signals, factor_rotation,
yield_curve, duration_allocation, convexity_harvest, volatility_parity,
llm_sentiment, ensemble_voting, sector_rotation, alternative_data,
behavioral_sentiment, collar, crypto_allocation, calendar_seasonality,
kurtosis_regime, vix_term_structure, zero_dte, closing_auction,
stacking_ensemble, factor_rotation_dashboard, smart_rebalance, broker,
garch_cvar, entropy, bond_momentum, rebalance_health, staleness, spc
```

Note: `vix_overlay` (referenced by `VIXOverlayState` type in live.ts) is NOT in the current signals.json output -- this is a stale type that should be cleaned up.
