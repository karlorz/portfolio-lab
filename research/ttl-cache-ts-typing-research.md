# Research Report: Price Cache Extension & TypeScript Type Safety for Portfolio-Lab

## TL;DR

1. **Price cache**: 8 modules independently re-read `prices.json` (2.4MB) with 4 distinct consumption patterns (raw dict, DataFrame pivot, numpy array, subset extraction). The existing `TTLCache(maxsize=1, ttl=30s)` + `threading.Lock` wrapper is correct but underutilized -- add a cached `get_prices_df(symbols=None)` method that returns a pivoted DataFrame, cutting ~19MB/s of redundant JSON parsing per cron cycle.

2. **TypeScript types**: 15 `as any` casts in `LiveDashboard.tsx` because `SignalsData` is missing 15+ signal panel interfaces (behavioral_sentiment, factor_rotation, etc.). Add Zod schema validation at the fetch boundary, use `z.infer` to generate types from the schema, and drop `as any` entirely. The TS 5.2.2 `satisfies` operator provides an intermediate step for partial migration.

3. **Backtest scripts should bypass caching**: Python backtest scripts (`scripts/optimize_portfolio.py`, `src/backtest/`) already use `load_prices(PRICES_JSON, symbols=...)` which is an intentional design choice for test isolation. Do NOT force these through the TTL cache.

4. **Zod is preferred over io-ts or TypeBox** for this codebase: zero existing FP patterns, 682 code snippets in docs, statically inferred types via `z.infer`, and a straightforward adoption curve.

5. **Incremental adoption matters more than perfection**: Both migrations should happen in stages -- first add `get_prices_df()` alongside the existing `get_prices()`, then port modules one at a time. For TS types, start with a single Zod schema for signals.json, validate at fetch, then delete `as any` casts per-panel.

---

## Part 1: Price Cache Extension

### Current Architecture

The project has a 2.4MB `public/data/prices.json` in compact format:

```json
{
  "SPY": [{"d": "2024-01-02", "p": 475.31}, ...],
  "GLD": [{"d": "2024-01-02", "p": 206.5}, ...],
  ...
}
```

The existing `src/data/price_cache.py` (lines 28-55):

```python
_PRICE_CACHE: TTLCache = TTLCache(maxsize=1, ttl=_PRICE_CACHE_TTL)
_PRICE_CACHE_LOCK = threading.Lock()

def get_prices() -> Dict[str, Any]:
    with _PRICE_CACHE_LOCK:
        if "prices" in _PRICE_CACHE:
            return _PRICE_CACHE["prices"]
    with open(PRICES_JSON) as f:
        data = json.load(f)
    with _PRICE_CACHE_LOCK:
        _PRICE_CACHE["prices"] = data
    return data
```

This is correct but incomplete. The file I/O is outside the lock (good practice), and the TTL prevents redundant reads within a cron cycle. However, every consumer still must parse the compact format themselves.

### The 4 Consumption Patterns Found

**Pattern 1: Dict-to-DataFrame pivot** (3 modules: `ensemble_voter`, `network_momentum_leadlag`, `risk_parity_weight_overlay`)

These all do the same idiom with minor variations -- open file, json.load, melt records, pivot into wide DataFrame, sort index. Repetitive code:

```python
with open(prices_path) as f:
    data = json.load(f)
records = []
for symbol, entries in data.items():
    for entry in entries:
        records.append({'date': entry['d'], 'ticker': symbol, 'price': entry['p']})
df = pd.DataFrame(records)
df = df.pivot(index='date', columns='ticker', values='price')
df = df.sort_index()
```

**Pattern 2: Dict-to-numpy arrays** (1 module: `risk_decomposition`)

Converts each symbol's data to `np.ndarray` of close prices, sorted chronologically. Totally different output format than Pattern 1.

**Pattern 3: Dict kept as dict** (2 modules: `adaptive_sizing`, `risk_agent_hmm`)

Keep the raw dict and access individual symbols like `self.prices[symbol]`, then extract to numpy arrays on demand. `adaptive_sizing` has its own per-instance cache (`_prices` attribute).

**Pattern 4: Subset extraction** (1 module: `unified_orchestrator`)

Only needs SPY/GLD/TLT, manually extracts their price lists from the raw dict, builds a small DataFrame from just those 3 symbols.

**Pattern 5: External helper** (1 module: `black_litterman_mapper`)

Uses `scripts/optimize_portfolio.load_prices(PRICES_JSON, symbols=DEFAULT_SYMBOLS)` which is a standalone helper that does the pivot.

### Cachetools TTLCache Behavior (from docs)

Key findings from the cachetools documentation:

- **TTLCache is NOT thread-safe**: The docs explicitly state "all these classes are not thread-safe. Access to a shared cache from multiple threads must be properly synchronized." The existing `threading.Lock` wrapping is correct.
- **No background expiration thread**: Expired items are only removed on the next mutating operation (`__setitem__`, `__delitem__`, or explicit `.expire()` call). For single-entry caches this is irrelevant, but worth knowing.
- **Expiration is evaluated lazily**: `item in cache` returns False after TTL, but the item still occupies memory until a mutating operation triggers cleanup. The `.expire()` method can be called explicitly.
- **maxsize must be positive**: Using `maxsize=1` for a single-entry cache (keyed as "prices") is correct and documented behavior. For a multi-entry cache (e.g. caching per-symbol DataFrames), set `maxsize` to the number of symbols.
- **The `@cached` decorator pattern**: cachetools also supports `@cached(cache=TTLCache(maxsize=..., ttl=...), lock=threading.Lock())` as a function-level decorator pattern.

### Recommendation for Portfolio-Lab

**Add `get_prices_df()` to the cache module** that returns a cached wide DataFrame, plus a `get_prices_arrays()` for the numpy array pattern:

```python
_PRICE_DF_CACHE: TTLCache = TTLCache(maxsize=1, ttl=_PRICE_CACHE_TTL)

def get_prices_df(
    symbols: Optional[set[str]] = None
) -> pd.DataFrame:
    """Return prices as a wide DataFrame (dates x symbols), cached.

    When called with the same `symbols` argument (including None = all),
    returns the cached DataFrame. Thread-safe.
    """
    cache_key = "all" if symbols is None else frozenset(symbols)
    with _PRICE_CACHE_LOCK:
        if cache_key in _PRICE_DF_CACHE:
            return _PRICE_DF_CACHE[cache_key].copy()

    raw = get_prices()
    records = []
    for symbol, entries in raw.items():
        if symbols is not None and symbol not in symbols:
            continue
        for entry in entries:
            records.append({"date": entry["d"], "ticker": symbol, "price": entry["p"]})

    df = pd.DataFrame(records)
    if df.empty:
        result = pd.DataFrame()
    else:
        result = df.pivot(index="date", columns="ticker", values="price").sort_index()

    with _PRICE_CACHE_LOCK:
        _PRICE_DF_CACHE[cache_key] = result
    return result.copy()  # Return copy to prevent mutation
```

Key design decisions:

- **Separate TTLCache for DataFrames** (not a combined cache): Prevents the DataFrame from being evicted when someone only needs the raw dict. Both share the same lock.
- **`symbols` parameter for partial reads**: Pass `symbols={"SPY", "GLD", "TLT"}` to only load and cache 3 symbols instead of 37. This saves memory when a module only needs a subset.
- **`.copy()` on return**: DataFrames are mutable. Returning a copy prevents callers from mutating the cached version (an issue the cachetools docs explicitly warn about).
- **Backtest scripts bypass**: The backtest scripts intentionally use `load_prices(PRICES_JSON, symbols=...)` from `scripts/optimize_portfolio.py` for isolation. Do NOT route them through the TTL cache -- backtest pipelines should read fresh data for correctness.

### Migration Order

| Module | Pattern | Migrate to | Priority |
|--------|---------|------------|----------|
| `ensemble_voter.py` | Pivot all 37 symbols | `get_prices_df()` | High (called every cron cycle) |
| `network_momentum_leadlag.py` | Pivot all 37 symbols | `get_prices_df(symbols=ASSETS)` | High |
| `risk_parity_weight_overlay.py` | Pivot all 37 symbols | `get_prices_df()` | High |
| `risk_decomposition.py` | numpy arrays per symbol | `get_prices_arrays()` | Medium |
| `adaptive_sizing.py` | Raw dict + extraction | `get_prices()` (already exists) + `get_prices_df(symbols=...)` | Medium |
| `risk_agent_hmm.py` | Raw dict + per-ticker extraction | `get_prices()` (already exists) | Low (ML-gated module) |
| `unified_orchestrator.py` | Subset SPY/GLD/TLT | `get_prices_df(symbols={"SPY","GLD","TLT"})` | High |
| `black_litterman_mapper.py` | External `load_prices()` helper | `get_prices_df(symbols=DEFAULT_SYMBOLS)` | Medium |

---

## Part 2: TypeScript Type Safety for Dashboard Signals

### Current Architecture

The `LiveDashboard.tsx` fetches `signals.json` at line 101:

```typescript
const s = await signalsRes.json();
setSignals(s);
```

The `SignalsData` interface in `src/types/live.ts` (lines 3-86) defines `~20` top-level fields but is missing 15+ signal panel fields. The `as any` casts at lines 420, 424, 429, 614, 615, 618, 619, 622, 623, 626, 629, 630, 671, 713, 716 work around the missing types:

```typescript
<BehavioralSentimentPanel data={(signals as any)?.behavioral_sentiment ?? null} />
<FactorRotationPanel data={(signals as any)?.factor_rotation ?? null} />
```

The 15 missing fields and their shapes (from inspection of the actual signals.json payload):

```typescript
behavioral_sentiment: {} | null
crypto_allocation: { active, btc_weight, eth_weight, total_crypto, btc_momentum_6m }
calendar_seasonality: { active, modifier, active_windows, next_window, days_to_next }
ensemble_voting: { regime, regime_confidence, weighted_consensus, agreement_ratio, action }
alternative_data: { regime, probability, confidence, timestamp, components }
factor_rotation: { selected_factors, allocation, signal_strength, recommendation }
stacking_ensemble: { active, stacking_available, prediction_direction, confidence, probability_bullish }
convexity_harvest: { date, allocation_pct, position_type, vix_level, contango_pct }
llm_sentiment: { timestamp, technical_regime, technical_confidence, sentiment_regime, sentiment_confidence }
sector_rotation: { timestamp, status, vix, regime, methodology }
factor_rotation_dashboard: { active, selected_factors, signal_strength, factor_allocations, backtest_finding }
collar: { active, regime, call_strike, put_strike, net_premium }
kurtosis_regime: { active, kurtosis_20d, kurtosis_60d, ker_ratio, regime }
volatility_parity: { date, target_volatility, spy_pct, gld_pct, tlt_pct }
```

### Zod Schema Validation: The Pattern

Based on Zod docs (v3.24.2, 682 code snippets, source reputation: High), the recommended pattern for this codebase is "validate at fetch boundary, type-infer everywhere else":

**Step 1: Define Zod schemas in `src/types/signals.schema.ts`**

```typescript
import { z } from 'zod';

// Per-signal panel schemas (nullable because signals may be absent)
export const BehavioralSentimentSchema = z.object({
  // Empty object when active, null when inactive
}).nullable().optional();

export const CryptoAllocationSchema = z.object({
  active: z.boolean(),
  btc_weight: z.number(),
  eth_weight: z.number(),
  total_crypto: z.number(),
  btc_momentum_6m: z.number(),
}).nullable().optional();

// ... same pattern for all 14 remaining panels

// The full signals response schema
export const SignalsResponseSchema = z.object({
  generated_at: z.string(),
  regime: z.object({
    regime: z.string(),
    vix: z.number().nullable(),
    detected: z.string().nullable(),
  }),
  target_allocations: z.record(z.string(), z.number()),
  current_positions: z.array(z.object({
    symbol: z.string(),
    shares: z.number(),
    value: z.number(),
    weight: z.number(),
    unrealized: z.number(),
  })),
  // ... existing typed fields ...
  latest_prices: z.record(z.string(), z.number()),
  // New signal panel fields
  behavioral_sentiment: BehavioralSentimentSchema,
  crypto_allocation: CryptoAllocationSchema,
  // ... all 15 panels ...
  collar: CollarSchema,
});

// Infer the type from the schema (single source of truth)
export type SignalsResponse = z.infer<typeof SignalsResponseSchema>;
```

**Step 2: Use at the fetch boundary**

```typescript
const signalsRes = await fetch('/data/signals.json');
const raw = await signalsRes.json();
const parsed = SignalsResponseSchema.parse(raw);  // throws if mismatch
setSignals(parsed);
```

Or use `.safeParse()` for graceful fallback:

```typescript
const result = SignalsResponseSchema.safeParse(raw);
if (!result.success) {
  console.error('Signals JSON validation failed:', result.error.issues);
  setSignals(raw as SignalsResponse);  // fallback but warn
} else {
  setSignals(result.data);
}
```

**Step 3: Drop `as any` casts**

Once `SignalsResponse` includes all signal panel fields as optional (`?.`), the `as any` casts become unnecessary:

```typescript
// Before:
<BehavioralSentimentPanel data={(signals as any)?.behavioral_sentiment ?? null} />

// After:
<BehavioralSentimentPanel data={signals?.behavioral_sentiment ?? null} />
```

### The `satisfies` Operator as an Intermediate Step

TypeScript 4.9+ `satisfies` operator (the project has TS 5.2.2) validates that a value's type satisfies an interface without changing its inferred type. This can help during incremental migration:

```typescript
// Before migration to Zod -- use satisfies to catch structural mismatches
const signalKeys = {
  behavioral_sentiment: signals?.behavioral_sentiment,
  crypto_allocation: signals?.crypto_allocation,
  // ... add fields as you go
} satisfies Partial<Record<keyof SignalsData, unknown>>;
```

However, `satisfies` does NOT add missing fields to the plucked value. It only validates that what you wrote conforms. So for the `data` props that need specific sub-interfaces, you would still need to cast or use a full type annotation. **Zod is the cleaner end-state solution.**

### Branded Types: Not Recommended Here

Branded types (`type Dollars = number & { readonly __brand: 'Dollars' }`) add compile-time nominal typing for domain values. For this codebase the analysis is:

- **Too much friction**: Every `parseFloat`, every JSON deserialization, every arithmetic operation would need unwrapping.
- **Financial panels already use `number`**: The panel components accept `data: SomeType` where all values are `number`. Branding `latest_prices` values wouldn't help because the dashboard never distinguishes "price dollars" from "weight percentage" at the type level.
- **Recommendation**: Skip branded types for now. The existing `Record<string, number>` for prices is sufficient.

### Pre-Existing Signal Panel Types

Some panels already have typed interfaces in their own files (e.g., `RegimeGateData`, `TSMOMData`, `CrossAssetRVData` imported at lines 49-55 of LiveDashboard.tsx). These are fetched from separate endpoints (`/data/regime_gate.json`, etc.) and are already typed. The Zod schema should only cover fields within `signals.json`, not duplicate these.

### Signals Data Dictionary

From analysis of the actual `public/data/signals.json` payload (36 top-level keys), the full type coverage map is:

| Field | In SignalsData? | Cast Needed? |
|-------|-----------------|--------------|
| generated_at | Yes | No |
| regime | Yes | No |
| target_allocations | Yes | No |
| current_positions | Yes | No |
| cash | Yes | No |
| total_value | Yes | No |
| latest_prices | Yes | No |
| recent_orders | Yes | No |
| ml_signals | Yes | No |
| yield_curve | Yes (optional) | No |
| duration_allocation | Yes (optional) | No |
| vix_term_structure | Yes (optional) | No |
| zero_dte | Yes (optional) | No |
| closing_auction | Yes (optional) | No |
| garch_cvar | Yes (optional) | No |
| entropy | Yes (optional) | No |
| bond_momentum | Yes (optional) | No |
| vix_overlay | Yes (optional) | No |
| smart_rebalance | Yes (optional) | No |
| broker | Yes (optional) | No |
| behavioral_sentiment | NO | as any |
| crypto_allocation | NO | as any |
| calendar_seasonality | NO | as any |
| ensemble_voting | NO | as any |
| alternative_data | NO | as any |
| factor_rotation | NO | as any |
| stacking_ensemble | NO | as any |
| convexity_harvest | NO | as any |
| llm_sentiment | NO | as any |
| sector_rotation | NO | as any |
| factor_rotation_dashboard | NO | as any |
| collar | NO | as any |
| kurtosis_regime | NO | as any |
| volatility_parity | NO | as any |
| rebalance_health | NO | (separate endpoint, already typed) |
| staleness | NO | (not rendered yet) |
| spc | NO | (not rendered yet) |

---

## Part 3: Implementation Considerations

### Thread Safety for Price Cache

The current `threading.Lock` wrapper around `TTLCache` is correct per cachetools docs. The key detail is that file I/O happens OUTSIDE the lock. If two threads race simultaneously:

1. Both check the cache under the lock, both miss
2. Both read the file outside the lock (duplicate I/O, but file reads are ~50ms)
3. Both write to the cache under the lock (the second write just overwrites)

This is acceptable for the cron workload (single writer, infrequent reads). For higher contention, the `@cached` decorator with `lock=threading.Lock()` is the recommended pattern from cachetools docs.

### Memory Budget for Price Cache

- Raw dict: ~2.4MB of JSON text in memory, plus Python object overhead (~4-6MB for dict of lists of dicts)
- Wide DataFrame (37 symbols x 5371 days): ~1.6MB for float64 data + index/column overhead
- Both caches simultaneously: ~8-10MB total -- negligible for a server with 2GB+ RAM

The `symbols` parameter limits DataFrame memory when only 3 symbols are needed (~130KB instead of 1.6MB).

### Zod Bundle Size

Zod v3 is ~12KB gzipped. The project's current `package.json` has React (~42KB gzipped) and Recharts (~120KB gzipped). Adding Zod is a ~9% increase in JS bundle but provides runtime validation that currently does not exist -- without it, malformed `signals.json` produces silent undefined errors on the dashboard.

### Validation at Build vs Runtime

The `signals.json` is produced by `generator.py` (Python), consumed by `LiveDashboard.tsx` (TypeScript). There are two validation layers:

1. **Pydantic (Python side)**: The generator should validate output shape before writing. Currently has no schema enforcement for the JSON output.
2. **Zod (TypeScript side)**: Validate at fetch time to catch schema drift between backend and frontend deployments.

Both are recommended. If only one is chosen, Zod provides more safety because it catches runtime drift (human edits to signals.json, partial cron failures, etc.).

### Risks and Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| DataFrame cache returns mutable reference | Silent data corruption | Return `.copy()` from `get_prices_df()` |
| Zod parse failure breaks dashboard | Red screen of death | Use `.safeParse()` with graceful fallback and console.error |
| Backtest scripts use cached stale data | Wrong research results | Backtest scripts bypass cache and read fresh file directly |
| Adding Zod changes TypeScript compilation errors | CI pipeline failures | Add schemas in parallel with existing types, swap fetch boundary last |
| Thread race on cache miss leads to double I/O | ~2x file reads under contention | Acceptable -- TTL prevents repeated reads, and cron workload has minimal contention |

---

## Verification Methods

### Price Cache Verification

1. **Unit test**: Create a test that calls `get_prices_df()` twice -- assert the second call does NOT trigger file I/O (mock `open` and count calls).
2. **Memory test**: Call `get_prices_df(symbols={"SPY","GLD","TLT"})` and assert the returned DataFrame has exactly 3 columns.
3. **Isolation test**: Verify that backtest pipeline (`scripts/optimize_portfolio.load_prices()`) does NOT call `get_prices()`.
4. **Thread safety test**: Call `get_prices()` and `get_prices_df()` concurrently from 4 threads -- verify no deadlocks and consistent results.

**Common wrong method**: Do NOT verify by checking CPU time (noisy in CI). Do NOT verify by asserting `sys.getrefcount` (implementation detail). Do NOT assume `get_prices().copy()` is needed -- the raw dict is also mutable, but the `.copy()` recommendation applies only to the DataFrame.

### TypeScript Type Safety Verification

1. **Compile check**: After adding Zod schemas and removing `as any`, run `bunx tsc --noEmit --strict` -- verify zero type errors.
2. **Runtime validation test**: Create a test signals.json with intentionally wrong shapes (e.g., `{behavioral_sentiment: "wrong_type"}`) and verify `.safeParse()` returns `success: false`.
3. **Snapshot test**: Assert that `z.infer<typeof SignalsResponseSchema>` produces a type structurally equivalent to the manually maintained `SignalsData` interface (use `ts-expect-error` pattern or type testing).
4. **Regression test**: Assert that the Dashboard renders all 15 panels without `as any` by checking the rendered component tree.

**Common wrong method**: Do NOT assume that removing `as any` is sufficient -- ensure Zod validation actually catches structural mismatches. Do NOT remove the `null` fallback (`?? null`) because signal panels can legitimately be absent from the JSON. Do NOT use `z.any()` to speed up migration -- this defeats the purpose.

---

## Sources

1. cachetools documentation -- TTLCache, thread safety, expire behavior. https://cachetools.readthedocs.io/ (accessed 2026-05-26)
2. cachetools source (tkem/cachetools) -- CHANGELOG.rst, README.rst, docs/index.md. https://github.com/tkem/cachetools (accessed 2026-05-26)
3. Zod v3 documentation -- z.infer, discriminatedUnion, safeParse, satisfies pattern. https://zod.dev/ (accessed 2026-05-26)
4. Zod source (colinhacks/zod) -- README.md, API docs, v3 docs. https://github.com/colinhacks/zod (accessed 2026-05-26)
5. TypeScript 4.9 release notes -- `satisfies` operator. https://devblogs.microsoft.com/typescript/announcing-typescript-4-9/#the-satisfies-operator (accessed 2026-05-26)
6. Portfolio-Lab codebase -- `src/data/price_cache.py`, `src/types/live.ts`, `src/components/LiveDashboard.tsx`, `public/data/signals.json` (accessed 2026-05-26)

---

Deep Research Complete
----------------------
Topic: Price Cache Extension + TypeScript Type Safety
Mode: file (research/ttl-cache-ts-typing-research.md)

Sources Queried:
  - Context7: /tkem/cachetools (cachetools), /colinhacks/zod (Zod) -- (model: sonnet)
  - Source reads: portfolio-lab codebase (price_cache.py, LiveDashboard.tsx, types/live.ts, signals.json, 6+ module files)
  - Web docs: cachetools README, Zod README, TypeScript satisfies docs

Synthesis: this agent (model: sonnet)
Refinement: skipped (manual synthesis)
Output: /root/projects/portfolio-lab/research/ttl-cache-ts-typing-research.md
Warnings: Agent tool unavailable -- all research conducted via Context7 CLI + direct source analysis + web doc fetches
