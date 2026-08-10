// Print the canonical TypeScript crypto symbol universe as a JSON array.
//
// The Python freshness layer mirrors this list (src/dashboard/data_freshness_section.py
// CRYPTO_SYMBOLS); the crypto parity test runs this script and asserts the two
// stay in sync. Canonical source of truth: src/data/symbol_universe.ts.
import { CRYPTO_SYMBOLS } from '../src/data/symbol_universe';

console.log(JSON.stringify(CRYPTO_SYMBOLS));
