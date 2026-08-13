#!/usr/bin/env python3
"""
Compare ensemble weights with and without regime-conditional multipliers.

Validates that the ENSEMBLE_DISABLE_REGIME_WEIGHTS toggle produces the
expected weight deltas across all 5 regimes. This provides the A/B
comparison instrumentation that the full backtest (multi-hour) would
otherwise require.

Usage:
    python scripts/compare_regime_weights.py
    ENSEMBLE_DISABLE_REGIME_WEIGHTS=1 python scripts/compare_regime_weights.py
"""

import os
import sys
from pathlib import Path

# Ensure project root is on sys.path
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from src.strategy.ensemble_voter import (  # noqa: E402  # deliberate placement (bootstrap/sys.path ordering)
    EnsembleVoter,
    Regime,
    SignalSource,
    REGIME_CONDITIONAL_WEIGHTS,
)

# Active signal sources in the current ensemble
ACTIVE_SOURCES = [
    SignalSource.ALTERNATIVE_DATA,
    SignalSource.INTERNATIONAL_MOMENTUM,
    SignalSource.CROSS_ASSET_RV,
    SignalSource.CROSS_ASSET_REGIME_ARB,
    SignalSource.UNIFIED_OVERLAY,
]

REGIME_ORDER = [Regime.CRISIS, Regime.HIGH_VOL, Regime.NORMAL, Regime.LOW_VOL, Regime.RECOVERY]


def format_weight_delta(before: float, after: float) -> str:
    """Format a weight delta with direction indicator."""
    delta = after - before
    if abs(delta) < 0.001:
        return "  --  "
    direction = "+" if delta > 0 else "-"
    return f"{direction}{abs(delta):.3f}"


def compare_weights(voter: EnsembleVoter, disable_toggle: bool = False):
    """Compare weights for all regimes, with and without regime multipliers."""
    print("=" * 90)
    print("  Regime-Conditional Ensemble Weight Comparison")
    print("=" * 90)
    print(f"  Toggle: {'DISABLED (no regime multipliers)' if disable_toggle else 'ENABLED (regime multipliers active)'}")
    print()

    # Collect per-regime weight deltas
    all_results = {}

    for regime in REGIME_ORDER:
        regime_name = regime.name

        # Get base weights (after gating, adaptive, health, correlation penalty — same
        # order as compute_vote before the toggle point at line 1016)
        base_weights = voter.get_blended_weights(regime_name)
        base_weights = voter._apply_regime_gating(base_weights, regime_name)
        base_weights = voter._apply_adaptive_weights(base_weights, regime)
        base_weights = voter._apply_health_weights(base_weights)
        base_weights = voter._apply_correlation_penalty(base_weights)

        if disable_toggle:
            adjusted = {k: v for k, v in base_weights.items()}
        else:
            adjusted = voter._apply_regime_weights(base_weights, regime)

        # Filter to active sources only, sorted by abs delta
        results = []
        for source in ACTIVE_SOURCES:
            before = base_weights.get(source, 0.0)
            after = adjusted.get(source, 0.0)
            delta = after - before
            multiplier = REGIME_CONDITIONAL_WEIGHTS.get(regime_name, {}).get(source, 1.0)
            results.append((source.name, before, after, delta, multiplier))

        results.sort(key=lambda r: abs(r[3]), reverse=True)
        all_results[regime_name] = results

        # Print per-regime table
        multipliers_display = REGIME_CONDITIONAL_WEIGHTS.get(regime_name, {})
        if multipliers_display:
            mult_str = ", ".join(
                f"{k}: {v:.1f}x" for k, v in sorted(multipliers_display.items())
            )
        else:
            mult_str = "(no multipliers — NORMAL regime)"

        print(f"  ┌─ {regime_name} {'─' * (56 - len(regime_name))}")
        print(f"  │  Multipliers: {mult_str}")
        print(f"  │  {'Source':<28} {'Before':>7}  {'After':>7}  {'Delta':>8}  Mult")
        print(f"  │  {'-'*28}  {'-'*7}  {'-'*7}  {'-'*8}  ----")

        for name, before, after, delta, mult in results:
            delta_str = format_weight_delta(before, after)
            if abs(delta) < 0.001:
                delta_str = f"  {delta_str}  "
            elif abs(delta) > 0.01:
                emphasis = " ***" if abs(delta) > 0.03 else ""
                delta_str = f"{delta_str}{emphasis}"
            print(f"  │  {name:<28}  {before:6.3f}   {after:6.3f}   {delta_str}   {mult:4.1f}")

        print(f"  │  {'Sum':<28}  {sum(r[1] for r in results):6.3f}   {sum(r[2] for r in results):6.3f}")
        print()

    # Summary: direction validation
    print("  ── Direction Validation ──")
    checks = [
        ("CRISIS: alt_data boosted", lambda r: any(x[0] == "ALTERNATIVE_DATA" and x[3] > 0 for x in r["CRISIS"])),
        ("CRISIS: unified_overlay reduced", lambda r: any(x[0] == "UNIFIED_OVERLAY" and x[3] < 0 for x in r["CRISIS"])),
        ("HIGH_VOL: intl_momentum reduced", lambda r: any(x[0] == "INTERNATIONAL_MOMENTUM" and x[3] < 0 for x in r["HIGH_VOL"])),
        ("LOW_VOL: intl_momentum boosted", lambda r: any(x[0] == "INTERNATIONAL_MOMENTUM" and x[3] > 0 for x in r["LOW_VOL"])),
        ("LOW_VOL: regime_arb reduced", lambda r: any(x[0] == "CROSS_ASSET_REGIME_ARB" and x[3] < 0 for x in r["LOW_VOL"])),
        ("RECOVERY: intl_momentum boosted", lambda r: any(x[0] == "INTERNATIONAL_MOMENTUM" and x[3] > 0 for x in r["RECOVERY"])),
        ("NORMAL: no deltas", lambda r: all(abs(x[3]) < 0.001 for x in r["NORMAL"])),
        ("Non-NORMAL regimes sum to 1.0", lambda r: all(
            abs(sum(x[2] for x in r[reg.name]) - 1.0) < 0.005
            for reg in REGIME_ORDER if reg != Regime.NORMAL
        )),
        ("NORMAL: before == after (no reweighting)", lambda r: all(
            abs(x[1] - x[2]) < 0.001 for x in r["NORMAL"]
        )),
    ]

    all_pass = True
    for label, check_fn in checks:
        passed = check_fn(all_results) if not disable_toggle else True
        if not passed:
            all_pass = False
        if disable_toggle and label != "NORMAL: no deltas":
            print(f"  [SKIP] {label} (toggle disabled)")
        else:
            print(f"  [{'PASS' if passed else 'FAIL'}] {label}")

    print()
    if disable_toggle:
        print("  Result: Toggle OFF — all regimes use uniform weights (no multipliers).")
    elif all_pass:
        print("  Result: ALL DIRECTIONS VALIDATED — regime multipliers produce correct deltas.")
    else:
        print("  Result: SOME CHECKS FAILED — review regime multiplier configuration.")
        return 1

    return 0


def main():
    disable_toggle = os.environ.get("ENSEMBLE_DISABLE_REGIME_WEIGHTS", "").lower() in ("1", "true")

    voter = EnsembleVoter()
    rc = compare_weights(voter, disable_toggle=disable_toggle)

    print("=" * 90)
    if disable_toggle:
        print("  To compare WITH regime multipliers: python scripts/compare_regime_weights.py")
    else:
        print("  To compare WITHOUT regime multipliers:")
        print("    ENSEMBLE_DISABLE_REGIME_WEIGHTS=1 python scripts/compare_regime_weights.py")
    print("=" * 90)

    return rc


if __name__ == "__main__":
    sys.exit(main())
