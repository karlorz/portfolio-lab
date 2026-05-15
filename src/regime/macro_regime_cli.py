#!/usr/bin/env python3
"""
Macro Regime Synthesizer CLI (v4.30)

CLI tool for managing cross-asset macro regime synthesis.

Commands:
    classify: Classify current regime from signal inputs
    overlay: Calculate allocation overlay for base portfolio
    history: Show recent regime history
    simulate: Simulate regime classification from historical signals
    backtest: Run walk-forward backtest with regime overlay

Usage:
    python -m src.regime.macro_regime_cli classify --signal fed_policy=easing --signal yield_curve=steep
    python -m src.regime.macro_regime_cli overlay --base spy=0.46,gld=0.38,tlt=0.16 --confidence 80
    python -m src.regime.macro_regime_cli history --days 30
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from src.regime.macro_regime import (
    MacroRegimeSynthesizer,
    MacroRegime,
    SignalState,
    SignalInput,
    classify_current_regime,
)


def parse_signal_arg(arg: str) -> tuple:
    """Parse --signal name=value argument."""
    parts = arg.split("=", 1)
    if len(parts) != 2:
        raise ValueError(f"Invalid signal format: {arg}. Expected: name=value")
    return parts[0], parts[1]


def parse_allocation_arg(arg: str) -> Dict[str, float]:
    """Parse --base spy=0.46,gld=0.38,tlt=0.16 argument."""
    result = {}
    parts = arg.split(",")
    for part in parts:
        if "=" in part:
            asset, weight = part.split("=", 1)
            result[asset.strip().lower()] = float(weight.strip())
    return result


def cmd_classify(args):
    """Classify current macro regime from signals."""
    # Parse signal arguments
    signals = {}
    for signal_arg in args.signal:
        name, value = parse_signal_arg(signal_arg)
        signals[name] = value
    
    # Classify regime
    result = classify_current_regime(signals, db_path=args.db_path)
    
    # Output
    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print(f"\n{'='*60}")
        print(f"Macro Regime Classification")
        print(f"{'='*60}")
        print(f"Timestamp:     {result['timestamp']}")
        print(f"Regime:        {result['regime_display']}")
        print(f"Confidence:    {result['confidence']:.1f}%")
        print(f"Agreement:     {result['signal_agreement']:.2f}")
        print(f"Strength:      {result['signal_strength']:.3f}")
        print(f"Weighted Sum:  {result['weighted_sum']:+.3f}")
        print(f"Duration:      {result['regime_duration_days']} days")
        print(f"\nRecommendation: {result['recommended_action']}")
        print(f"\nAllocation Shifts:")
        for asset, shift in result['allocation_shifts'].items():
            print(f"  {asset.upper()}: {shift*100:+.1f}%")
        print(f"{'='*60}\n")


def cmd_overlay(args):
    """Calculate allocation overlay for base portfolio."""
    # Parse base allocation
    base = parse_allocation_arg(args.base)
    
    # Create synthesizer
    synth = MacroRegimeSynthesizer(db_path=args.db_path)
    
    # Determine regime and confidence
    if args.regime:
        regime = MacroRegime(args.regime)
        confidence = args.confidence
    else:
        # Use last classification from database
        history = synth.get_regime_history(days=1)
        if history:
            regime = MacroRegime(history[0]['regime'])
            confidence = history[0]['confidence']
        else:
            print("Error: No recent regime classification found. Use --regime option.")
            sys.exit(1)
    
    # Calculate overlay
    overlay = synth.get_allocation_overlay(regime, confidence, base)
    
    # Calculate changes
    changes = {k: overlay[k] - base.get(k, 0) for k in overlay}
    
    # Output
    if args.json:
        result = {
            "regime": regime.value,
            "confidence": confidence,
            "base_allocation": base,
            "adjusted_allocation": overlay,
            "changes": changes,
        }
        print(json.dumps(result, indent=2))
    else:
        print(f"\n{'='*60}")
        print(f"Portfolio Allocation Overlay")
        print(f"{'='*60}")
        print(f"Regime:     {regime.value.replace('_', ' ').title()}")
        print(f"Confidence: {confidence:.1f}%")
        print(f"\n{'Asset':<12} {'Base':>10} {'Adjusted':>12} {'Change':>10}")
        print(f"{'-'*46}")
        for asset in overlay:
            base_val = base.get(asset, 0) * 100
            adj_val = overlay[asset] * 100
            chg_val = changes[asset] * 100
            print(f"{asset.upper():<12} {base_val:>9.1f}% {adj_val:>11.1f}% {chg_val:>+9.1f}%")
        print(f"{'='*60}\n")


def cmd_history(args):
    """Show recent regime history."""
    synth = MacroRegimeSynthesizer(db_path=args.db_path)
    history = synth.get_regime_history(days=args.days)
    
    if not history:
        print(f"No regime classifications found in last {args.days} days.")
        return
    
    if args.json:
        print(json.dumps(history, indent=2))
    else:
        print(f"\n{'='*80}")
        print(f"Macro Regime History (Last {args.days} days)")
        print(f"{'='*80}")
        print(f"{'Timestamp':<22} {'Regime':<20} {'Conf':>8} {'Weighted':>10}")
        print(f"{'-'*80}")
        for entry in history:
            regime_display = entry['regime'].replace('_', ' ')[:18]
            print(f"{entry['timestamp'][:19]:<22} {regime_display:<20} "
                  f"{entry['confidence']:>7.1f}% {entry['weighted_sum']:>+9.3f}")
        print(f"{'='*80}\n")


def cmd_simulate(args):
    """Simulate regime classification with multiple signal scenarios."""
    scenarios = {
        "bull_market": {
            "fed_policy": "easing",
            "yield_curve": "steep",
            "credit_spread": "normal",
            "equity_tsmom": "risk_on",
        },
        "late_cycle": {
            "fed_policy": "tightening",
            "yield_curve": "flat",
            "credit_spread": "normal",
            "equity_tsmom": "risk_on",
        },
        "defensive": {
            "yield_curve": "inverted",
            "credit_spread": "elevated",
            "bond_momentum": "tlt",
            "equity_tsmom": "risk_off",
        },
        "crisis": {
            "fed_policy": "easing",
            "yield_curve": "inverted",
            "credit_spread": "distressed",
            "fx_carry": "unwind_risk",
            "equity_tsmom": "risk_off",
        },
        "recovery": {
            "fed_policy": "easing",
            "yield_curve": "steep",
            "credit_spread": "normal",
            "fx_carry": "safe",
            "equity_tsmom": "risk_on",
        },
    }
    
    print(f"\n{'='*80}")
    print(f"Macro Regime Simulation Scenarios")
    print(f"{'='*80}")
    
    for scenario_name, signals in scenarios.items():
        result = classify_current_regime(signals)
        
        print(f"\nScenario: {scenario_name.replace('_', ' ').title()}")
        print(f"  Signals: {', '.join(f'{k}={v}' for k, v in signals.items())}")
        print(f"  Regime: {result['regime_display']}")
        print(f"  Confidence: {result['confidence']:.1f}%")
        print(f"  Recommendation: {result['recommended_action']}")
    
    print(f"\n{'='*80}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Macro Regime Synthesizer CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s classify --signal fed_policy=easing --signal yield_curve=steep
  %(prog)s overlay --base spy=0.46,gld=0.38,tlt=0.16 --regime risk_on_growth --confidence 80
  %(prog)s history --days 30
  %(prog)s simulate
        """
    )
    
    parser.add_argument(
        "--db-path",
        default="data/macro_regime_history.db",
        help="Path to regime history database"
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output as JSON"
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Classify command
    classify_parser = subparsers.add_parser("classify", help="Classify current regime")
    classify_parser.add_argument(
        "--signal",
        action="append",
        default=[],
        help="Signal input (format: name=value, e.g., fed_policy=easing)"
    )
    classify_parser.set_defaults(func=cmd_classify)
    
    # Overlay command
    overlay_parser = subparsers.add_parser("overlay", help="Calculate allocation overlay")
    overlay_parser.add_argument(
        "--base",
        required=True,
        help="Base allocation (format: asset=weight,asset=weight, e.g., spy=0.46,gld=0.38,tlt=0.16)"
    )
    overlay_parser.add_argument(
        "--regime",
        help="Regime to use (if not using last classification)"
    )
    overlay_parser.add_argument(
        "--confidence",
        type=float,
        default=75.0,
        help="Confidence level for overlay (default: 75)"
    )
    overlay_parser.set_defaults(func=cmd_overlay)
    
    # History command
    history_parser = subparsers.add_parser("history", help="Show regime history")
    history_parser.add_argument(
        "--days",
        type=int,
        default=30,
        help="Number of days to show (default: 30)"
    )
    history_parser.set_defaults(func=cmd_history)
    
    # Simulate command
    simulate_parser = subparsers.add_parser("simulate", help="Run simulation scenarios")
    simulate_parser.set_defaults(func=cmd_simulate)
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    args.func(args)


if __name__ == "__main__":
    main()
