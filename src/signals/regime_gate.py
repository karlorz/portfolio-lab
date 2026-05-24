"""Regime-adaptive signal gating — hard on/off per signal per market regime.

Sits before BanditWeighter in the compute_vote pipeline. Gates eliminate
negative-Sharpe signals per regime (regime-mismatch), while the bandit
optimizes allocation among positive survivors. Gating and weighting are
complementary, not competing.

Gating rules derived from deep research (2026-05-23/24):
- MULTI_SPEED_MOM: OFF in HIGH_VOL and CRISIS (Sharpe -0.5 or worse)
- INTERNATIONAL_MOMENTUM: OFF in CRISIS (correlated with equity drawdowns)
- BEHAVIORAL_SENTIMENT: OFF in NORMAL/HIGH_VOL/CRISIS (-0.216 Sharpe, 65.8% FP)
- CROSS_ASSET_REGIME_ARB: OFF in LOW_VOL (marginal when markets are calm)

Hysteresis: minimum 20-day dwell time per regime prevents chattering
on rapid regime transitions (AlphaCrafter, Yuan et al. 2026).

Data-driven updates: update_from_performance() can refine gate rules
based on rolling per-signal Sharpe by regime from the health tracker.
"""
from typing import Dict, List, Optional, Set



__all__ = ['RegimeGate']

class RegimeGate:
    """Hard gate that enables/disables signals per market regime.

    Usage:
        gate = RegimeGate()
        active = gate.gate("HIGH_VOL")  # returns list of active signal names
        gate.gate_with_hysteresis("CRISIS", "HIGH_VOL", days_in_regime=5)
    """

    # Gating rules: signal_name -> set of regimes where signal is OFF
    # Based on regime-signal Sharpe analysis from deep research.
    # A signal not in this dict is ON in all regimes.
    #
    # Updated 2026-05-24 with comprehensive rules from signal-ensemble
    # optimization deep research:
    # - MSM: net-negative (-0.012 Sharpe) overall; OFF in HIGH_VOL and CRISIS
    # - INTL_MOM: correlated with equity drawdowns in CRISIS
    # - BEHAVIORAL_SENTIMENT: -0.216 Sharpe, 65.8% false positive rate
    # - CROSS_ASSET_REGIME_ARB: marginal in LOW_VOL sideways markets
    GATE_RULES: Dict[str, Set[str]] = {
        "multi_speed_momentum": {"HIGH_VOL", "CRISIS"},
        "international_momentum": {"CRISIS"},
        "behavioral_sentiment": {"NORMAL", "HIGH_VOL", "CRISIS"},  # net-negative in all but LOW_VOL
        "cross_asset_regime_arb": {"LOW_VOL"},  # marginal when markets are calm
    }

    # Minimum days in a regime before allowing gate changes (hysteresis)
    DEFAULT_MIN_DWELL_DAYS = 20

    def __init__(
        self,
        gate_rules: Optional[Dict[str, Set[str]]] = None,
        min_dwell_days: int = DEFAULT_MIN_DWELL_DAYS,
    ):
        self.gate_rules = gate_rules or {k: set(v) for k, v in self.GATE_RULES.items()}
        self.min_dwell_days = min_dwell_days

    def is_active(self, signal_name: str, regime_name: str) -> bool:
        """Check if a signal is active in a given regime."""
        off_regimes = self.gate_rules.get(signal_name, set())
        return regime_name not in off_regimes

    def gate(self, regime_name: str) -> List[str]:
        """Return list of signal names that are active in the given regime.

        Only returns signals that have explicit gate rules. Signals not in
        gate_rules are implicitly ON in all regimes and are NOT included
        in the return value (caller should treat them as always-active).
        """
        return [
            sig for sig in self.gate_rules
            if self.is_active(sig, regime_name)
        ]

    def gate_with_hysteresis(
        self,
        current_regime: str,
        prev_regime: Optional[str] = None,
        days_in_regime: int = 999,
    ) -> List[str]:
        """Gate signals with hysteresis to prevent chattering.

        If the regime just changed and we haven't been in the new regime
        for min_dwell_days, we use the PREVIOUS regime's gating rules.
        This prevents rapid toggling on short-lived regime transitions.

        Args:
            current_regime: Current detected regime
            prev_regime: Previous regime (None if first observation)
            days_in_regime: Days since regime transition

        Returns:
            List of gated signal names that are active
        """
        if prev_regime is not None and days_in_regime < self.min_dwell_days:
            # Hysteresis: use previous regime's gating
            return self.gate(prev_regime)
        return self.gate(current_regime)

    def filter_weights(self, weights: Dict, regime_name: str) -> Dict:
        """Zero out weights for gated-off signals in a regime.

        Takes a dict of {signal: weight} and returns a new dict where
        gated-off signals have weight=0.0. Caller should renormalize.
        """
        filtered = {}
        for signal, weight in weights.items():
            # Handle both Enum keys and string keys
            signal_name = signal.value if hasattr(signal, 'value') else str(signal)
            if self.is_active(signal_name, regime_name):
                filtered[signal] = weight
            else:
                filtered[signal] = 0.0
        return filtered

    def get_active_signal_names(self, all_signals: List[str], regime_name: str) -> List[str]:
        """Return all signal names that are active in a regime.

        Unlike gate(), this includes signals NOT in gate_rules
        (they are implicitly ON).
        """
        return [
            sig for sig in all_signals
            if self.is_active(sig, regime_name)
        ]

    def update_from_performance(
        self,
        regime_signal_sharpe: Dict[str, Dict[str, float]],
        sharpe_threshold: float = 0.0,
    ) -> None:
        """Update gate rules from rolling per-signal Sharpe by regime.

        This enables data-driven gate refinement: if a signal has negative
        rolling Sharpe in a regime, it's gated OFF. If it recovers, it's
        re-enabled (subject to hysteresis).

        Args:
            regime_signal_sharpe: {regime: {signal: sharpe_ratio}}.
                Computed from health tracker's IC data or attribution.
            sharpe_threshold: Minimum Sharpe to keep a signal ON.
                Default 0.0 means any positive Sharpe passes.
        """
        for regime, signals in regime_signal_sharpe.items():
            for signal, sharpe in signals.items():
                signal_name = signal.lower().replace(" ", "_")
                if sharpe < sharpe_threshold:
                    # Gate OFF: add regime to signal's off-set
                    if signal_name not in self.gate_rules:
                        self.gate_rules[signal_name] = set()
                    self.gate_rules[signal_name].add(regime)
                else:
                    # Gate ON: remove regime from signal's off-set
                    if signal_name in self.gate_rules:
                        self.gate_rules[signal_name].discard(regime)
                        # Remove empty entries
                        if not self.gate_rules[signal_name]:
                            del self.gate_rules[signal_name]

    def get_gate_summary(self) -> Dict[str, List[str]]:
        """Return a summary of all gate rules for logging/diagnostics.

        Returns:
            Dict of {signal_name: [list of OFF regimes]}.
        """
        return {sig: sorted(regimes) for sig, regimes in self.gate_rules.items()}
