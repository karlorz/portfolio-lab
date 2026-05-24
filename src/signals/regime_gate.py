"""Regime-adaptive signal gating — hard on/off per signal per market regime.

Sits before BanditWeighter in the compute_vote pipeline. Gates eliminate
negative-Sharpe signals per regime (regime-mismatch), while the bandit
optimizes allocation among positive survivors. Gating and weighting are
complementary, not competing.

Gating rules derived from deep research (2026-05-23):
- MULTI_SPEED_MOM: OFF in HIGH_VOL and CRISIS (Sharpe -0.5 or worse)
- INTERNATIONAL_MOMENTUM: OFF in CRISIS (correlated with equity drawdowns)
- All other signals: ON in all regimes

Hysteresis: minimum 20-day dwell time per regime prevents chattering
on rapid regime transitions (AlphaCrafter, Yuan et al. 2026).
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
    GATE_RULES: Dict[str, Set[str]] = {
        "multi_speed_momentum": {"HIGH_VOL", "CRISIS"},
        "international_momentum": {"CRISIS"},
    }

    # Minimum days in a regime before allowing gate changes (hysteresis)
    DEFAULT_MIN_DWELL_DAYS = 20

    def __init__(
        self,
        gate_rules: Optional[Dict[str, Set[str]]] = None,
        min_dwell_days: int = DEFAULT_MIN_DWELL_DAYS,
    ):
        self.gate_rules = gate_rules or dict(self.GATE_RULES)
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
