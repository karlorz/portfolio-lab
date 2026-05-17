"""
Macro Regime Meta-Synthesis — v8.07 Implementation

Polls all 5 regime detectors in src/regime/ and produces a weighted consensus
regime with confidence score. Provides a single source of truth for "what
regime are we in?" to the EnsembleVoter.

Regime detectors polled:
1. macro_regime.py     — FRED/VIX based macro signals (no ML)
2. kurtosis_regime.py  — Distribution tail shape (no ML, pure numpy)
3. vol_volume_gap.py   — Volatility-Volume-Gap day classifier (no ML)
4. regime_hmm.py       — 5-state GaussianHMM (ML-gated, graceful fallback)
5. transformer_regime.py — Attention-based (ML-gated, graceful fallback)

Canonical regimes: bull, bear, neutral, high_vol, crisis

Usage:
    python -m src.signals.macro_regime_synthesis status
    python -m src.signals.macro_regime_synthesis poll
    python -m src.signals.macro_regime_synthesis explain
"""

import json
import logging
import os
import sys
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# --- Constants ---

CANONICAL_REGIMES = ["bull", "bear", "neutral", "high_vol", "crisis"]

# Detector weights (confidence in each detector's accuracy)
DETECTOR_WEIGHTS = {
    "macro_regime": 0.30,          # Broad macro context, most reliable
    "kurtosis_regime": 0.20,       # Distributional regime, good signal
    "vol_volume_gap": 0.20,        # Day-level classification
    "regime_hmm": 0.15,            # HMM (ML-gated, lower weight due to fallback risk)
    "transformer_regime": 0.15,    # Transformer (ML-gated, lower weight)
}

# Map detector-specific regime names to canonical regimes
REGIME_MAPPING = {
    # macro_regime outputs
    "risk_on_growth": "bull",
    "risk_on_late": "bull",
    "risk_on": "bull",
    "neutral": "neutral",
    "risk_off_rotation": "bear",
    "defensive": "bear",
    "crisis": "crisis",
    "risk_off": "bear",
    # kurtosis_regime outputs
    "low_kurtosis": "bull",
    "normal": "neutral",
    "high_kurtosis": "high_vol",
    "extreme_kurtosis": "crisis",
    # vol_volume_gap outputs
    "trend_up": "bull",
    "trend_down": "bear",
    "mean_revert": "neutral",
    "high_vol": "high_vol",
    "crisis": "crisis",
    # HMM regime outputs
    "bull": "bull",
    "bear": "bear",
    "neutral": "neutral",
    "high_vol": "high_vol",
    "crisis": "crisis",
    # transformer_regime outputs
    "trend_up": "bull",
    "trend_down": "bear",
    "mean_revert": "neutral",
    "high_vol": "high_vol",
    "crisis": "crisis",
}

# Canonical regime to ensemble signal value
REGIME_SIGNAL_VALUE = {
    "bull": 0.6,       # Strong risk-on
    "neutral": 0.0,    # No directional bias
    "bear": -0.4,      # Moderate risk-off
    "high_vol": -0.2,  # Slight risk-off (uncertainty)
    "crisis": -0.8,    # Strong risk-off
}

# Detector name to its detection function path
DETECTOR_NAMES = ["macro_regime", "kurtosis_regime", "vol_volume_gap",
                  "regime_hmm", "transformer_regime"]


@dataclass
class DetectorVote:
    """A single regime detector's output after mapping to canonical regime."""
    detector_name: str
    raw_regime: str
    canonical_regime: str
    confidence: float  # 0-1
    available: bool    # True if detector responded


@dataclass
class MetaRegimeConsensus:
    """Complete meta-regime synthesis result."""
    timestamp: str
    consensus_regime: str
    consensus_confidence: float
    regime_signal: float        # -1 to +1
    vote_details: List[DetectorVote]
    agreement_ratio: float      # % of detectors agreeing with consensus
    num_active_detectors: int
    num_total_detectors: int


# Minimum seconds between full re-polls. EnsembleVoter calls this on every
# collect_signals(), which can fire multiple times per minute during volatile
# markets. A 60s TTL avoids redundant detector re-invocation and repeated
# JSON state writes when nothing has changed.
POLL_TTL_SECONDS = 60


class MetaRegimeSynthesizer:
    """
    Polls all regime detectors and produces weighted consensus.

    Handles ML-gated detectors gracefully — if a detector raises ImportError
    (ML libs not available), it's skipped and remaining detectors are reweighted.

    Results are cached for POLL_TTL_SECONDS to avoid re-polling all 5 detectors
    on every EnsembleVoter signal collection cycle.
    """

    def __init__(self, data_dir: Optional[Path] = None):
        self.data_dir = data_dir or Path(__file__).parent.parent.parent / "data"
        self.state_path = self.data_dir / "macro_regime_state.json"
        self.state: Dict = {}
        self._cached_consensus: Optional[MetaRegimeConsensus] = None
        self._last_poll_time: float = 0.0
        self._load_state()

    # --- Public API ---

    def get_ensemble_signal(self) -> float:
        """Returns -1 to +1 signal value for the EnsembleVoter."""
        consensus = self.poll_regimes()
        return consensus.regime_signal

    def get_regime_for_ensemble_voter(self) -> Tuple[str, float]:
        """
        Returns (regime_name, confidence) for EnsembleVoter's detect_regime.
        Maps 5 canonical regimes to EnsembleVoter's 4-regime system.
        """
        consensus = self.poll_regimes()

        if consensus.consensus_regime == "crisis":
            ensemble_regime = "crisis"
        elif consensus.consensus_regime == "high_vol":
            ensemble_regime = "high_vol"
        else:
            ensemble_regime = "normal"

        return ensemble_regime, consensus.consensus_confidence

    def poll_regimes(self, force: bool = False) -> MetaRegimeConsensus:
        """Poll all detectors and compute consensus.

        Results are cached for POLL_TTL_SECONDS. Pass force=True to bypass.
        """
        now = time.time()
        if not force and self._cached_consensus is not None:
            if now - self._last_poll_time < POLL_TTL_SECONDS:
                return self._cached_consensus

        votes: List[DetectorVote] = []

        for detector_name in DETECTOR_NAMES:
            vote = self._poll_detector(detector_name)
            votes.append(vote)

        available = [v for v in votes if v.available]
        num_active = len(available)

        if num_active == 0:
            consensus = MetaRegimeConsensus(
                timestamp=datetime.now().isoformat(),
                consensus_regime="neutral",
                consensus_confidence=0.0,
                regime_signal=0.0,
                vote_details=votes,
                agreement_ratio=0.0,
                num_active_detectors=0,
                num_total_detectors=len(DETECTOR_NAMES),
            )
            self._cached_consensus = consensus
            self._last_poll_time = now
            return consensus

        regime_scores: Dict[str, float] = {r: 0.0 for r in CANONICAL_REGIMES}
        total_weight = 0.0

        for v in available:
            base_weight = DETECTOR_WEIGHTS.get(v.detector_name, 0.15)
            adjusted_weight = base_weight * v.confidence
            regime_scores[v.canonical_regime] = (
                regime_scores.get(v.canonical_regime, 0.0) + adjusted_weight
            )
            total_weight += adjusted_weight

        if total_weight > 0:
            for r in regime_scores:
                regime_scores[r] /= total_weight

        winner = max(sorted(regime_scores.keys()), key=regime_scores.get)
        winner_score = regime_scores[winner]

        participation_ratio = num_active / len(DETECTOR_NAMES)
        confidence_multiplier = min(1.0, participation_ratio * 2.0)
        winner_score = min(winner_score, confidence_multiplier)

        agreeing = sum(1 for v in available if v.canonical_regime == winner)
        agreement_ratio = agreeing / num_active

        consensus = MetaRegimeConsensus(
            timestamp=datetime.now().isoformat(),
            consensus_regime=winner,
            consensus_confidence=winner_score,
            regime_signal=REGIME_SIGNAL_VALUE.get(winner, 0.0),
            vote_details=votes,
            agreement_ratio=agreement_ratio,
            num_active_detectors=num_active,
            num_total_detectors=len(DETECTOR_NAMES),
        )

        # Only persist if the consensus regime changed or TTL expired
        prev_regime = self.state.get("consensus_regime")
        if prev_regime != winner:
            self._save_state(consensus)

        self._cached_consensus = consensus
        self._last_poll_time = now
        return consensus

    def explain(self) -> str:
        """Generate human-readable explanation of current regime synthesis."""
        consensus = self.poll_regimes()
        lines = [
            f"Meta-Regime Synthesis — {consensus.timestamp}",
            f"  Consensus: {consensus.consensus_regime.upper()} "
            f"(confidence: {consensus.consensus_confidence:.1%})",
            f"  Signal: {consensus.regime_signal:+.3f}",
            f"  Agreement: {consensus.agreement_ratio:.0%} "
            f"({consensus.num_active_detectors}/{consensus.num_total_detectors} active)",
            "",
            "  Per-Detector Breakdown:",
        ]

        for v in consensus.vote_details:
            status = "●" if v.available else "○"
            lines.append(
                f"    {status} {v.detector_name:20s} → "
                f"{v.canonical_regime:12s} "
                f"(raw: {v.raw_regime}, conf: {v.confidence:.2f})"
            )

        lines.extend([
            "",
            "  Regime Scores:",
        ])

        # Recompute scores for display
        available = [v for v in consensus.vote_details if v.available]
        regime_scores = {r: 0.0 for r in CANONICAL_REGIMES}
        total_weight = 0.0
        for v in available:
            base_weight = DETECTOR_WEIGHTS.get(v.detector_name, 0.15)
            adjusted = base_weight * v.confidence
            regime_scores[v.canonical_regime] += adjusted
            total_weight += adjusted
        if total_weight > 0:
            for r in regime_scores:
                regime_scores[r] /= total_weight

        for regime, score in sorted(regime_scores.items(), key=lambda x: -x[1]):
            bar_len = int(score * 40)
            bar = "█" * bar_len + "░" * (40 - bar_len)
            lines.append(f"    {regime:12s} {bar} {score:.1%}")

        return "\n".join(lines)

    # --- Detector Polling ---

    def _poll_detector(self, detector_name: str) -> DetectorVote:
        """Poll a single regime detector and map its output to canonical regime."""
        try:
            if detector_name == "macro_regime":
                return self._poll_macro_regime()
            elif detector_name == "kurtosis_regime":
                return self._poll_kurtosis_regime()
            elif detector_name == "vol_volume_gap":
                return self._poll_vol_volume_gap()
            elif detector_name == "regime_hmm":
                return self._poll_regime_hmm()
            elif detector_name == "transformer_regime":
                return self._poll_transformer_regime()
            else:
                return DetectorVote(
                    detector_name=detector_name,
                    raw_regime="unknown",
                    canonical_regime="neutral",
                    confidence=0.0,
                    available=False,
                )
        except (ImportError, ModuleNotFoundError) as e:
            # ML-gated detector not available — graceful skip
            logger.debug(f"Detector {detector_name} unavailable: {e}")
            return DetectorVote(
                detector_name=detector_name,
                raw_regime="unavailable",
                canonical_regime="neutral",
                confidence=0.0,
                available=False,
            )
        except Exception as e:
            logger.warning(f"Detector {detector_name} error: {e}")
            return DetectorVote(
                detector_name=detector_name,
                raw_regime="error",
                canonical_regime="neutral",
                confidence=0.0,
                available=False,
            )

    def _poll_macro_regime(self) -> DetectorVote:
        """Poll macro_regime classifier.

        MacroRegimeSynthesizer requires constructing 9 signal inputs,
        which needs live data fetchers. We mark it as requiring data polling
        and return a neutral vote with low confidence for now.
        """
        try:
            from src.regime.macro_regime import MacroRegimeSynthesizer

            # Macro regime needs signals dict — requires data fetchers
            # Mark as unavailable until data polling is wired up
            return DetectorVote(
                detector_name="macro_regime",
                raw_regime="requires_data_polling",
                canonical_regime="neutral",
                confidence=0.0,
                available=False,
            )
        except ImportError:
            raise

    def _poll_kurtosis_regime(self) -> DetectorVote:
        """Poll kurtosis_regime detector."""
        from src.regime.kurtosis_regime import detect_kurtosis_regime
        signal = detect_kurtosis_regime()

        # signal.regime may be an enum or a string
        regime_obj = signal.regime if hasattr(signal, 'regime') else getattr(signal, 'regime', 'normal')
        regime_str = regime_obj.value if hasattr(regime_obj, 'value') else str(regime_obj).lower()
        canonical = REGIME_MAPPING.get(regime_str, "neutral")
        confidence = signal.confidence if hasattr(signal, 'confidence') else 0.5
        # Normalize: kurtosis confidence is 0-100, convert to 0-1
        confidence = float(confidence) / 100.0 if confidence > 1.0 else float(confidence)

        return DetectorVote(
            detector_name="kurtosis_regime",
            raw_regime=regime_str,
            canonical_regime=canonical,
            confidence=float(confidence),
            available=True,
        )

    def _poll_vol_volume_gap(self) -> DetectorVote:
        """Poll vol_volume_gap day classifier."""
        from src.regime.vol_volume_gap import detect_regime
        result = detect_regime()

        regime_str = result.get("regime", "mean_revert") if isinstance(result, dict) else str(result).lower()
        canonical = REGIME_MAPPING.get(regime_str, "neutral")
        confidence = result.get("confidence", 0.5) if isinstance(result, dict) else 0.5

        return DetectorVote(
            detector_name="vol_volume_gap",
            raw_regime=regime_str,
            canonical_regime=canonical,
            confidence=float(confidence),
            available=True,
        )

    def _poll_regime_hmm(self) -> DetectorVote:
        """Poll HMM regime detector (ML-gated)."""
        # Check ML gate first
        if os.environ.get("PORTFOLIO_LAB_ENABLE_ML") != "1":
            raise ImportError("HMM regime requires PORTFOLIO_LAB_ENABLE_ML=1")

        from src.strategy.regime_hmm import WassersteinHMMDetector
        detector = WassersteinHMMDetector()
        result = detector.detect_current_regime()

        regime_str = result.get("regime", "neutral") if isinstance(result, dict) else str(result).lower()
        canonical = REGIME_MAPPING.get(regime_str, "neutral")
        confidence = result.get("confidence", 0.5) if isinstance(result, dict) else 0.5

        return DetectorVote(
            detector_name="regime_hmm",
            raw_regime=regime_str,
            canonical_regime=canonical,
            confidence=float(confidence),
            available=True,
        )

    def _poll_transformer_regime(self) -> DetectorVote:
        """Poll transformer regime detector (ML-gated)."""
        # Check ML gate first
        if os.environ.get("PORTFOLIO_LAB_ENABLE_ML") != "1":
            raise ImportError("Transformer regime requires PORTFOLIO_LAB_ENABLE_ML=1")

        from src.regime.transformer_regime import detect_transformer_regime
        result = detect_transformer_regime()

        regime_str = result.get("regime", "mean_revert") if isinstance(result, dict) else str(result).lower()
        canonical = REGIME_MAPPING.get(regime_str, "neutral")
        confidence = result.get("confidence", 0.5) if isinstance(result, dict) else 0.5

        return DetectorVote(
            detector_name="transformer_regime",
            raw_regime=regime_str,
            canonical_regime=canonical,
            confidence=float(confidence),
            available=True,
        )

    # --- State Persistence ---

    def _load_state(self):
        """Load persisted state."""
        if self.state_path.exists():
            try:
                with open(self.state_path) as f:
                    self.state = json.load(f)
            except (json.JSONDecodeError, OSError):
                self.state = {}
        else:
            self.state = {}

    def _save_state(self, consensus: MetaRegimeConsensus) -> None:
        """Persist current consensus to disk using dataclass serialization."""
        self.state = asdict(consensus)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        with open(self.state_path, "w") as f:
            json.dump(self.state, f, indent=2)

    def get_status(self) -> Dict:
        """Return current state dict."""
        if not self.state:
            return {"status": "idle", "message": "No state available. Run poll first."}
        return {
            "status": "active",
            "timestamp": self.state.get("timestamp"),
            "consensus_regime": self.state.get("consensus_regime"),
            "consensus_confidence": self.state.get("consensus_confidence"),
            "regime_signal": self.state.get("regime_signal"),
            "agreement_ratio": self.state.get("agreement_ratio"),
        }


# --- CLI ---

def main():
    """CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Macro Regime Meta-Synthesis")
    parser.add_argument(
        "command",
        nargs="?",
        default="status",
        choices=["status", "poll", "explain"],
        help="Command to execute",
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="Save output to JSON",
    )

    args = parser.parse_args()
    synthesizer = MetaRegimeSynthesizer()

    if args.command == "status":
        status = synthesizer.get_status()
        if status["status"] == "idle":
            print(f"Status: {status['message']}")
        else:
            print(f"  Last Poll:  {status['timestamp']}")
            print(f"  Regime:     {status['consensus_regime']}")
            print(f"  Confidence: {status['consensus_confidence']:.1%}")
            print(f"  Signal:     {status['regime_signal']:+.4f}")
            print(f"  Agreement:  {status['agreement_ratio']:.0%}")
    elif args.command == "poll":
        consensus = synthesizer.poll_regimes()
        print(f"  Timestamp: {consensus.timestamp}")
        print(f"  Consensus: {consensus.consensus_regime}")
        print(f"  Confidence: {consensus.consensus_confidence:.1%}")
        print(f"  Signal:     {consensus.regime_signal:+.4f}")
        print(f"  Agreement:  {consensus.agreement_ratio:.0%}")
        print(f"  Active:     {consensus.num_active_detectors}/{consensus.num_total_detectors}")
    elif args.command == "explain":
        explanation = synthesizer.explain()
        print(explanation)

    if args.save:
        output_path = Path(f"data/macro_regime_synthesis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        with open(output_path, "w") as f:
            json.dump(synthesizer.state, f, indent=2)
        print(f"\nSaved to {output_path}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
    main()
