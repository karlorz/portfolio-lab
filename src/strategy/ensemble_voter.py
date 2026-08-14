"""
Portfolio-Lab v2.58: Ensemble Signal Voter

Multi-source signal aggregation with regime-dependent weighting and health-adjusted weighting.
Implements soft voting with confidence-based consensus for portfolio decisions.

Active Sources (6):
- Multi-Speed Momentum (v2.56) - Speed-diversified trends
- Cross-Asset Relative Value (v5.71) - Mean-reversion triggers
- International Equity Momentum (v3.13) - EFA/EEM vs SPY
- Alternative Data (v9.00) - SEC EDGAR, NewsAPI, jobs
- Cross-Asset Regime Arbitrage (v8.09) - Divergence detection
- Unified Overlay (v4.90) - Collar + bond + crypto + calendar

Weight Adjustments (applied in order):
1. Static REGIME_WEIGHTS (per-regime allocation)
2. Adaptive ensemble weighting (v6.09, from attribution data)
3. Health-adjusted weighting (v3.12, from signal health scores)
4. Correlation penalty (v2.59, from IC prediction correlations)
5. Regime-conditional weights (v2.60, per-regime signal multipliers)
6. Utility-based reweighting (v2.58, Sharpe contribution + hit rate)
7. Exploration noise (v2.57, Dirichlet sampling)
8. Turnover-aware validation (v8.01, with basis-pursuit + regret-weighted)

# Online IC-based weight learning (new, gated by ENSEMBLE_USE_IC_WEIGHTS)

Consensus threshold: 2/3 weighted signals agree for action

Usage:
    python -m src.strategy.ensemble_voter vote
    python -m src.strategy.ensemble_voter recommend --portfolio 46/38/16
    python -m src.strategy.ensemble_voter explain
"""

import json

import os






from typing import Dict, Optional

from dataclasses import dataclass

from pathlib import Path

import logging

from src.paths import (
    DATA_DIR, ENSEMBLE_CRISIS_VOL_THRESHOLD, ENSEMBLE_CRISIS_DRAWDOWN_THRESHOLD,
    ENSEMBLE_HIGH_VOL_VOL_THRESHOLD, ENSEMBLE_HIGH_VOL_DRAWDOWN_THRESHOLD,
    ENSEMBLE_LOW_VOL_VOL_THRESHOLD, ENSEMBLE_LOW_VOL_MOM_THRESHOLD,
    ENSEMBLE_RECOVERY_DRAWDOWN_THRESHOLD, ENSEMBLE_RECOVERY_MOM_THRESHOLD,
)




from src.strategy.signal_aggregator import SignalAggregator

__all__ = ['Regime', 'SignalSource', 'SignalReading', 'EnsembleVote', 'REGIME_WEIGHTS', 'REGIME_CONDITIONAL_WEIGHTS', 'REGIME_CONSENSUS_THRESHOLDS', 'DEFAULT_DIVERSITY_FLOOR', 'BanditWeighter', 'SignalAggregator', 'EnsembleVoter', 'compute_signal_correlation_matrix']
from src.strategy.ensemble_support import (  # noqa: E402, F401  # re-export hub
    _get_health_tracker,
    EnsembleVote,
    _load_regime_conditional_weights,
    _extract_signal_predictions,
    _rank_prediction_matrix,
    _rank_correlation_from_matrix,
    compute_signal_correlation_matrix,
    BanditWeighter,
)

logger = logging.getLogger(__name__)

BANDIT_MAX_BLEND: float = float(os.environ.get("ENSEMBLE_BANDIT_MAX_BLEND", "0.7"))

BANDIT_WARMUP_DAYS: int = int(os.environ.get("ENSEMBLE_BANDIT_WARMUP_DAYS", "252"))

BANDIT_REWARD_NOISE_FLOOR: float = float(
    os.environ.get("ENSEMBLE_BANDIT_REWARD_NOISE_FLOOR", "1e-6")
)

DEFAULT_DIVERSITY_FLOOR: float = float(os.environ.get("ENSEMBLE_DIVERSITY_FLOOR", "0.05"))

REGIME_CONSENSUS_THRESHOLDS: dict = {
    "CRISIS": 0.50,
    "HIGH_VOL": 0.55,
    "LOW_VOL": 0.67,
    "NORMAL": 0.75,
    "RECOVERY": 0.60,
}

from src.signals.signal_source import SignalSource  # noqa: E402  # canonical, consolidated May 2026 (lazy init above)

from src.signals.regime_spec import (  # noqa: E402, F401
    Regime,
    SignalReading,
    REGIME_WEIGHTS,
    _load_regime_weights,
    _build_hardcoded_weights,
)

REGIME_CONDITIONAL_WEIGHTS = _load_regime_conditional_weights()

from src.strategy.ensemble_voter_collect import CollectMixin  # noqa: E402  # mixin
from src.strategy.ensemble_voter_bandit import BanditMixin  # noqa: E402  # mixin
from src.strategy.ensemble_voter_vote import VoteMixin  # noqa: E402  # mixin
from src.strategy.ensemble_voter_weights import WeightsMixin  # noqa: E402  # mixin

class EnsembleVoterBase:
    """
    Multi-source signal ensemble with regime-adaptive weighting.

    Collects signals from all strategy modules, applies regime-dependent
    weighting, and produces consensus recommendations.
    """
    CRISIS_VOL_THRESHOLD = ENSEMBLE_CRISIS_VOL_THRESHOLD
    CRISIS_DRAWDOWN_THRESHOLD = ENSEMBLE_CRISIS_DRAWDOWN_THRESHOLD
    HIGH_VOL_VOL_THRESHOLD = ENSEMBLE_HIGH_VOL_VOL_THRESHOLD
    HIGH_VOL_DRAWDOWN_THRESHOLD = ENSEMBLE_HIGH_VOL_DRAWDOWN_THRESHOLD
    HIGH_VOL_MOM_THRESHOLD = 0.0       # Negative momentum with drawdown → HIGH_VOL
    LOW_VOL_VOL_THRESHOLD = ENSEMBLE_LOW_VOL_VOL_THRESHOLD
    LOW_VOL_MOM_THRESHOLD = ENSEMBLE_LOW_VOL_MOM_THRESHOLD
    RECOVERY_DRAWDOWN_THRESHOLD = ENSEMBLE_RECOVERY_DRAWDOWN_THRESHOLD
    RECOVERY_MOM_THRESHOLD = ENSEMBLE_RECOVERY_MOM_THRESHOLD
    @dataclass
    class _ConsensusResult:
        """Internal intermediate result from consensus computation."""
        weighted_consensus: float
        agreement: float
        equity_bias: float
        duration_bias: float
        gold_bias: float
        action: str
        action_confidence: float
    def __init__(
        self,
        data_path: Optional[Path] = None,
        regime_detector: Optional[str] = None
    ):
        self.data_path = data_path or DATA_DIR
        self.db_path = self.data_path / "ensemble_signals.db"
        self._init_db()

        # Current readings cache
        self.current_readings: Dict[SignalSource, SignalReading] = {}
        self.current_regime: Regime = Regime.NORMAL
        self.current_regime_confidence: float = 0.5

        # Bandit weighter for dynamic signal weight adaptation
        self.bandit = BanditWeighter(
            signals=[s.value for s in SignalSource],
            epsilon=0.1,
            window=252,
        )
        self.bandit_observations: int = 0
        # Calendar reward steps for warmup blend (not arm×day updates)
        self.bandit_days: int = 0
        self.bandit_state_path = self.data_path / "ensemble_bandit_state.json"
        self._load_bandit_state()

        # Online IC weighter for IC-based ensemble weight learning
        # Gated by ENSEMBLE_USE_IC_WEIGHTS env var (default: off)
        self._use_ic_weights = os.environ.get("ENSEMBLE_USE_IC_WEIGHTS", "0").lower() in ("1", "true")
        self._ic_weighter = None
        if self._use_ic_weights:
            try:
                from src.strategy.online_ic_weighter import OnlineICWeighter
                self._ic_weighter = OnlineICWeighter()
                # Load persisted IC weighter state if available
                ic_weighter_state = self.data_path / "ic_weighter_state.json"
                if ic_weighter_state.exists():
                    with open(ic_weighter_state) as f:
                        self._ic_weighter.load_state(json.load(f))
                    logger.info("OnlineICWeighter state loaded from %s", ic_weighter_state)
            except Exception as e:
                logger.warning("Failed to initialize OnlineICWeighter: %s", e)
                self._ic_weighter = None

        # Regime gate — disables signals in regimes where they are net-negative
        from src.signals.regime_gate import RegimeGate
        self.regime_gate = RegimeGate()

        # Load data-driven gate rules if available (computed by DashboardGenerator)
        try:
            from src.monitor.regime_sharpe_matrix import load_persisted_gate_rules
            persist_path = self.data_path / "regime_gate_persisted.json"
            data_rules = load_persisted_gate_rules(persist_path)
            if data_rules:
                self.regime_gate.gate_rules.update(data_rules)
                logger.info(
                    "Loaded %d data-driven gate rules from persisted file",
                    len(data_rules),
                )
        except (ImportError, Exception) as e:
            logger.debug("Data-driven gate loading skipped: %s", e)

        self._prev_regime: Optional[str] = None
        self._days_in_regime: int = 999  # Start assuming stable regime

        # Signal collection collaborator (extractable / injectable for tests)
        self.signal_aggregator = SignalAggregator(
            # Lambda so instance patches of _load_price_data still apply.
            load_price_data=lambda: self._load_price_data(),
            regime_weights=REGIME_WEIGHTS,
        )

class EnsembleVoter(CollectMixin, BanditMixin, VoteMixin, WeightsMixin, EnsembleVoterBase):
    """Multi-source signal ensemble with regime-adaptive weighting (mixin chain)."""
    pass

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Ensemble Signal Voter')
    subparsers = parser.add_subparsers(dest='command')
    
    # Vote command
    vote_parser = subparsers.add_parser('vote', help='Compute ensemble vote')
    vote_parser.add_argument('--date', help='Date for signal (default: latest)')
    
    # Recommend command
    rec_parser = subparsers.add_parser('recommend', help='Generate allocation recommendation')
    rec_parser.add_argument('--portfolio', default='46/38/16', help='Base allocation SPY/GLD/TLT')
    rec_parser.add_argument('--max-shift', type=float, default=0.10, help='Max allocation shift')
    
    # Explain command
    subparsers.add_parser('explain', help='Explain current vote reasoning')

    args = parser.parse_args()
    
    voter = EnsembleVoter()
    
    if args.command == 'vote':
        readings = voter.collect_signals(args.date)
        vote = voter.compute_vote(readings)

        logger.info("Ensemble Vote")
        logger.info("Timestamp: %s", vote.timestamp)
        logger.info("Regime: %s (confidence: %.1f%%)", vote.regime.value.upper(), vote.regime_confidence * 100)
        logger.info("Sources: %d", vote.num_sources)
        logger.info("Consensus: %+.3f", vote.weighted_consensus)
        logger.info("Agreement: %.1f%%", vote.agreement_ratio * 100)
        logger.info("Asset Biases:")
        logger.info("  Equity (SPY):   %+.3f", vote.equity_bias)
        logger.info("  Duration (TLT): %+.3f", vote.duration_bias)
        logger.info("  Gold (GLD):     %+.3f", vote.gold_bias)
        logger.info("Recommended Action: %s", vote.action.upper())
        logger.info("Confidence: %.1f%%", vote.confidence * 100)

    elif args.command == 'recommend':
        weights = [float(w) / 100 for w in args.portfolio.split('/')]
        base = {'SPY': weights[0], 'GLD': weights[1], 'TLT': weights[2]}

        vote = voter.compute_vote()
        rec = voter.recommend_allocation(base, vote, args.max_shift)

        logger.info("Allocation Recommendation")
        logger.info("Base: %s", args.portfolio)
        logger.info("Regime: %s (confidence: %.1f%%)", rec['regime'].upper(), rec['confidence'] * 100)
        logger.info("Consensus: %+.3f", rec['consensus'])
        logger.info("Recommended Allocation:")
        for asset, data in rec['assets'].items():
            logger.info("  %s: %.1f%% -> %.1f%% (shift: %+.1f%%)",
                        asset, data['base'] * 100, data['new'] * 100, data['normalized_shift'] * 100)

    elif args.command == 'explain':
        vote = voter.compute_vote()

        logger.info("Ensemble Vote Explanation")
        logger.info(vote.reasoning)
        logger.info("Active Sources (%d):", len(vote.source_votes))
        for src in vote.source_votes:
            logger.info("  %25s | value: %+.3f | weight: %.2f | conf: %.1f%%",
                        src.source.value, src.value, src.weight, src.confidence * 100)
    
    else:
        parser.print_help()

if __name__ == '__main__':
    from src.utils.log_config import configure_logging

    configure_logging()
    main()
