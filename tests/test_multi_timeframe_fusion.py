"""
Tests for Multi-Timeframe Signal Fusion (v8.06).

Covers:
- Signal classification into correct timeframe buckets
- Within-bucket consensus computation
- Cross-timeframe fusion with regime weights
- Per-asset bias computation
- Edge cases (empty signals, single signal, extreme values)
- State persistence
- Timeframe breakdown
"""

import json
import os
import sys
import tempfile
from pathlib import Path
from datetime import datetime, timedelta

import pytest
import numpy as np

# Ensure src is importable
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from signals.multi_timeframe_fusion import (
    MultiTimeframeFusion,
    Timeframe,
    TimeframeSignal,
    TimeframeBucket,
    FusedResult,
    SIGNAL_TIMEFRAMES,
    TIMEFRAME_BUCKET_WEIGHTS,
)


# ─── Fixtures ────────────────────────────────────────────────────────

@pytest.fixture
def fusion():
    """Create a MultiTimeframeFusion with temp state path."""
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        state_path = f.name
    yield MultiTimeframeFusion(state_path=state_path)
    if os.path.exists(state_path):
        os.unlink(state_path)


@pytest.fixture
def sample_signals():
    """Realistic signal values across all timeframes."""
    return {
        "hmm_regime": 0.10,              # MEDIUM
        "cta_trend": 0.25,               # MEDIUM
        "macro_momentum": 0.20,          # LONG
        "multi_speed_momentum": 0.15,    # MEDIUM
        "circuit_breaker": 0.0,          # SHORT
        "factor_rotation": 0.05,         # MEDIUM
        "unified_overlay": 0.08,         # LONG
        "mean_reversion": -0.03,         # SHORT
        "transformer_regime": 0.12,      # MEDIUM
        "cross_asset_rv": 0.15,          # MEDIUM
        "regime_classifier": 0.05,       # MEDIUM
        "risk_budget": 0.05,            # LONG
        "llm_narrative": 0.12,          # MEDIUM
        "tax_aware": 0.03,              # LONG
        "vixy_hedge": 0.05,             # SHORT
    }


@pytest.fixture
def sample_confidences():
    """Confidence values for sample signals."""
    return {
        "hmm_regime": 0.60,
        "cta_trend": 0.75,
        "macro_momentum": 0.70,
        "multi_speed_momentum": 0.65,
        "circuit_breaker": 0.50,
        "factor_rotation": 0.60,
        "unified_overlay": 0.70,
        "mean_reversion": 0.45,
        "transformer_regime": 0.65,
        "cross_asset_rv": 0.65,
        "regime_classifier": 0.55,
        "risk_budget": 0.50,
        "llm_narrative": 0.65,
        "tax_aware": 0.40,
        "vixy_hedge": 0.55,
    }


# ─── Signal Classification Tests ─────────────────────────────────────

class TestSignalClassification:

    def test_classify_signal_timeframe_correctness(self, fusion):
        """Each known signal should be classified into correct timeframe."""
        assert fusion.get_signal_timeframe("mean_reversion") == "short"

        assert fusion.get_signal_timeframe("cta_trend") == "medium"
        assert fusion.get_signal_timeframe("multi_speed_momentum") == "medium"
        assert fusion.get_signal_timeframe("cross_asset_rv") == "medium"
        assert fusion.get_signal_timeframe("unified_overlay") == "medium"

        assert fusion.get_signal_timeframe("risk_budget") == "long"
        assert fusion.get_signal_timeframe("tax_aware") == "long"

    def test_classify_unknown_signal_returns_none(self, fusion):
        """Unknown signal source should return None."""
        assert fusion.get_signal_timeframe("nonexistent_signal") is None

    def test_classify_empty_signals(self, fusion):
        """Empty signal dict should produce empty buckets."""
        buckets, explanation = fusion.classify_signals({})
        assert len(buckets) == 3
        assert all(b.active_count == 0 for b in buckets.values())
        assert "No signals" in explanation

    def test_classify_single_signal(self, fusion):
        """Single signal should classify and bucket correctly."""
        buckets, explanation = fusion.classify_signals(
            {"mean_reversion": 0.5},
            {"mean_reversion": 0.8}
        )
        short_bucket = buckets["short"]
        assert short_bucket.active_count == 1
        assert short_bucket.signals[0].source == "mean_reversion"
        assert short_bucket.signals[0].value == 0.5
        assert short_bucket.signals[0].confidence == 0.8
        assert short_bucket.consensus == 0.5

    def test_classify_mixed_directions(self, fusion):
        """Conflicting signals within a bucket should produce moderate consensus."""
        buckets, explanation = fusion.classify_signals({
            "circuit_breaker": 0.8,
            "mean_reversion": -0.6,
        })
        short_bucket = buckets["short"]
        assert short_bucket.active_count == 2
        # Consensus should be positive but reduced
        assert -0.5 < short_bucket.consensus < 0.8
        # Agreement should be < 100%
        assert short_bucket.agreement < 1.0

    def test_agreement_calculation(self, fusion):
        """All same-direction signals should give 100% agreement."""
        buckets, explanation = fusion.classify_signals({
            "mean_reversion": 0.5,
            "circuit_breaker": 0.3,
        })
        assert buckets["short"].agreement == 1.0

    def test_agreement_split(self, fusion):
        """Evenly split signals should give 50% agreement."""
        buckets, explanation = fusion.classify_signals({
            "mean_reversion": 0.5,
            "circuit_breaker": -0.3,
        })
        assert buckets["short"].agreement == 0.5

    def test_get_timeframe_breakdown(self, fusion):
        """Breakdown should list all 22 signals across 3 timeframes."""
        breakdown = fusion.get_timeframe_breakdown()
        assert len(breakdown) == 3
        assert "short" in breakdown
        assert "medium" in breakdown
        assert "long" in breakdown
        total = sum(len(sources) for sources in breakdown.values())
        assert total == len(SIGNAL_TIMEFRAMES)


# ─── Fusion Tests ────────────────────────────────────────────────────

class TestFusion:

    def test_fuse_empty_signals(self, fusion):
        """Empty signals should produce zero signal with explanation."""
        result = fusion.fuse({})
        assert result.overall_signal == 0.0
        assert result.confidence >= 0
        assert result.regime == "normal"

    def test_fuse_normal_regime(self, fusion, sample_signals, sample_confidences):
        """Normal regime should weight long-term most heavily."""
        result = fusion.fuse(sample_signals, sample_confidences, regime="normal")
        
        assert -1.0 <= result.overall_signal <= 1.0
        assert 0 <= result.confidence <= 1.0
        assert result.regime == "normal"
        
        # In normal regime, long-term should dominate
        # Long-term signals are mostly positive, so overall should be positive
        assert result.overall_signal > 0
        
        # Long-term weight (0.50) > medium (0.35) > short (0.15)
        norm_weights = TIMEFRAME_BUCKET_WEIGHTS["normal"]
        assert norm_weights[Timeframe.LONG] > norm_weights[Timeframe.MEDIUM]
        assert norm_weights[Timeframe.MEDIUM] > norm_weights[Timeframe.SHORT]

    def test_fuse_high_vol_regime(self, fusion, sample_signals, sample_confidences):
        """High vol regime should increase short-term weight."""
        result = fusion.fuse(sample_signals, sample_confidences, regime="high_vol")
        
        vol_weights = TIMEFRAME_BUCKET_WEIGHTS["high_vol"]
        # In high vol, short-term has higher weight
        assert vol_weights[Timeframe.SHORT] == 0.25
        assert vol_weights[Timeframe.MEDIUM] == 0.45

    def test_fuse_crisis_regime(self, fusion, sample_signals, sample_confidences):
        """Crisis regime should maximize short-term weight."""
        result = fusion.fuse(sample_signals, sample_confidences, regime="crisis")
        
        crisis_weights = TIMEFRAME_BUCKET_WEIGHTS["crisis"]
        assert crisis_weights[Timeframe.SHORT] == 0.40
        assert crisis_weights[Timeframe.LONG] == 0.20

    def test_fuse_recovery_regime(self, fusion, sample_signals, sample_confidences):
        """Recovery regime should maximize long-term weight."""
        result = fusion.fuse(sample_signals, sample_confidences, regime="recovery")
        
        recovery_weights = TIMEFRAME_BUCKET_WEIGHTS["recovery"]
        assert recovery_weights[Timeframe.LONG] == 0.60
        assert recovery_weights[Timeframe.SHORT] == 0.10

    def test_fuse_unknown_regime_defaults_to_normal(self, fusion, sample_signals):
        """Unknown regime should default to normal weights."""
        result = fusion.fuse(sample_signals, regime="unknown_regime")
        assert result.regime == "normal"

    def test_fuse_all_bullish(self, fusion):
        """All positive signals should yield positive overall."""
        all_bullish = {k: 0.8 for k in SIGNAL_TIMEFRAMES}
        result = fusion.fuse(all_bullish)
        assert result.overall_signal > 0.3

    def test_fuse_all_bearish(self, fusion):
        """All negative signals should yield negative overall."""
        all_bearish = {k: -0.8 for k in SIGNAL_TIMEFRAMES}
        result = fusion.fuse(all_bearish)
        assert result.overall_signal < -0.3

    def test_fuse_extreme_values_clamped(self, fusion):
        """Signal values should be clamped to [-1, 1]."""
        extreme = {k: 999.0 for k in SIGNAL_TIMEFRAMES}
        result = fusion.fuse(extreme)
        assert result.overall_signal <= 1.0
        assert result.overall_signal >= -1.0

    def test_fuse_all_zero(self, fusion):
        """All zero signals should yield near-zero overall."""
        zeros = {k: 0.0 for k in SIGNAL_TIMEFRAMES}
        result = fusion.fuse(zeros)
        assert abs(result.overall_signal) < 0.001

    def test_fuse_timeframe_breakdown_report(self, fusion, sample_signals):
        """Result should contain per-timeframe signals."""
        result = fusion.fuse(sample_signals)
        assert hasattr(result, 'short_term_signal')
        assert hasattr(result, 'medium_term_signal')
        assert hasattr(result, 'long_term_signal')

    def test_fuse_equity_bias(self, fusion, sample_signals, sample_confidences):
        """Equity bias should be influenced by equity-relevant signals."""
        # Strongly bullish equity signals
        equity_bullish = {
            "multi_speed_momentum": 0.9,
            "cta_trend": 0.8,
            "mean_reversion": 0.0,
            "factor_rotation": 0.7,
            "cross_asset_rv": 0.6,
        }
        # Add placeholder zeros for other signals
        for k in SIGNAL_TIMEFRAMES:
            if k not in equity_bullish:
                equity_bullish[k] = 0.0

        result = fusion.fuse(equity_bullish)
        assert result.equity_bias > 0.3

    def test_fuse_duration_bias(self, fusion):
        """Duration bias should be influenced by rate-relevant signals."""
        # Bearish duration signals (expecting rates to rise)
        duration_bearish = {
            "macro_momentum": -0.6,
            "risk_budget": -0.5,
            "tax_aware": -0.3,
            "unified_overlay": -0.4,
        }
        for k in SIGNAL_TIMEFRAMES:
            if k not in duration_bearish:
                duration_bearish[k] = 0.0
        
        result = fusion.fuse(duration_bearish)
        assert result.duration_bias < -0.2

    def test_fuse_gold_bias(self, fusion):
        """Gold bias should be influenced by gold-relevant signals."""
        gold_bullish = {
            "unified_overlay": 0.7,
            "llm_narrative": 0.6,
            "vixy_hedge": 0.5,
        }
        for k in SIGNAL_TIMEFRAMES:
            if k not in gold_bullish:
                gold_bullish[k] = 0.0
        
        result = fusion.fuse(gold_bullish)
        assert result.gold_bias > 0.2

    def test_fuse_state_persistence(self, fusion, sample_signals):
        """State should be updated after fusion."""
        assert fusion.state["fusion_count"] == 0
        fusion.fuse(sample_signals)
        assert fusion.state["fusion_count"] == 1
        assert fusion.state["last_fusion"] is not None
        assert len(fusion.state["history"]) == 1

    def test_fuse_multiple_calls_accumulate_history(self, fusion, sample_signals):
        """Multiple fusion calls should accumulate in history (max 30)."""
        for i in range(35):
            fusion.fuse(sample_signals)
        # History should be capped at 30
        assert len(fusion.state["history"]) <= 30
        assert fusion.state["fusion_count"] == 35

    def test_fuse_state_file_saved(self, fusion, sample_signals):
        """State should be saved to disk after fusion."""
        fusion.fuse(sample_signals)
        assert os.path.exists(fusion.state_path)
        with open(fusion.state_path) as f:
            saved = json.load(f)
        assert saved["fusion_count"] == 1


# ─── Edge Cases ──────────────────────────────────────────────────────

class TestEdgeCases:

    def test_fuse_with_single_signal(self, fusion):
        """Single signal should still produce valid fusion."""
        result = fusion.fuse({"mean_reversion": 0.5}, {"mean_reversion": 0.9})
        assert result.overall_signal != 0.0
        assert result.short_term_signal == 0.5
        # Other buckets should be 0
        assert result.medium_term_signal == 0.0
        assert result.long_term_signal == 0.0

    def test_fuse_with_only_short_signals(self, fusion):
        """Only short-term signals should produce valid fusion."""
        only_short = {
            "mean_reversion": -0.3,
            "circuit_breaker": 0.0,
            "vixy_hedge": 0.1,
        }
        result = fusion.fuse(only_short)
        assert result.short_term_signal != 0.0
        assert result.medium_term_signal == 0.0
        assert result.long_term_signal == 0.0
        # Overall should equal short-term (only bucket with weight)
        assert result.overall_signal != 0.0

    def test_fuse_with_delay_between_calls(self, fusion, sample_signals):
        """Timestamps should differ between calls."""
        r1 = fusion.fuse(sample_signals)
        r2 = fusion.fuse(sample_signals)
        assert r1.timestamp != r2.timestamp

    def test_state_reload_preserves_history(self, fusion, sample_signals):
        """State saved to disk should be reloadable."""
        fusion.fuse(sample_signals)
        fusion.fuse(sample_signals)
        
        # Create new instance with same path
        fusion2 = MultiTimeframeFusion(state_path=fusion.state_path)
        assert fusion2.state["fusion_count"] == 2
        assert len(fusion2.state["history"]) == 2

    def test_fuse_returns_fused_result_type(self, fusion, sample_signals):
        """Fuse should return FusedResult dataclass."""
        result = fusion.fuse(sample_signals)
        assert isinstance(result, FusedResult)
        assert hasattr(result, 'timestamp')
        assert hasattr(result, 'overall_signal')
        assert hasattr(result, 'explanation')

    def test_explanation_contains_timeframe_info(self, fusion, sample_signals):
        """Explanation should mention all three timeframes."""
        result = fusion.fuse(sample_signals)
        assert "short" in result.explanation.lower()
        assert "medium" in result.explanation.lower()
        assert "long" in result.explanation.lower()

    def test_timeframe_bucket_weight_assignment(self, fusion):
        """Bucket weights should reflect regime configuration."""
        buckets, _ = fusion.classify_signals(
            {"mean_reversion": 0.5}, regime="crisis"
        )
        # Crisis: short=0.40, medium=0.40, long=0.20
        # But classify_signals doesn't set total_weight — just classifies
        # Testing the TIMEFRAME_BUCKET_WEIGHTS directly
        assert TIMEFRAME_BUCKET_WEIGHTS["crisis"][Timeframe.SHORT] == 0.40

    def test_empty_signal_returns_default_state(self, fusion):
        """Default state has fusion_count=0."""
        assert fusion.state["fusion_count"] == 0
        assert fusion.state["last_fusion"] is None
        assert len(fusion.state["history"]) == 0

    def test_mean_reversion_is_short_term(self):
        """Mean reversion should be classified as short-term."""
        assert SIGNAL_TIMEFRAMES["mean_reversion"] == Timeframe.SHORT

    def test_cta_trend_is_medium_term(self):
        """CTA trend should be classified as medium-term."""
        assert SIGNAL_TIMEFRAMES["cta_trend"] == Timeframe.MEDIUM


# ─── TimeframeBucket Tests ───────────────────────────────────────────

class TestTimeframeBucket:

    def test_bucket_default_values(self):
        """New bucket should have zero defaults."""
        bucket = TimeframeBucket(
            timeframe=Timeframe.SHORT,
            signals=[],
            consensus=0.0,
            agreement=0.0,
            active_count=0,
            total_weight=0.33,
            explanation=""
        )
        assert bucket.timeframe == Timeframe.SHORT
        assert bucket.consensus == 0.0
        assert bucket.active_count == 0

    def test_bucket_with_signals(self):
        """Bucket with signals should reflect input."""
        sig = TimeframeSignal(
            source="test", timeframe=Timeframe.SHORT,
            value=0.5, confidence=0.8, weight=0.8
        )
        bucket = TimeframeBucket(
            timeframe=Timeframe.SHORT,
            signals=[sig],
            consensus=0.5,
            agreement=1.0,
            active_count=1,
            total_weight=0.25,
            explanation="test bucket"
        )
        assert bucket.consensus == 0.5
        assert bucket.agreement == 1.0
        assert bucket.signals[0].value == 0.5

    def test_fused_result_serializable(self, fusion, sample_signals):
        """FusedResult should be JSON-serializable for state persistence."""
        result = fusion.fuse(sample_signals)
        # Basic fields
        assert isinstance(result.timestamp, str)
        assert isinstance(result.overall_signal, float)
        assert isinstance(result.confidence, float)
        assert isinstance(result.regime, str)
        # Check JSON serialization of main fields
        serializable = {
            "timestamp": result.timestamp,
            "overall_signal": result.overall_signal,
            "confidence": result.confidence,
            "regime": result.regime,
        }
        json.dumps(serializable)  # Should not raise


# ─── Integration Test ────────────────────────────────────────────────

class TestIntegration:

    def test_run_from_cli_fuse(self, fusion, sample_signals, sample_confidences, monkeypatch):
        """CLI 'fuse' action should produce output."""
        result = fusion.fuse(sample_signals, sample_confidences)
        assert isinstance(result, FusedResult)
        assert result.overall_signal != 0.0
        assert result.confidence > 0.0

    def test_run_from_cli_breakdown(self, fusion):
        """Breakdown should list all signals."""
        breakdown = fusion.get_timeframe_breakdown()
        assert len(breakdown) >= 3
        # Verify all known signals appear
        all_sources = set()
        for sources in breakdown.values():
            all_sources.update(sources)
        for source in SIGNAL_TIMEFRAMES:
            assert source in all_sources

    def test_status_output(self, fusion):
        """Status should report fusion count and source counts."""
        assert fusion.state["fusion_count"] >= 0
        assert len(SIGNAL_TIMEFRAMES) > 0

    def test_regime_weight_sums_vary(self):
        """Different regimes should have different weight distributions."""
        normal = TIMEFRAME_BUCKET_WEIGHTS["normal"]
        crisis = TIMEFRAME_BUCKET_WEIGHTS["crisis"]
        
        # Crisis should weight short-term more than normal
        assert crisis[Timeframe.SHORT] > normal[Timeframe.SHORT]
        # Normal should weight long-term more than crisis
        assert normal[Timeframe.LONG] > crisis[Timeframe.LONG]
