"""Tests for the ensemble signal collection collaborator."""

from src.strategy.ensemble_voter import (
    EnsembleVoter,
    Regime,
    SignalReading,
    SignalSource,
)
from src.strategy.signal_aggregator import SignalAggregator


def _reading(source: SignalSource) -> SignalReading:
    return SignalReading(
        source=source,
        timestamp="2026-01-01",
        value=0.25,
        confidence=0.8,
        weight=0.0,
        regime_fit="all",
        asset_signals={"SPY": 0.25, "TLT": -0.1, "GLD": 0.05},
        explanation="test",
    )


def test_signal_aggregator_derives_active_sources_from_regime_weights():
    aggregator = SignalAggregator(
        load_price_data=lambda: None,
        regime_weights={
            Regime.LOW_VOL: {
                SignalSource.MULTI_SPEED_MOM: 0.0,
                SignalSource.CROSS_ASSET_RV: 0.4,
            }
        },
    )

    assert aggregator.active_sources_for(Regime.LOW_VOL) == {
        SignalSource.CROSS_ASSET_RV
    }


def test_ensemble_voter_delegates_signal_collection_to_aggregator(tmp_path):
    voter = EnsembleVoter(data_path=tmp_path)
    expected = {SignalSource.CROSS_ASSET_RV: _reading(SignalSource.CROSS_ASSET_RV)}

    class StubAggregator:
        def collect(self, date=None, regime=None):
            assert date == "2026-01-02"
            assert regime == Regime.NORMAL
            return expected

    voter.signal_aggregator = StubAggregator()

    result = voter.collect_signals(date="2026-01-02", regime=Regime.NORMAL)

    assert result == expected
    assert voter.current_readings == expected
