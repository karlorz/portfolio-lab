"""Tests for the ensemble signal collection collaborator."""

from datetime import datetime, timezone

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
    """Batch DJ: zero-weight soft-delete arms stay on the collect roster."""
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
        SignalSource.MULTI_SPEED_MOM,
        SignalSource.CROSS_ASSET_RV,
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


def test_aggregator_round_trips_production_alt_data_envelope(monkeypatch, tmp_path, caplog):
    """Runtime provenance must not suppress the valid advisory alt-data reading."""
    from src.dashboard import public_projection
    from src.signals.alternative_data_signal import (
        AlternativeDataComposite,
        AlternativeDataSignalGenerator,
    )

    generator = AlternativeDataSignalGenerator()
    generator.signals_dir = tmp_path
    generator.state_dir = tmp_path
    timestamp = datetime.now(timezone.utc).isoformat()
    composite = AlternativeDataComposite(
        timestamp=timestamp,
        composite_score=-0.25,
        confidence=0.8,
        regime="risk_off",
        z_score=-0.5,
        components={"test": -0.25},
        component_confidences={"test": 0.8},
        weights={"test": 1.0},
        data_freshness_hours=1.0,
        sources_count=1,
        symbol_coverage=["SPY"],
    )
    signal = generator.to_ensemble_signal(composite)
    monkeypatch.setenv("PORTFOLIO_LAB_ALT_DATA_AUTO_PROJECT", "0")
    monkeypatch.setattr(public_projection, "_is_known_runtime_path", lambda _path: True)
    generator._save_signal(composite, signal)
    monkeypatch.setattr(
        "src.signals.alternative_data_signal.AlternativeDataSignalGenerator",
        lambda: generator,
    )

    aggregator = SignalAggregator(
        load_price_data=lambda: None,
        regime_weights={Regime.NORMAL: {SignalSource.ALTERNATIVE_DATA: 0.5}},
    )
    with caplog.at_level("WARNING"):
        readings = aggregator.collect(regime=Regime.NORMAL)

    assert SignalSource.ALTERNATIVE_DATA in readings
    assert readings[SignalSource.ALTERNATIVE_DATA].is_active is True
    assert "unexpected keyword" not in caplog.text.lower()
    # Aggregator output is an advisory SignalReading, not an allocation or
    # order-router payload.
    assert set(readings) == {SignalSource.ALTERNATIVE_DATA}
