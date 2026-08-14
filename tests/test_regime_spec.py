"""Direct unit tests for src/signals/regime_spec.py.

The module was extracted from the ensemble voter (A1-CYCLE-BREAK s2, 96837cd);
test_integrator pins only the re-exported constant. These tests lock the
extraction contract directly: the Regime/SignalReading types, the
REGIME_WEIGHTS shape, and every reachable ``_load_regime_weights`` validation
branch. Test-only item: no src/ behavior change.

Note on defensive branches: the ``Regime()`` ValueError branch (regime_spec.py
:260-267) and the final missing-regimes branch (:304-311) are unreachable
through ``_load_regime_weights`` because business keys are validated against
the same enum set before the loop (:190, :248). They are therefore not
individually exercised.
"""

import json
import math

import pytest

from src.signals.regime_spec import (
    ENSEMBLE_WEIGHT_METADATA_KEYS,
    REGIME_WEIGHTS,
    Regime,
    SignalReading,
    _build_hardcoded_weights,
    _load_regime_weights,
)
from src.signals.signal_source import SignalSource


def _valid_payload() -> dict:
    """Payload that passes every validation branch (5 regimes x full map)."""
    return {r.value: {s.value: 0.1 for s in SignalSource} for r in Regime}


def _write(tmp_path, payload) -> str:
    path = tmp_path / "ensemble_weights.json"
    path.write_text(json.dumps(payload))
    return str(path)


class TestRegimeContract:
    def test_regime_enum_has_five_values(self):
        assert len(Regime) == 5
        assert {r.value for r in Regime} == {
            "low_vol",
            "normal",
            "high_vol",
            "crisis",
            "recovery",
        }

    def test_regime_weights_shape(self):
        """5 regime keys, each mapping every SignalSource to a finite weight >= 0."""
        assert set(REGIME_WEIGHTS) == set(Regime)
        for regime in Regime:
            sources = REGIME_WEIGHTS[regime]
            assert set(sources) == set(SignalSource)
            for weight in sources.values():
                assert weight >= 0

    def test_integrator_reexport_identity(self):
        """Lock the A1 extraction seam: integrator re-exports the same object."""
        from src.signals.integrator import CANONICAL_REGIME_WEIGHTS

        assert CANONICAL_REGIME_WEIGHTS is REGIME_WEIGHTS

    def test_signal_reading_defaults(self):
        reading = SignalReading(
            source=SignalSource.GOOGLE_TRENDS,
            timestamp="2026-08-13T00:00:00Z",
            value=0.5,
            confidence=0.8,
            weight=0.05,
            regime_fit="normal",
        )
        assert reading.asset_signals is None
        assert reading.explanation == ""
        assert reading.is_active is True
        assert reading.metadata is None

    def test_signal_reading_custom_metadata(self):
        reading = SignalReading(
            source=SignalSource.CROSS_ASSET_RV,
            timestamp="2026-08-13T00:00:00Z",
            value=-0.25,
            confidence=0.6,
            weight=0.1,
            regime_fit="crisis",
            asset_signals={"SPY": 0.5},
            explanation="RV spike",
            is_active=False,
            metadata={"pattern": "vol_breakout"},
        )
        assert reading.asset_signals == {"SPY": 0.5}
        assert reading.is_active is False
        assert reading.metadata == {"pattern": "vol_breakout"}

    def test_metadata_allowlist_is_frozenset(self):
        assert isinstance(ENSEMBLE_WEIGHT_METADATA_KEYS, frozenset)
        assert "generated_at" in ENSEMBLE_WEIGHT_METADATA_KEYS
        assert "schema_version" in ENSEMBLE_WEIGHT_METADATA_KEYS


class TestFallbackNormalization:
    """Clean-checkout fallback (weights JSON absent) must be a valid 1.0 map.

    fda0020 untracked data/ensemble_weights.json; every hardcoded regime
    table then served as the fallback. VIX_TERM_STRUCTURE was appended at
    83a56eb without scaling down the older entries, so each table summed to
    1.05. These tests force the fallback through a missing weights file (never
    the local data dir) and pin the normalization contract.
    """

    def test_fallback_regimes_sum_to_one_when_weights_json_absent(self, tmp_path):
        """Every fallback regime map must sum to exactly 1.0."""
        missing = str(tmp_path / "does_not_exist.json")
        fallback = _load_regime_weights(weights_file=missing)
        for regime in Regime:
            total = sum(fallback[regime].values())
            assert math.isclose(total, 1.0, rel_tol=0.0, abs_tol=1e-9), (
                f"{regime.value} fallback weights sum to {total!r}"
            )

    def test_fallback_preserves_relative_ratios_of_nonzero_sources(self, tmp_path):
        """Normalization must keep pairwise ratios of the legacy raw table.

        Raw LOW_VOL table (legacy relative values): CROSS_ASSET_RV 0.1350,
        ALTERNATIVE_DATA 0.2650, INTERNATIONAL_MOMENTUM 0.2520, UNIFIED_OVERLAY
        0.1980, MULTI_TIMEFRAME_FUSION 0.1000, GOOGLE_TRENDS 0.0500,
        VIX_TERM_STRUCTURE 0.0500 -> total 1.05. Each normalized weight must
        equal raw / 1.05, so the VIX-term 0.05 contribution keeps its share
        relative to the former values.
        """
        raw = {
            SignalSource.CROSS_ASSET_RV: 0.1350,
            SignalSource.ALTERNATIVE_DATA: 0.2650,
            SignalSource.INTERNATIONAL_MOMENTUM: 0.2520,
            SignalSource.UNIFIED_OVERLAY: 0.1980,
            SignalSource.MULTI_TIMEFRAME_FUSION: 0.1000,
            SignalSource.GOOGLE_TRENDS: 0.0500,
            SignalSource.VIX_TERM_STRUCTURE: 0.0500,
        }
        raw_total = sum(raw.values())
        assert math.isclose(raw_total, 1.05, rel_tol=0.0, abs_tol=1e-12)

        missing = str(tmp_path / "does_not_exist.json")
        low_vol = _load_regime_weights(weights_file=missing)[Regime.LOW_VOL]
        for source, raw_weight in raw.items():
            assert low_vol[source] == pytest.approx(raw_weight / raw_total), (
                f"{source.value} ratio to total not preserved"
            )

    def test_fallback_normalization_handles_all_zero_map(self, tmp_path):
        """An all-zero map must be returned unchanged, not divided by zero."""
        from src.signals.regime_spec import _normalize_weights

        zero_map = {SignalSource.GOOGLE_TRENDS: 0.0}
        assert _normalize_weights(zero_map) == zero_map


class TestLoadRegimeWeights:
    def test_valid_payload_loads(self, tmp_path):
        payload = _valid_payload()
        result = _load_regime_weights(weights_file=_write(tmp_path, payload))
        assert set(result) == set(Regime)
        assert result[Regime.LOW_VOL][SignalSource.GOOGLE_TRENDS] == 0.1

    def test_metadata_keys_are_ignored(self, tmp_path):
        """Additive runtime provenance keys must not invalidate the payload."""
        payload = _valid_payload()
        payload["generated_at"] = "2026-08-13T00:00:00Z"
        payload["artifact_id"] = "abc"
        payload["_private"] = {"nested": True}
        result = _load_regime_weights(weights_file=_write(tmp_path, payload))
        assert set(result) == set(Regime)

    def test_missing_file_falls_back_to_hardcoded(self, tmp_path):
        missing = str(tmp_path / "does_not_exist.json")
        assert _load_regime_weights(weights_file=missing) == _build_hardcoded_weights()

    def test_invalid_json_falls_back_to_hardcoded(self, tmp_path):
        path = tmp_path / "bad.json"
        path.write_text("{not json")
        assert _load_regime_weights(weights_file=str(path)) == _build_hardcoded_weights()

    def test_oserror_falls_back_to_hardcoded(self, tmp_path):
        """A directory path raises IsADirectoryError (OSError) on open."""
        assert _load_regime_weights(weights_file=str(tmp_path)) == _build_hardcoded_weights()

    @pytest.mark.parametrize(
        "payload",
        [
            pytest.param([], id="raw-not-a-dict"),
            pytest.param({"low_vol": [1, 2, 3]}, id="regime-map-not-a-dict"),
        ],
    )
    def test_non_object_payload_falls_back(self, tmp_path, payload):
        assert _load_regime_weights(weights_file=_write(tmp_path, payload)) == _build_hardcoded_weights()

    @pytest.mark.parametrize(
        "mutator",
        [
            pytest.param(lambda p: p.pop("normal"), id="missing-regime"),
            pytest.param(lambda p: p.update({"extra_regime": {s.value: 0.1 for s in SignalSource}}), id="extra-regime"),
            pytest.param(lambda p: p.update({"extra_key": "not-a-regime"}), id="unknown-key"),
        ],
    )
    def test_regime_set_mismatch_falls_back(self, tmp_path, mutator):
        payload = _valid_payload()
        mutator(payload)
        assert _load_regime_weights(weights_file=_write(tmp_path, payload)) == _build_hardcoded_weights()

    @pytest.mark.parametrize(
        "mutator",
        [
            pytest.param(lambda p: p["low_vol"].pop("google_trends"), id="missing-source"),
            pytest.param(lambda p: p["low_vol"].update({"extra_source": 0.1}), id="extra-source"),
        ],
    )
    def test_source_map_mismatch_falls_back(self, tmp_path, mutator):
        payload = _valid_payload()
        mutator(payload)
        assert _load_regime_weights(weights_file=_write(tmp_path, payload)) == _build_hardcoded_weights()

    @pytest.mark.parametrize(
        "bad_weight",
        [
            pytest.param("abc", id="non-numeric-string"),
            pytest.param(None, id="none"),
            pytest.param(-0.1, id="negative"),
            pytest.param(float("nan"), id="nan"),
            pytest.param(float("inf"), id="infinity"),
        ],
    )
    def test_invalid_weight_falls_back(self, tmp_path, bad_weight):
        payload = _valid_payload()
        payload["low_vol"]["google_trends"] = bad_weight
        assert _load_regime_weights(weights_file=_write(tmp_path, payload)) == _build_hardcoded_weights()

    def test_env_var_override_is_honored(self, tmp_path, monkeypatch):
        """ENSEMBLE_WEIGHTS_FILE override applies when weights_file is None."""
        payload = _valid_payload()
        path = _write(tmp_path, payload)
        monkeypatch.setenv("ENSEMBLE_WEIGHTS_FILE", path)
        result = _load_regime_weights()
        assert set(result) == set(Regime)
        assert result[Regime.CRISIS][SignalSource.CROSS_ASSET_RV] == 0.1

    def test_default_path_matches_import_time_load(self, monkeypatch):
        """No override: loads DATA_DIR/ensemble_weights.json like module import."""
        monkeypatch.delenv("ENSEMBLE_WEIGHTS_FILE", raising=False)
        assert _load_regime_weights() == REGIME_WEIGHTS
