"""Signal collection collaborator for the ensemble voter."""

import logging
from datetime import datetime
from typing import Any, Callable, Dict, Mapping, Optional, Set

import pandas as pd

from src.signals.signal_source import SignalSource

logger = logging.getLogger(__name__)

SIGNAL_EXCEPTIONS = (AttributeError, KeyError, ValueError, TypeError, OSError, RuntimeError)


class SignalAggregator:
    """Collect active ensemble signals without owning vote weighting logic."""

    def __init__(
        self,
        load_price_data: Callable[[], Optional[pd.DataFrame]],
        regime_weights: Mapping[Any, Mapping[SignalSource, float]],
    ) -> None:
        self._load_price_data = load_price_data
        self.regime_weights = regime_weights

    def active_sources_for(self, regime: Optional[Any]) -> Optional[Set[SignalSource]]:
        if regime is None:
            return None
        return {
            source
            for source, weight in self.regime_weights.get(regime, {}).items()
            if weight > 0
        }

    def should_skip(
        self,
        source: SignalSource,
        active_sources: Optional[Set[SignalSource]],
        regime: Optional[Any],
    ) -> bool:
        if active_sources is not None and source not in active_sources:
            regime_name = regime.value if regime is not None and hasattr(regime, "value") else "?"
            logger.debug("Skipping %s: zero weight for regime=%s", source.value, regime_name)
            return True
        return False

    def collect(
        self,
        date: Optional[str] = None,
        regime: Optional[Any] = None,
    ) -> Dict[SignalSource, Any]:
        active_sources = self.active_sources_for(regime)
        readings: Dict[SignalSource, Any] = {}
        self._collect_msm_signal(readings, active_sources, regime, date)
        self._collect_cross_asset_rv_signal(readings, active_sources, regime)
        self._collect_intl_momentum_signal(readings, active_sources, regime)
        self._collect_alt_data_signal(readings, active_sources, regime)
        self._collect_regime_arb_signal(readings, active_sources, regime)
        self._collect_unified_overlay_signal(readings, active_sources, regime)
        self._collect_mtf_signal(readings, active_sources, regime, date)
        self._collect_google_trends(readings, active_sources, regime)
        self._collect_vix_term_structure_signal(readings, active_sources, regime)
        return readings

    @staticmethod
    def _store_active_snapshot(
        readings: Dict[SignalSource, Any],
        source: SignalSource,
        snapshot: Any,
    ) -> None:
        if snapshot.is_active:
            readings[source] = snapshot.to_signal_reading()

    def _collect_msm_signal(
        self,
        readings: Dict[SignalSource, Any],
        active_sources: Optional[Set[SignalSource]],
        regime: Optional[Any],
        date: Optional[str],
    ) -> None:
        if self.should_skip(SignalSource.MULTI_SPEED_MOM, active_sources, regime):
            return
        try:
            from src.signals.multi_speed_momentum import MultiSpeedMomentum

            snapshot = MultiSpeedMomentum().get_signal_snapshot(
                tickers=["SPY", "TLT", "GLD"], date=date
            )
            self._store_active_snapshot(readings, SignalSource.MULTI_SPEED_MOM, snapshot)
        except ImportError:
            pass
        except SIGNAL_EXCEPTIONS as e:
            logger.warning("Multi-speed momentum unavailable: %s", e)

    def _collect_cross_asset_rv_signal(
        self,
        readings: Dict[SignalSource, Any],
        active_sources: Optional[Set[SignalSource]],
        regime: Optional[Any],
    ) -> None:
        if self.should_skip(SignalSource.CROSS_ASSET_RV, active_sources, regime):
            return
        try:
            from src.signals.cross_asset_relative_value import CrossAssetRVScanner

            snapshot = CrossAssetRVScanner().get_signal_snapshot()
            self._store_active_snapshot(readings, SignalSource.CROSS_ASSET_RV, snapshot)
        except ImportError:
            pass
        except SIGNAL_EXCEPTIONS as e:
            logger.warning("Cross-asset RV unavailable: %s", e)

    def _collect_intl_momentum_signal(
        self,
        readings: Dict[SignalSource, Any],
        active_sources: Optional[Set[SignalSource]],
        regime: Optional[Any],
    ) -> None:
        if self.should_skip(SignalSource.INTERNATIONAL_MOMENTUM, active_sources, regime):
            return
        try:
            from src.signals.international_momentum import InternationalMomentumGenerator

            data = self._build_international_momentum_data()
            if data is None:
                return
            signal = InternationalMomentumGenerator().generate_signal(data)
            self._store_active_snapshot(
                readings,
                SignalSource.INTERNATIONAL_MOMENTUM,
                signal.to_signal_snapshot(),
            )
        except ImportError:
            pass
        except SIGNAL_EXCEPTIONS as e:
            logger.warning("International momentum unavailable: %s", e)

    def _build_international_momentum_data(self) -> Optional[Dict[str, Any]]:
        price_data = self._load_price_data()
        if price_data is None or price_data.empty:
            return None

        required_cols = [c for c in ["SPY", "EFA", "EEM"] if c in price_data.columns]
        if len(required_cols) < 2:
            return None
        recent = price_data[required_cols].iloc[-126:] if len(price_data) >= 126 else price_data[required_cols]
        if len(recent) < 20:
            return None

        spy_mom = (recent["SPY"].iloc[-1] / recent["SPY"].iloc[0] - 1) * 100
        efa_mom = self._momentum_or_zero(recent, "EFA")
        eem_mom = self._momentum_or_zero(recent, "EEM")
        return {
            "timestamp": str(datetime.now()),
            "relative": {
                "efa_momentum_6m": efa_mom,
                "eem_momentum_6m": eem_mom,
                "spy_momentum_6m": spy_mom,
                "efa_vs_spy": efa_mom - spy_mom,
                "eem_vs_spy": eem_mom - spy_mom,
            },
            "data_fresh": True,
        }

    @staticmethod
    def _momentum_or_zero(recent: pd.DataFrame, column: str) -> float:
        if column not in recent:
            return 0.0
        return (recent[column].iloc[-1] / recent[column].iloc[0] - 1) * 100

    def _collect_alt_data_signal(
        self,
        readings: Dict[SignalSource, Any],
        active_sources: Optional[Set[SignalSource]],
        regime: Optional[Any],
    ) -> None:
        if self.should_skip(SignalSource.ALTERNATIVE_DATA, active_sources, regime):
            return
        try:
            from src.signals.alternative_data_signal import AlternativeDataSignalGenerator

            snapshot = AlternativeDataSignalGenerator().get_signal_snapshot()
            self._store_active_snapshot(readings, SignalSource.ALTERNATIVE_DATA, snapshot)
        except ImportError:
            pass
        except SIGNAL_EXCEPTIONS as e:
            logger.warning("Alternative data unavailable: %s", e)

    def _collect_regime_arb_signal(
        self,
        readings: Dict[SignalSource, Any],
        active_sources: Optional[Set[SignalSource]],
        regime: Optional[Any],
    ) -> None:
        if self.should_skip(SignalSource.CROSS_ASSET_REGIME_ARB, active_sources, regime):
            return
        try:
            from src.signals.cross_asset_regime_arb import CrossAssetRegimeArbDetector

            snapshot = CrossAssetRegimeArbDetector().get_signal_snapshot()
            self._store_active_snapshot(readings, SignalSource.CROSS_ASSET_REGIME_ARB, snapshot)
        except ImportError:
            logger.warning("Cross-asset regime arb module not available")
        except SIGNAL_EXCEPTIONS as e:
            logger.warning("Cross-asset regime arb unavailable: %s", e)

    def _collect_unified_overlay_signal(
        self,
        readings: Dict[SignalSource, Any],
        active_sources: Optional[Set[SignalSource]],
        regime: Optional[Any],
    ) -> None:
        if self.should_skip(SignalSource.UNIFIED_OVERLAY, active_sources, regime):
            return
        try:
            from .orchestrator_ensemble_bridge import OrchestratorEnsembleBridge

            readings[SignalSource.UNIFIED_OVERLAY] = OrchestratorEnsembleBridge().get_ensemble_reading()
        except ImportError:
            pass
        except SIGNAL_EXCEPTIONS as e:
            logger.warning("Unified overlay unavailable: %s", e)

    def _collect_mtf_signal(
        self,
        readings: Dict[SignalSource, Any],
        active_sources: Optional[Set[SignalSource]],
        regime: Optional[Any],
        date: Optional[str] = None,
    ) -> None:
        if self.should_skip(SignalSource.MULTI_TIMEFRAME_FUSION, active_sources, regime):
            return
        try:
            from src.signals.multi_timeframe_fusion import MultiTimeframeFusion

            snapshot = MultiTimeframeFusion(
                prices_df=self._load_price_data()
            ).get_signal_snapshot(
                tickers=["SPY", "GLD", "TLT"],
                date=date,
                regime=regime.value if regime else "normal",
            )
            self._store_active_snapshot(readings, SignalSource.MULTI_TIMEFRAME_FUSION, snapshot)
        except ImportError:
            pass
        except SIGNAL_EXCEPTIONS as e:
            logger.warning("Multi-timeframe fusion unavailable: %s", e)

    def _collect_google_trends(
        self,
        readings: Dict[SignalSource, Any],
        active_sources: Optional[Set[SignalSource]],
        regime: Optional[Any],
    ) -> None:
        if self.should_skip(SignalSource.GOOGLE_TRENDS, active_sources, regime):
            return
        try:
            from src.signals.google_trends_signal import GoogleTrendsSignal

            snapshot = GoogleTrendsSignal().get_signal_snapshot()
            self._store_active_snapshot(readings, SignalSource.GOOGLE_TRENDS, snapshot)
        except ImportError:
            pass
        except SIGNAL_EXCEPTIONS as e:
            logger.warning("Google Trends signal unavailable: %s", e)

    def _collect_vix_term_structure_signal(
        self,
        readings: Dict[SignalSource, Any],
        active_sources: Optional[Set[SignalSource]],
        regime: Optional[Any],
    ) -> None:
        if self.should_skip(SignalSource.VIX_TERM_STRUCTURE, active_sources, regime):
            return
        try:
            from src.signals.vix_term_structure import VIXTermStructureSignalGenerator

            signal = VIXTermStructureSignalGenerator().generate_signal()
            if signal.is_valid:
                readings[SignalSource.VIX_TERM_STRUCTURE] = (
                    signal.to_signal_snapshot().to_signal_reading()
                )
        except ImportError:
            pass
        except SIGNAL_EXCEPTIONS as e:
            logger.warning("VIX term structure signal unavailable: %s", e)
