"""Canonical SignalSource enum — single source of truth.

Three modules previously had duplicate definitions with divergent member sets:
- the ensemble voter (7 members: added multi_timeframe_fusion)
- stacking_feature_engine.py (6 members)
- health_tracker.py (6 members)

Consolidated here to prevent further divergence.
"""

from enum import Enum


class SignalSource(Enum):
    """Available signal sources used across ensemble voting, feature stacking,
    and health tracking."""

    MULTI_SPEED_MOM = "multi_speed_momentum"
    CROSS_ASSET_RV = "cross_asset_rv"
    INTERNATIONAL_MOMENTUM = "international_momentum"
    ALTERNATIVE_DATA = "alternative_data"
    CROSS_ASSET_REGIME_ARB = "cross_asset_regime_arb"
    UNIFIED_OVERLAY = "unified_overlay"
    MULTI_TIMEFRAME_FUSION = "multi_timeframe_fusion"  # v806
    GOOGLE_TRENDS = "google_trends"  # replaces behavioral_sentiment
    VIX_TERM_STRUCTURE = "vix_term_structure"  # v3.23 intraday vol timing


__all__ = ["SignalSource"]
