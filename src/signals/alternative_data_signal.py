"""
Alternative Data Signal Generator — v9.00 Expansion
====================================================
Refactored to use **free data sources from the existing pipeline** only.
No TypeScript dependencies, no mock data, no API keys needed.

Data Sources:
  1. Treasury Curve — TLT/SHY price ratio (yield curve slope proxy)
  2. Sector Rotation — XLF vs XLY momentum differential (risk appetite)
  3. Credit Spread — AGG/IEF ratio (credit cycle proxy)
  4. Tail Risk — SPY realized vol vs long-term average
  5. Broad Momentum — SPY 3-month trend (market health)
  6. Crypto Sentiment — BTC momentum + volatility (sentiment proxy)

Output is compatible with the existing ensemble_voter integration
which reads data/signals/alternative_data_latest.json.
"""

import json
import logging
import math
import statistics
from dataclasses import dataclass, asdict
from datetime import datetime
import numpy as np
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.paths import PRICES_JSON, SIGNALS_DIR, DATA_DIR
from src.backtest.metrics import save_results_json
from src.data.price_cache import get_prices


__all__ = ['SYMBOLS_REQUIRED', 'ComponentSignal', 'AlternativeDataComposite', 'EnsembleSignal', 'AlternativeDataSignalGenerator']

logger = logging.getLogger(__name__)

PRICES_PATH = PRICES_JSON

# Symbols needed for each component
SYMBOLS_REQUIRED = ["SPY", "TLT", "SHY", "XLF", "XLY", "AGG", "IEF", "BTC-USD"]

# Default weights for the 6-component composite (must sum to 1.0)
COMPONENT_WEIGHTS: Dict[str, float] = {
    "treasury_curve": 0.20,
    "sector_rotation": 0.18,
    "credit_spread": 0.18,
    "tail_risk": 0.15,
    "broad_momentum": 0.18,
    "crypto_sentiment": 0.11,
}


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class ComponentSignal:
    """Single component score (-1 to +1)."""
    name: str
    value: float       # -1 to +1
    confidence: float   # 0 to 1
    raw_inputs: Dict[str, Any]


@dataclass
class AlternativeDataComposite:
    """Full alternative data signal."""
    timestamp: str
    composite_score: float          # -1 to +1
    confidence: float               # 0 to 1
    regime: str                     # 'risk_on' | 'neutral' | 'risk_off'
    z_score: float
    components: Dict[str, float]    # Individual component values
    component_confidences: Dict[str, float]
    weights: Dict[str, float]
    data_freshness_hours: float
    sources_count: int
    symbol_coverage: List[str]


@dataclass
class EnsembleSignal:
    """Format that ensemble_voter expects."""
    source: str
    regime: str     # 'bull', 'bear', 'neutral', 'crisis'
    probability: float
    confidence: float
    timestamp: str
    raw_data: dict


# ---------------------------------------------------------------------------
# Signal Generator
# ---------------------------------------------------------------------------

class AlternativeDataSignalGenerator:
    """Generates alternative data signals from existing price pipeline."""

    def __init__(self, prices_path: Path = PRICES_PATH):
        self.prices_path = prices_path
        self.signals_dir = SIGNALS_DIR
        self.signals_dir.mkdir(parents=True, exist_ok=True)
        self.state_dir = DATA_DIR
        self._prices: Optional[Dict[str, List[Dict]]] = None
        self.weights = dict(COMPONENT_WEIGHTS)

    # ---- Data loading ----

    def _load_prices(self) -> Dict[str, List[Dict]]:
        """Load price data from JSON (TTL-cached)."""
        if self._prices is None:
            self._prices = get_prices()
        assert self._prices is not None, "Prices failed to load"
        return self._prices

    def _get_prices(self, symbol: str, days: int = 252) -> List[float]:
        """Get closing prices for a symbol."""
        data = self._load_prices()
        if symbol not in data:
            return []
        prices = [d["p"] for d in data[symbol][-days:]]
        return prices

    def _returns(self, prices: List[float], period: int) -> Optional[float]:
        """Compute total return over a period from a price list."""
        if len(prices) < period:
            return None
        if prices[-period] == 0:
            return None
        return (prices[-1] / prices[-period]) - 1.0

    # ---- Component 1: Treasury Curve ----

    def _treasury_curve_signal(self) -> ComponentSignal:
        """Yield curve slope proxy via TLT/SHY price ratio.
        
        TLT = long-term bonds (20yr), SHY = short-term (1-3yr)
        When TLT outperforms SHY → curve steepening → economic optimism
        When SHY outperforms TLT → curve flattening/inversion → caution
        """
        tlt = self._get_prices("TLT", 126)  # 6 months
        shy = self._get_prices("SHY", 126)

        if len(tlt) < 60 or len(shy) < 60:
            return ComponentSignal("treasury_curve", 0.0, 0.3, {"error": "insufficient data"})

        # Ratio of TLT/SHY prices
        ratio_now = tlt[-1] / shy[-1]
        ratio_60d = tlt[-61] / shy[-61] if len(tlt) > 60 and len(shy) > 60 else ratio_now
        ratio_126d = tlt[0] / shy[0] if len(tlt) >= 126 else ratio_now

        # 60-day and 126-day momentum of the ratio
        mom_60 = (ratio_now / ratio_60d) - 1.0 if ratio_60d > 0 else 0.0
        mom_126 = (ratio_now / ratio_126d) - 1.0 if ratio_126d > 0 else 0.0

        # Blend both timeframes (weight recent more)
        blended = 0.6 * mom_60 + 0.4 * mom_126

        # Scale to -1..+1 (typical max ratio move is ~5-8% in 60 days)
        value = max(-1.0, min(1.0, blended / 0.04))

        # Confidence: higher when TLT and SHY data are fresh and ratio is decisive
        tlt_vol = statistics.stdev([p / tlt[0] for p in tlt[-60:]]) * 100
        confidence = min(1.0, max(0.3, abs(value) * 0.6 + tlt_vol * 0.02))

        return ComponentSignal(
            name="treasury_curve",
            value=value,
            confidence=round(confidence, 4),
            raw_inputs={
                "tlt_price": tlt[-1],
                "shy_price": shy[-1],
                "ratio": ratio_now,
                "ratio_mom_60d": round(mom_60 * 100, 2),
                "ratio_mom_126d": round(mom_126 * 100, 2),
            },
        )

    # ---- Component 2: Sector Rotation ----

    def _sector_rotation_signal(self) -> ComponentSignal:
        """Risk appetite via XLF (financials) vs XLY (consumer disc) momentum.
        
        Financials lead in risk-on periods (rising rates, economic expansion).
        Consumer discretionary leads in consumer confidence.
        When both strong → strong risk-on.
        When XLF weak → potential stress in the financial system.
        """
        xlf = self._get_prices("XLF", 126)
        xly = self._get_prices("XLY", 126)

        if len(xlf) < 60 or len(xly) < 60:
            return ComponentSignal("sector_rotation", 0.0, 0.3, {"error": "insufficient data"})

        # 3-month momentum for each
        xlf_3m = self._returns(xlf, 63) or 0.0
        xly_3m = self._returns(xly, 63) or 0.0

        # 1-month momentum
        xlf_1m = self._returns(xlf, 21) or 0.0
        xly_1m = self._returns(xly, 21) or 0.0

        # Composite: average of both, with XLF leading indicator bias
        avg_3m = (xlf_3m + xly_3m) / 2.0
        avg_1m = (xlf_1m + xly_1m) / 2.0
        blended = 0.5 * avg_3m + 0.5 * avg_1m

        # Scale to -1..+1 (typical sector 3m momentum range ±10-15%)
        value = max(-1.0, min(1.0, blended / 0.08))

        # Confidence: higher when both sectors agree
        same_direction = (xlf_3m > 0) == (xly_3m > 0)
        confidence = 0.6 + 0.3 * (1.0 if same_direction else 0.0)

        return ComponentSignal(
            name="sector_rotation",
            value=value,
            confidence=round(confidence, 4),
            raw_inputs={
                "xlf_3m": round(xlf_3m * 100, 2),
                "xly_3m": round(xly_3m * 100, 2),
                "xlf_1m": round(xlf_1m * 100, 2),
                "xly_1m": round(xly_1m * 100, 2),
                "blended": round(blended * 100, 2),
            },
        )

    # ---- Component 3: Credit Spread Proxy ----

    def _credit_spread_signal(self) -> ComponentSignal:
        """Credit cycle via AGG/IEF ratio.
        
        AGG is the US Aggregate Bond Index (corporate + government).
        IEF is Treasury-only (7-10yr).
        When AGG outperforms IEF → credit tightening, spreads narrowing → risk-on.
        When AGG underperforms → credit widening → stress → risk-off.
        """
        agg = self._get_prices("AGG", 126)
        ief = self._get_prices("IEF", 126)

        if len(agg) < 60 or len(ief) < 60:
            return ComponentSignal("credit_spread", 0.0, 0.3, {"error": "insufficient data"})

        # Ratio of AGG/IEF
        ratio_now = agg[-1] / ief[-1]
        ratio_63d = agg[-64] / ief[-64] if len(agg) > 63 and len(ief) > 63 else ratio_now

        # 63-day momentum of the ratio
        mom = (ratio_now / ratio_63d) - 1.0 if ratio_63d > 0 else 0.0

        # Scale: AGG/IEF ratio moves are typically small (0.5-2% over 3 months)
        value = max(-1.0, min(1.0, mom / 0.012))

        # Confidence: higher with larger spread moves
        agg_vol = statistics.stdev([p / agg[0] for p in agg[-60:]]) * 100
        confidence = min(1.0, max(0.4, abs(value) * 0.5 + agg_vol * 0.015))

        return ComponentSignal(
            name="credit_spread",
            value=value,
            confidence=round(confidence, 4),
            raw_inputs={
                "agg_price": agg[-1],
                "ief_price": ief[-1],
                "ratio": ratio_now,
                "ratio_mom_63d": round(mom * 100, 2),
                "agg_vol_60d": round(agg_vol, 2),
            },
        )

    # ---- Component 4: Tail Risk ----

    def _tail_risk_signal(self) -> ComponentSignal:
        """SPY realized volatility vs long-term average.
        
        When short-term vol is low relative to history → benign → risk-on.
        When short-term vol spikes → fear → risk-off.
        This mirrors the VIX/VVIX relationship.
        """
        spy = self._get_prices("SPY", 504)  # 2 years

        if len(spy) < 252:
            return ComponentSignal("tail_risk", 0.0, 0.3, {"error": "insufficient data"})

        # Compute daily returns
        daily_returns = [
            (spy[i] / spy[i - 1]) - 1.0
            for i in range(1, len(spy))
        ]

        # 21-day realized vol (annualized)
        short_vol = statistics.stdev(daily_returns[-21:]) * math.sqrt(252)

        # 252-day realized vol (annualized)
        long_vol = statistics.stdev(daily_returns[-252:]) * math.sqrt(252)

        if long_vol == 0:
            return ComponentSignal("tail_risk", 0.0, 0.4, {"error": "zero vol"})

        vol_ratio = short_vol / long_vol

        # vol_ratio > 1.0 = elevated vol = risk-off
        # vol_ratio < 1.0 = low vol = risk-on
        # Scale: typical vol ratio ranges from 0.3 to 3.0
        raw_value = 1.0 - (vol_ratio - 1.0)  # inverse: low vol → positive
        value = max(-1.0, min(1.0, raw_value * 0.6))

        confidence = min(1.0, max(0.4, 1.0 - abs(vol_ratio - 1.0) * 0.3))

        return ComponentSignal(
            name="tail_risk",
            value=value,
            confidence=round(confidence, 4),
            raw_inputs={
                "short_vol_21d": round(short_vol * 100, 2),
                "long_vol_252d": round(long_vol * 100, 2),
                "vol_ratio": round(vol_ratio, 4),
            },
        )

    # ---- Component 5: Broad Momentum ----

    def _broad_momentum_signal(self) -> ComponentSignal:
        """SPY broad market trend as a market health catch-all.
        
        When SPY has strong positive momentum → risk-on.
        When SPY is declining → risk-off.
        Combines 1m, 3m, and 6m timeframes.
        """
        spy = self._get_prices("SPY", 252)

        if len(spy) < 63:
            return ComponentSignal("broad_momentum", 0.0, 0.3, {"error": "insufficient data"})

        mom_1m = self._returns(spy, 21) or 0.0
        mom_3m = self._returns(spy, 63) or 0.0
        mom_6m = self._returns(spy, 126) or 0.0

        # Weighted blend: more weight on medium term
        blended = 0.3 * mom_1m + 0.4 * mom_3m + 0.3 * mom_6m

        # Scale to -1..+1 (SPY typical 3m momentum ±5-12%)
        value = max(-1.0, min(1.0, blended / 0.06))

        # Confidence: higher with consistent direction across timeframes
        directions = [mom_1m > 0, mom_3m > 0, mom_6m > 0]
        consistency = sum(directions) / len(directions)
        confidence = 0.5 + 0.4 * abs(consistency - 0.5) * 2  # 0.5 to 0.9

        return ComponentSignal(
            name="broad_momentum",
            value=value,
            confidence=round(confidence, 4),
            raw_inputs={
                "spy_1m": round(mom_1m * 100, 2),
                "spy_3m": round(mom_3m * 100, 2),
                "spy_6m": round(mom_6m * 100, 2),
                "direction_consistency": consistency,
            },
        )

    # ---- Component 6: Crypto Sentiment ----

    def _crypto_sentiment_signal(self) -> ComponentSignal:
        """Crypto Fear & Greed Index as sentiment proxy.
        
        Uses BTC momentum and realized volatility as a proxy for crypto market
        sentiment. Extreme fear in crypto often precedes broader market risk-on
        moves, while extreme greed often precedes market corrections.
        
        Note: BTC-USD may not be in prices.json pipeline. When unavailable,
        returns neutral signal with low confidence.
        """
        btc = self._get_prices("BTC-USD", 126)
        
        # If BTC data not available, return neutral signal
        if not btc or len(btc) < 60:
            return ComponentSignal("crypto_sentiment", 0.0, 0.1, {"error": "BTC data unavailable", "fallback": "neutral"})
        
        # 20-day momentum
        mom_20 = self._returns(btc, 20) or 0.0
        
        # 60-day realized volatility
        daily_returns = [(btc[i] / btc[i-1]) - 1.0 for i in range(1, len(btc))]
        vol_60 = statistics.stdev(daily_returns[-60:]) * math.sqrt(252) if len(daily_returns) >= 60 else 0.0
        
        # Compute a fear/greed score from momentum and vol
        # High momentum + low vol = greed (positive)
        # Low momentum + high vol = fear (negative)
        mom_score = mom_20 / 0.10  # Normalize: 10% move = 1.0
        vol_score = 1.0 - (vol_60 / 0.80)  # 80% annual vol = 0.0, 0% = 1.0
        
        # Blend: momentum signals direction, vol signals confidence
        value = mom_score * 0.6 + vol_score * 0.4
        value = max(-1.0, min(1.0, value))
        
        # Confidence: higher when both signals agree
        confidence = 0.5 + 0.3 * abs(value)
        
        return ComponentSignal(
            name="crypto_sentiment",
            value=value,
            confidence=round(confidence, 4),
            raw_inputs={
                "btc_price": btc[-1],
                "btc_mom_20d": round(mom_20 * 100, 2),
                "btc_vol_60d": round(vol_60 * 100, 2),
                "vol_score": round(vol_score, 4),
            },
        )

    # ---- Composite calculation ----

    def _compute_z_score(self, composite_score: float) -> float:
        """Estimate z-score assuming std dev of 0.3."""
        return composite_score / 0.3

    def _determine_regime(self, composite_score: float) -> str:
        """Map composite score to regime label."""
        if composite_score > 0.15:
            return "risk_on"
        elif composite_score < -0.15:
            return "risk_off"
        return "neutral"

    def calculate_composite(self, components: List[ComponentSignal]) -> AlternativeDataComposite:
        """Weighted composite from all component signals."""
        comp_map = {c.name: c for c in components}

        total_weight = sum(self.weights.values())
        if total_weight == 0:
            total_weight = 1.0

        composite = 0.0
        confidence = 0.0
        weight_sum = 0.0

        comp_values = {}
        comp_confidences = {}

        for name, weight in self.weights.items():
            if name in comp_map:
                sig = comp_map[name]
                comp_values[name] = sig.value
                comp_confidences[name] = sig.confidence
                composite += sig.value * weight
                confidence += sig.confidence * weight
                weight_sum += weight

        if weight_sum > 0:
            composite /= weight_sum
            confidence /= weight_sum

        return AlternativeDataComposite(
            timestamp=datetime.now().isoformat(),
            composite_score=round(composite, 4),
            confidence=round(confidence, 4),
            regime=self._determine_regime(composite),
            z_score=round(self._compute_z_score(composite), 4),
            components=comp_values,
            component_confidences=comp_confidences,
            weights=dict(self.weights),
            data_freshness_hours=12.0,
            sources_count=len(self.weights),
            symbol_coverage=SYMBOLS_REQUIRED,
        )

    # ---- Output formatting ----

    def to_ensemble_signal(self, composite: AlternativeDataComposite) -> EnsembleSignal:
        """Convert to ensemble voter compatible format."""
        regime_map = {
            "risk_on": "bull",
            "neutral": "neutral",
            "risk_off": "bear",
        }

        def normal_cdf(x):
            return 0.5 * (1 + math.erf(x / math.sqrt(2)))

        if composite.regime == "risk_on":
            probability = normal_cdf(composite.z_score)
        elif composite.regime == "risk_off":
            probability = normal_cdf(-composite.z_score)
        else:
            probability = 2 * (0.5 - abs(normal_cdf(composite.z_score) - 0.5))

        return EnsembleSignal(
            source="alternative_data",
            regime=regime_map.get(composite.regime, "neutral"),
            probability=round(probability, 4),
            confidence=composite.confidence,
            timestamp=composite.timestamp,
            raw_data=asdict(composite),
        )

    # ---- Persistence ----

    def _save_signal(self, composite: AlternativeDataComposite, signal: EnsembleSignal):
        """Save signals to disk and update state."""
        # Ensemble format (for ensemble_voter)
        signal_file = self.signals_dir / "alternative_data_latest.json"
        save_results_json(asdict(signal), output_path=str(signal_file))

        # State file (for status tracking)
        state = {
            "last_update": composite.timestamp,
            "composite_score": composite.composite_score,
            "confidence": composite.confidence,
            "regime": composite.regime,
            "z_score": composite.z_score,
            "components": composite.components,
            "component_confidences": composite.component_confidences,
        }
        state_file = self.state_dir / "alternative_data_state.json"
        save_results_json(state, output_path=str(state_file))

    def load_latest_signal(self) -> Optional[EnsembleSignal]:
        """Load most recent signal from disk."""
        latest_file = self.signals_dir / "alternative_data_latest.json"
        if not latest_file.exists():
            return None
        with open(latest_file) as f:
            data = json.load(f)
        return EnsembleSignal(**data)

    def validate_signal(self, signal: EnsembleSignal) -> bool:
        """Validate signal meets quality criteria."""
        if signal.confidence < 0.3:
            return False
        signal_time = datetime.fromisoformat(signal.timestamp)
        age_hours = (datetime.now() - signal_time).total_seconds() / 3600
        if age_hours > 48:
            return False
        return True

    def get_signal_snapshot(self):
        """Return latest signal as canonical SignalSnapshot for typed pipeline consumption."""
        from src.signals.signal_snapshot import SignalSnapshot
        signal = self.load_latest_signal()
        if signal is None:
            return SignalSnapshot(
                source="alternative_data",
                timestamp=datetime.now().isoformat(),
                value=0.0,
                confidence=0.0,
                is_active=False,
                explanation="Alternative data signal unavailable",
            )
        composite_score = signal.raw_data.get("composite_score", 0.0) if signal.raw_data else 0.0
        value = float(np.clip(composite_score, -1, 1)) if composite_score else 0.0
        if value == 0.0:
            regime_map = {"bull": 0.4, "bear": -0.4, "neutral": 0.0, "crisis": -0.7}
            value = regime_map.get(signal.regime, 0.0)
        return SignalSnapshot(
            source="alternative_data",
            timestamp=signal.timestamp,
            value=value,
            confidence=signal.confidence,
            asset_signals={"SPY": value},
            regime_fit="all",
            is_active=True,
            explanation=f"Alt Data: regime={signal.regime}, composite={composite_score:.4f}, "
                        f"prob={signal.probability:.2f}, conf={signal.confidence:.2f}",
            metadata={"regime": signal.regime, "probability": signal.probability,
                      "raw_data": signal.raw_data},
        )

    # ---- Main pipeline ----

    def generate_signal(self) -> EnsembleSignal:
        """Main signal generation pipeline."""
        components = [
            self._treasury_curve_signal(),
            self._sector_rotation_signal(),
            self._credit_spread_signal(),
            self._tail_risk_signal(),
            self._broad_momentum_signal(),
            self._crypto_sentiment_signal(),
        ]

        composite = self.calculate_composite(components)
        signal = self.to_ensemble_signal(composite)
        self._save_signal(composite, signal)
        return signal


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    import argparse
    from src.utils.log_config import configure_logging
    configure_logging()

    parser = argparse.ArgumentParser(
        description="Alternative Data Signal Generator (v9.00)"
    )
    parser.add_argument("--generate", action="store_true", help="Generate new signal")
    parser.add_argument("--status", action="store_true", help="Show latest signal status")
    parser.add_argument("--validate", action="store_true", help="Validate latest signal")

    args = parser.parse_args()
    generator = AlternativeDataSignalGenerator()

    if args.generate:
        signal = generator.generate_signal()
        raw = signal.raw_data
        logger.info("\n%s", "=" * 50)
        logger.info("  Alternative Data Signal — %s", signal.timestamp)
        logger.info("%s", "=" * 50)
        logger.info("  Regime:       %s", signal.regime.upper())
        logger.info("  Score:        %+.4f", raw['composite_score'])
        logger.info("  Confidence:   %.2f%%", raw['confidence'] * 100)
        logger.info("  Z-Score:      %.2f", raw['z_score'])
        logger.info("\n  Components:")
        for name in sorted(raw.get('components', {}).keys()):
            val = raw['components'].get(name, 0)
            conf = raw['component_confidences'].get(name, 0)
            logger.info("    %-20s: %+.4f  (conf=%.2f%%)", name, val, conf * 100)
        logger.info("\n  Symbols:       %s", len(raw.get('symbol_coverage', [])))
        logger.info("  Freshness:    %sh", raw.get('data_freshness_hours', '?'))
        logger.info("%s", "=" * 50)

    elif args.status:
        signal = generator.load_latest_signal()
        if signal:
            raw = signal.raw_data
            logger.info("Latest Signal (%s):", signal.timestamp)
            logger.info("  Regime:       %s", signal.regime.upper())
            logger.info("  Score:        %s", raw.get('composite_score', '?'))
            logger.info("  Confidence:   %.2f%%", signal.confidence * 100)
            logger.info("  Probability:  %.2f%%", signal.probability * 100)
            logger.info("\n  Components:")
            for name in sorted(raw.get('components', {}).keys()):
                val = raw['components'].get(name, 0)
                logger.info("    %-20s: %+.4f", name, val)
        else:
            logger.info("No signal found. Run with --generate first.")

    elif args.validate:
        signal = generator.load_latest_signal()
        if signal:
            is_valid = generator.validate_signal(signal)
            logger.info("Signal valid: %s", 'Yes' if is_valid else 'No')
        else:
            logger.info("No signal to validate.")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
