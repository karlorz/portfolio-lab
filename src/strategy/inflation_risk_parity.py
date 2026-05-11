"""
Inflation-Aware Risk Parity Strategy (v2.11)
Risk parity allocation with inflation regime detection and dynamic tilting

Research: Goldman Sachs (Oct 2025), iShares Fall 2025, LongTail Alpha
- Risk parity ETFs gained up to 19% in 2025's inflationary environment
- Equal risk contributions provide better inflation resilience
- TIPS and commodities should be overweighted in high inflation regimes
"""

import os
import sys
import json
import sqlite3
import numpy as np
import argparse
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path


@dataclass
class InflationRegime:
    """Detected inflation regime with confidence"""
    regime: str  # "low_inflation", "rising_inflation", "high_inflation", "disinflation"
    confidence: float  # 0-1
    signals: Dict[str, float]  # Individual signal values
    

@dataclass
class RiskParityAllocation:
    """Risk parity allocation result"""
    timestamp: str
    base_weights: Dict[str, float]  # Inverse volatility weights
    tilted_weights: Dict[str, float]  # After regime tilt
    regime: InflationRegime
    volatilities: Dict[str, float]
    risk_contributions: Dict[str, float]


class InflationRiskParityEngine:
    """
    Inflation-Aware Risk Parity Strategy Engine
    
    Base: Inverse volatility weighted risk parity
    Overlay: Dynamic tilting based on inflation regime
    
    Assets:
    - SPY: US equity (vol ~16%, inflation beta 0.3)
    - VTV: Value equity (vol ~15%, inflation beta 0.2)
    - GLD: Gold (vol ~15%, inflation beta 0.8)
    - DBC: Commodities (vol ~20%, inflation beta 1.2)
    - TLT: Nominal bonds (vol ~12%, inflation beta -0.4)
    - SPTL: Long bonds (vol ~13%, inflation beta -0.5)
    """
    
    ASSETS = {
        "SPY": {"class": "equity", "vol": 0.16, "inflation_beta": 0.3},
        "VTV": {"class": "value_equity", "vol": 0.15, "inflation_beta": 0.2},
        "GLD": {"class": "gold", "vol": 0.15, "inflation_beta": 0.8},
        "DBC": {"class": "commodities", "vol": 0.20, "inflation_beta": 1.2},
        "TLT": {"class": "nominal_bonds", "vol": 0.12, "inflation_beta": -0.4},
        "SPTL": {"class": "long_bonds", "vol": 0.13, "inflation_beta": -0.5},
    }
    
    REGIME_TILTS = {
        "low_inflation": {
            "TLT": +0.15, "SPTL": +0.10, "SPY": +0.05,
            "GLD": -0.10, "DBC": -0.05
        },
        "rising_inflation": {
            "DBC": +0.15, "GLD": +0.10, "VTV": +0.05,
            "TLT": -0.15, "SPTL": -0.10
        },
        "high_inflation": {
            "DBC": +0.20, "GLD": +0.15,
            "SPY": -0.10, "TLT": -0.20
        },
        "disinflation": {
            "SPY": +0.15, "TLT": +0.10,
            "GLD": -0.10, "DBC": -0.10
        }
    }
    
    def __init__(
        self,
        db_path: Path = None
    ):
        if db_path is None:
            self.db_path = Path("/root/projects/portfolio-lab/data/market.db")
        else:
            self.db_path = Path(db_path).expanduser()
        self.vol_lookback = 60  # 60-day volatility
        
    def _fetch_data(self, symbol: str, days: int = 100) -> List[Dict]:
        """Fetch historical price data from SQLite"""
        if not self.db_path.exists():
            print(f"Database not found: {self.db_path}")
            return []
        
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        try:
            cursor.execute("""
                SELECT date, close, volume
                FROM prices
                WHERE symbol = ?
                ORDER BY date DESC
                LIMIT ?
            """, (symbol, days))
            
            rows = cursor.fetchall()
        except sqlite3.Error as e:
            print(f"Database error: {e}")
            rows = []
        finally:
            conn.close()
        
        return [
            {"date": row[0], "close": row[1], "volume": row[2]}
            for row in reversed(rows)
        ]
    
    def _calculate_volatility(self, prices: np.ndarray, days: int = 60) -> float:
        """Calculate annualized realized volatility"""
        if len(prices) < days + 1:
            return 0.15  # Default 15%
        
        returns = np.diff(prices[-days-1:]) / prices[-days-1:-1]
        daily_vol = np.std(returns)
        annual_vol = daily_vol * np.sqrt(252)
        
        return max(annual_vol, 0.05)  # Floor at 5%
    
    def _calculate_inverse_vol_weights(self, volatilities: Dict[str, float]) -> Dict[str, float]:
        """
        Risk parity: Inverse volatility weighting
        Higher volatility assets get lower weights
        """
        inverse_vols = {}
        for sym, vol in volatilities.items():
            if vol > 0:
                inverse_vols[sym] = 1.0 / vol
            else:
                inverse_vols[sym] = 1.0 / 0.15  # Default
        
        total = sum(inverse_vols.values())
        if total == 0:
            return {sym: 1.0 / len(volatilities) for sym in volatilities}
        
        return {sym: w / total for sym, w in inverse_vols.items()}
    
    def _detect_inflation_regime(self) -> InflationRegime:
        """
        Detect inflation regime from market data
        
        Signals used:
        - Gold trend (proxy for inflation expectations)
        - Commodity trend (DBC)
        - Bond yields (if available)
        - Equity value vs growth (VTV vs VUG proxy)
        """
        signals = {}
        
        # Gold trend (proxy for inflation expectations)
        gld_data = self._fetch_data("GLD", 90)
        if len(gld_data) >= 60:
            gld_prices = np.array([d["close"] for d in gld_data])
            gld_20d = np.mean(gld_prices[-20:])
            gld_60d = np.mean(gld_prices[-60:])
            gld_trend = (gld_20d / gld_60d - 1) * 100
            signals["gold_trend"] = gld_trend
        else:
            signals["gold_trend"] = 0.0
        
        # Commodity trend
        dbc_data = self._fetch_data("DBC", 90)
        if len(dbc_data) >= 60:
            dbc_prices = np.array([d["close"] for d in dbc_data])
            dbc_20d = np.mean(dbc_prices[-20:])
            dbc_60d = np.mean(dbc_prices[-60:])
            dbc_trend = (dbc_20d / dbc_60d - 1) * 100
            signals["commodity_trend"] = dbc_trend
        else:
            signals["commodity_trend"] = 0.0
        
        # Bond yield proxy (using TLT price inverse)
        tlt_data = self._fetch_data("TLT", 90)
        if len(tlt_data) >= 60:
            tlt_prices = np.array([d["close"] for d in tlt_data])
            tlt_20d = np.mean(tlt_prices[-20:])
            tlt_60d = np.mean(tlt_prices[-60:])
            tlt_trend = (tlt_20d / tlt_60d - 1) * 100
            # Invert: falling TLT = rising yields = inflation
            signals["yield_proxy"] = -tlt_trend
        else:
            signals["yield_proxy"] = 0.0
        
        # Calculate composite inflation score
        # Gold and commodities up + yields up = rising inflation
        inflation_score = (
            signals.get("gold_trend", 0) * 0.4 +
            signals.get("commodity_trend", 0) * 0.4 +
            signals.get("yield_proxy", 0) * 0.2
        )
        
        # Determine regime
        if inflation_score > 5:
            regime = "high_inflation"
            confidence = min(1.0, abs(inflation_score) / 10)
        elif inflation_score > 2:
            regime = "rising_inflation"
            confidence = min(1.0, abs(inflation_score) / 5)
        elif inflation_score < -5:
            regime = "disinflation"
            confidence = min(1.0, abs(inflation_score) / 10)
        elif inflation_score < -2:
            regime = "low_inflation"
            confidence = min(1.0, abs(inflation_score) / 5)
        else:
            # Neutral zone - check individual signals
            if signals.get("gold_trend", 0) > 3 and signals.get("commodity_trend", 0) > 2:
                regime = "rising_inflation"
                confidence = 0.6
            elif signals.get("gold_trend", 0) < -3 and signals.get("commodity_trend", 0) < -2:
                regime = "disinflation"
                confidence = 0.6
            else:
                regime = "low_inflation"  # Default
                confidence = 0.5
        
        return InflationRegime(
            regime=regime,
            confidence=confidence,
            signals=signals
        )
    
    def _apply_regime_tilt(
        self,
        base_weights: Dict[str, float],
        regime: str,
        confidence: float = 1.0
    ) -> Dict[str, float]:
        """
        Apply regime-based allocation tilt
        
        Confidence scales the tilt strength (0.5 to 1.0)
        """
        tilt = self.REGIME_TILTS.get(regime, {})
        tilted = {}
        
        # Scale tilt by confidence
        scale = 0.5 + (confidence * 0.5)  # 0.5 to 1.0
        
        for sym, weight in base_weights.items():
            tilt_amount = tilt.get(sym, 0) * scale
            tilted[sym] = max(0.02, weight + tilt_amount)  # Min 2% per asset
        
        # Renormalize to sum to 1.0
        total = sum(tilted.values())
        if total > 0:
            tilted = {sym: w / total for sym, w in tilted.items()}
        
        return tilted
    
    def evaluate(self) -> RiskParityAllocation:
        """
        Run full inflation-aware risk parity evaluation
        
        Returns allocation recommendation with regime detection
        """
        timestamp = datetime.now().isoformat()
        
        # Calculate realized volatilities for each asset
        volatilities = {}
        for symbol in self.ASSETS.keys():
            data = self._fetch_data(symbol, self.vol_lookback + 10)
            if len(data) >= self.vol_lookback:
                prices = np.array([d["close"] for d in data])
                vol = self._calculate_volatility(prices, self.vol_lookback)
                volatilities[symbol] = vol
            else:
                # Use default volatility
                volatilities[symbol] = self.ASSETS[symbol]["vol"]
        
        # Calculate base risk parity weights (inverse volatility)
        base_weights = self._calculate_inverse_vol_weights(volatilities)
        
        # Detect inflation regime
        regime_info = self._detect_inflation_regime()
        
        # Apply regime tilt
        tilted_weights = self._apply_regime_tilt(
            base_weights,
            regime_info.regime,
            regime_info.confidence
        )
        
        # Calculate risk contributions (simplified)
        risk_contributions = {}
        total_risk = 0
        for sym, weight in tilted_weights.items():
            risk = weight * volatilities.get(sym, 0.15)
            risk_contributions[sym] = risk
            total_risk += risk
        
        # Normalize risk contributions
        if total_risk > 0:
            risk_contributions = {
                sym: r / total_risk for sym, r in risk_contributions.items()
            }
        
        return RiskParityAllocation(
            timestamp=timestamp,
            base_weights=base_weights,
            tilted_weights=tilted_weights,
            regime=regime_info,
            volatilities=volatilities,
            risk_contributions=risk_contributions
        )
    
    def get_allocation_summary(self) -> Dict:
        """Get JSON-serializable allocation summary"""
        result = self.evaluate()
        
        return {
            "timestamp": result.timestamp,
            "strategy": "inflation_risk_parity",
            "version": "2.11",
            "regime": {
                "name": result.regime.regime,
                "confidence": round(result.regime.confidence, 2),
                "signals": {k: round(v, 3) for k, v in result.regime.signals.items()}
            },
            "allocation": {
                "base_weights": {k: round(v, 4) for k, v in result.base_weights.items()},
                "tilted_weights": {k: round(v, 4) for k, v in result.tilted_weights.items()}
            },
            "volatilities": {k: round(v, 4) for k, v in result.volatilities.items()},
            "risk_contributions": {k: round(v, 4) for k, v in result.risk_contributions.items()},
            "asset_classes": {
                sym: info["class"] for sym, info in self.ASSETS.items()
            }
        }


def main():
    """CLI interface"""
    parser = argparse.ArgumentParser(description="Inflation-Aware Risk Parity Strategy")
    parser.add_argument("--current", action="store_true", help="Get current allocation")
    parser.add_argument("--db", type=str, default="/root/projects/portfolio-lab/data/market.db", help="Database path")
    parser.add_argument("--json", action="store_true", help="Output raw JSON only")
    
    args = parser.parse_args()
    
    engine = InflationRiskParityEngine(db_path=Path(args.db))
    
    if args.current or not args.json:
        result = engine.get_allocation_summary()
        
        if args.json:
            print(json.dumps(result, indent=2))
        else:
            print(f"\n{'='*70}")
            print("INFLATION-AWARE RISK PARITY (v2.11)")
            print(f"{'='*70}")
            print(f"Timestamp: {result['timestamp']}")
            print(f"Regime: {result['regime']['name']} (confidence: {result['regime']['confidence']:.0%})")
            
            print(f"\n{'-'*70}")
            print("INFLATION SIGNALS")
            print(f"{'-'*70}")
            for signal, value in result['regime']['signals'].items():
                print(f"  {signal}: {value:+.2f}%")
            
            print(f"\n{'-'*70}")
            print("ALLOCATION COMPARISON")
            print(f"{'-'*70}")
            print(f"{'Asset':<10} {'Base':<10} {'Tilted':<10} {'Change':<10} {'Risk %':<10}")
            print(f"{'-'*70}")
            
            all_assets = sorted(set(result['allocation']['base_weights'].keys()) | 
                              set(result['allocation']['tilted_weights'].keys()))
            
            for asset in all_assets:
                base = result['allocation']['base_weights'].get(asset, 0)
                tilted = result['allocation']['tilted_weights'].get(asset, 0)
                change = tilted - base
                risk = result['risk_contributions'].get(asset, 0)
                print(f"{asset:<10} {base:>9.1%} {tilted:>9.1%} {change:>+9.1%} {risk:>9.1%}")
            
            print(f"{'='*70}\n")
            
            # Also output JSON for integration
            print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
