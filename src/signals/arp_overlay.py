"""
v2.60 Alternative Risk Premia (ARP) Overlay
Implements carry and value signals based on AQR research:
- "Understanding Alternative Risk Premia" (AQR Whitepaper)
- "Value and Momentum Everywhere" (Asness, Moskowitz, Pedersen)

Carry: Buy higher-yielding assets, sell lower-yielding
Value: Buy cheap assets (high yield/book-to-price), sell expensive
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import json


@dataclass
class CarrySignal:
    """Carry signal for an asset"""
    symbol: str
    carry_yield: float  # Annualized carry yield (%)
    percentile_1y: float  # Percentile vs 1-year history
    percentile_5y: float  # Percentile vs 5-year history
    signal_score: float  # Normalized -1 to 1
    confidence: float
    regime: str  # 'high_carry', 'neutral', 'low_carry'


@dataclass
class ValueSignal:
    """Value signal for an asset"""
    symbol: str
    metric_type: str  # 'pe_ratio', 'pb_ratio', 'dividend_yield', 'real_yield'
    current_value: float
    percentile_1y: float
    percentile_5y: float
    percentile_10y: float
    signal_score: float  # Normalized -1 to 1 (negative = cheap = buy)
    confidence: float
    regime: str  # 'cheap', 'fair', 'expensive'


class CarryCalculator:
    """
    Calculate carry signals across asset classes.
    
    Bond Carry: Real yield spread vs risk-free
    Equity Carry: Dividend yield + buyback yield - risk-free
    Gold Carry: Opportunity cost (real yield inverse)
    """
    
    def __init__(self, risk_free_rate: float = 0.045):
        self.risk_free_rate = risk_free_rate
        self.carry_history: Dict[str, List[Tuple[datetime, float]]] = {}
        
    def calculate_bond_carry(self, 
                            symbol: str,
                            yield_to_maturity: float,
                            inflation_expectation: float = 0.025) -> CarrySignal:
        """
        Calculate bond carry as real yield.
        
        Real Yield = Nominal Yield - Inflation Expectation
        Higher real yield = higher carry = buy signal
        """
        real_yield = yield_to_maturity - inflation_expectation
        carry_spread = real_yield - self.risk_free_rate
        
        # Store for percentile calculation
        if symbol not in self.carry_history:
            self.carry_history[symbol] = []
        self.carry_history[symbol].append((datetime.now(), real_yield))
        
        # Keep only 5 years of history
        cutoff = datetime.now() - timedelta(days=5*365)
        self.carry_history[symbol] = [
            (d, v) for d, v in self.carry_history[symbol] if d > cutoff
        ]
        
        # Calculate percentiles
        pctl_1y = self._percentile(symbol, real_yield, days=365)
        pctl_5y = self._percentile(symbol, real_yield, days=5*365)
        
        # Signal score: normalize percentiles to -1 to 1
        # High percentile (high yield) = positive signal (buy)
        signal_score = (pctl_5y - 0.5) * 2
        
        # Regime classification
        if pctl_5y > 0.75:
            regime = 'high_carry'
        elif pctl_5y < 0.25:
            regime = 'low_carry'
        else:
            regime = 'neutral'
            
        # Confidence based on data availability
        confidence = min(1.0, len(self.carry_history.get(symbol, [])) / 252)
        
        return CarrySignal(
            symbol=symbol,
            carry_yield=real_yield * 100,  # Convert to percentage
            percentile_1y=pctl_1y,
            percentile_5y=pctl_5y,
            signal_score=signal_score,
            confidence=confidence,
            regime=regime
        )
    
    def calculate_equity_carry(self,
                              symbol: str,
                              dividend_yield: float,
                              buyback_yield: float = 0.015,
                              earnings_growth: float = 0.05) -> CarrySignal:
        """
        Calculate equity carry as shareholder yield + growth - risk-free.
        
        Equity Carry = Dividend Yield + Buyback Yield + Earnings Growth - Risk-Free
        """
        shareholder_yield = dividend_yield + buyback_yield
        total_carry = shareholder_yield + earnings_growth - self.risk_free_rate
        
        if symbol not in self.carry_history:
            self.carry_history[symbol] = []
        self.carry_history[symbol].append((datetime.now(), total_carry))
        
        cutoff = datetime.now() - timedelta(days=5*365)
        self.carry_history[symbol] = [
            (d, v) for d, v in self.carry_history[symbol] if d > cutoff
        ]
        
        pctl_1y = self._percentile(symbol, total_carry, days=365)
        pctl_5y = self._percentile(symbol, total_carry, days=5*365)
        
        signal_score = (pctl_5y - 0.5) * 2
        
        if pctl_5y > 0.75:
            regime = 'high_carry'
        elif pctl_5y < 0.25:
            regime = 'low_carry'
        else:
            regime = 'neutral'
            
        confidence = min(1.0, len(self.carry_history.get(symbol, [])) / 252)
        
        return CarrySignal(
            symbol=symbol,
            carry_yield=total_carry * 100,
            percentile_1y=pctl_1y,
            percentile_5y=pctl_5y,
            signal_score=signal_score,
            confidence=confidence,
            regime=regime
        )
    
    def calculate_gold_carry(self,
                           symbol: str = 'GLD',
                           real_yield_10y: float = 0.02,
                           storage_cost: float = 0.0025) -> CarrySignal:
        """
        Calculate gold carry as negative of real yield (opportunity cost).
        
        Gold has no yield, so carry = -real_yield - storage_cost
        High real yields = negative gold carry = sell signal
        """
        gold_carry = -real_yield_10y - storage_cost
        
        if symbol not in self.carry_history:
            self.carry_history[symbol] = []
        self.carry_history[symbol].append((datetime.now(), gold_carry))
        
        cutoff = datetime.now() - timedelta(days=5*365)
        self.carry_history[symbol] = [
            (d, v) for d, v in self.carry_history[symbol] if d > cutoff
        ]
        
        pctl_1y = self._percentile(symbol, gold_carry, days=365)
        pctl_5y = self._percentile(symbol, gold_carry, days=5*365)
        
        # For gold: low carry (very negative) is actually a buying opportunity
        # because it means real yields are high (gold is cheap)
        signal_score = -(pctl_5y - 0.5) * 2  # Inverted
        
        if pctl_5y < 0.25:  # Very negative carry = high real yields = gold cheap
            regime = 'high_carry'  # Good time to buy gold
        elif pctl_5y > 0.75:
            regime = 'low_carry'
        else:
            regime = 'neutral'
            
        confidence = min(1.0, len(self.carry_history.get(symbol, [])) / 252)
        
        return CarrySignal(
            symbol=symbol,
            carry_yield=gold_carry * 100,
            percentile_1y=pctl_1y,
            percentile_5y=pctl_5y,
            signal_score=signal_score,
            confidence=confidence,
            regime=regime
        )
    
    def _percentile(self, symbol: str, current: float, days: int) -> float:
        """Calculate percentile of current value vs history"""
        if symbol not in self.carry_history:
            return 0.5
            
        cutoff = datetime.now() - timedelta(days=days)
        history = [v for d, v in self.carry_history[symbol] if d > cutoff]
        
        if len(history) < 30:
            return 0.5
            
        return np.sum(np.array(history) < current) / len(history)


class ValueCalculator:
    """
    Calculate value signals across asset classes.
    
    Based on "Value and Momentum Everywhere" (Asness, Moskowitz, Pedersen):
    - Value: Buy assets with high book-to-price (low P/B), sell low B/P
    - Applied across equities, bonds, currencies, commodities
    """
    
    def __init__(self):
        self.value_history: Dict[str, Dict[str, List[Tuple[datetime, float]]]] = {}
        
    def calculate_equity_value(self,
                              symbol: str,
                              pe_ratio: float,
                              pb_ratio: float,
                              dividend_yield: float) -> ValueSignal:
        """
        Calculate equity value signal.
        
        Uses composite of P/E and P/B percentiles.
        Lower P/E and P/B = higher value = buy signal (negative score)
        """
        # Composite value metric (inverse of P/E normalized)
        if pe_ratio > 0:
            earnings_yield = 1.0 / pe_ratio
        else:
            earnings_yield = 0.0
            
        book_to_market = 1.0 / pb_ratio if pb_ratio > 0 else 0.0
        
        # Composite value score
        value_score = (earnings_yield + book_to_market + dividend_yield) / 3.0
        
        # Store history
        if symbol not in self.value_history:
            self.value_history[symbol] = {}
        if 'composite' not in self.value_history[symbol]:
            self.value_history[symbol]['composite'] = []
            
        self.value_history[symbol]['composite'].append((datetime.now(), value_score))
        
        # Clean old data
        cutoff = datetime.now() - timedelta(days=10*365)
        self.value_history[symbol]['composite'] = [
            (d, v) for d, v in self.value_history[symbol]['composite'] if d > cutoff
        ]
        
        # Calculate percentiles
        pctl_1y = self._percentile(symbol, 'composite', value_score, days=365)
        pctl_5y = self._percentile(symbol, 'composite', value_score, days=5*365)
        pctl_10y = self._percentile(symbol, 'composite', value_score, days=10*365)
        
        # Signal score: Low percentile (cheap) = positive signal
        # Invert so cheap (low percentile) = positive score
        signal_score = -(pctl_5y - 0.5) * 2
        
        # Regime
        if pctl_5y < 0.25:
            regime = 'cheap'
        elif pctl_5y > 0.75:
            regime = 'expensive'
        else:
            regime = 'fair'
            
        confidence = min(1.0, len(self.value_history[symbol].get('composite', [])) / 252)
        
        return ValueSignal(
            symbol=symbol,
            metric_type='composite',
            current_value=value_score,
            percentile_1y=pctl_1y,
            percentile_5y=pctl_5y,
            percentile_10y=pctl_10y,
            signal_score=signal_score,
            confidence=confidence,
            regime=regime
        )
    
    def calculate_bond_value(self,
                           symbol: str,
                           real_yield: float,
                           term_premium: float = 0.02) -> ValueSignal:
        """
        Calculate bond value signal based on real yield percentile.
        
        Higher real yield = cheaper bonds = buy signal
        """
        # Bond value = real yield (higher is better/cheaper)
        value_score = real_yield
        
        if symbol not in self.value_history:
            self.value_history[symbol] = {}
        if 'real_yield' not in self.value_history[symbol]:
            self.value_history[symbol]['real_yield'] = []
            
        self.value_history[symbol]['real_yield'].append((datetime.now(), value_score))
        
        cutoff = datetime.now() - timedelta(days=10*365)
        self.value_history[symbol]['real_yield'] = [
            (d, v) for d, v in self.value_history[symbol]['real_yield'] if d > cutoff
        ]
        
        pctl_1y = self._percentile(symbol, 'real_yield', value_score, days=365)
        pctl_5y = self._percentile(symbol, 'real_yield', value_score, days=5*365)
        pctl_10y = self._percentile(symbol, 'real_yield', value_score, days=10*365)
        
        # High percentile (high yield) = cheap = buy signal (positive)
        signal_score = (pctl_5y - 0.5) * 2
        
        if pctl_5y > 0.75:
            regime = 'cheap'
        elif pctl_5y < 0.25:
            regime = 'expensive'
        else:
            regime = 'fair'
            
        confidence = min(1.0, len(self.value_history[symbol].get('real_yield', [])) / 252)
        
        return ValueSignal(
            symbol=symbol,
            metric_type='real_yield',
            current_value=value_score,
            percentile_1y=pctl_1y,
            percentile_5y=pctl_5y,
            percentile_10y=pctl_10y,
            signal_score=signal_score,
            confidence=confidence,
            regime=regime
        )
    
    def _percentile(self, symbol: str, metric: str, current: float, days: int) -> float:
        """Calculate percentile of current value vs history"""
        if symbol not in self.value_history or metric not in self.value_history[symbol]:
            return 0.5
            
        cutoff = datetime.now() - timedelta(days=days)
        history = [v for d, v in self.value_history[symbol][metric] if d > cutoff]
        
        if len(history) < 30:
            return 0.5
            
        return np.sum(np.array(history) < current) / len(history)


class ARPOverlay:
    """
    Alternative Risk Premia Overlay combining carry and value signals.
    
    Implements AQR-style multi-style, cross-asset approach:
    - Carry: Bond yields, equity shareholder yield, gold opportunity cost
    - Value: P/E, P/B, dividend yield percentiles
    
    Target weight in SignalIntegrator: 8-10%
    """
    
    def __init__(self, risk_free_rate: float = 0.045):
        self.carry_calc = CarryCalculator(risk_free_rate)
        self.value_calc = ValueCalculator()
        
        # Current market data (would be fetched from data source in production)
        self.current_data = {
            'SPY': {'pe': 22.5, 'pb': 4.1, 'div_yield': 0.013, 'buyback_yield': 0.018},
            'QQQ': {'pe': 28.3, 'pb': 5.8, 'div_yield': 0.006, 'buyback_yield': 0.015},
            'TLT': {'yield': 0.045, 'duration': 17.5},
            'IEF': {'yield': 0.042, 'duration': 7.5},
            'GLD': {'storage_cost': 0.0025},
            'EFA': {'pe': 15.2, 'pb': 1.8, 'div_yield': 0.028},
            'DBC': {'roll_yield': -0.015},  # Typically negative for commodities
        }
        
    def generate_signals(self) -> Dict[str, Dict]:
        """Generate carry and value signals for all tracked assets"""
        signals = {}
        
        # SPY signals
        spy_carry = self.carry_calc.calculate_equity_carry(
            'SPY',
            dividend_yield=self.current_data['SPY']['div_yield'],
            buyback_yield=self.current_data['SPY']['buyback_yield']
        )
        spy_value = self.value_calc.calculate_equity_value(
            'SPY',
            pe_ratio=self.current_data['SPY']['pe'],
            pb_ratio=self.current_data['SPY']['pb'],
            dividend_yield=self.current_data['SPY']['div_yield']
        )
        signals['SPY'] = {
            'carry': spy_carry,
            'value': spy_value,
            'composite_score': (spy_carry.signal_score * 0.5 + spy_value.signal_score * 0.5),
            'composite_confidence': (spy_carry.confidence + spy_value.confidence) / 2
        }
        
        # TLT signals
        tlt_carry = self.carry_calc.calculate_bond_carry(
            'TLT',
            yield_to_maturity=self.current_data['TLT']['yield']
        )
        tlt_value = self.value_calc.calculate_bond_value(
            'TLT',
            real_yield=self.current_data['TLT']['yield'] - 0.025  # Approx real yield
        )
        signals['TLT'] = {
            'carry': tlt_carry,
            'value': tlt_value,
            'composite_score': (tlt_carry.signal_score * 0.6 + tlt_value.signal_score * 0.4),
            'composite_confidence': (tlt_carry.confidence + tlt_value.confidence) / 2
        }
        
        # GLD signals (carry-based on real yields)
        real_yield_10y = 0.02  # Approx current 10Y TIPS yield
        gld_carry = self.carry_calc.calculate_gold_carry(
            'GLD',
            real_yield_10y=real_yield_10y
        )
        signals['GLD'] = {
            'carry': gld_carry,
            'value': None,  # Gold has no traditional value metric
            'composite_score': gld_carry.signal_score,
            'composite_confidence': gld_carry.confidence
        }
        
        # EFA (international) signals
        efa_carry = self.carry_calc.calculate_equity_carry(
            'EFA',
            dividend_yield=self.current_data['EFA']['div_yield'],
            buyback_yield=0.01  # Lower buyback in international
        )
        efa_value = self.value_calc.calculate_equity_value(
            'EFA',
            pe_ratio=self.current_data['EFA']['pe'],
            pb_ratio=self.current_data['EFA']['pb'],
            dividend_yield=self.current_data['EFA']['div_yield']
        )
        signals['EFA'] = {
            'carry': efa_carry,
            'value': efa_value,
            'composite_score': (efa_carry.signal_score * 0.5 + efa_value.signal_score * 0.5),
            'composite_confidence': (efa_carry.confidence + efa_value.confidence) / 2
        }
        
        return signals
    
    def get_allocation_adjustments(self, 
                                 base_allocation: Dict[str, float]) -> Dict[str, float]:
        """
        Generate allocation adjustments based on ARP signals.
        
        Returns delta adjustments (positive = increase allocation)
        """
        signals = self.generate_signals()
        adjustments = {}
        
        for symbol, base_weight in base_allocation.items():
            if symbol in signals:
                sig = signals[symbol]
                score = sig['composite_score']
                conf = sig['composite_confidence']
                
                # Scale adjustment by confidence and base weight
                # Max adjustment: +/- 20% of base weight
                max_adjust = base_weight * 0.2
                adjustment = score * max_adjust * conf
                adjustments[symbol] = adjustment
            else:
                adjustments[symbol] = 0.0
                
        return adjustments
    
    def get_signal_summary(self) -> Dict:
        """Get summary of current ARP signals for reporting"""
        signals = self.generate_signals()
        
        return {
            'timestamp': datetime.now().isoformat(),
            'risk_free_rate': self.carry_calc.risk_free_rate,
            'signals': {
                symbol: {
                    'carry_regime': s['carry'].regime if s['carry'] else None,
                    'carry_score': round(s['carry'].signal_score, 3) if s['carry'] else None,
                    'carry_yield': round(s['carry'].carry_yield, 2) if s['carry'] else None,
                    'value_regime': s['value'].regime if s['value'] else None,
                    'value_score': round(s['value'].signal_score, 3) if s['value'] else None,
                    'composite_score': round(s['composite_score'], 3),
                    'confidence': round(s['composite_confidence'], 2)
                }
                for symbol, s in signals.items()
            },
            'top_carry_trades': self._get_top_signals(signals, 'carry', 3),
            'top_value_trades': self._get_top_signals(signals, 'value', 3)
        }
    
    def _get_top_signals(self, 
                        signals: Dict, 
                        signal_type: str, 
                        n: int) -> List[Dict]:
        """Get top N signals by absolute score"""
        scored = []
        for symbol, s in signals.items():
            if signal_type == 'carry' and s['carry']:
                scored.append({
                    'symbol': symbol,
                    'score': s['carry'].signal_score,
                    'regime': s['carry'].regime,
                    'yield': s['carry'].carry_yield if hasattr(s['carry'], 'carry_yield') else None
                })
            elif signal_type == 'value' and s['value']:
                scored.append({
                    'symbol': symbol,
                    'score': s['value'].signal_score,
                    'regime': s['value'].regime,
                    'value': s['value'].current_value
                })
        
        scored.sort(key=lambda x: abs(x['score']), reverse=True)
        return scored[:n]


def main():
    """CLI for ARP Overlay"""
    import argparse
    
    parser = argparse.ArgumentParser(description='v2.60 ARP Overlay')
    parser.add_argument('--mode', choices=['signals', 'adjustments', 'summary'], 
                       default='summary')
    parser.add_argument('--portfolio', default='46/38/16',
                       help='Base allocation SPY/GLD/TLT')
    parser.add_argument('--risk-free', type=float, default=0.045)
    
    args = parser.parse_args()
    
    # Parse portfolio
    weights = args.portfolio.split('/')
    base_alloc = {
        'SPY': float(weights[0]) / 100 if len(weights) > 0 else 0.46,
        'GLD': float(weights[1]) / 100 if len(weights) > 1 else 0.38,
        'TLT': float(weights[2]) / 100 if len(weights) > 2 else 0.16
    }
    
    arp = ARPOverlay(risk_free_rate=args.risk_free)
    
    if args.mode == 'signals':
        signals = arp.generate_signals()
        for symbol, sig in signals.items():
            print(f"\n{symbol}:")
            if sig['carry']:
                print(f"  Carry: {sig['carry'].regime} (score: {sig['carry'].signal_score:.3f})")
            if sig['value']:
                print(f"  Value: {sig['value'].regime} (score: {sig['value'].signal_score:.3f})")
            print(f"  Composite: {sig['composite_score']:.3f} (conf: {sig['composite_confidence']:.2f})")
    
    elif args.mode == 'adjustments':
        adjustments = arp.get_allocation_adjustments(base_alloc)
        print(f"\nBase allocation: {base_alloc}")
        print(f"ARP adjustments:")
        for symbol, adj in adjustments.items():
            new_weight = base_alloc.get(symbol, 0) + adj
            print(f"  {symbol}: {base_alloc.get(symbol, 0)*100:.1f}% → {new_weight*100:.1f}% (Δ{adj*100:+.1f}%)")
    
    elif args.mode == 'summary':
        summary = arp.get_signal_summary()
        print(json.dumps(summary, indent=2, default=str))


if __name__ == '__main__':
    main()


# ---------------------------------------------------------------------------
# SignalIntegrator Adapter Classes
# ---------------------------------------------------------------------------

class SignalSource:
    """Abstract base class for signal sources (mirrors integrator interface)"""
    def generate_signal(self, ticker: str):
        raise NotImplementedError


class CarrySignalAdapter(SignalSource):
    """
    Adapter for Carry signals to integrate with SignalIntegrator.
    v2.60 AQR-style carry signal source.
    """
    
    def __init__(self):
        self.carry_calc = CarryCalculator()
        self.source_name = "aqr_carry_premium"
        
    def generate_signal(self, ticker: str):
        """Generate carry signal for integrator interface"""
        from dataclasses import dataclass
        
        @dataclass
        class SignalResult:
            source_type: str
            source_name: str
            signal: float
            confidence: float
            raw_score: float
            raw_unit: str
            historical_accuracy: float
            sample_count: int
            metadata: dict
            
        # Map tickers to asset types
        asset_type = 'equity'
        if ticker in ['TLT', 'IEF', 'AGG']:
            asset_type = 'bond'
        elif ticker == 'GLD':
            asset_type = 'gold'
        elif ticker in ['EFA', 'VEA', 'VXUS']:
            asset_type = 'equity_intl'
            
        # Mock data for demonstration (would fetch from data source in production)
        mock_data = {
            'SPY': {'div_yield': 0.013, 'buyback': 0.018},
            'QQQ': {'div_yield': 0.006, 'buyback': 0.015},
            'TLT': {'yield': 0.045},
            'IEF': {'yield': 0.042},
            'GLD': {'real_yield_10y': 0.02},
            'EFA': {'div_yield': 0.028, 'buyback': 0.010}
        }
        
        if ticker not in mock_data:
            return None
            
        # Generate signal based on asset type
        if asset_type == 'equity':
            carry = self.carry_calc.calculate_equity_carry(
                ticker,
                dividend_yield=mock_data[ticker]['div_yield'],
                buyback_yield=mock_data[ticker]['buyback']
            )
        elif asset_type == 'bond':
            carry = self.carry_calc.calculate_bond_carry(
                ticker,
                yield_to_maturity=mock_data[ticker]['yield']
            )
        elif asset_type == 'gold':
            carry = self.carry_calc.calculate_gold_carry(
                ticker,
                real_yield_10y=mock_data[ticker]['real_yield_10y']
            )
        else:
            return None
            
        return SignalResult(
            source_type='carry_arp',
            source_name=self.source_name,
            signal=carry.signal_score,
            confidence=carry.confidence,
            raw_score=carry.carry_yield,
            raw_unit='pct_annual',
            historical_accuracy=0.55,  # AQR research shows positive expected returns
            sample_count=120,
            metadata={
                'regime': carry.regime,
                'percentile_1y': carry.percentile_1y,
                'percentile_5y': carry.percentile_5y
            }
        )


class ValueSignalAdapter(SignalSource):
    """
    Adapter for Value signals to integrate with SignalIntegrator.
    v2.60 AQR-style value signal source.
    """
    
    def __init__(self):
        self.value_calc = ValueCalculator()
        self.source_name = "aqr_value_premium"
        
    def generate_signal(self, ticker: str):
        """Generate value signal for integrator interface"""
        from dataclasses import dataclass
        
        @dataclass
        class SignalResult:
            source_type: str
            source_name: str
            signal: float
            confidence: float
            raw_score: float
            raw_unit: str
            historical_accuracy: float
            sample_count: int
            metadata: dict
            
        # Mock data for demonstration
        mock_data = {
            'SPY': {'pe': 22.5, 'pb': 4.1, 'div_yield': 0.013},
            'QQQ': {'pe': 28.3, 'pb': 5.8, 'div_yield': 0.006},
            'TLT': {'yield': 0.045, 'inflation': 0.025},
            'IEF': {'yield': 0.042, 'inflation': 0.025},
            'EFA': {'pe': 15.2, 'pb': 1.8, 'div_yield': 0.028},
            'VLUE': {'pe': 12.5, 'pb': 1.5, 'div_yield': 0.025}
        }
        
        if ticker not in mock_data:
            return None
            
        data = mock_data[ticker]
        
        # Generate signal based on asset type
        if 'pe' in data:  # Equity
            value = self.value_calc.calculate_equity_value(
                ticker,
                pe_ratio=data['pe'],
                pb_ratio=data['pb'],
                dividend_yield=data['div_yield']
            )
        elif 'yield' in data:  # Bond
            real_yield = data['yield'] - data.get('inflation', 0.025)
            value = self.value_calc.calculate_bond_value(
                ticker,
                real_yield=real_yield
            )
        else:
            return None
            
        return SignalResult(
            source_type='value_arp',
            source_name=self.source_name,
            signal=value.signal_score,
            confidence=value.confidence,
            raw_score=value.current_value,
            raw_unit='composite_value_score',
            historical_accuracy=0.52,  # AQR "Value and Momentum Everywhere" results
            sample_count=240,
            metadata={
                'regime': value.regime,
                'percentile_1y': value.percentile_1y,
                'percentile_5y': value.percentile_5y
            }
        )
