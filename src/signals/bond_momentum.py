"""
Bond Momentum Signal Module

Implements time-series momentum (TSMOM) for Treasury ETFs to provide
fixed-income tactical overlay. Research shows momentum works best for
short-duration bonds (SHY, IEF) vs long-duration (TLT).

Based on Phase 1 research (v3.30):
- 18-month formation optimal for TLT crisis detection
- 12-month formation for SHY/IEF allocation timing
- 6% vol target appropriate for bond volatility
- Long-only (no shorting) — appropriate for bonds
- 2-3% weight in ensemble voter

Author: Autonomous Agent
Version: v3.30
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, Tuple, Literal
from dataclasses import dataclass
from pathlib import Path
import json
from datetime import datetime, timedelta


@dataclass
class BondMomentumSignal:
    """Container for bond momentum signal output"""
    etf: str
    timestamp: datetime
    
    # Signal value (-1 to +1, but bond momentum is long-only so 0 to +1)
    signal: float
    
    # Position sizing (0 to 2x based on volatility targeting)
    position_size: float
    
    # Momentum metrics
    formation_return: float  # Trailing return over formation period
    realized_vol: float    # Annualized realized volatility
    
    # Metadata
    formation_months: int
    volatility_target: float
    confidence: str  # 'strong', 'moderate', 'weak' based on signal magnitude
    
    # Recommendation
    action: Literal['increase', 'hold', 'reduce', 'avoid']
    weight_delta: float  # Suggested allocation delta (-0.05 to +0.05)


class BondMomentumCalculator:
    """
    Calculates momentum signals for Treasury ETFs.
    
    Uses TSMOM-style approach adapted for fixed income:
    - Long-only signals (no shorting bonds)
    - Volatility-scaled position sizing
    - Multiple formation periods per ETF type
    """
    
    # Formation period recommendations from Phase 1 research
    DEFAULT_CONFIG = {
        'SHY': {'formation_months': 12, 'vol_target': 0.06},   # Short: strong momentum
        'IEF': {'formation_months': 12, 'vol_target': 0.06},  # Intermediate: moderate
        'TLT': {'formation_months': 18, 'vol_target': 0.06},   # Long: 18m better than 12m
        'BIL': {'formation_months': 12, 'vol_target': 0.04},  # T-Bills: conservative
    }
    
    # Trading days per month (approximate)
    DAYS_PER_MONTH = 21
    
    def __init__(
        self,
        config: Optional[Dict[str, Dict]] = None,
        prices: Optional[pd.DataFrame] = None
    ):
        """
        Initialize calculator with configuration and price data.
        
        Args:
            config: ETF-specific configuration overrides
            prices: DataFrame with columns [SHY, IEF, TLT, BIL] indexed by date
        """
        self.config = config or self.DEFAULT_CONFIG.copy()
        self.prices = prices
        self.last_updated: Optional[datetime] = None
        
    def load_prices_from_file(self, data_path: Optional[Path] = None) -> bool:
        """
        Load Treasury ETF prices from prices.json.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            if data_path is None:
                data_path = Path(__file__).parent.parent.parent / "public" / "data" / "prices.json"
            
            with open(data_path) as f:
                data = json.load(f)
            
            treasury_etfs = ['TLT', 'IEF', 'SHY', 'BIL']
            records = []
            
            for etf in treasury_etfs:
                if etf in data:
                    for entry in data[etf]:
                        records.append({
                            'date': entry['d'],
                            'etf': etf,
                            'price': entry['p']
                        })
            
            df = pd.DataFrame(records)
            df['date'] = pd.to_datetime(df['date'])
            df = df.pivot(index='date', columns='etf', values='price')
            
            self.prices = df
            self.last_updated = datetime.now()
            return True
            
        except Exception as e:
            print(f"Error loading prices: {e}")
            return False
    
    def calculate_momentum(
        self,
        etf: str,
        formation_months: Optional[int] = None,
        skip_months: int = 1,
        volatility_target: Optional[float] = None
    ) -> Optional[BondMomentumSignal]:
        """
        Calculate momentum signal for a single Treasury ETF.
        
        Args:
            etf: ETF symbol (SHY, IEF, TLT, BIL)
            formation_months: Lookback period (uses config default if None)
            skip_months: Skip most recent N months (default 1)
            volatility_target: Target volatility (uses config default if None)
            
        Returns:
            BondMomentumSignal with signal value and metadata
        """
        if self.prices is None or etf not in self.prices.columns:
            return None
        
        # Use defaults from config
        etf_config = self.config.get(etf, self.DEFAULT_CONFIG.get(etf, {}))
        formation_months = formation_months or etf_config.get('formation_months', 12)
        volatility_target = volatility_target or etf_config.get('vol_target', 0.06)
        
        prices = self.prices[etf].dropna()
        if len(prices) < formation_months * self.DAYS_PER_MONTH + 10:
            return None  # Insufficient data
        
        # Calculate formation period return (skip most recent month)
        formation_days = formation_months * self.DAYS_PER_MONTH
        skip_days = skip_months * self.DAYS_PER_MONTH
        
        current_price = prices.iloc[-1]
        formation_start_price = prices.iloc[-(formation_days + skip_days)]
        
        formation_return = (current_price / formation_start_price) - 1
        
        # Calculate realized volatility (63-day = ~3 months)
        returns = prices.pct_change().dropna()
        realized_vol = returns.iloc[-63:].std() * np.sqrt(252)
        
        # Momentum signal: +1 for positive momentum, 0 otherwise (long-only)
        raw_signal = 1.0 if formation_return > 0 else 0.0
        
        # Volatility-scaled position sizing
        # Position = target_vol / realized_vol, capped at 2x
        if realized_vol > 0:
            position_size = volatility_target / realized_vol
            position_size = min(position_size, 2.0)  # Max 2x leverage
        else:
            position_size = 1.0
        
        # Final signal (0 to 2.0 range due to position sizing)
        final_signal = raw_signal * position_size
        
        # Confidence level
        if abs(formation_return) > 0.10:  # >10% return
            confidence = 'strong'
        elif abs(formation_return) > 0.05:  # >5% return
            confidence = 'moderate'
        else:
            confidence = 'weak'
        
        # Action recommendation
        if etf == 'TLT':
            # TLT: use as crisis indicator (reduce on negative momentum)
            if formation_return < -0.05:
                action = 'reduce'
                weight_delta = -0.03
            elif formation_return > 0.05:
                action = 'hold'
                weight_delta = 0.0
            else:
                action = 'avoid'
                weight_delta = -0.02
        else:
            # SHY, IEF: tactical allocation
            if formation_return > 0.08:
                action = 'increase'
                weight_delta = 0.03
            elif formation_return > 0.03:
                action = 'hold'
                weight_delta = 0.0
            elif formation_return > 0:
                action = 'reduce'
                weight_delta = -0.02
            else:
                action = 'avoid'
                weight_delta = -0.03
        
        return BondMomentumSignal(
            etf=etf,
            timestamp=datetime.now(),
            signal=final_signal,
            position_size=position_size,
            formation_return=formation_return,
            realized_vol=realized_vol,
            formation_months=formation_months,
            volatility_target=volatility_target,
            confidence=confidence,
            action=action,
            weight_delta=weight_delta
        )
    
    def calculate_all(
        self,
        etfs: Optional[list] = None
    ) -> Dict[str, BondMomentumSignal]:
        """
        Calculate momentum signals for all configured ETFs.
        
        Args:
            etfs: List of ETFs to calculate (default: all in config)
            
        Returns:
            Dict mapping ETF symbol to BondMomentumSignal
        """
        etfs = etfs or list(self.config.keys())
        results = {}
        
        for etf in etfs:
            signal = self.calculate_momentum(etf)
            if signal:
                results[etf] = signal
        
        return results
    
    def get_ensemble_recommendation(
        self,
        current_allocation: Optional[Dict[str, float]] = None
    ) -> Dict:
        """
        Generate ensemble-ready recommendation for fixed income allocation.
        
        Returns a signal compatible with the ensemble voter at 2-3% weight.
        
        Args:
            current_allocation: Current bond allocations (e.g., {'SHY': 0.10, 'TLT': 0.16})
            
        Returns:
            Dict with ensemble signal format
        """
        signals = self.calculate_all()
        
        if not signals:
            return {
                'timestamp': datetime.now().isoformat(),
                'signal_value': 0.0,
                'confidence': 'none',
                'recommendation': 'neutral',
                'details': {}
            }
        
        # Aggregate signals
        # Weight by ETF importance in typical portfolio
        weights = {'SHY': 0.3, 'IEF': 0.3, 'TLT': 0.3, 'BIL': 0.1}
        
        weighted_signal = 0.0
        signal_count = 0
        
        details = {}
        for etf, signal in signals.items():
            weight = weights.get(etf, 0.25)
            # Normalize signal to -1 to +1 range for ensemble
            # (our signal is 0 to 2, so subtract 1)
            normalized = signal.signal - 1.0
            weighted_signal += normalized * weight
            signal_count += 1
            
            details[etf] = {
                'signal': round(signal.signal, 3),
                'formation_return': round(signal.formation_return, 4),
                'action': signal.action,
                'weight_delta': round(signal.weight_delta, 3)
            }
        
        # Final ensemble signal (-1 to +1)
        ensemble_signal = weighted_signal if signal_count > 0 else 0.0
        
        # Confidence based on signal dispersion
        if abs(ensemble_signal) > 0.5:
            confidence = 'high'
        elif abs(ensemble_signal) > 0.2:
            confidence = 'moderate'
        else:
            confidence = 'low'
        
        # Recommendation
        if ensemble_signal > 0.3:
            recommendation = 'overweight_bonds'
        elif ensemble_signal < -0.3:
            recommendation = 'underweight_bonds'
        else:
            recommendation = 'neutral'
        
        return {
            'timestamp': datetime.now().isoformat(),
            'signal_value': round(ensemble_signal, 3),
            'confidence': confidence,
            'recommendation': recommendation,
            'weight_recommendation': 0.025,  # 2.5% ensemble weight
            'details': details,
            'source': 'bond_momentum'
        }


def get_bond_momentum_status() -> Dict:
    """
    Get current bond momentum status for dashboard/API.
    
    Returns:
        Dict with current signals and recommendations
    """
    calc = BondMomentumCalculator()
    
    if not calc.load_prices_from_file():
        return {
            'status': 'error',
            'message': 'Failed to load price data',
            'timestamp': datetime.now().isoformat()
        }
    
    signals = calc.calculate_all()
    ensemble = calc.get_ensemble_recommendation()
    
    return {
        'status': 'ok',
        'timestamp': datetime.now().isoformat(),
        'last_data_date': str(calc.prices.index[-1])[:10] if calc.prices is not None else None,
        'individual_signals': {
            etf: {
                'signal': round(s.signal, 3),
                'formation_return': round(s.formation_return * 100, 2),  # as %
                'realized_vol': round(s.realized_vol * 100, 2),  # as %
                'position_size': round(s.position_size, 2),
                'confidence': s.confidence,
                'action': s.action,
                'formation_months': s.formation_months
            }
            for etf, s in signals.items()
        },
        'ensemble': ensemble
    }


if __name__ == '__main__':
    # CLI for testing
    import argparse
    
    parser = argparse.ArgumentParser(description='Bond Momentum Signal')
    parser.add_argument('--etf', type=str, help='Specific ETF to analyze')
    parser.add_argument('--all', action='store_true', help='Analyze all ETFs')
    parser.add_argument('--json', action='store_true', help='Output as JSON')
    
    args = parser.parse_args()
    
    calc = BondMomentumCalculator()
    if not calc.load_prices_from_file():
        print("Error: Could not load price data")
        exit(1)
    
    if args.json:
        import json
        status = get_bond_momentum_status()
        print(json.dumps(status, indent=2))
    elif args.etf:
        signal = calc.calculate_momentum(args.etf.upper())
        if signal:
            print(f"\n{signal.etf} Bond Momentum Signal")
            print("=" * 40)
            print(f"Signal Value: {signal.signal:.3f}")
            print(f"Formation Return ({signal.formation_months}m): {signal.formation_return:.2%}")
            print(f"Realized Volatility: {signal.realized_vol:.2%}")
            print(f"Position Size: {signal.position_size:.2f}x")
            print(f"Confidence: {signal.confidence}")
            print(f"Action: {signal.action}")
            print(f"Weight Delta: {signal.weight_delta:+.1%}")
        else:
            print(f"Error: Could not calculate signal for {args.etf}")
    else:
        # Default: show all
        status = get_bond_momentum_status()
        print("\nBond Momentum Signal Status")
        print("=" * 50)
        print(f"Last Data: {status['last_data_date']}")
        print(f"Timestamp: {status['timestamp']}")
        print()
        
        for etf, data in status['individual_signals'].items():
            print(f"\n{etf}:")
            print(f"  Signal: {data['signal']:.3f}")
            print(f"  Formation Return: {data['formation_return']:+.2f}%")
            print(f"  Action: {data['action']}")
        
        print(f"\nEnsemble Recommendation: {status['ensemble']['recommendation']}")
        print(f"Ensemble Signal: {status['ensemble']['signal_value']:+.3f}")
        print(f"Confidence: {status['ensemble']['confidence']}")
