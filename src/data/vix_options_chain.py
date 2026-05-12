"""
VIX Options Chain Fetcher
Fetches VIX options data from CBOE/Yahoo Finance for insurance overlay strategy.

Part of v2.44: VIX Call Spread Insurance Overlay
"""

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class VIXOption:
    """Represents a VIX option contract"""
    strike: float
    expiration: datetime
    option_type: str  # 'call' or 'put'
    bid: float
    ask: float
    last_price: float
    volume: int
    open_interest: int
    implied_vol: float
    delta: Optional[float] = None
    
    @property
    def mid_price(self) -> float:
        return (self.bid + self.ask) / 2
    
    @property
    def days_to_expiration(self) -> int:
        return (self.expiration - datetime.now()).days


@dataclass
class VIXOptionChain:
    """Complete VIX options chain for a given date"""
    timestamp: datetime
    spot_vix: float
    front_month_expiry: datetime
    calls: List[VIXOption]
    puts: List[VIXOption]
    
    def get_calls_by_delta(self, target_delta: float, tolerance: float = 0.05) -> List[VIXOption]:
        """Get calls closest to target delta"""
        if not self.calls:
            return []
        
        # Sort by delta distance from target
        calls_with_delta = [c for c in self.calls if c.delta is not None]
        if not calls_with_delta:
            # Fallback to strike-based approximation
            atm_strike = min(self.calls, key=lambda x: abs(x.strike - self.spot_vix))
            otm_calls = [c for c in self.calls if c.strike > self.spot_vix]
            if not otm_calls:
                return []
            
            # Approximate 30-delta is ~10% OTM for 60-day options
            target_strike = self.spot_vix * 1.10
            closest = min(otm_calls, key=lambda x: abs(x.strike - target_strike))
            return [closest]
        
        calls_with_delta.sort(key=lambda x: abs(x.delta - target_delta))
        return [c for c in calls_with_delta if abs(c.delta - target_delta) <= tolerance][:3]
    
    def get_atm_call(self) -> Optional[VIXOption]:
        """Get at-the-money call"""
        if not self.calls:
            return None
        return min(self.calls, key=lambda x: abs(x.strike - self.spot_vix))


class VIXOptionsFetcher:
    """
    Fetches and stores VIX options chain data.
    Uses Yahoo Finance as primary source with CBOE fallback.
    """
    
    def __init__(self, data_dir: str = "data/vix_options"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # VIX futures months
        self.futures_months = ['F', 'G', 'H', 'J', 'K', 'M', 'N', 'Q', 'U', 'V', 'X', 'Z']
        self.month_map = {
            'F': 1, 'G': 2, 'H': 3, 'J': 4, 'K': 5, 'M': 6,
            'N': 7, 'Q': 8, 'U': 9, 'V': 10, 'X': 11, 'Z': 12
        }
    
    def fetch_spot_vix(self) -> Optional[float]:
        """Fetch current VIX spot price from Yahoo Finance"""
        try:
            import yfinance as yf
            vix = yf.Ticker("^VIX")
            hist = vix.history(period="1d")
            if not hist.empty:
                return float(hist['Close'].iloc[-1])
        except Exception as e:
            logger.warning(f"Failed to fetch VIX spot: {e}")
        return None
    
    def fetch_option_chain(self) -> Optional[VIXOptionChain]:
        """
        Fetch VIX options chain.
        Note: VIX options trade under ticker symbol VIX (not ^VIX)
        """
        try:
            import yfinance as yf
            
            # VIX options trade on CBOE under symbol 'VIX'
            vix_options = yf.Ticker("VIX")
            
            # Get options expiration dates
            expirations = vix_options.options
            if not expirations or len(expirations) < 2:
                logger.warning("No VIX options expirations available")
                return None
            
            # Get chain for 2nd expiration (avoiding front week gamma)
            # Standard VIX options expire on Wednesday 30 days before S&P expiration
            target_expiry = expirations[min(1, len(expirations)-1)]
            
            chain = vix_options.option_chain(target_expiry)
            
            spot_vix = self.fetch_spot_vix()
            if spot_vix is None:
                # Use ATM strike as proxy for spot
                spot_vix = chain.calls['strike'].median()
            
            # Parse calls
            calls = []
            for _, row in chain.calls.iterrows():
                call = VIXOption(
                    strike=float(row['strike']),
                    expiration=datetime.strptime(target_expiry, '%Y-%m-%d'),
                    option_type='call',
                    bid=float(row['bid']) if 'bid' in row else 0,
                    ask=float(row['ask']) if 'ask' in row else 0,
                    last_price=float(row['lastPrice']),
                    volume=int(row['volume']) if 'volume' in row else 0,
                    open_interest=int(row['openInterest']) if 'openInterest' in row else 0,
                    implied_vol=float(row['impliedVolatility']) if 'impliedVolatility' in row else 0,
                    delta=float(row['delta']) if 'delta' in row else None
                )
                calls.append(call)
            
            # Parse puts (for completeness)
            puts = []
            for _, row in chain.puts.iterrows():
                put = VIXOption(
                    strike=float(row['strike']),
                    expiration=datetime.strptime(target_expiry, '%Y-%m-%d'),
                    option_type='put',
                    bid=float(row['bid']) if 'bid' in row else 0,
                    ask=float(row['ask']) if 'ask' in row else 0,
                    last_price=float(row['lastPrice']),
                    volume=int(row['volume']) if 'volume' in row else 0,
                    open_interest=int(row['openInterest']) if 'openInterest' in row else 0,
                    implied_vol=float(row['impliedVolatility']) if 'impliedVolatility' in row else 0,
                    delta=float(row['delta']) if 'delta' in row else None
                )
                puts.append(put)
            
            # Determine front month expiry
            expiry_dt = datetime.strptime(target_expiry, '%Y-%m-%d')
            
            return VIXOptionChain(
                timestamp=datetime.now(),
                spot_vix=spot_vix,
                front_month_expiry=expiry_dt,
                calls=calls,
                puts=puts
            )
            
        except Exception as e:
            logger.error(f"Failed to fetch VIX option chain: {e}")
            return None
    
    def save_chain(self, chain: VIXOptionChain) -> str:
        """Save option chain to JSON"""
        timestamp_str = chain.timestamp.strftime('%Y%m%d_%H%M%S')
        filepath = self.data_dir / f"vix_chain_{timestamp_str}.json"
        
        data = {
            'timestamp': chain.timestamp.isoformat(),
            'spot_vix': chain.spot_vix,
            'front_month_expiry': chain.front_month_expiry.isoformat(),
            'calls': [
                {
                    'strike': c.strike,
                    'expiration': c.expiration.isoformat(),
                    'option_type': c.option_type,
                    'bid': c.bid,
                    'ask': c.ask,
                    'last_price': c.last_price,
                    'volume': c.volume,
                    'open_interest': c.open_interest,
                    'implied_vol': c.implied_vol,
                    'delta': c.delta,
                    'mid_price': c.mid_price,
                    'days_to_expiration': c.days_to_expiration
                }
                for c in chain.calls
            ],
            'puts': [
                {
                    'strike': p.strike,
                    'expiration': p.expiration.isoformat(),
                    'option_type': p.option_type,
                    'bid': p.bid,
                    'ask': p.ask,
                    'last_price': p.last_price,
                    'volume': p.volume,
                    'open_interest': p.open_interest,
                    'implied_vol': p.implied_vol,
                    'delta': p.delta,
                    'mid_price': p.mid_price,
                    'days_to_expiration': p.days_to_expiration
                }
                for p in chain.puts
            ]
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Saved VIX option chain to {filepath}")
        return str(filepath)
    
    def get_insurance_candidate(self, chain: VIXOptionChain, 
                                 target_delta: float = 0.30,
                                 max_premium_pct: float = 0.01) -> Optional[Dict]:
        """
        Get best VIX call option for insurance overlay.
        
        Args:
            chain: VIX option chain
            target_delta: Target delta (0.30 = 30-delta)
            max_premium_pct: Max premium as % of portfolio (default 1%)
        
        Returns:
            Dict with option details or None if no suitable option found
        """
        if chain.spot_vix >= 22:
            logger.info(f"VIX {chain.spot_vix:.2f} too high for insurance entry")
            return None
        
        candidates = chain.get_calls_by_delta(target_delta, tolerance=0.10)
        if not candidates:
            logger.warning("No options found near target delta")
            return None
        
        # Select best candidate (liquidity + cost)
        best = max(candidates, key=lambda x: x.volume * x.open_interest)
        
        # Check if reasonable premium
        premium = best.mid_price
        
        # VIX options are $100 multiplier (but quoted in VIX points)
        notional_per_contract = 100 * premium
        
        # Calculate how many contracts for 1% portfolio allocation
        # Assuming $100K portfolio, 1% = $1,000 premium budget
        portfolio_value = 100000  # Default assumption
        budget = portfolio_value * max_premium_pct
        contracts = int(budget / notional_per_contract)
        
        if contracts < 1:
            logger.warning(f"Premium too high: ${notional_per_contract:.2f} per contract")
            return None
        
        return {
            'spot_vix': chain.spot_vix,
            'strike': best.strike,
            'expiration': best.expiration.isoformat(),
            'days_to_expiration': best.days_to_expiration,
            'premium': premium,
            'delta': best.delta,
            'implied_vol': best.implied_vol,
            'volume': best.volume,
            'open_interest': best.open_interest,
            'contracts': contracts,
            'total_premium': contracts * notional_per_contract,
            'breakeven': best.strike + premium,
            'target_delta': target_delta
        }
    
    def run(self) -> Optional[Dict]:
        """Main entry point - fetch chain and get insurance candidate"""
        chain = self.fetch_option_chain()
        if chain is None:
            return None
        
        # Save raw data
        self.save_chain(chain)
        
        # Get insurance candidate
        candidate = self.get_insurance_candidate(chain)
        
        if candidate:
            # Save signal
            signal_file = self.data_dir / 'latest_insurance_signal.json'
            with open(signal_file, 'w') as f:
                json.dump({
                    'timestamp': datetime.now().isoformat(),
                    'signal_type': 'vix_insurance',
                    'vix_level': chain.spot_vix,
                    'recommendation': 'ENTER' if chain.spot_vix < 20 else 'HOLD',
                    'candidate': candidate
                }, f, indent=2)
            
            logger.info(f"VIX Insurance Signal: {candidate['recommendation']} at VIX={chain.spot_vix:.2f}")
        
        return candidate


def main():
    """CLI entry point"""
    fetcher = VIXOptionsFetcher()
    result = fetcher.run()
    
    if result:
        print(f"\n{'='*60}")
        print(f"VIX INSURANCE CANDIDATE")
        print(f"{'='*60}")
        print(f"Spot VIX: {result['spot_vix']:.2f}")
        print(f"Strike: {result['strike']:.2f}")
        print(f"Expiration: {result['expiration']}")
        print(f"DTE: {result['days_to_expiration']}")
        print(f"Premium: ${result['premium']:.2f}")
        print(f"Delta: {result['delta']:.3f}" if result['delta'] else "Delta: N/A")
        print(f"Contracts: {result['contracts']}")
        print(f"Total Cost: ${result['total_premium']:.2f}")
        print(f"Breakeven VIX: {result['breakeven']:.2f}")
        print(f"{'='*60}")
    else:
        print("No insurance candidate available (VIX too high or data unavailable)")


if __name__ == '__main__':
    main()
