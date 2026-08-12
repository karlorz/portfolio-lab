"""
VIX Futures Data Infrastructure
Fetches and manages VIX futures term structure data for convexity harvesting strategies.

Based on CBOE VIX futures methodology and multi-asset volatility parity research (v2.21).
"""

import json
import logging
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple

from src.paths import DATA_DIR
from src.backtest.metrics import save_results_json

logger = logging.getLogger(__name__)


@dataclass
class VIXTermStructure:
    """VIX futures term structure snapshot"""
    date: str
    vix_spot: float
    front_month: float      # VX1 - near-term
    second_month: float     # VX2 - next-term
    third_month: float      # VX3
    contango_1m_2m: float    # (VX2/VX1 - 1) * 100
    contango_spot_1m: float # (VX1/VIX - 1) * 100
    is_contango: bool
    days_to_expiry_front: int
    
    def to_dict(self) -> Dict:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'VIXTermStructure':
        # Filter out unexpected keys to maintain backward compatibility
        valid_fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered_data = {k: v for k, v in data.items() if k in valid_fields}
        front_month = float(filtered_data["front_month"])
        if filtered_data.get("second_month") is None:
            contango_1m_2m = filtered_data.get("contango_1m_2m")
            if contango_1m_2m is not None:
                filtered_data["second_month"] = front_month * (1 + float(contango_1m_2m) / 100)
            elif filtered_data.get("third_month") is not None:
                filtered_data["second_month"] = float(filtered_data["third_month"])
            else:
                filtered_data["second_month"] = front_month
        if filtered_data.get("third_month") is None:
            filtered_data["third_month"] = filtered_data["second_month"]
        if filtered_data.get("days_to_expiry_front") is None:
            filtered_data["days_to_expiry_front"] = 0

        # market.db / VIX3M proxy rows often omit derived contango fields —
        # compute them so cache load does not skip every bar (empty self.data
        # forced convexity_harvest status=unavailable and sticky kill).
        vix_spot = float(filtered_data["vix_spot"])
        second_month = float(filtered_data["second_month"])
        third_month = float(filtered_data["third_month"])
        if filtered_data.get("contango_1m_2m") is None:
            if front_month:
                filtered_data["contango_1m_2m"] = (second_month / front_month - 1.0) * 100.0
            else:
                filtered_data["contango_1m_2m"] = 0.0
        if filtered_data.get("contango_spot_1m") is None:
            if vix_spot:
                filtered_data["contango_spot_1m"] = (front_month / vix_spot - 1.0) * 100.0
            else:
                filtered_data["contango_spot_1m"] = 0.0
        if filtered_data.get("is_contango") is None:
            filtered_data["is_contango"] = float(filtered_data["contango_spot_1m"]) >= 0.0

        for field_name in (
            "vix_spot",
            "front_month",
            "second_month",
            "third_month",
            "contango_1m_2m",
            "contango_spot_1m",
        ):
            filtered_data[field_name] = float(filtered_data[field_name])
        filtered_data["days_to_expiry_front"] = int(filtered_data["days_to_expiry_front"])
        filtered_data["is_contango"] = bool(filtered_data["is_contango"])
        return cls(**filtered_data)


class VIXDataManager:
    """
    Manages VIX futures term structure data.

    For production: Connects to CBOE or data provider API
    For backtesting: Uses historical simulation or proxy data
    """

    DATA_DIR = DATA_DIR
    VIX_FILE = DATA_DIR / 'vix_term_structure.json'
    
    def __init__(self):
        self.data: Dict[str, VIXTermStructure] = {}
        self._ensure_data_dir()
        self._load_cached_data()
    
    def _ensure_data_dir(self):
        """Ensure data directory exists"""
        self.DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    def _load_cached_data(self):
        """Load cached VIX data if available"""
        if self.VIX_FILE.exists():
            try:
                with open(self.VIX_FILE, 'r') as f:
                    raw_data = json.load(f)
                if not isinstance(raw_data, dict):
                    raise ValueError("VIX cache must contain a date-to-record object")
                loaded = {}
                skipped = []
                for date, ts in raw_data.items():
                    # Batch BV: skip provenance meta / non-date keys
                    if str(date).startswith("_") or str(date) in {"meta", "schema"}:
                        continue
                    if not isinstance(ts, dict):
                        skipped.append((date, "record is not an object"))
                        continue
                    # Date-like keys only (YYYY-MM-DD)
                    if len(str(date)) < 10 or str(date)[4:5] != "-":
                        skipped.append((date, "key is not a calendar date"))
                        continue
                    try:
                        loaded[date] = VIXTermStructure.from_dict(ts)
                    except (KeyError, ValueError, TypeError) as e:
                        skipped.append((date, str(e)))
                self.data = loaded
                logger.info("Loaded %d VIX term structure records", len(self.data))
                if skipped:
                    first_date, first_error = skipped[0]
                    logger.warning(
                        "Skipped %d invalid VIX cache rows; first invalid row %s: %s",
                        len(skipped),
                        first_date,
                        first_error,
                    )
            except (OSError, json.JSONDecodeError, KeyError, ValueError, TypeError) as e:
                logger.warning("Error loading VIX cache: %s", e)
    
    def _save_cached_data(self):
        """Save VIX data to cache"""
        try:
            save_results_json({date: ts.to_dict() for date, ts in self.data.items()}, output_path=str(self.VIX_FILE))
        except (OSError, TypeError) as e:
            logger.warning("Error saving VIX cache: %s", e)
    
    def generate_historical_proxy(
        self,
        start_date: str = '2006-01-01',
        end_date: str = None
    ) -> List[VIXTermStructure]:
        """
        Generate historical VIX term structure proxy data.
        
        Uses historical VIX levels and typical term structure patterns:
        - Contango (80% of time): VX1 > VIX spot by 5-15%
        - Backwardation (20% of time): VX1 < VIX spot during stress
        
        For production, replace with actual CBOE VIX futures data.
        """
        if end_date is None:
            end_date = datetime.now().strftime('%Y-%m-%d')
        
        # Historical VIX levels by year (approximate monthly averages)
        historical_vix = {
            '2006': 12.8, '2007': 17.5, '2008': 32.7, '2009': 31.5,
            '2010': 22.5, '2011': 24.4, '2012': 17.8, '2013': 14.2,
            '2014': 14.2, '2015': 17.0, '2016': 15.8, '2017': 11.1,
            '2018': 16.6, '2019': 15.4, '2020': 29.3, '2021': 19.7,
            '2022': 25.6, '2023': 17.1, '2024': 14.5, '2025': 18.0, '2026': 20.0,
        }
        
        results = []
        start = datetime.strptime(start_date, '%Y-%m-%d')
        end = datetime.strptime(end_date, '%Y-%m-%d')
        
        current = start
        while current <= end:
            date_str = current.strftime('%Y-%m-%d')
            year = str(current.year)
            
            # Base VIX level for this year
            base_vix = historical_vix.get(year, 20.0)
            
            # Add seasonal and random variation
            # Higher vol in Oct-Dec (year-end), lower in summer
            month = current.month
            seasonal_factor = 1.2 if month in [10, 11, 12] else 0.9 if month in [6, 7, 8] else 1.0
            
            # Random daily variation
            daily_noise = (hash(date_str) % 100 - 50) / 200  # -0.25 to +0.25
            
            vix_spot = base_vix * seasonal_factor * (1 + daily_noise)
            
            # Generate term structure based on VIX level
            # Higher VIX = more likely backwardation
            stress_prob = min(1.0, vix_spot / 30)  # 30+ VIX = likely backwardation
            is_backwardation = (hash(date_str + 'stress') % 100) / 100 < stress_prob
            
            if is_backwardation:
                # Backwardation: futures < spot
                front_month = vix_spot * (0.85 + (hash(date_str + 'front') % 15) / 100)
                second_month = front_month * (0.90 + (hash(date_str + 'second') % 10) / 100)
                third_month = second_month * (0.92 + (hash(date_str + 'third') % 8) / 100)
            else:
                # Contango: futures > spot
                contango_front = 0.05 + (hash(date_str + 'cont') % 15) / 100  # 5-20%
                contango_second = 0.10 + (hash(date_str + 'cont2') % 15) / 100  # 10-25%
                contango_third = 0.15 + (hash(date_str + 'cont3') % 15) / 100   # 15-30%
                
                front_month = vix_spot * (1 + contango_front)
                second_month = vix_spot * (1 + contango_second)
                third_month = vix_spot * (1 + contango_third)
            
            # Calculate contango metrics
            contango_1m_2m = (second_month / front_month - 1) * 100
            contango_spot_1m = (front_month / vix_spot - 1) * 100
            
            # Days to expiry (simplified: assume monthly expiration ~21st)
            days_to_expiry = max(0, 21 - current.day)
            
            ts = VIXTermStructure(
                date=date_str,
                vix_spot=vix_spot,
                front_month=front_month,
                second_month=second_month,
                third_month=third_month,
                contango_1m_2m=contango_1m_2m,
                contango_spot_1m=contango_spot_1m,
                is_contango=not is_backwardation,
                days_to_expiry_front=days_to_expiry
            )
            
            results.append(ts)
            self.data[date_str] = ts
            
            current += timedelta(days=1)
        
        self._save_cached_data()
        return results
    
    def get_term_structure(self, date: str) -> Optional[VIXTermStructure]:
        """Get VIX term structure for a specific date"""
        return self.data.get(date)
    
    def get_contango_signal(self, date: str) -> Optional[Dict]:
        """
        Generate contango/backwardation signal for trading decisions.
        
        Returns:
            - signal: 'strong_contango', 'contango', 'flat', 'backwardation', 'strong_backwardation'
            - strength: 0-1 score
            - annualized_roll_yield: Expected roll yield if holding short position
        """
        ts = self.get_term_structure(date)
        if not ts:
            return None
        
        spot_1m = ts.contango_spot_1m
        
        # Classify signal
        if spot_1m > 10:
            signal = 'strong_contango'
            strength = min(1.0, spot_1m / 20)
        elif spot_1m > 5:
            signal = 'contango'
            strength = spot_1m / 10
        elif spot_1m > -2:
            signal = 'flat'
            strength = 0.3
        elif spot_1m > -8:
            signal = 'backwardation'
            strength = abs(spot_1m) / 10
        else:
            signal = 'strong_backwardation'
            strength = min(1.0, abs(spot_1m) / 15)
        
        # Annualized roll yield approximation
        # (Assume holding ~30 days, collecting the contango)
        days_per_month = 30
        periods_per_year = 365 / days_per_month
        annualized_roll_yield = spot_1m * periods_per_year if ts.is_contango else spot_1m * periods_per_year * 2
        
        return {
            'date': date,
            'signal': signal,
            'strength': strength,
            'contango_spot_1m': spot_1m,
            'contango_1m_2m': ts.contango_1m_2m,
            'is_contango': ts.is_contango,
            'annualized_roll_yield': annualized_roll_yield,
            'vix_level': ts.vix_spot
        }
    
    def get_data_range(self) -> Tuple[str, str]:
        """Get date range of available data"""
        if not self.data:
            return ('', '')
        dates = sorted(self.data.keys())
        return (dates[0], dates[-1])


def fetch_vix_futures_data(
    start_date: str = '2006-01-01',
    end_date: str = None,
    use_cache: bool = True
) -> List[VIXTermStructure]:
    """
    Main entry point to fetch VIX futures term structure data.
    
    For production: Implement actual CBOE API or data provider connection.
    For now: Uses historical proxy generation.
    """
    manager = VIXDataManager()
    
    # Check if we have cached data covering the range
    if use_cache and manager.data:
        cache_start, cache_end = manager.get_data_range()
        if cache_start <= start_date and cache_end >= (end_date or datetime.now().strftime('%Y-%m-%d')):
            logger.info("Using cached VIX data (%s to %s)", cache_start, cache_end)
            return [
                manager.data[d] for d in sorted(manager.data.keys())
                if start_date <= d <= (end_date or cache_end)
            ]

    logger.info("Generating VIX term structure proxy data (%s to %s)", start_date, end_date or 'now')
    return manager.generate_historical_proxy(start_date, end_date)


if __name__ == '__main__':
    # Test the data manager
    manager = VIXDataManager()
    
    # Generate data if not cached
    if not manager.data:
        data = manager.generate_historical_proxy('2020-01-01', '2024-12-31')
        logger.info("Generated %d VIX term structure records", len(data))

    # Test contango signals
    test_dates = ['2020-03-15', '2021-06-01', '2022-01-01', '2023-10-15']
    for date in test_dates:
        signal = manager.get_contango_signal(date)
        if signal:
            logger.info("%s: %s", date, signal['signal'])
            logger.info("  VIX: %.2f", signal['vix_level'])
            logger.info("  Contango (spot-1m): %.2f%%", signal['contango_spot_1m'])
            logger.info("  Annualized roll yield: %.1f%%", signal['annualized_roll_yield'])
