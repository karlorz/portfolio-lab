#!/usr/bin/env python3
"""
Portfolio-Lab v2.23: Alternative Data Module

Satellite imagery, credit card transactions, and supply chain data adapters
for alpha generation. Based on Q3 2026 research synthesis.

Sources:
- Satellite: RS Metrics-style parking lot occupancy → retail revenue prediction
- Credit Card: Consumer Edge-style spending trends → earnings surprise prediction  
- Supply Chain: Flexport-style shipping/container metrics → inventory/production signals

Usage:
    from src.data.alternative_data import AlternativeDataClient
    
    client = AlternativeDataClient()
    
    # Satellite signal for retail ticker
    signal = client.get_satellite_signal("AAPL", days=30)
    
    # Composite alternative data score
    composite = client.get_composite_signal(["AAPL", "AMZN", "TGT"])

CLI:
    python -m src.data.alternative_data fetch --ticker AAPL --source satellite
    python -m src.data.alternative_data composite --tickers AAPL,AMZN,TGT
    python -m src.data.alternative_data backtest --tickers AAPL,AMZN --start 2025-01-01
"""

import json
import sqlite3
import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Optional
from collections import defaultdict
import statistics

# ---------------------------------------------------------------------------
# Data Classes
# ---------------------------------------------------------------------------

@dataclass
class AlternativeDataSignal:
    """Single alternative data signal."""
    ticker: str
    source: str  # satellite, credit_card, supply_chain
    signal_type: str  # momentum, level, surprise, trend
    
    # Signal value: -1.0 (bearish) to +1.0 (bullish)
    score: float
    confidence: float  # 0.0 to 1.0
    
    # Raw metrics (source-specific)
    raw_value: float
    raw_unit: str  # e.g., "pct_change", "index", "count"
    period_days: int
    
    # Historical context
    z_score: float  # vs 90-day history
    percentile: float  # 0-100
    trend_direction: str  # improving, deteriorating, stable
    
    # Metadata
    data_timestamp: str
    signal_generated: str = field(default_factory=lambda: datetime.now().isoformat())
    model_version: str = "v2.23.0"
    
    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class CompositeSignal:
    """Composite alternative data signal across sources."""
    ticker: str
    
    # Component signals
    satellite_score: Optional[float] = None
    credit_card_score: Optional[float] = None
    supply_chain_score: Optional[float] = None
    
    # Composite
    composite_score: float = 0.0  # Weighted average of available signals
    composite_confidence: float = 0.0
    
    # Attribution
    primary_driver: str = ""  # Which source has highest confidence
    signal_agreement: str = "neutral"  # aligned, mixed, conflicting
    
    # Historical performance
    historical_accuracy: Optional[float] = None  # Backtested prediction accuracy
    
    # Metadata
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class EarningsPrediction:
    """Earnings prediction based on alternative data."""
    ticker: str
    quarter: str  # Q4-2025 format
    
    # Revenue prediction
    predicted_revenue_growth: float  # YoY %
    revenue_surprise_probability: float  # 0-1
    revenue_direction: str  # beat, miss, inline
    
    # Confidence
    confidence: float
    primary_signals: list[str]  # Which alt data sources drove prediction
    
    # Historical accuracy for this ticker
    historical_accuracy: Optional[float] = None
    
    # Metadata
    prediction_date: str = field(default_factory=lambda: datetime.now().isoformat())
    earnings_date: Optional[str] = None
    
    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Database Setup
# ---------------------------------------------------------------------------

DATA_DIR = Path("~/projects/portfolio-lab/data").expanduser()
ALT_DATA_DB = DATA_DIR / "alternative_data.db"


def init_database():
    """Initialize SQLite database for alternative data storage."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    conn = sqlite3.connect(ALT_DATA_DB)
    cursor = conn.cursor()
    
    # Satellite data table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS satellite_data (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker TEXT NOT NULL,
            date TEXT NOT NULL,
            parking_occupancy_pct REAL,
            occupancy_vs_last_year_pct REAL,
            store_count INTEGER,
            data_quality_score REAL,
            source TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(ticker, date)
        )
    """)
    
    # Credit card data table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS credit_card_data (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker TEXT NOT NULL,
            date TEXT NOT NULL,
            spending_growth_yoy REAL,
            spending_growth_mom REAL,
            transaction_volume_index REAL,
            avg_ticket_size REAL,
            category_rank_pct REAL,
            data_quality_score REAL,
            source TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(ticker, date)
        )
    """)
    
    # Supply chain data table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS supply_chain_data (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker TEXT NOT NULL,
            date TEXT NOT NULL,
            container_throughput_index REAL,
            inventory_days_coverage REAL,
            supplier_lead_time_days REAL,
            shipping_cost_index REAL,
            data_quality_score REAL,
            source TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(ticker, date)
        )
    """)
    
    # Signal history table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS alt_data_signals (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker TEXT NOT NULL,
            source TEXT NOT NULL,
            signal_type TEXT NOT NULL,
            score REAL,
            confidence REAL,
            raw_value REAL,
            period_days INTEGER,
            z_score REAL,
            percentile REAL,
            trend_direction TEXT,
            data_timestamp TEXT,
            signal_generated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    # Prediction accuracy tracking
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS prediction_accuracy (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker TEXT NOT NULL,
            quarter TEXT NOT NULL,
            prediction_date TEXT,
            predicted_revenue_growth REAL,
            actual_revenue_growth REAL,
            prediction_error REAL,
            primary_signals TEXT,
            accuracy_score REAL,
            recorded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(ticker, quarter)
        )
    """)
    
    conn.commit()
    conn.close()


# ---------------------------------------------------------------------------
# Base Adapter
# ---------------------------------------------------------------------------

class AlternativeDataAdapter(ABC):
    """Base class for alternative data adapters."""
    
    def __init__(self, source_name: str, db_path: Path = ALT_DATA_DB):
        self.source_name = source_name
        self.db_path = db_path
        init_database()
    
    @abstractmethod
    def fetch_data(self, ticker: str, days: int = 90) -> list[dict]:
        """Fetch raw data for ticker."""
        pass
    
    @abstractmethod
    def calculate_signal(self, ticker: str, days: int = 30) -> AlternativeDataSignal:
        """Calculate signal from stored data."""
        pass
    
    def _get_db_connection(self) -> sqlite3.Connection:
        """Get database connection."""
        return sqlite3.connect(self.db_path)


# ---------------------------------------------------------------------------
# Satellite Data Adapter
# ---------------------------------------------------------------------------

class SatelliteDataAdapter(AlternativeDataAdapter):
    """
    Satellite parking lot occupancy → retail revenue prediction.
    
    Based on RS Metrics methodology:
    - Parking lot occupancy % vs same time last year
    - Leading indicator: 2-4 weeks ahead of earnings
    - Best for: Retail, restaurants, consumer discretionary
    """
    
    RETAIL_TICKERS = {
        "AAPL", "AMZN", "WMT", "TGT", "COST", "HD", "LOW", "NKE", "MCD", "SBUX",
        "TJX", "ROST", "DG", "DLTR", "BBY", "KSS", "JWN", "M", "GPS", "URBN",
    }
    
    def __init__(self):
        super().__init__("satellite")
    
    def fetch_data(self, ticker: str, days: int = 90) -> list[dict]:
        """
        Fetch satellite parking data for ticker.
        
        In production, this would call:
        - RS Metrics API
        - Orbital Insight API
        - Custom satellite imagery analysis
        
        For now, generates synthetic data based on ticker characteristics.
        """
        if ticker not in self.RETAIL_TICKERS:
            return []  # No satellite data for non-retail
        
        conn = self._get_db_connection()
        cursor = conn.cursor()
        
        # Check if we have recent data
        cursor.execute("""
            SELECT ticker, date, parking_occupancy_pct, occupancy_vs_last_year_pct,
                   store_count, data_quality_score, source
            FROM satellite_data 
            WHERE ticker = ? 
            AND date >= date('now', '-{} days')
            ORDER BY date DESC
        """.format(days), (ticker,))
        
        rows = cursor.fetchall()
        
        if not rows:
            # Generate synthetic data for testing
            # In production, this would fetch from API
            rows = self._generate_synthetic_data(ticker, days)
            self._store_data(rows)
        
        conn.close()
        
        return [{
            "ticker": row[0],
            "date": row[1],
            "parking_occupancy_pct": float(row[2]) if row[2] is not None else None,
            "occupancy_vs_last_year_pct": float(row[3]) if row[3] is not None else None,
            "store_count": int(row[4]) if row[4] is not None else None,
            "data_quality_score": float(row[5]) if row[5] is not None else 0.8,
            "source": row[6],
        } for row in rows]
    
    def _generate_synthetic_data(self, ticker: str, days: int) -> list[tuple]:
        """Generate synthetic satellite data for testing."""
        import random
        random.seed(hash(ticker) % 2**32)
        
        data = []
        base_occupancy = {
            "AAPL": 75.0, "AMZN": 60.0, "WMT": 80.0, "TGT": 70.0,
            "COST": 85.0, "HD": 72.0, "LOW": 68.0, "NKE": 65.0,
        }.get(ticker, 70.0)
        
        for i in range(days):
            date = (datetime.now() - timedelta(days=i)).strftime("%Y-%m-%d")
            
            # Simulate trend with noise
            trend = 0.001 * (days - i)  # Slight upward trend
            noise = random.gauss(0, 0.05)
            
            occupancy = base_occupancy * (1 + trend + noise)
            vs_last_year = trend * 100 + random.gauss(5, 10)  # Usually positive YoY
            
            data.append((
                ticker,
                date,
                max(0, min(100, occupancy)),
                vs_last_year,
                random.randint(100, 5000),  # Store count
                0.85,  # Data quality
                "synthetic"
            ))
        
        return data
    
    def _store_data(self, rows: list[tuple]):
        """Store data in database."""
        conn = self._get_db_connection()
        cursor = conn.cursor()
        
        cursor.executemany("""
            INSERT OR REPLACE INTO satellite_data 
            (ticker, date, parking_occupancy_pct, occupancy_vs_last_year_pct, 
             store_count, data_quality_score, source)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, rows)
        
        conn.commit()
        conn.close()
    
    def calculate_signal(self, ticker: str, days: int = 30) -> AlternativeDataSignal:
        """Calculate satellite-based revenue signal."""
        data = self.fetch_data(ticker, days=90)
        
        if not data:
            return AlternativeDataSignal(
                ticker=ticker,
                source="satellite",
                signal_type="revenue_momentum",
                score=0.0,
                confidence=0.0,
                raw_value=0.0,
                raw_unit="pct_change",
                period_days=days,
                z_score=0.0,
                percentile=50.0,
                trend_direction="insufficient_data",
                data_timestamp=datetime.now().isoformat(),
            )
        
        # Calculate metrics
        recent = data[:days]
        historical = data[days:90] if len(data) > days else data[-30:]
        
        recent_yoy = [d["occupancy_vs_last_year_pct"] for d in recent if d["occupancy_vs_last_year_pct"]]
        hist_yoy = [d["occupancy_vs_last_year_pct"] for d in historical if d["occupancy_vs_last_year_pct"]]
        
        if not recent_yoy or not hist_yoy:
            return AlternativeDataSignal(
                ticker=ticker,
                source="satellite",
                signal_type="revenue_momentum",
                score=0.0,
                confidence=0.3,
                raw_value=0.0,
                raw_unit="pct_change",
                period_days=days,
                z_score=0.0,
                percentile=50.0,
                trend_direction="insufficient_data",
                data_timestamp=datetime.now().isoformat(),
            )
        
        current_avg = statistics.mean(recent_yoy)
        hist_avg = statistics.mean(hist_yoy)
        hist_std = statistics.stdev(hist_yoy) if len(hist_yoy) > 1 else 1.0
        
        # Z-score
        z_score = (current_avg - hist_avg) / hist_std if hist_std > 0 else 0.0
        
        # Convert to signal (-1 to +1)
        # Parking occupancy YoY > 10% = strong bullish
        # Parking occupancy YoY < -5% = bearish
        if current_avg > 10:
            score = min(1.0, 0.5 + current_avg / 20)
        elif current_avg > 5:
            score = 0.3 + (current_avg - 5) / 25
        elif current_avg > 0:
            score = current_avg / 15
        elif current_avg > -5:
            score = current_avg / 10
        else:
            score = max(-1.0, -0.5 + (current_avg + 5) / 10)
        
        # Determine trend
        if z_score > 1.0:
            trend = "improving"
        elif z_score < -1.0:
            trend = "deteriorating"
        else:
            trend = "stable"
        
        # Calculate percentile
        all_values = recent_yoy + hist_yoy
        percentile = sum(1 for v in all_values if v <= current_avg) / len(all_values) * 100 if all_values else 50.0
        
        # Confidence based on data quality and sample size
        data_quality = statistics.mean([d.get("data_quality_score", 0.8) for d in recent])
        confidence = min(0.9, data_quality * (0.5 + len(recent_yoy) / 60))
        
        signal = AlternativeDataSignal(
            ticker=ticker,
            source="satellite",
            signal_type="revenue_momentum",
            score=round(score, 3),
            confidence=round(confidence, 3),
            raw_value=round(current_avg, 2),
            raw_unit="pct_change",
            period_days=days,
            z_score=round(z_score, 2),
            percentile=round(percentile, 1),
            trend_direction=trend,
            data_timestamp=data[0]["date"] if data else datetime.now().isoformat(),
        )
        
        # Store signal
        self._store_signal(signal)
        
        return signal
    
    def _store_signal(self, signal: AlternativeDataSignal):
        """Store signal in database."""
        conn = self._get_db_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO alt_data_signals 
            (ticker, source, signal_type, score, confidence, raw_value, 
             period_days, z_score, percentile, trend_direction, data_timestamp)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            signal.ticker, signal.source, signal.signal_type,
            signal.score, signal.confidence, signal.raw_value,
            signal.period_days, signal.z_score, signal.percentile,
            signal.trend_direction, signal.data_timestamp
        ))
        
        conn.commit()
        conn.close()


# ---------------------------------------------------------------------------
# Credit Card Data Adapter
# ---------------------------------------------------------------------------

class CreditCardAdapter(AlternativeDataAdapter):
    """
    Credit card transaction data → earnings surprise prediction.
    
    Based on Consumer Edge methodology:
    - YoY spending growth by merchant/ticker
    - Transaction volume and ticket size trends
    - 2-4 week lead time for quarterly earnings
    """
    
    CONSUMER_TICKERS = {
        "AAPL", "AMZN", "V", "MA", "PYPL", "SQ", "SHOP", "UBER", "LYFT",
        "WMT", "TGT", "COST", "HD", "LOW", "NKE", "MCD", "SBUX", "BKNG", "ABNB",
    }
    
    def __init__(self):
        super().__init__("credit_card")
    
    def fetch_data(self, ticker: str, days: int = 90) -> list[dict]:
        """Fetch credit card spending data."""
        if ticker not in self.CONSUMER_TICKERS:
            return []
        
        conn = self._get_db_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT ticker, date, spending_growth_yoy, spending_growth_mom,
                   transaction_volume_index, avg_ticket_size, category_rank_pct,
                   data_quality_score, source
            FROM credit_card_data 
            WHERE ticker = ? 
            AND date >= date('now', '-{} days')
            ORDER BY date DESC
        """.format(days), (ticker,))
        
        rows = cursor.fetchall()
        
        if not rows:
            rows = self._generate_synthetic_data(ticker, days)
            self._store_data(rows)
        
        conn.close()
        
        return [{
            "ticker": row[0],
            "date": row[1],
            "spending_growth_yoy": float(row[2]) if row[2] is not None else None,
            "spending_growth_mom": float(row[3]) if row[3] is not None else None,
            "transaction_volume_index": float(row[4]) if row[4] is not None else None,
            "avg_ticket_size": float(row[5]) if row[5] is not None else None,
            "category_rank_pct": float(row[6]) if row[6] is not None else None,
            "data_quality_score": float(row[7]) if row[7] is not None else 0.8,
            "source": row[8],
        } for row in rows]
    
    def _generate_synthetic_data(self, ticker: str, days: int) -> list[tuple]:
        """Generate synthetic credit card data."""
        import random
        random.seed(hash(ticker + "cc") % 2**32)
        
        data = []
        base_growth = {
            "AAPL": 8.0, "AMZN": 15.0, "V": 12.0, "MA": 10.0,
            "WMT": 5.0, "TGT": 3.0, "COST": 7.0, "NKE": 6.0,
        }.get(ticker, 5.0)
        
        for i in range(days):
            date = (datetime.now() - timedelta(days=i)).strftime("%Y-%m-%d")
            
            trend = random.gauss(0, 0.02)
            yoy = base_growth + trend * 100 + random.gauss(0, 3)
            mom = yoy / 12 + random.gauss(0, 1)
            
            data.append((
                ticker,
                date,
                yoy,
                mom,
                100.0 + random.gauss(0, 5),  # Volume index
                50.0 + random.gauss(0, 5),   # Ticket size
                random.uniform(60, 90),      # Category rank %
                0.82,  # Data quality
                "synthetic"
            ))
        
        return data
    
    def _store_data(self, rows: list[tuple]):
        """Store credit card data."""
        conn = self._get_db_connection()
        cursor = conn.cursor()
        
        cursor.executemany("""
            INSERT OR REPLACE INTO credit_card_data 
            (ticker, date, spending_growth_yoy, spending_growth_mom,
             transaction_volume_index, avg_ticket_size, category_rank_pct,
             data_quality_score, source)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, rows)
        
        conn.commit()
        conn.close()
    
    def calculate_signal(self, ticker: str, days: int = 30) -> AlternativeDataSignal:
        """Calculate credit card spending signal."""
        data = self.fetch_data(ticker, days=90)
        
        if not data:
            return AlternativeDataSignal(
                ticker=ticker,
                source="credit_card",
                signal_type="spending_momentum",
                score=0.0,
                confidence=0.0,
                raw_value=0.0,
                raw_unit="pct_change",
                period_days=days,
                z_score=0.0,
                percentile=50.0,
                trend_direction="insufficient_data",
                data_timestamp=datetime.now().isoformat(),
            )
        
        recent = data[:days]
        historical = data[days:90] if len(data) > days else data[-30:]
        
        recent_yoy = [d["spending_growth_yoy"] for d in recent if d["spending_growth_yoy"]]
        hist_yoy = [d["spending_growth_yoy"] for d in historical if d["spending_growth_yoy"]]
        
        if not recent_yoy:
            return AlternativeDataSignal(
                ticker=ticker,
                source="credit_card",
                signal_type="spending_momentum",
                score=0.0,
                confidence=0.3,
                raw_value=0.0,
                raw_unit="pct_change",
                period_days=days,
                z_score=0.0,
                percentile=50.0,
                trend_direction="insufficient_data",
                data_timestamp=datetime.now().isoformat(),
            )
        
        current_avg = statistics.mean(recent_yoy)
        hist_avg = statistics.mean(hist_yoy) if hist_yoy else current_avg
        hist_std = statistics.stdev(hist_yoy) if len(hist_yoy) > 1 else 5.0
        
        z_score = (current_avg - hist_avg) / hist_std if hist_std > 0 else 0.0
        
        # Spending growth signal mapping
        # > 15% YoY = very bullish
        # 10-15% = bullish
        # 5-10% = mildly bullish
        # 0-5% = neutral
        # < 0% = bearish
        if current_avg > 15:
            score = min(1.0, 0.6 + (current_avg - 15) / 25)
        elif current_avg > 10:
            score = 0.4 + (current_avg - 10) / 25
        elif current_avg > 5:
            score = 0.2 + (current_avg - 5) / 25
        elif current_avg > 0:
            score = current_avg / 25
        else:
            score = max(-1.0, current_avg / 10)
        
        if z_score > 1.0:
            trend = "improving"
        elif z_score < -1.0:
            trend = "deteriorating"
        else:
            trend = "stable"
        
        all_values = recent_yoy + hist_yoy
        percentile = sum(1 for v in all_values if v <= current_avg) / len(all_values) * 100 if all_values else 50.0
        
        data_quality = statistics.mean([d.get("data_quality_score", 0.8) for d in recent])
        confidence = min(0.85, data_quality * (0.5 + len(recent_yoy) / 60))
        
        signal = AlternativeDataSignal(
            ticker=ticker,
            source="credit_card",
            signal_type="spending_momentum",
            score=round(score, 3),
            confidence=round(confidence, 3),
            raw_value=round(current_avg, 2),
            raw_unit="pct_change",
            period_days=days,
            z_score=round(z_score, 2),
            percentile=round(percentile, 1),
            trend_direction=trend,
            data_timestamp=data[0]["date"] if data else datetime.now().isoformat(),
        )
        
        self._store_signal(signal)
        return signal
    
    def _store_signal(self, signal: AlternativeDataSignal):
        """Store signal."""
        conn = self._get_db_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO alt_data_signals 
            (ticker, source, signal_type, score, confidence, raw_value, 
             period_days, z_score, percentile, trend_direction, data_timestamp)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            signal.ticker, signal.source, signal.signal_type,
            signal.score, signal.confidence, signal.raw_value,
            signal.period_days, signal.z_score, signal.percentile,
            signal.trend_direction, signal.data_timestamp
        ))
        
        conn.commit()
        conn.close()


# ---------------------------------------------------------------------------
# Supply Chain Adapter
# ---------------------------------------------------------------------------

class SupplyChainAdapter(AlternativeDataAdapter):
    """
    Supply chain metrics → inventory/production signals.
    
    Based on Flexport methodology:
    - Container throughput vs historical norms
    - Inventory days coverage trends
    - Supplier lead times
    """
    
    SUPPLY_CHAIN_TICKERS = {
        "AAPL", "AMZN", "WMT", "HD", "LOW", "TGT", "COST", "NKE",
        "CAT", "DE", "GE", "HON", "UPS", "FDX", "CSX", "UNP",
    }
    
    def __init__(self):
        super().__init__("supply_chain")
    
    def fetch_data(self, ticker: str, days: int = 90) -> list[dict]:
        """Fetch supply chain data."""
        if ticker not in self.SUPPLY_CHAIN_TICKERS:
            return []
        
        conn = self._get_db_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT ticker, date, container_throughput_index, inventory_days_coverage,
                   supplier_lead_time_days, shipping_cost_index, data_quality_score, source
            FROM supply_chain_data 
            WHERE ticker = ? 
            AND date >= date('now', '-{} days')
            ORDER BY date DESC
        """.format(days), (ticker,))
        
        rows = cursor.fetchall()
        
        if not rows:
            rows = self._generate_synthetic_data(ticker, days)
            self._store_data(rows)
        
        conn.close()
        
        return [{
            "ticker": row[0],
            "date": row[1],
            "container_throughput_index": float(row[2]) if row[2] is not None else None,
            "inventory_days_coverage": float(row[3]) if row[3] is not None else None,
            "supplier_lead_time_days": float(row[4]) if row[4] is not None else None,
            "shipping_cost_index": float(row[5]) if row[5] is not None else None,
            "data_quality_score": float(row[6]) if row[6] is not None else 0.75,
            "source": row[7],
        } for row in rows]
    
    def _generate_synthetic_data(self, ticker: str, days: int) -> list[tuple]:
        """Generate synthetic supply chain data."""
        import random
        random.seed(hash(ticker + "sc") % 2**32)
        
        data = []
        
        for i in range(days):
            date = (datetime.now() - timedelta(days=i)).strftime("%Y-%m-%d")
            
            data.append((
                ticker,
                date,
                100.0 + random.gauss(0, 8),  # Container throughput
                45.0 + random.gauss(0, 5),     # Inventory days
                21.0 + random.gauss(0, 3),     # Lead time
                150.0 + random.gauss(0, 20),   # Shipping cost index
                0.75,  # Data quality
                "synthetic"
            ))
        
        return data
    
    def _store_data(self, rows: list[tuple]):
        """Store supply chain data."""
        conn = self._get_db_connection()
        cursor = conn.cursor()
        
        cursor.executemany("""
            INSERT OR REPLACE INTO supply_chain_data 
            (ticker, date, container_throughput_index, inventory_days_coverage,
             supplier_lead_time_days, shipping_cost_index, data_quality_score, source)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, rows)
        
        conn.commit()
        conn.close()
    
    def calculate_signal(self, ticker: str, days: int = 30) -> AlternativeDataSignal:
        """Calculate supply chain signal."""
        data = self.fetch_data(ticker, days=90)
        
        if not data:
            return AlternativeDataSignal(
                ticker=ticker,
                source="supply_chain",
                signal_type="operational_efficiency",
                score=0.0,
                confidence=0.0,
                raw_value=0.0,
                raw_unit="composite_index",
                period_days=days,
                z_score=0.0,
                percentile=50.0,
                trend_direction="insufficient_data",
                data_timestamp=datetime.now().isoformat(),
            )
        
        recent = data[:days]
        historical = data[days:90] if len(data) > days else data[-30:]
        
        # Calculate composite metric
        # High throughput + low inventory days + low lead time = bullish (efficient)
        # Low throughput + high inventory + high lead time = bearish (supply constraints)
        
        recent_scores = []
        for d in recent:
            throughput = d.get("container_throughput_index", 100)
            inventory = d.get("inventory_days_coverage", 45)
            lead_time = d.get("supplier_lead_time_days", 21)
            
            # Normalize and combine
            # Throughput: higher is better
            # Inventory days: lower is better (efficient turnover)
            # Lead time: lower is better
            score = (throughput - 100) / 10 - (inventory - 45) / 5 - (lead_time - 21) / 3
            recent_scores.append(score)
        
        current_avg = statistics.mean(recent_scores)
        
        if len(historical) > 1:
            hist_scores = []
            for d in historical:
                throughput = d.get("container_throughput_index", 100)
                inventory = d.get("inventory_days_coverage", 45)
                lead_time = d.get("supplier_lead_time_days", 21)
                score = (throughput - 100) / 10 - (inventory - 45) / 5 - (lead_time - 21) / 3
                hist_scores.append(score)
            
            hist_avg = statistics.mean(hist_scores)
            hist_std = statistics.stdev(hist_scores) if len(hist_scores) > 1 else 1.0
        else:
            hist_avg = current_avg
            hist_std = 1.0
        
        z_score = (current_avg - hist_avg) / hist_std if hist_std > 0 else 0.0
        
        # Map to -1 to +1 signal
        score = max(-1.0, min(1.0, z_score / 2))
        
        if z_score > 1.0:
            trend = "improving"
        elif z_score < -1.0:
            trend = "deteriorating"
        else:
            trend = "stable"
        
        all_values = recent_scores + (hist_scores if 'hist_scores' in dir() else [])
        percentile = sum(1 for v in all_values if v <= current_avg) / len(all_values) * 100 if all_values else 50.0
        
        data_quality = statistics.mean([d.get("data_quality_score", 0.75) for d in recent])
        confidence = min(0.75, data_quality * (0.5 + len(recent_scores) / 60))
        
        signal = AlternativeDataSignal(
            ticker=ticker,
            source="supply_chain",
            signal_type="operational_efficiency",
            score=round(score, 3),
            confidence=round(confidence, 3),
            raw_value=round(current_avg, 2),
            raw_unit="composite_index",
            period_days=days,
            z_score=round(z_score, 2),
            percentile=round(percentile, 1),
            trend_direction=trend,
            data_timestamp=data[0]["date"] if data else datetime.now().isoformat(),
        )
        
        self._store_signal(signal)
        return signal
    
    def _store_signal(self, signal: AlternativeDataSignal):
        """Store signal."""
        conn = self._get_db_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO alt_data_signals 
            (ticker, source, signal_type, score, confidence, raw_value, 
             period_days, z_score, percentile, trend_direction, data_timestamp)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            signal.ticker, signal.source, signal.signal_type,
            signal.score, signal.confidence, signal.raw_value,
            signal.period_days, signal.z_score, signal.percentile,
            signal.trend_direction, signal.data_timestamp
        ))
        
        conn.commit()
        conn.close()


# ---------------------------------------------------------------------------
# Main Client
# ---------------------------------------------------------------------------

class AlternativeDataClient:
    """Unified client for all alternative data sources."""
    
    # Source weights for composite signal
    SOURCE_WEIGHTS = {
        "satellite": 0.35,
        "credit_card": 0.40,
        "supply_chain": 0.25,
    }
    
    def __init__(self):
        self.satellite = SatelliteDataAdapter()
        self.credit_card = CreditCardAdapter()
        self.supply_chain = SupplyChainAdapter()
        init_database()
    
    def get_satellite_signal(self, ticker: str, days: int = 30) -> Optional[AlternativeDataSignal]:
        """Get satellite parking lot signal."""
        return self.satellite.calculate_signal(ticker, days)
    
    def get_credit_card_signal(self, ticker: str, days: int = 30) -> Optional[AlternativeDataSignal]:
        """Get credit card spending signal."""
        return self.credit_card.calculate_signal(ticker, days)
    
    def get_supply_chain_signal(self, ticker: str, days: int = 30) -> Optional[AlternativeDataSignal]:
        """Get supply chain signal."""
        return self.supply_chain.calculate_signal(ticker, days)
    
    def get_composite_signal(self, ticker: str, days: int = 30) -> CompositeSignal:
        """
        Calculate composite signal from all available sources.
        
        Weights:
        - Credit card: 40% (highest accuracy for earnings)
        - Satellite: 35% (good for retail/restaurants)
        - Supply chain: 25% (operational efficiency)
        """
        signals = {
            "satellite": self.get_satellite_signal(ticker, days),
            "credit_card": self.get_credit_card_signal(ticker, days),
            "supply_chain": self.get_supply_chain_signal(ticker, days),
        }
        
        # Extract scores and confidences
        scores = {}
        confidences = {}
        
        for source, signal in signals.items():
            if signal and signal.confidence > 0.3:
                scores[source] = signal.score
                confidences[source] = signal.confidence
        
        # Calculate weighted composite
        weighted_sum = 0.0
        weight_total = 0.0
        
        for source, score in scores.items():
            weight = self.SOURCE_WEIGHTS.get(source, 0.33) * confidences[source]
            weighted_sum += score * weight
            weight_total += weight
        
        if weight_total > 0:
            composite_score = weighted_sum / weight_total
            composite_confidence = min(0.9, weight_total / sum(self.SOURCE_WEIGHTS.values()))
        else:
            composite_score = 0.0
            composite_confidence = 0.0
        
        # Determine agreement
        if len(scores) >= 2:
            score_values = list(scores.values())
            if all(s > 0.2 for s in score_values) or all(s < -0.2 for s in score_values):
                agreement = "aligned"
            elif any(s > 0.3 for s in score_values) and any(s < -0.3 for s in score_values):
                agreement = "conflicting"
            else:
                agreement = "mixed"
        else:
            agreement = "insufficient_data"
        
        # Primary driver
        if confidences:
            primary = max(confidences.items(), key=lambda x: x[1])[0]
        else:
            primary = "none"
        
        return CompositeSignal(
            ticker=ticker,
            satellite_score=scores.get("satellite"),
            credit_card_score=scores.get("credit_card"),
            supply_chain_score=scores.get("supply_chain"),
            composite_score=round(composite_score, 3),
            composite_confidence=round(composite_confidence, 3),
            primary_driver=primary,
            signal_agreement=agreement,
        )
    
    def get_earnings_prediction(self, ticker: str, quarter: str) -> Optional[EarningsPrediction]:
        """Generate earnings prediction based on alternative data."""
        composite = self.get_composite_signal(ticker)
        
        if composite.composite_confidence < 0.4:
            return None
        
        # Map composite score to revenue growth prediction
        # Score > 0.5 → predict 10%+ growth
        # Score 0.2-0.5 → predict 5-10% growth
        # Score -0.2 to 0.2 → inline with consensus
        # Score < -0.2 → predict miss
        
        score = composite.composite_score
        
        if score > 0.5:
            predicted_growth = 12.0 + (score - 0.5) * 10
            direction = "beat"
            surprise_prob = min(0.9, 0.6 + score * 0.3)
        elif score > 0.2:
            predicted_growth = 5.0 + (score - 0.2) * 23
            direction = "beat"
            surprise_prob = 0.5 + score * 0.3
        elif score > -0.2:
            predicted_growth = score * 15
            direction = "inline"
            surprise_prob = 0.3
        else:
            predicted_growth = -5.0 + (score + 0.2) * 20
            direction = "miss"
            surprise_prob = min(0.8, 0.5 - score * 0.3)
        
        # Primary signals used
        signals_used = []
        if composite.satellite_score is not None:
            signals_used.append("satellite")
        if composite.credit_card_score is not None:
            signals_used.append("credit_card")
        if composite.supply_chain_score is not None:
            signals_used.append("supply_chain")
        
        return EarningsPrediction(
            ticker=ticker,
            quarter=quarter,
            predicted_revenue_growth=round(predicted_growth, 1),
            revenue_surprise_probability=round(surprise_prob, 2),
            revenue_direction=direction,
            confidence=round(composite.composite_confidence, 2),
            primary_signals=signals_used,
        )
    
    def get_batch_signals(self, tickers: list[str], days: int = 30) -> dict[str, CompositeSignal]:
        """Get composite signals for multiple tickers."""
        return {ticker: self.get_composite_signal(ticker, days) for ticker in tickers}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    if len(sys.argv) < 2:
        print("""Usage:
  python -m src.data.alternative_data fetch --ticker AAPL --source satellite
  python -m src.data.alternative_data composite --ticker AAPL
  python -m src.data.alternative_data batch --tickers AAPL,AMZN,TGT
  python -m src.data.alternative_data earnings --ticker AAPL --quarter Q4-2025
        """)
        sys.exit(1)
    
    command = sys.argv[1]
    client = AlternativeDataClient()
    
    if command == "fetch":
        ticker = None
        source = "all"
        days = 30
        
        i = 2
        while i < len(sys.argv):
            if sys.argv[i] == "--ticker" and i + 1 < len(sys.argv):
                ticker = sys.argv[i + 1]
                i += 2
            elif sys.argv[i] == "--source" and i + 1 < len(sys.argv):
                source = sys.argv[i + 1]
                i += 2
            elif sys.argv[i] == "--days" and i + 1 < len(sys.argv):
                days = int(sys.argv[i + 1])
                i += 2
            else:
                i += 1
        
        if not ticker:
            print("Error: --ticker required")
            sys.exit(1)
        
        if source == "satellite" or source == "all":
            signal = client.get_satellite_signal(ticker, days)
            if signal:
                print(f"📡 Satellite Signal for {ticker}:")
                print(json.dumps(signal.to_dict(), indent=2))
        
        if source == "credit_card" or source == "all":
            signal = client.get_credit_card_signal(ticker, days)
            if signal:
                print(f"💳 Credit Card Signal for {ticker}:")
                print(json.dumps(signal.to_dict(), indent=2))
        
        if source == "supply_chain" or source == "all":
            signal = client.get_supply_chain_signal(ticker, days)
            if signal:
                print(f"🚢 Supply Chain Signal for {ticker}:")
                print(json.dumps(signal.to_dict(), indent=2))
        
        return
    
    if command == "composite":
        ticker = None
        days = 30
        
        i = 2
        while i < len(sys.argv):
            if sys.argv[i] == "--ticker" and i + 1 < len(sys.argv):
                ticker = sys.argv[i + 1]
                i += 2
            elif sys.argv[i] == "--days" and i + 1 < len(sys.argv):
                days = int(sys.argv[i + 1])
                i += 2
            else:
                i += 1
        
        if not ticker:
            print("Error: --ticker required")
            sys.exit(1)
        
        composite = client.get_composite_signal(ticker, days)
        print(f"🔀 Composite Alternative Data Signal for {ticker}:")
        print(json.dumps(composite.to_dict(), indent=2))
        return
    
    if command == "batch":
        tickers = []
        days = 30
        
        i = 2
        while i < len(sys.argv):
            if sys.argv[i] == "--tickers" and i + 1 < len(sys.argv):
                tickers = sys.argv[i + 1].split(",")
                i += 2
            elif sys.argv[i] == "--days" and i + 1 < len(sys.argv):
                days = int(sys.argv[i + 1])
                i += 2
            else:
                i += 1
        
        if not tickers:
            print("Error: --tickers required (comma-separated)")
            sys.exit(1)
        
        signals = client.get_batch_signals(tickers, days)
        print(f"📊 Batch Alternative Data Signals:")
        for ticker, signal in signals.items():
            print(f"\n{ticker}:")
            print(f"  Composite Score: {signal.composite_score:+.3f}")
            print(f"  Confidence: {signal.composite_confidence:.1%}")
            print(f"  Primary Driver: {signal.primary_driver}")
            print(f"  Agreement: {signal.signal_agreement}")
        return
    
    if command == "earnings":
        ticker = None
        quarter = None
        
        i = 2
        while i < len(sys.argv):
            if sys.argv[i] == "--ticker" and i + 1 < len(sys.argv):
                ticker = sys.argv[i + 1]
                i += 2
            elif sys.argv[i] == "--quarter" and i + 1 < len(sys.argv):
                quarter = sys.argv[i + 1]
                i += 2
            else:
                i += 1
        
        if not ticker or not quarter:
            print("Error: --ticker and --quarter required")
            sys.exit(1)
        
        prediction = client.get_earnings_prediction(ticker, quarter)
        if prediction:
            print(f"📈 Earnings Prediction for {ticker} {quarter}:")
            print(json.dumps(prediction.to_dict(), indent=2))
        else:
            print(f"Insufficient data for {ticker} earnings prediction")
        return
    
    print(f"Unknown command: {command}")
    sys.exit(1)


if __name__ == "__main__":
    main()
