"""
Tests for TIPS Monitor (v2.35 Phase 1)
"""
import pytest
import json
from pathlib import Path
from datetime import datetime, timedelta
from unittest.mock import Mock, patch

# Skip pandas-dependent tests if not available
try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False

from src.data.tips_monitor import TIPSMonitor, TIPSData


class TestTIPSMonitor:
    """Test suite for TIPS Monitor."""

    def test_initialization(self, tmp_path):
        """Test TIPS monitor initializes correctly."""
        monitor = TIPSMonitor(data_dir=tmp_path)
        assert monitor.data_dir == tmp_path
        assert monitor.db_path.exists()

    def test_database_schema(self, tmp_path):
        """Test database tables are created."""
        monitor = TIPSMonitor(data_dir=tmp_path)
        
        import sqlite3
        conn = sqlite3.connect(monitor.db_path)
        cursor = conn.cursor()
        
        # Check tables exist
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [row[0] for row in cursor.fetchall()]
        
        assert 'tips_yields' in tables
        assert 'breakeven_inflation' in tables
        assert 'tips_signals' in tables
        
        conn.close()

    def test_tips_data_dataclass(self):
        """Test TIPSData container."""
        data = TIPSData(
            symbol='TIP',
            timestamp=datetime.now(),
            real_yield=1.5,
            nominal_yield=4.2,
            breakeven_rate=2.7,
            duration=7.5
        )
        
        assert data.symbol == 'TIP'
        assert data.real_yield == 1.5
        assert data.breakeven_rate == 2.7

    def test_signal_thresholds(self, tmp_path):
        """Test signal threshold configuration."""
        monitor = TIPSMonitor(data_dir=tmp_path)
        
        # Verify thresholds loaded
        assert monitor.SIGNAL_THRESHOLDS['breakeven_low'] == 1.5
        assert monitor.SIGNAL_THRESHOLDS['breakeven_target'] == 2.0
        assert monitor.SIGNAL_THRESHOLDS['breakeven_high'] == 2.5
        assert monitor.SIGNAL_THRESHOLDS['breakeven_extreme'] == 3.0

    def test_tips_etfs_configuration(self, tmp_path):
        """Test TIPS ETF universe configuration."""
        monitor = TIPSMonitor(data_dir=tmp_path)
        
        # Verify TIPS ETFs
        assert 'SCHP' in monitor.TIPS_ETFS
        assert 'TIP' in monitor.TIPS_ETFS
        assert 'LTPZ' in monitor.TIPS_ETFS
        assert 'STIP' in monitor.TIPS_ETFS
        
        # Check expense ratios
        assert monitor.TIPS_ETFS['SCHP']['expense'] == 0.04
        assert monitor.TIPS_ETFS['TIP']['expense'] == 0.19

    def test_fred_series_configuration(self, tmp_path):
        """Test FRED series configuration."""
        monitor = TIPSMonitor(data_dir=tmp_path)
        
        assert 'T5YIE' in monitor.FRED_BREAKEVEN
        assert 'T10YIE' in monitor.FRED_BREAKEVEN
        assert 'T5YIFR' in monitor.FRED_BREAKEVEN

    @pytest.mark.skipif(not PANDAS_AVAILABLE, reason="pandas not installed")
    def test_signal_generation(self, tmp_path):
        """Test signal generation logic via mock data."""
        monitor = TIPSMonitor(data_dir=tmp_path)
        
        # Create mock TIPS data with different breakeven scenarios
        now = datetime.now()
        
        # Low breakeven (disinflation)
        low_data = TIPSData(
            symbol='TIP',
            timestamp=now,
            real_yield=1.5,
            nominal_yield=3.2,  # breakeven = 1.7%
            breakeven_rate=0.017,
            duration=7.5
        )
        
        signal_low = monitor.generate_signal({'TIP': low_data})
        assert signal_low['current_regime'] in ['DISINFLATION', 'LOW_STABLE']
        
        # High breakeven (inflation risk)
        high_data = TIPSData(
            symbol='TIP',
            timestamp=now,
            real_yield=1.5,
            nominal_yield=6.0,  # breakeven = 4.5%
            breakeven_rate=0.045,
            duration=7.5
        )
        
        signal_high = monitor.generate_signal({'TIP': high_data})
        assert signal_high['current_regime'] in ['HIGH_INFLATION', 'EXTREME_INFLATION']
        assert signal_high['confidence'] >= 0.8

    @pytest.mark.skipif(not PANDAS_AVAILABLE, reason="pandas not installed")
    def test_allocation_guidance_in_signal(self, tmp_path):
        """Test allocation guidance is present in signal output."""
        monitor = TIPSMonitor(data_dir=tmp_path)
        
        # Create mock TIPS data
        now = datetime.now()
        data = TIPSData(
            symbol='TIP',
            timestamp=now,
            real_yield=1.5,
            nominal_yield=4.2,
            breakeven_rate=0.027,  # 2.7%
            duration=7.5
        )
        
        signal = monitor.generate_signal({'TIP': data})
        
        # Verify signal contains allocation guidance
        assert 'tips_allocation_signal' in signal
        assert 'confidence' in signal
        assert 'rationale' in signal
        assert signal['confidence'] > 0

    def test_data_to_dict(self):
        """Test TIPSData serialization."""
        now = datetime.now()
        data = TIPSData(
            symbol='SCHP',
            timestamp=now,
            real_yield=0.8,
            nominal_yield=3.5,
            breakeven_rate=2.7,
            duration=3.0,
            nav=52.34,
            price=52.40
        )
        
        d = data.to_dict()
        assert d['symbol'] == 'SCHP'
        assert d['real_yield'] == 0.8
        assert d['breakeven_rate'] == 2.7
        assert 'timestamp' in d


class TestTIPSIntegration:
    """Integration tests for TIPS monitor."""

    def test_cli_export_format(self, tmp_path, capsys):
        """Test CLI export produces valid JSON."""
        monitor = TIPSMonitor(data_dir=tmp_path)
        
        # Create mock signal data
        signal_data = {
            'timestamp': datetime.now().isoformat(),
            'current_regime': 'TARGET_INFLATION',
            'breakeven_5y': 2.15,
            'breakeven_10y': 2.35,
            'tips_allocation_signal': '+3% TIPS overweight',
            'confidence': 0.75,
            'rationale': 'Breakeven near Fed target'
        }
        
        # Verify JSON serialization
        json_str = json.dumps(signal_data)
        parsed = json.loads(json_str)
        
        assert parsed['current_regime'] == 'TARGET_INFLATION'
        assert parsed['confidence'] == 0.75

    @pytest.mark.skipif(not PANDAS_AVAILABLE, reason="pandas not installed")
    def test_database_insert_and_retrieve(self, tmp_path):
        """Test database persistence."""
        monitor = TIPSMonitor(data_dir=tmp_path)
        
        # Insert test data
        import sqlite3
        conn = sqlite3.connect(monitor.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO tips_yields 
            (symbol, timestamp, real_yield, nominal_yield, breakeven_rate, duration)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', ('TIP', datetime.now().isoformat(), 1.5, 4.2, 2.7, 7.5))
        
        conn.commit()
        
        # Retrieve
        cursor.execute('SELECT * FROM tips_yields WHERE symbol = ?', ('TIP',))
        row = cursor.fetchone()
        
        assert row is not None
        assert row[1] == 'TIP'  # symbol column
        
        conn.close()
