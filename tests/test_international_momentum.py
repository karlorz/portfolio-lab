"""
Tests for International Equity Data Fetcher and Momentum Signal Generator
"""

import unittest
import json
import sqlite3
import tempfile
from pathlib import Path
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
import sys
import math

# Mock dependencies BEFORE importing our modules
mock_yf = MagicMock()
mock_pd = MagicMock()
mock_np = MagicMock()
mock_np.sqrt = math.sqrt
mock_np.nan = float('nan')

sys.modules['yfinance'] = mock_yf
sys.modules['pandas'] = mock_pd
sys.modules['numpy'] = mock_np

# Add src to path
sys.path.insert(0, '/root/projects/portfolio-lab/src')

# Now import our modules
from data.international_fetcher import (
    MomentumMetrics, 
    RelativeMomentum, 
    InternationalData,
    InternationalDataFetcher
)
from signals.international_momentum import (
    SignalType,
    ConfidenceLevel,
    InternationalMomentumSignal,
    InternationalMomentumGenerator
)


class TestMomentumMetrics(unittest.TestCase):
    """Test MomentumMetrics dataclass"""
    
    def test_to_dict(self):
        """Test conversion to dictionary"""
        metrics = MomentumMetrics(
            symbol='EFA',
            price=75.50,
            momentum_1m=0.03,
            momentum_3m=0.08,
            momentum_6m=0.124,
            volatility_20d=0.15,
            sharpe_6m=0.83,
            timestamp='2026-05-14T10:00:00'
        )
        
        d = metrics.to_dict()
        self.assertEqual(d['symbol'], 'EFA')
        self.assertEqual(d['price'], 75.50)
        self.assertEqual(d['momentum_6m'], 0.124)
    
    def test_momentum_calculation(self):
        """Test momentum values are stored correctly"""
        metrics = MomentumMetrics(
            symbol='EEM',
            price=42.30,
            momentum_1m=-0.02,
            momentum_3m=0.05,
            momentum_6m=0.081,
            volatility_20d=0.22,
            sharpe_6m=0.37,
            timestamp='2026-05-14T10:00:00'
        )
        
        self.assertLess(metrics.momentum_1m, 0)  # Negative momentum
        self.assertGreater(metrics.volatility_20d, 0.20)  # Higher vol for EM


class TestRelativeMomentum(unittest.TestCase):
    """Test RelativeMomentum dataclass"""
    
    def test_signal_neutral(self):
        """Test neutral signal when no outperformance"""
        rel = RelativeMomentum(
            symbol='relative_momentum',
            efa_momentum_6m=0.10,
            eem_momentum_6m=0.08,
            spy_momentum_6m=0.15,  # SPY leading
            efa_vs_spy=-0.05,
            eem_vs_spy=-0.07,
            signal='neutral',
            confidence=0.0,
            timestamp='2026-05-14T10:00:00'
        )
        
        self.assertEqual(rel.signal, 'neutral')
        self.assertEqual(rel.confidence, 0.0)
    
    def test_signal_efa_lead(self):
        """Test EFA lead signal"""
        rel = RelativeMomentum(
            symbol='relative_momentum',
            efa_momentum_6m=0.20,
            eem_momentum_6m=0.08,
            spy_momentum_6m=0.12,
            efa_vs_spy=0.08,  # EFA outperforming by 8%
            eem_vs_spy=-0.04,
            signal='efa_lead',
            confidence=0.80,
            timestamp='2026-05-14T10:00:00'
        )
        
        self.assertEqual(rel.signal, 'efa_lead')
        self.assertGreater(rel.confidence, 0.5)


class TestInternationalMomentumSignal(unittest.TestCase):
    """Test InternationalMomentumSignal"""
    
    def test_is_active_neutral(self):
        """Test neutral signal is not active"""
        signal = InternationalMomentumSignal(
            timestamp='2026-05-14T10:00:00',
            signal_type='neutral',
            confidence=0.0,
            confidence_level='low',
            efa_momentum_6m=0.12,
            eem_momentum_6m=0.08,
            spy_momentum_6m=0.15,
            efa_vs_spy=-0.03,
            eem_vs_spy=-0.07,
            spy_shift=0.0,
            efa_shift=0.0,
            eem_shift=0.0,
            max_allocation_efa=0.05,
            max_allocation_eem=0.03,
            holding_period_days=30,
            data_fresh=True,
            vix_filter_active=False,
            correlation_override=False
        )
        
        self.assertFalse(signal.is_active())
        self.assertEqual(signal.get_allocation_delta(), {'SPY': 0.0, 'EFA': 0.0, 'EEM': 0.0})
    
    def test_is_active_efa(self):
        """Test active EFA signal"""
        signal = InternationalMomentumSignal(
            timestamp='2026-05-14T10:00:00',
            signal_type='efa_lead',
            confidence=0.65,
            confidence_level='medium',
            efa_momentum_6m=0.20,
            eem_momentum_6m=0.08,
            spy_momentum_6m=0.12,
            efa_vs_spy=0.08,
            eem_vs_spy=-0.04,
            spy_shift=0.04,
            efa_shift=0.04,
            eem_shift=0.0,
            max_allocation_efa=0.05,
            max_allocation_eem=0.03,
            holding_period_days=30,
            data_fresh=True,
            vix_filter_active=False,
            correlation_override=False
        )
        
        self.assertTrue(signal.is_active())
        delta = signal.get_allocation_delta()
        self.assertLess(delta['SPY'], 0)  # Reduce SPY
        self.assertGreater(delta['EFA'], 0)  # Add EFA
    
    def test_is_active_vix_filtered(self):
        """Test signal filtered by high VIX"""
        signal = InternationalMomentumSignal(
            timestamp='2026-05-14T10:00:00',
            signal_type='efa_lead',
            confidence=0.65,
            confidence_level='medium',
            efa_momentum_6m=0.20,
            eem_momentum_6m=0.08,
            spy_momentum_6m=0.12,
            efa_vs_spy=0.08,
            eem_vs_spy=-0.04,
            spy_shift=0.04,
            efa_shift=0.04,
            eem_shift=0.0,
            max_allocation_efa=0.05,
            max_allocation_eem=0.03,
            holding_period_days=30,
            data_fresh=True,
            vix_filter_active=True,  # High VIX filter
            correlation_override=False
        )
        
        self.assertFalse(signal.is_active())  # Should be inactive
    
    def test_is_active_correlation_override(self):
        """Test signal disabled by high correlation"""
        signal = InternationalMomentumSignal(
            timestamp='2026-05-14T10:00:00',
            signal_type='efa_lead',
            confidence=0.65,
            confidence_level='medium',
            efa_momentum_6m=0.20,
            eem_momentum_6m=0.08,
            spy_momentum_6m=0.12,
            efa_vs_spy=0.08,
            eem_vs_spy=-0.04,
            spy_shift=0.04,
            efa_shift=0.04,
            eem_shift=0.0,
            max_allocation_efa=0.05,
            max_allocation_eem=0.03,
            holding_period_days=30,
            data_fresh=True,
            vix_filter_active=False,
            correlation_override=True  # High correlation
        )
        
        self.assertFalse(signal.is_active())  # Should be inactive
    
    def test_low_confidence_inactive(self):
        """Test low confidence signal is not active"""
        signal = InternationalMomentumSignal(
            timestamp='2026-05-14T10:00:00',
            signal_type='efa_lead',
            confidence=0.30,  # Below 0.5 threshold
            confidence_level='low',
            efa_momentum_6m=0.15,
            eem_momentum_6m=0.08,
            spy_momentum_6m=0.12,
            efa_vs_spy=0.03,
            eem_vs_spy=-0.04,
            spy_shift=0.015,
            efa_shift=0.015,
            eem_shift=0.0,
            max_allocation_efa=0.05,
            max_allocation_eem=0.03,
            holding_period_days=30,
            data_fresh=True,
            vix_filter_active=False,
            correlation_override=False
        )
        
        self.assertFalse(signal.is_active())  # Low confidence = inactive


class TestInternationalMomentumGenerator(unittest.TestCase):
    """Test InternationalMomentumGenerator"""
    
    def setUp(self):
        """Create temporary database for testing"""
        self.temp_db = tempfile.NamedTemporaryFile(suffix='.db', delete=False)
        self.generator = InternationalMomentumGenerator(cache_db=Path(self.temp_db.name))
    
    def tearDown(self):
        """Clean up temporary database"""
        self.temp_db.close()
        Path(self.temp_db.name).unlink(missing_ok=True)
    
    def test_determine_signal_type_neutral(self):
        """Test neutral signal determination"""
        signal_type, confidence = self.generator._determine_signal_type(
            efa_vs_spy=0.02,  # Below 5% threshold
            eem_vs_spy=-0.05
        )
        
        self.assertEqual(signal_type, SignalType.NEUTRAL)
        self.assertEqual(confidence, 0.0)
    
    def test_determine_signal_type_efa(self):
        """Test EFA lead signal determination"""
        signal_type, confidence = self.generator._determine_signal_type(
            efa_vs_spy=0.07,  # Above 5% threshold
            eem_vs_spy=0.02
        )
        
        self.assertEqual(signal_type, SignalType.EFA_LEAD)
        self.assertGreater(confidence, 0.5)
    
    def test_determine_signal_type_eem(self):
        """Test EEM lead signal determination"""
        signal_type, confidence = self.generator._determine_signal_type(
            efa_vs_spy=0.02,
            eem_vs_spy=0.10  # Above 8% threshold
        )
        
        self.assertEqual(signal_type, SignalType.EEM_LEAD)
        self.assertGreater(confidence, 0.5)
    
    def test_allocation_shifts_neutral(self):
        """Test no allocation shifts for neutral signal"""
        spy_shift, efa_shift, eem_shift = self.generator._calculate_allocation_shifts(
            SignalType.NEUTRAL,
            confidence=0.0
        )
        
        self.assertEqual(spy_shift, 0.0)
        self.assertEqual(efa_shift, 0.0)
        self.assertEqual(eem_shift, 0.0)
    
    def test_allocation_shifts_efa(self):
        """Test EFA allocation shifts"""
        spy_shift, efa_shift, eem_shift = self.generator._calculate_allocation_shifts(
            SignalType.EFA_LEAD,
            confidence=0.80
        )
        
        self.assertGreater(spy_shift, 0)  # Reduce SPY
        self.assertEqual(spy_shift, efa_shift)  # Same shift
        self.assertEqual(eem_shift, 0.0)  # No EEM shift
        self.assertLessEqual(spy_shift, self.generator.MAX_EFA_ALLOCATION)
    
    def test_allocation_shifts_eem(self):
        """Test EEM allocation shifts"""
        spy_shift, efa_shift, eem_shift = self.generator._calculate_allocation_shifts(
            SignalType.EEM_LEAD,
            confidence=0.60
        )
        
        self.assertGreater(spy_shift, 0)  # Reduce SPY
        self.assertEqual(efa_shift, 0.0)  # No EFA shift
        self.assertEqual(spy_shift, eem_shift)  # Same shift
        self.assertLessEqual(spy_shift, self.generator.MAX_EEM_ALLOCATION)
    
    def test_generate_signal_neutral(self):
        """Test generating neutral signal from data"""
        data = {
            'timestamp': '2026-05-14T10:00:00',
            'data_fresh': True,
            'relative': {
                'efa_momentum_6m': 0.10,
                'eem_momentum_6m': 0.08,
                'spy_momentum_6m': 0.15,  # SPY leading
                'efa_vs_spy': -0.05,
                'eem_vs_spy': -0.07
            }
        }
        
        signal = self.generator.generate_signal(data)
        
        self.assertEqual(signal.signal_type, 'neutral')
        self.assertEqual(signal.confidence, 0.0)
        self.assertFalse(signal.is_active())
    
    def test_generate_signal_efa_lead(self):
        """Test generating EFA lead signal"""
        data = {
            'timestamp': '2026-05-14T10:00:00',
            'data_fresh': True,
            'relative': {
                'efa_momentum_6m': 0.20,
                'eem_momentum_6m': 0.08,
                'spy_momentum_6m': 0.12,
                'efa_vs_spy': 0.08,  # EFA leading by 8%
                'eem_vs_spy': -0.04
            }
        }
        
        with patch.object(self.generator, '_get_vix_level', return_value=20.0):
            with patch.object(self.generator, '_get_correlation', return_value=0.85):
                signal = self.generator.generate_signal(data)
        
        self.assertEqual(signal.signal_type, 'efa_lead')
        self.assertGreater(signal.confidence, 0.5)
        self.assertTrue(signal.is_active())
    
    def test_generate_signal_vix_filtered(self):
        """Test signal filtered by high VIX"""
        data = {
            'timestamp': '2026-05-14T10:00:00',
            'data_fresh': True,
            'relative': {
                'efa_momentum_6m': 0.20,
                'eem_momentum_6m': 0.08,
                'spy_momentum_6m': 0.12,
                'efa_vs_spy': 0.08,
                'eem_vs_spy': -0.04
            }
        }
        
        with patch.object(self.generator, '_get_vix_level', return_value=35.0):  # High VIX
            with patch.object(self.generator, '_get_correlation', return_value=0.85):
                signal = self.generator.generate_signal(data)
        
        self.assertTrue(signal.vix_filter_active)
        self.assertFalse(signal.is_active())  # Should be inactive due to VIX
    
    def test_save_and_retrieve_signal(self):
        """Test saving and retrieving signal from database"""
        # Generate and save signal
        data = {
            'timestamp': '2026-05-14T10:00:00',
            'data_fresh': True,
            'relative': {
                'efa_momentum_6m': 0.20,
                'eem_momentum_6m': 0.08,
                'spy_momentum_6m': 0.12,
                'efa_vs_spy': 0.08,
                'eem_vs_spy': -0.04
            }
        }
        
        with patch.object(self.generator, '_get_vix_level', return_value=20.0):
            with patch.object(self.generator, '_get_correlation', return_value=0.85):
                signal = self.generator.generate_signal(data)
        
        # Retrieve from database
        retrieved = self.generator.get_current_signal()
        
        self.assertIsNotNone(retrieved)
        self.assertEqual(retrieved.signal_type, 'efa_lead')
    
    def test_signal_statistics(self):
        """Test signal statistics calculation"""
        # Generate several signals
        for i in range(5):
            data = {
                'timestamp': f'2026-05-{10+i}T10:00:00',
                'data_fresh': True,
                'relative': {
                    'efa_momentum_6m': 0.20,
                    'eem_momentum_6m': 0.08,
                    'spy_momentum_6m': 0.12,
                    'efa_vs_spy': 0.08 if i < 3 else -0.02,  # 3 EFA, 2 neutral
                    'eem_vs_spy': -0.04
                }
            }
            
            with patch.object(self.generator, '_get_vix_level', return_value=20.0):
                with patch.object(self.generator, '_get_correlation', return_value=0.85):
                    self.generator.generate_signal(data)
        
        stats = self.generator.get_signal_statistics(days=30)
        
        self.assertEqual(stats['total_signals'], 5)
        self.assertEqual(stats['efa_lead_count'], 3)
        self.assertEqual(stats['neutral_count'], 2)
        self.assertGreater(stats['activation_rate'], 0)


class TestInternationalDataFetcher(unittest.TestCase):
    """Test InternationalDataFetcher (mocked)"""
    
    def setUp(self):
        """Create temporary database"""
        self.temp_db = tempfile.NamedTemporaryFile(suffix='.db', delete=False)
        self.fetcher = InternationalDataFetcher(cache_db=Path(self.temp_db.name))
    
    def tearDown(self):
        """Clean up"""
        self.temp_db.close()
        Path(self.temp_db.name).unlink(missing_ok=True)
    
    @patch('yfinance.Ticker')
    def test_fetch_symbol_success(self, mock_ticker_class):
        """Test successful symbol fetch"""
        # Mock yfinance response
        mock_ticker = MagicMock()
        mock_hist = MagicMock()
        mock_hist.empty = False
        mock_hist.index = [datetime(2026, 5, i) for i in range(1, 15)]
        mock_hist.__getitem__ = MagicMock(side_effect=lambda key: {
            'Open': [100.0] * 14,
            'High': [101.0] * 14,
            'Low': [99.0] * 14,
            'Close': [100.5] * 14,
            'Volume': [1000000] * 14
        }[key])
        mock_ticker.history.return_value = mock_hist
        mock_ticker_class.return_value = mock_ticker
        
        df = self.fetcher.fetch_symbol('EFA', period='1mo')
        
        self.assertIsNotNone(df)
        self.assertEqual(len(df), 14)
    
    def test_calculate_momentum(self):
        """Test momentum calculation from dataframe"""
        import pandas as pd
        import numpy as np
        
        # Create synthetic price data with known momentum
        dates = pd.date_range('2025-11-01', '2026-05-14', freq='B')  # ~126 business days
        prices = 100 * (1 + np.linspace(0, 0.15, len(dates)))  # 15% gain over 6 months
        
        df = pd.DataFrame({
            'date': dates,
            'open': prices,
            'high': prices * 1.01,
            'low': prices * 0.99,
            'close': prices,
            'volume': [1000000] * len(dates)
        })
        
        metrics = self.fetcher.calculate_momentum(df, 'EFA')
        
        self.assertEqual(metrics.symbol, 'EFA')
        self.assertAlmostEqual(metrics.momentum_6m, 0.15, places=1)
        self.assertGreater(metrics.price, 0)
        self.assertGreater(metrics.volatility_20d, 0)
    
    def test_calculate_relative_momentum(self):
        """Test relative momentum calculation"""
        efa = MomentumMetrics(
            symbol='EFA',
            price=75.0,
            momentum_1m=0.02,
            momentum_3m=0.06,
            momentum_6m=0.20,
            volatility_20d=0.15,
            sharpe_6m=1.33,
            timestamp='2026-05-14T10:00:00'
        )
        
        eem = MomentumMetrics(
            symbol='EEM',
            price=42.0,
            momentum_1m=0.01,
            momentum_3m=0.03,
            momentum_6m=0.08,
            volatility_20d=0.22,
            sharpe_6m=0.36,
            timestamp='2026-05-14T10:00:00'
        )
        
        spy = MomentumMetrics(
            symbol='SPY',
            price=450.0,
            momentum_1m=0.015,
            momentum_3m=0.04,
            momentum_6m=0.12,
            volatility_20d=0.12,
            sharpe_6m=1.0,
            timestamp='2026-05-14T10:00:00'
        )
        
        relative = self.fetcher.calculate_relative_momentum(efa, eem, spy)
        
        self.assertEqual(relative.signal, 'efa_lead')  # EFA beating SPY by 8%
        self.assertGreater(relative.confidence, 0.5)
        self.assertEqual(relative.efa_vs_spy, 0.08)
    
    def test_relative_momentum_eem_lead(self):
        """Test relative momentum when EEM leads"""
        efa = MomentumMetrics(
            symbol='EFA',
            price=75.0,
            momentum_1m=0.02,
            momentum_3m=0.05,
            momentum_6m=0.10,
            volatility_20d=0.15,
            sharpe_6m=0.67,
            timestamp='2026-05-14T10:00:00'
        )
        
        eem = MomentumMetrics(
            symbol='EEM',
            price=42.0,
            momentum_1m=0.03,
            momentum_3m=0.08,
            momentum_6m=0.22,  # EEM outperforming
            volatility_20d=0.22,
            sharpe_6m=1.0,
            timestamp='2026-05-14T10:00:00'
        )
        
        spy = MomentumMetrics(
            symbol='SPY',
            price=450.0,
            momentum_1m=0.015,
            momentum_3m=0.04,
            momentum_6m=0.12,
            volatility_20d=0.12,
            sharpe_6m=1.0,
            timestamp='2026-05-14T10:00:00'
        )
        
        relative = self.fetcher.calculate_relative_momentum(efa, eem, spy)
        
        self.assertEqual(relative.signal, 'eem_lead')  # EEM beating SPY by 10%
        self.assertGreater(relative.confidence, 0.5)


class TestIntegration(unittest.TestCase):
    """Integration tests for fetcher + generator"""
    
    def setUp(self):
        """Create temporary database"""
        self.temp_db = tempfile.NamedTemporaryFile(suffix='.db', delete=False)
        self.fetcher = InternationalDataFetcher(cache_db=Path(self.temp_db.name))
        self.generator = InternationalMomentumGenerator(cache_db=Path(self.temp_db.name))
    
    def tearDown(self):
        """Clean up"""
        self.temp_db.close()
        Path(self.temp_db.name).unlink(missing_ok=True)
    
    def test_end_to_end_signal_generation(self):
        """Test complete pipeline from metrics to signal"""
        import pandas as pd
        import numpy as np
        
        # Create synthetic data showing EFA outperformance
        dates = pd.date_range('2025-11-01', '2026-05-14', freq='B')
        
        # EFA: +20% over 6 months
        efa_prices = 75 * (1 + np.linspace(0, 0.20, len(dates)))
        # SPY: +12% over 6 months
        spy_prices = 450 * (1 + np.linspace(0, 0.12, len(dates)))
        # EEM: +8% over 6 months
        eem_prices = 42 * (1 + np.linspace(0, 0.08, len(dates)))
        
        efa_df = pd.DataFrame({
            'date': dates,
            'open': efa_prices,
            'high': efa_prices * 1.01,
            'low': efa_prices * 0.99,
            'close': efa_prices,
            'volume': [1000000] * len(dates)
        })
        
        spy_df = pd.DataFrame({
            'date': dates,
            'open': spy_prices,
            'high': spy_prices * 1.01,
            'low': spy_prices * 0.99,
            'close': spy_prices,
            'volume': [5000000] * len(dates)
        })
        
        eem_df = pd.DataFrame({
            'date': dates,
            'open': eem_prices,
            'high': eem_prices * 1.02,
            'low': eem_prices * 0.98,
            'close': eem_prices,
            'volume': [2000000] * len(dates)
        })
        
        # Calculate metrics
        efa_metrics = self.fetcher.calculate_momentum(efa_df, 'EFA')
        spy_metrics = self.fetcher.calculate_momentum(spy_df, 'SPY')
        eem_metrics = self.fetcher.calculate_momentum(eem_df, 'EEM')
        
        # Calculate relative momentum
        relative = self.fetcher.calculate_relative_momentum(efa_metrics, eem_metrics, spy_metrics)
        
        # Build data dict for generator
        data = {
            'timestamp': '2026-05-14T10:00:00',
            'data_fresh': True,
            'relative': relative.to_dict()
        }
        
        # Generate signal
        with patch.object(self.generator, '_get_vix_level', return_value=20.0):
            with patch.object(self.generator, '_get_correlation', return_value=0.85):
                signal = self.generator.generate_signal(data)
        
        # Verify signal
        self.assertEqual(signal.signal_type, 'efa_lead')
        self.assertTrue(signal.is_active())
        self.assertGreater(signal.efa_vs_spy, 0.05)  # Above threshold
        
        # Check allocation deltas
        delta = signal.get_allocation_delta()
        self.assertLess(delta['SPY'], 0)  # Reduce SPY
        self.assertGreater(delta['EFA'], 0)  # Increase EFA
        self.assertEqual(delta['EEM'], 0)  # No EEM change


if __name__ == '__main__':
    unittest.main(verbosity=2)
