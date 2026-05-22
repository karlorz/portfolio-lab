"""
Tests for International Momentum Signal Generator
"""

import unittest
import tempfile
from pathlib import Path
from unittest.mock import patch

import sys
sys.path.insert(0, '/root/projects/portfolio-lab/src')

from signals.international_momentum import (
    SignalType,
    ConfidenceLevel,
    InternationalMomentumSignal,
    InternationalMomentumGenerator
)


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


if __name__ == '__main__':
    unittest.main(verbosity=2)
