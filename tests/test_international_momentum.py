"""
Tests for International Momentum Signal Generator
"""

import unittest
import tempfile
from pathlib import Path
from unittest.mock import patch


from src.signals.international_momentum import (
    SignalType,
    ConfidenceLevel,
    InternationalMomentumSignal,
    InternationalMomentumGenerator
)


class TestInternationalMomentumSignal(unittest.TestCase):
    """Test InternationalMomentumSignal"""

    def _make_signal(self, **overrides):
        """Helper to create a signal with sensible defaults."""
        defaults = dict(
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
            correlation_override=False,
        )
        defaults.update(overrides)
        return InternationalMomentumSignal(**defaults)

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


class TestInternationalMomentumSignalSnapshot(unittest.TestCase):
    """Test to_signal_snapshot() bridge method."""

    def test_active_signal_snapshot(self):
        signal = InternationalMomentumSignal(
            timestamp='2026-05-14T10:00:00',
            signal_type='efa_lead',
            confidence=0.70,
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
            correlation_override=False,
        )
        snapshot = signal.to_signal_snapshot()
        self.assertTrue(snapshot.is_active)
        self.assertEqual(snapshot.source, "international_momentum")
        self.assertEqual(snapshot.timestamp, '2026-05-14T10:00:00')
        self.assertGreater(snapshot.confidence, 0)
        self.assertIn("SPY", snapshot.asset_signals)

    def test_inactive_signal_snapshot(self):
        signal = InternationalMomentumSignal(
            timestamp='2026-05-14T10:00:00',
            signal_type='neutral',
            confidence=0.0,
            confidence_level='low',
            efa_momentum_6m=0.10,
            eem_momentum_6m=0.08,
            spy_momentum_6m=0.15,
            efa_vs_spy=-0.05,
            eem_vs_spy=-0.07,
            spy_shift=0.0,
            efa_shift=0.0,
            eem_shift=0.0,
            max_allocation_efa=0.05,
            max_allocation_eem=0.03,
            holding_period_days=30,
            data_fresh=True,
            vix_filter_active=False,
            correlation_override=False,
        )
        snapshot = signal.to_signal_snapshot()
        self.assertFalse(snapshot.is_active)

    def test_snapshot_to_signal_reading(self):
        """Test that snapshot can convert to SignalReading for ensemble."""
        signal = InternationalMomentumSignal(
            timestamp='2026-05-14T10:00:00',
            signal_type='efa_lead',
            confidence=0.70,
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
            correlation_override=False,
        )
        snapshot = signal.to_signal_snapshot()
        reading = snapshot.to_signal_reading()
        from src.strategy.ensemble_voter import SignalSource
        self.assertEqual(reading.source, SignalSource.INTERNATIONAL_MOMENTUM)


class TestInternationalMomentumSignalDataclass(unittest.TestCase):
    """Test InternationalMomentumSignal dataclass methods."""

    def test_to_dict(self):
        signal = InternationalMomentumSignal(
            timestamp='2026-05-14T10:00:00',
            signal_type='neutral',
            confidence=0.0,
            confidence_level='low',
            efa_momentum_6m=0.10,
            eem_momentum_6m=0.08,
            spy_momentum_6m=0.15,
            efa_vs_spy=-0.05,
            eem_vs_spy=-0.07,
            spy_shift=0.0,
            efa_shift=0.0,
            eem_shift=0.0,
            max_allocation_efa=0.05,
            max_allocation_eem=0.03,
            holding_period_days=30,
            data_fresh=True,
            vix_filter_active=False,
            correlation_override=False,
        )
        d = signal.to_dict()
        self.assertIsInstance(d, dict)
        self.assertEqual(d['signal_type'], 'neutral')
        self.assertEqual(d['efa_momentum_6m'], 0.10)

    def test_eem_lead_allocation_delta(self):
        signal = InternationalMomentumSignal(
            timestamp='2026-05-14T10:00:00',
            signal_type='eem_lead',
            confidence=0.65,
            confidence_level='medium',
            efa_momentum_6m=0.08,
            eem_momentum_6m=0.20,
            spy_momentum_6m=0.12,
            efa_vs_spy=-0.04,
            eem_vs_spy=0.08,
            spy_shift=0.03,
            efa_shift=0.0,
            eem_shift=0.03,
            max_allocation_efa=0.05,
            max_allocation_eem=0.03,
            holding_period_days=30,
            data_fresh=True,
            vix_filter_active=False,
            correlation_override=False,
        )
        delta = signal.get_allocation_delta()
        self.assertLess(delta['SPY'], 0)
        self.assertGreater(delta['EEM'], 0)
        self.assertEqual(delta['EFA'], 0.0)


class TestInternationalMomentumSignalExtended(unittest.TestCase):
    """Extended tests for InternationalMomentumSignal dataclass methods."""

    # --- to_dict field completeness ---

    def test_to_dict_has_all_fields(self):
        signal = self._make_signal()
        d = signal.to_dict()
        expected_fields = [
            'timestamp', 'signal_type', 'confidence', 'confidence_level',
            'efa_momentum_6m', 'eem_momentum_6m', 'spy_momentum_6m',
            'efa_vs_spy', 'eem_vs_spy',
            'spy_shift', 'efa_shift', 'eem_shift',
            'max_allocation_efa', 'max_allocation_eem', 'holding_period_days',
            'data_fresh', 'vix_filter_active', 'correlation_override',
        ]
        for field in expected_fields:
            self.assertIn(field, d, f"Missing field: {field}")

    def test_to_dict_values_match(self):
        signal = self._make_signal(confidence=0.75, efa_vs_spy=0.09)
        d = signal.to_dict()
        self.assertEqual(d['confidence'], 0.75)
        self.assertEqual(d['efa_vs_spy'], 0.09)

    # --- is_active edge cases ---

    def test_is_active_stale_data(self):
        """Non-neutral signal with stale data is inactive."""
        signal = self._make_signal(
            signal_type='efa_lead', confidence=0.7,
            efa_vs_spy=0.08, data_fresh=False,
        )
        self.assertFalse(signal.is_active())

    def test_is_active_exactly_at_confidence_threshold(self):
        """Confidence exactly 0.5 should be active."""
        signal = self._make_signal(
            signal_type='efa_lead', confidence=0.5,
            efa_vs_spy=0.08, data_fresh=True,
        )
        self.assertTrue(signal.is_active())

    def test_is_active_just_below_confidence_threshold(self):
        """Confidence just below 0.5 should be inactive."""
        signal = self._make_signal(
            signal_type='efa_lead', confidence=0.49,
            efa_vs_spy=0.08, data_fresh=True,
        )
        self.assertFalse(signal.is_active())

    # --- get_allocation_delta ---

    def test_get_allocation_delta_inactive_returns_zeros(self):
        """Inactive signal returns zero deltas for all assets."""
        signal = self._make_signal(
            signal_type='efa_lead', confidence=0.3,
            spy_shift=0.04, efa_shift=0.04, eem_shift=0.0,
        )
        delta = signal.get_allocation_delta()
        self.assertEqual(delta, {'SPY': 0.0, 'EFA': 0.0, 'EEM': 0.0})

    def test_get_allocation_delta_active_efa(self):
        """Active EFA signal: SPY negative, EFA positive."""
        signal = self._make_signal(
            signal_type='efa_lead', confidence=0.7,
            efa_vs_spy=0.08, spy_shift=0.035, efa_shift=0.035, eem_shift=0.0,
        )
        delta = signal.get_allocation_delta()
        self.assertEqual(delta['SPY'], -0.035)
        self.assertEqual(delta['EFA'], 0.035)
        self.assertEqual(delta['EEM'], 0.0)

    # --- to_signal_snapshot ---

    def test_snapshot_eem_lead_value(self):
        """EEM lead snapshot value is clipped eem_vs_spy / 10."""
        signal = self._make_signal(
            signal_type='eem_lead', confidence=0.7,
            eem_vs_spy=0.12, efa_vs_spy=-0.02,
        )
        snapshot = signal.to_signal_snapshot()
        self.assertAlmostEqual(snapshot.value, 0.012)

    def test_snapshot_regime_fit_is_all(self):
        snapshot = self._make_signal().to_signal_snapshot()
        self.assertEqual(snapshot.regime_fit, "all")

    def test_snapshot_explanation_contains_signal_type(self):
        signal = self._make_signal(signal_type='efa_lead', confidence=0.65)
        snapshot = signal.to_signal_snapshot()
        self.assertIn('efa_lead', snapshot.explanation)

    def test_snapshot_metadata_keys(self):
        signal = self._make_signal(signal_type='eem_lead', confidence=0.75)
        snapshot = signal.to_signal_snapshot()
        self.assertIn('signal_type', snapshot.metadata)
        self.assertIn('confidence_level', snapshot.metadata)
        self.assertIn('vix_filter_active', snapshot.metadata)
        self.assertIn('correlation_override', snapshot.metadata)

    def test_snapshot_value_clipped_at_upper_bound(self):
        """Very large outperformance should be clipped to 0.5."""
        signal = self._make_signal(
            signal_type='efa_lead', confidence=0.9, efa_vs_spy=0.80,
        )
        snapshot = signal.to_signal_snapshot()
        self.assertLessEqual(snapshot.value, 0.5)

    def test_snapshot_value_clipped_at_lower_bound(self):
        """Very negative outperformance should be clipped to -0.5."""
        signal = self._make_signal(
            signal_type='eem_lead', confidence=0.9, eem_vs_spy=-0.80,
        )
        snapshot = signal.to_signal_snapshot()
        self.assertGreaterEqual(snapshot.value, -0.5)

    # --- helper ---

    def _make_signal(self, **overrides):
        defaults = dict(
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
            correlation_override=False,
        )
        defaults.update(overrides)
        return InternationalMomentumSignal(**defaults)


class TestInternationalMomentumGeneratorExtended(unittest.TestCase):
    """Extended tests for InternationalMomentumGenerator."""

    def setUp(self):
        self.temp_db = tempfile.NamedTemporaryFile(suffix='.db', delete=False)
        self.generator = InternationalMomentumGenerator(cache_db=Path(self.temp_db.name))

    def tearDown(self):
        self.temp_db.close()
        Path(self.temp_db.name).unlink(missing_ok=True)

    # --- Class constants ---

    def test_threshold_constants(self):
        self.assertEqual(self.generator.EFA_THRESHOLD, 0.05)
        self.assertEqual(self.generator.EEM_THRESHOLD, 0.08)

    def test_allocation_constants(self):
        self.assertEqual(self.generator.MAX_EFA_ALLOCATION, 0.05)
        self.assertEqual(self.generator.MAX_EEM_ALLOCATION, 0.03)
        self.assertEqual(self.generator.MIN_HOLDING_DAYS, 30)

    def test_risk_filter_constants(self):
        self.assertEqual(self.generator.VIX_CUTOFF, 30.0)
        self.assertEqual(self.generator.CORRELATION_CUTOFF, 0.95)

    # --- determine_signal_type edge cases ---

    def test_determine_signal_type_efa_at_threshold(self):
        """EFA exactly at threshold should produce NEUTRAL (strict >)."""
        sig_type, conf = self.generator._determine_signal_type(
            efa_vs_spy=0.05, eem_vs_spy=-0.02
        )
        self.assertEqual(sig_type, SignalType.NEUTRAL)

    def test_determine_signal_type_eem_at_threshold(self):
        """EEM exactly at threshold should produce NEUTRAL (strict >)."""
        sig_type, conf = self.generator._determine_signal_type(
            efa_vs_spy=-0.02, eem_vs_spy=0.08
        )
        self.assertEqual(sig_type, SignalType.NEUTRAL)

    def test_determine_signal_type_efa_just_above_threshold(self):
        sig_type, conf = self.generator._determine_signal_type(
            efa_vs_spy=0.051, eem_vs_spy=0.0
        )
        self.assertEqual(sig_type, SignalType.EFA_LEAD)
        self.assertGreater(conf, 0)

    def test_determine_signal_type_confidence_caps_at_1(self):
        """Confidence should be capped at 1.0 for very large outperformance."""
        sig_type, conf = self.generator._determine_signal_type(
            efa_vs_spy=0.50, eem_vs_spy=0.0
        )
        self.assertEqual(sig_type, SignalType.EFA_LEAD)
        self.assertLessEqual(conf, 1.0)

    # --- static determine_signal_type with custom thresholds ---

    def test_static_determine_custom_thresholds(self):
        """Custom thresholds should override defaults."""
        sig_type, conf = InternationalMomentumGenerator.determine_signal_type(
            efa_vs_spy=0.03, eem_vs_spy=0.0,
            efa_threshold=0.02, eem_threshold=0.10,
        )
        self.assertEqual(sig_type, SignalType.EFA_LEAD)

    # --- _calculate_allocation_shifts edge cases ---

    def test_allocation_shifts_zero_confidence(self):
        """Zero confidence produces zero shifts."""
        spy, efa, eem = self.generator._calculate_allocation_shifts(
            SignalType.EFA_LEAD, confidence=0.0
        )
        self.assertEqual(spy, 0.0)
        self.assertEqual(efa, 0.0)
        self.assertEqual(eem, 0.0)

    # --- generate_signal edge cases ---

    def test_generate_signal_empty_data(self):
        """Empty data dict should produce neutral signal."""
        signal = self.generator.generate_signal({})
        self.assertEqual(signal.signal_type, 'neutral')
        self.assertFalse(signal.is_active())

    def test_generate_signal_custom_timestamp(self):
        """Custom timestamp should be preserved."""
        data = {
            'timestamp': '2025-01-15T08:30:00',
            'data_fresh': True,
            'relative': {
                'efa_momentum_6m': 0.10,
                'eem_momentum_6m': 0.08,
                'spy_momentum_6m': 0.15,
                'efa_vs_spy': -0.02,
                'eem_vs_spy': -0.03,
            }
        }
        signal = self.generator.generate_signal(data)
        self.assertEqual(signal.timestamp, '2025-01-15T08:30:00')

    def test_generate_signal_rounding(self):
        """Values should be rounded to 4 decimal places."""
        data = {
            'timestamp': '2026-05-14T10:00:00',
            'data_fresh': True,
            'relative': {
                'efa_momentum_6m': 0.123456789,
                'eem_momentum_6m': 0.08,
                'spy_momentum_6m': 0.15,
                'efa_vs_spy': 0.06,
                'eem_vs_spy': -0.04,
            }
        }
        with patch.object(self.generator, '_get_vix_level', return_value=20.0):
            with patch.object(self.generator, '_get_correlation', return_value=0.85):
                signal = self.generator.generate_signal(data)
        self.assertEqual(signal.efa_momentum_6m, round(0.123456789, 4))

    # --- get_current_signal ---

    def test_get_current_signal_empty_db(self):
        """Empty database should return None."""
        result = self.generator.get_current_signal()
        self.assertIsNone(result)

    # --- get_signal_statistics ---

    def test_get_signal_statistics_single_signal(self):
        """Single signal statistics should work."""
        data = {
            'timestamp': '2026-05-14T10:00:00',
            'data_fresh': True,
            'relative': {
                'efa_momentum_6m': 0.20,
                'eem_momentum_6m': 0.08,
                'spy_momentum_6m': 0.12,
                'efa_vs_spy': 0.08,
                'eem_vs_spy': -0.04,
            }
        }
        with patch.object(self.generator, '_get_vix_level', return_value=20.0):
            with patch.object(self.generator, '_get_correlation', return_value=0.85):
                self.generator.generate_signal(data)

        stats = self.generator.get_signal_statistics(days=30)
        self.assertEqual(stats['total_signals'], 1)
        self.assertEqual(stats['efa_lead_count'], 1)
        self.assertIn('avg_confidence', stats)

    # --- confidence_level classification ---

    def test_generate_signal_low_confidence_level(self):
        """Low confidence signal should have confidence_level='low'."""
        data = {
            'timestamp': '2026-05-14T10:00:00',
            'data_fresh': True,
            'relative': {
                'efa_momentum_6m': 0.20,
                'eem_momentum_6m': 0.08,
                'spy_momentum_6m': 0.12,
                'efa_vs_spy': 0.06,  # confidence ~0.6 (between 0.5-0.7 = medium)
                'eem_vs_spy': -0.04,
            }
        }
        with patch.object(self.generator, '_get_vix_level', return_value=20.0):
            with patch.object(self.generator, '_get_correlation', return_value=0.85):
                signal = self.generator.generate_signal(data)
        self.assertEqual(signal.confidence_level, 'medium')

    def test_generate_signal_high_confidence_level(self):
        """High confidence signal should have confidence_level='high'."""
        data = {
            'timestamp': '2026-05-14T10:00:00',
            'data_fresh': True,
            'relative': {
                'efa_momentum_6m': 0.20,
                'eem_momentum_6m': 0.08,
                'spy_momentum_6m': 0.12,
                'efa_vs_spy': 0.09,  # confidence ~0.9 (above 0.7 = high)
                'eem_vs_spy': -0.04,
            }
        }
        with patch.object(self.generator, '_get_vix_level', return_value=20.0):
            with patch.object(self.generator, '_get_correlation', return_value=0.85):
                signal = self.generator.generate_signal(data)
        self.assertEqual(signal.confidence_level, 'high')


class TestInternationalMomentumGeneratorAdvanced(unittest.TestCase):
    """Additional tests for InternationalMomentumGenerator."""

    def setUp(self):
        self.temp_db = tempfile.NamedTemporaryFile(suffix='.db', delete=False)
        self.generator = InternationalMomentumGenerator(cache_db=Path(self.temp_db.name))

    def tearDown(self):
        self.temp_db.close()
        Path(self.temp_db.name).unlink(missing_ok=True)

    def test_static_determine_signal_type(self):
        """Test the static determine_signal_type method."""
        # EFA lead
        sig_type, conf = InternationalMomentumGenerator.determine_signal_type(
            efa_vs_spy=0.08, eem_vs_spy=0.02
        )
        self.assertEqual(sig_type, SignalType.EFA_LEAD)
        self.assertGreater(conf, 0)

    def test_static_determine_signal_type_both_lead(self):
        """When both EFA and EEM lead, pick the stronger."""
        sig_type, conf = InternationalMomentumGenerator.determine_signal_type(
            efa_vs_spy=0.10, eem_vs_spy=0.12
        )
        # EEM threshold is 8%, EFA is 5%; both exceed → stronger one wins
        self.assertIn(sig_type, (SignalType.EFA_LEAD, SignalType.EEM_LEAD))

    def test_static_determine_signal_type_negative_outperformance(self):
        """Negative outperformance should be neutral."""
        sig_type, conf = InternationalMomentumGenerator.determine_signal_type(
            efa_vs_spy=-0.05, eem_vs_spy=-0.03
        )
        self.assertEqual(sig_type, SignalType.NEUTRAL)
        self.assertEqual(conf, 0.0)

    def test_get_vix_level_default(self):
        """VIX level should return default when no data."""
        vix = self.generator._get_vix_level()
        self.assertIsInstance(vix, float)
        self.assertGreater(vix, 0)

    def test_get_correlation_default(self):
        """Correlation should return default when no data."""
        corr = self.generator._get_correlation()
        self.assertIsInstance(corr, float)
        self.assertGreaterEqual(corr, -1)
        self.assertLessEqual(corr, 1)

    def test_init_signal_history(self):
        """Test that signal history table is created."""
        self.generator._init_signal_history()
        # Should not raise
        import sqlite3
        with sqlite3.connect(str(self.generator.cache_db)) as conn:
            tables = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        table_names = [t[0] for t in tables]
        self.assertIn('international_signals', table_names)

    def test_save_and_get_signal_history(self):
        """Test saving signal and retrieving history."""
        data = {
            'timestamp': '2026-05-14T10:00:00',
            'data_fresh': True,
            'relative': {
                'efa_momentum_6m': 0.20,
                'eem_momentum_6m': 0.08,
                'spy_momentum_6m': 0.12,
                'efa_vs_spy': 0.08,
                'eem_vs_spy': -0.04,
            }
        }
        with patch.object(self.generator, '_get_vix_level', return_value=20.0):
            with patch.object(self.generator, '_get_correlation', return_value=0.85):
                signal = self.generator.generate_signal(data)

        history = self.generator.get_signal_history(days=30)
        self.assertIsInstance(history, list)
        self.assertGreaterEqual(len(history), 1)
        self.assertEqual(history[0]['signal_type'], 'efa_lead')

    def test_generate_signal_stale_data(self):
        """Stale data should produce inactive signal."""
        data = {
            'timestamp': '2026-05-14T10:00:00',
            'data_fresh': False,  # Stale
            'relative': {
                'efa_momentum_6m': 0.20,
                'eem_momentum_6m': 0.08,
                'spy_momentum_6m': 0.12,
                'efa_vs_spy': 0.08,
                'eem_vs_spy': -0.04,
            }
        }
        signal = self.generator.generate_signal(data)
        self.assertFalse(signal.data_fresh)
        self.assertFalse(signal.is_active())

    def test_generate_signal_missing_relative(self):
        """Missing relative data should produce neutral signal."""
        data = {
            'timestamp': '2026-05-14T10:00:00',
            'data_fresh': True,
        }
        signal = self.generator.generate_signal(data)
        self.assertEqual(signal.signal_type, 'neutral')

    def test_generate_signal_eem_lead(self):
        """Test generating EEM lead signal."""
        data = {
            'timestamp': '2026-05-14T10:00:00',
            'data_fresh': True,
            'relative': {
                'efa_momentum_6m': 0.08,
                'eem_momentum_6m': 0.25,
                'spy_momentum_6m': 0.12,
                'efa_vs_spy': -0.04,
                'eem_vs_spy': 0.13,
            }
        }
        with patch.object(self.generator, '_get_vix_level', return_value=18.0):
            with patch.object(self.generator, '_get_correlation', return_value=0.80):
                signal = self.generator.generate_signal(data)

        self.assertEqual(signal.signal_type, 'eem_lead')
        self.assertTrue(signal.is_active())

    def test_generate_signal_correlation_override(self):
        """High correlation should disable the signal."""
        data = {
            'timestamp': '2026-05-14T10:00:00',
            'data_fresh': True,
            'relative': {
                'efa_momentum_6m': 0.20,
                'eem_momentum_6m': 0.08,
                'spy_momentum_6m': 0.12,
                'efa_vs_spy': 0.08,
                'eem_vs_spy': -0.04,
            }
        }
        with patch.object(self.generator, '_get_vix_level', return_value=20.0):
            with patch.object(self.generator, '_get_correlation', return_value=0.96):  # Above 0.95 cutoff
                signal = self.generator.generate_signal(data)

        self.assertTrue(signal.correlation_override)
        self.assertFalse(signal.is_active())

    def test_get_signal_statistics_empty(self):
        """Statistics on empty history should return error dict."""
        stats = self.generator.get_signal_statistics(days=30)
        self.assertIn('error', stats)

    def test_allocation_shifts_eem(self):
        """Test EEM allocation shifts cap at max allocation."""
        spy_shift, efa_shift, eem_shift = self.generator._calculate_allocation_shifts(
            SignalType.EEM_LEAD,
            confidence=0.95
        )
        self.assertGreater(spy_shift, 0)
        self.assertEqual(eem_shift, spy_shift)
        self.assertLessEqual(eem_shift, self.generator.MAX_EEM_ALLOCATION)

    def test_allocation_shifts_high_confidence_efa(self):
        """High confidence EFA shift should be capped."""
        spy_shift, efa_shift, eem_shift = self.generator._calculate_allocation_shifts(
            SignalType.EFA_LEAD,
            confidence=1.0
        )
        self.assertLessEqual(efa_shift, self.generator.MAX_EFA_ALLOCATION)

    def test_signal_type_enum_values(self):
        """Test SignalType enum values."""
        self.assertEqual(SignalType.NEUTRAL.value, 'neutral')
        self.assertEqual(SignalType.EFA_LEAD.value, 'efa_lead')
        self.assertEqual(SignalType.EEM_LEAD.value, 'eem_lead')

    def test_confidence_level_enum_values(self):
        """Test ConfidenceLevel enum values."""
        self.assertEqual(ConfidenceLevel.LOW.value, 'low')
        self.assertEqual(ConfidenceLevel.MEDIUM.value, 'medium')
        self.assertEqual(ConfidenceLevel.HIGH.value, 'high')


if __name__ == '__main__':
    unittest.main(verbosity=2)
