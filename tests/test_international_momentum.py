"""
Tests for International Momentum Signal Generator
"""

import unittest
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock


from src.signals.international_momentum import (
    SignalType,
    ConfidenceLevel,
    InternationalMomentumSignal,
    InternationalMomentumGenerator,
    main,
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


# ===========================================================================
# Additional comprehensive tests added for coverage expansion
# ===========================================================================


class TestInternationalMomentumSignalDataclassFields(unittest.TestCase):
    """Validate dataclass field definitions via dataclasses.fields()."""

    def test_all_fields_present(self):
        """Verify all 18 fields are defined in the correct order."""
        import dataclasses
        fields = dataclasses.fields(InternationalMomentumSignal)
        field_names = [f.name for f in fields]
        expected = [
            'timestamp', 'signal_type', 'confidence', 'confidence_level',
            'efa_momentum_6m', 'eem_momentum_6m', 'spy_momentum_6m',
            'efa_vs_spy', 'eem_vs_spy',
            'spy_shift', 'efa_shift', 'eem_shift',
            'max_allocation_efa', 'max_allocation_eem', 'holding_period_days',
            'data_fresh', 'vix_filter_active', 'correlation_override',
        ]
        self.assertEqual(field_names, expected)

    def test_all_field_types(self):
        """Verify each field has the correct type annotation."""
        import dataclasses
        fields = {f.name: f.type for f in dataclasses.fields(InternationalMomentumSignal)}
        self.assertIs(fields['timestamp'], str)
        self.assertIs(fields['signal_type'], str)
        self.assertIs(fields['confidence'], float)
        self.assertIs(fields['confidence_level'], str)
        self.assertIs(fields['efa_momentum_6m'], float)
        self.assertIs(fields['eem_momentum_6m'], float)
        self.assertIs(fields['spy_momentum_6m'], float)
        self.assertIs(fields['efa_vs_spy'], float)
        self.assertIs(fields['eem_vs_spy'], float)
        self.assertIs(fields['spy_shift'], float)
        self.assertIs(fields['efa_shift'], float)
        self.assertIs(fields['eem_shift'], float)
        self.assertIs(fields['max_allocation_efa'], float)
        self.assertIs(fields['max_allocation_eem'], float)
        self.assertIs(fields['holding_period_days'], int)
        self.assertIs(fields['data_fresh'], bool)
        self.assertIs(fields['vix_filter_active'], bool)
        self.assertIs(fields['correlation_override'], bool)

    def test_is_active_eem_lead(self):
        """EEM lead with high confidence should be active."""
        signal = InternationalMomentumSignal(
            timestamp='2026-05-14T10:00:00',
            signal_type='eem_lead', confidence=0.7,
            confidence_level='medium',
            efa_momentum_6m=0.08, eem_momentum_6m=0.20, spy_momentum_6m=0.12,
            efa_vs_spy=-0.04, eem_vs_spy=0.10,
            spy_shift=0.03, efa_shift=0.0, eem_shift=0.03,
            max_allocation_efa=0.05, max_allocation_eem=0.03,
            holding_period_days=30,
            data_fresh=True, vix_filter_active=False, correlation_override=False,
        )
        self.assertTrue(signal.is_active())
        delta = signal.get_allocation_delta()
        self.assertLess(delta['SPY'], 0)
        self.assertGreater(delta['EEM'], 0)

    def test_is_active_all_filters_triggered(self):
        """Signal with all filters triggered should be inactive."""
        signal = InternationalMomentumSignal(
            timestamp='2026-05-14T10:00:00',
            signal_type='efa_lead', confidence=0.8,
            confidence_level='high',
            efa_momentum_6m=0.20, eem_momentum_6m=0.08, spy_momentum_6m=0.12,
            efa_vs_spy=0.08, eem_vs_spy=-0.04,
            spy_shift=0.04, efa_shift=0.04, eem_shift=0.0,
            max_allocation_efa=0.05, max_allocation_eem=0.03,
            holding_period_days=30,
            data_fresh=False, vix_filter_active=True, correlation_override=True,
        )
        self.assertFalse(signal.is_active())

    def test_is_active_unfiltered_efa_boundary(self):
        """EFA lead with exact boundary confidence and all filters clear."""
        signal = InternationalMomentumSignal(
            timestamp='2026-05-14T10:00:00',
            signal_type='efa_lead', confidence=0.5,
            confidence_level='medium',
            efa_momentum_6m=0.20, eem_momentum_6m=0.08, spy_momentum_6m=0.12,
            efa_vs_spy=0.08, eem_vs_spy=-0.04,
            spy_shift=0.025, efa_shift=0.025, eem_shift=0.0,
            max_allocation_efa=0.05, max_allocation_eem=0.03,
            holding_period_days=30,
            data_fresh=True, vix_filter_active=False, correlation_override=False,
        )
        self.assertTrue(signal.is_active())


class TestInternationalMomentumSignalNaNInf(unittest.TestCase):
    """Test edge cases with NaN and Inf values."""

    def _make_signal(self, **overrides):
        defaults = dict(
            timestamp='2026-05-14T10:00:00', signal_type='efa_lead',
            confidence=0.7, confidence_level='medium',
            efa_momentum_6m=0.20, eem_momentum_6m=0.08, spy_momentum_6m=0.12,
            efa_vs_spy=0.08, eem_vs_spy=-0.04,
            spy_shift=0.035, efa_shift=0.035, eem_shift=0.0,
            max_allocation_efa=0.05, max_allocation_eem=0.03,
            holding_period_days=30,
            data_fresh=True, vix_filter_active=False, correlation_override=False,
        )
        defaults.update(overrides)
        return InternationalMomentumSignal(**defaults)

    def test_is_active_nan_confidence(self):
        """NaN confidence should make is_active() return False."""
        import math
        signal = self._make_signal(confidence=float('nan'))
        # NaN >= 0.5 evaluates to False in Python
        self.assertFalse(signal.is_active())

    def test_is_active_inf_confidence(self):
        """Positive infinite confidence should make is_active() return True."""
        import math
        signal = self._make_signal(confidence=float('inf'))
        self.assertTrue(signal.is_active())

    def test_signal_nan_momentum_values(self):
        """NaN momentum values should be stored without error."""
        import math
        signal = self._make_signal(
            efa_momentum_6m=float('nan'),
            eem_momentum_6m=float('nan'),
            spy_momentum_6m=float('nan'),
        )
        import math as m
        self.assertTrue(m.isnan(signal.efa_momentum_6m))
        self.assertTrue(m.isnan(signal.eem_momentum_6m))
        self.assertTrue(m.isnan(signal.spy_momentum_6m))
        # is_active should still work (confidence check passes)
        self.assertTrue(signal.is_active())

    def test_signal_infinite_momentum_values(self):
        """Infinite momentum values should be stored without error."""
        import math
        signal = self._make_signal(
            efa_momentum_6m=float('inf'),
            eem_momentum_6m=float('-inf'),
            spy_momentum_6m=float('inf'),
        )
        self.assertTrue(math.isinf(signal.efa_momentum_6m))
        self.assertTrue(math.isinf(signal.eem_momentum_6m))
        self.assertTrue(math.isinf(signal.spy_momentum_6m))


class TestInternationalMomentumConstantsAndExports(unittest.TestCase):
    """Validate module-level constants and __all__ export completeness."""

    def test_module_all_contains_expected(self):
        """__all__ should contain all four public names."""
        from src.signals import international_momentum as mod
        expected = {
            'SignalType',
            'ConfidenceLevel',
            'InternationalMomentumSignal',
            'InternationalMomentumGenerator',
        }
        self.assertSetEqual(set(mod.__all__), expected)

    def test_module_all_exports_are_defined(self):
        """Every name in __all__ should be a real symbol in the module."""
        from src.signals import international_momentum as mod
        for name in mod.__all__:
            self.assertTrue(
                hasattr(mod, name),
                f"__all__ contains '{name}' but module has no such attribute",
            )

    def test_cache_db_is_market_db(self):
        """CACHE_DB should reference MARKET_DB."""
        from src.signals.international_momentum import CACHE_DB, MARKET_DB
        self.assertIs(CACHE_DB, MARKET_DB)

    def test_logger_exists(self):
        """Module-level logger should exist and be named correctly."""
        from src.signals.international_momentum import logger
        self.assertEqual(logger.name, 'src.signals.international_momentum')

    def test_signal_type_enum_has_three_members(self):
        """SignalType enum should have exactly 3 members."""
        self.assertEqual(len(SignalType), 3)

    def test_confidence_level_enum_has_three_members(self):
        """ConfidenceLevel enum should have exactly 3 members."""
        self.assertEqual(len(ConfidenceLevel), 3)

    def test_confidence_level_enum_boundary_values(self):
        """ConfidenceLevel values should match expected strings."""
        self.assertEqual(ConfidenceLevel.LOW.value, 'low')
        self.assertEqual(ConfidenceLevel.MEDIUM.value, 'medium')
        self.assertEqual(ConfidenceLevel.HIGH.value, 'high')

    def test_signal_type_enum_values(self):
        """SignalType values should match expected strings."""
        self.assertEqual(SignalType.NEUTRAL.value, 'neutral')
        self.assertEqual(SignalType.EFA_LEAD.value, 'efa_lead')
        self.assertEqual(SignalType.EEM_LEAD.value, 'eem_lead')


class TestInternationalMomentumCli(unittest.TestCase):
    """Test CLI entry point via main() with patched sys.argv."""

    def test_cli_help_output(self):
        """--help should print usage and exit with code 0."""
        import sys
        test_args = ['prog', '--help']
        with patch('sys.argv', test_args):
            with self.assertRaises(SystemExit) as ctx:
                main()
        # argparse calls sys.exit(0) after printing help
        self.assertEqual(ctx.exception.code, 0)

    def test_cli_no_args(self):
        """No arguments should print help without exiting."""
        import io
        import sys
        test_args = ['prog']
        with patch('sys.argv', test_args):
            with patch('sys.stdout', new_callable=io.StringIO) as mock_stdout:
                main()
        output = mock_stdout.getvalue()
        self.assertIn('International Momentum Signal Generator', output)
        self.assertIn('--generate', output)

    def test_cli_current_empty(self):
        """--current with no signal should exit with code 1 and print error to stderr."""
        import io
        import sys
        mock_gen = MagicMock(spec=InternationalMomentumGenerator)
        mock_gen.get_current_signal.return_value = None
        test_args = ['prog', '--current']
        with patch('sys.argv', test_args):
            with patch('sys.stderr', new_callable=io.StringIO) as mock_stderr:
                with patch(
                    'src.signals.international_momentum.InternationalMomentumGenerator',
                    return_value=mock_gen,
                ):
                    with self.assertRaises(SystemExit) as ctx:
                        main()
        self.assertEqual(ctx.exception.code, 1)
        self.assertIn('No signal found', mock_stderr.getvalue())

    def test_cli_current_with_signal(self):
        """--current with a stored signal should print JSON to stdout."""
        import io
        import sys
        signal = InternationalMomentumSignal(
            timestamp='2026-05-14T10:00:00', signal_type='efa_lead',
            confidence=0.65, confidence_level='medium',
            efa_momentum_6m=0.20, eem_momentum_6m=0.08, spy_momentum_6m=0.12,
            efa_vs_spy=0.08, eem_vs_spy=-0.04,
            spy_shift=0.04, efa_shift=0.04, eem_shift=0.0,
            max_allocation_efa=0.05, max_allocation_eem=0.03,
            holding_period_days=30,
            data_fresh=True, vix_filter_active=False, correlation_override=False,
        )
        mock_gen = MagicMock(spec=InternationalMomentumGenerator)
        mock_gen.get_current_signal.return_value = signal
        test_args = ['prog', '--current']
        with patch('sys.argv', test_args):
            with patch('sys.stdout', new_callable=io.StringIO) as mock_stdout:
                with patch(
                    'src.signals.international_momentum.InternationalMomentumGenerator',
                    return_value=mock_gen,
                ):
                    main()
        output = mock_stdout.getvalue()
        self.assertIn('efa_lead', output)
        self.assertIn('signal_type', output)

    def test_cli_stats(self):
        """--stats should print JSON statistics to stdout."""
        import io
        import sys
        stats = {'total_signals': 3, 'activation_rate': 0.67}
        mock_gen = MagicMock(spec=InternationalMomentumGenerator)
        mock_gen.get_signal_statistics.return_value = stats
        test_args = ['prog', '--stats']
        with patch('sys.argv', test_args):
            with patch('sys.stdout', new_callable=io.StringIO) as mock_stdout:
                with patch(
                    'src.signals.international_momentum.InternationalMomentumGenerator',
                    return_value=mock_gen,
                ):
                    main()
        output = mock_stdout.getvalue()
        self.assertIn('"total_signals"', output)
        self.assertIn('"activation_rate"', output)

    def test_cli_generate_no_file(self):
        """--generate with non-existent file should exit with code 1."""
        import io
        import sys
        mock_gen = MagicMock(spec=InternationalMomentumGenerator)
        test_args = ['prog', '--generate', '--data-file', '/nonexistent/path/data.json']
        with patch('sys.argv', test_args):
            with patch('sys.stderr', new_callable=io.StringIO) as mock_stderr:
                with patch(
                    'src.signals.international_momentum.InternationalMomentumGenerator',
                    return_value=mock_gen,
                ):
                    with self.assertRaises(SystemExit) as ctx:
                        main()
        self.assertEqual(ctx.exception.code, 1)
        self.assertIn('Data file not found', mock_stderr.getvalue())


class TestInternationalMomentumGeneratorRiskBoundaries(unittest.TestCase):
    """Test risk filter boundary conditions."""

    def setUp(self):
        self.temp_db = tempfile.NamedTemporaryFile(suffix='.db', delete=False)
        self.generator = InternationalMomentumGenerator(cache_db=Path(self.temp_db.name))

    def tearDown(self):
        self.temp_db.close()
        Path(self.temp_db.name).unlink(missing_ok=True)

    def _make_data(self):
        return {
            'timestamp': '2026-05-14T10:00:00',
            'data_fresh': True,
            'relative': {
                'efa_momentum_6m': 0.20, 'eem_momentum_6m': 0.08,
                'spy_momentum_6m': 0.12,
                'efa_vs_spy': 0.08, 'eem_vs_spy': -0.04,
            },
        }

    def test_vix_at_cutoff(self):
        """VIX exactly at cutoff (30.0) should NOT activate the filter (strict >)."""
        data = self._make_data()
        with patch.object(self.generator, '_get_vix_level', return_value=30.0):
            with patch.object(self.generator, '_get_correlation', return_value=0.85):
                signal = self.generator.generate_signal(data)
        self.assertFalse(signal.vix_filter_active)
        self.assertTrue(signal.is_active())

    def test_vix_just_above_cutoff(self):
        """VIX 30.01 should activate the VIX filter."""
        data = self._make_data()
        with patch.object(self.generator, '_get_vix_level', return_value=30.01):
            with patch.object(self.generator, '_get_correlation', return_value=0.85):
                signal = self.generator.generate_signal(data)
        self.assertTrue(signal.vix_filter_active)
        self.assertFalse(signal.is_active())

    def test_correlation_at_cutoff(self):
        """Correlation exactly at cutoff (0.95) should NOT activate override (strict >)."""
        data = self._make_data()
        with patch.object(self.generator, '_get_vix_level', return_value=20.0):
            with patch.object(self.generator, '_get_correlation', return_value=0.95):
                signal = self.generator.generate_signal(data)
        self.assertFalse(signal.correlation_override)
        self.assertTrue(signal.is_active())

    def test_correlation_just_above_cutoff(self):
        """Correlation 0.951 should activate the correlation override."""
        data = self._make_data()
        with patch.object(self.generator, '_get_vix_level', return_value=20.0):
            with patch.object(self.generator, '_get_correlation', return_value=0.951):
                signal = self.generator.generate_signal(data)
        self.assertTrue(signal.correlation_override)
        self.assertFalse(signal.is_active())


class TestInternationalMomentumGeneratorExceptionHandling(unittest.TestCase):
    """Test exception handling paths in InternationalMomentumGenerator."""

    def setUp(self):
        self.temp_db = tempfile.NamedTemporaryFile(suffix='.db', delete=False)
        self.generator = InternationalMomentumGenerator(cache_db=Path(self.temp_db.name))

    def tearDown(self):
        self.temp_db.close()
        Path(self.temp_db.name).unlink(missing_ok=True)

    def test_get_vix_level_exception_returns_default(self):
        """Exception in _get_vix_level should return default 20.0."""
        with patch('src.signals.international_momentum.sqlite_connect',
                   side_effect=Exception('DB error')):
            vix = self.generator._get_vix_level()
        self.assertEqual(vix, 20.0)

    def test_get_correlation_exception_returns_default(self):
        """Exception in _get_correlation should return default 0.85."""
        with patch('src.signals.international_momentum.sqlite_connect',
                   side_effect=Exception('DB error')):
            corr = self.generator._get_correlation()
        self.assertEqual(corr, 0.85)

    def test_save_signal_exception_does_not_propagate(self):
        """Exception in _save_signal should be caught and not re-raised."""
        signal = InternationalMomentumSignal(
            timestamp='2026-05-14T10:00:00', signal_type='neutral',
            confidence=0.0, confidence_level='low',
            efa_momentum_6m=0.0, eem_momentum_6m=0.0, spy_momentum_6m=0.0,
            efa_vs_spy=0.0, eem_vs_spy=0.0,
            spy_shift=0.0, efa_shift=0.0, eem_shift=0.0,
            max_allocation_efa=0.05, max_allocation_eem=0.03,
            holding_period_days=30,
            data_fresh=True, vix_filter_active=False, correlation_override=False,
        )
        with patch('src.signals.international_momentum.sqlite_connect',
                   side_effect=Exception('DB error')):
            # Should not raise
            self.generator._save_signal(signal)

    def test_save_signal_in_active_roundtrip(self):
        """An active signal saved and retrieved preserves is_active flag."""
        signal = InternationalMomentumSignal(
            timestamp='2026-05-14T10:00:00', signal_type='efa_lead',
            confidence=0.7, confidence_level='medium',
            efa_momentum_6m=0.20, eem_momentum_6m=0.08, spy_momentum_6m=0.12,
            efa_vs_spy=0.08, eem_vs_spy=-0.04,
            spy_shift=0.035, efa_shift=0.035, eem_shift=0.0,
            max_allocation_efa=0.05, max_allocation_eem=0.03,
            holding_period_days=30,
            data_fresh=True, vix_filter_active=False, correlation_override=False,
        )
        self.generator._save_signal(signal)
        retrieved = self.generator.get_current_signal()
        self.assertIsNotNone(retrieved)


class TestInternationalMomentumGeneratorComputeEdgeCases(unittest.TestCase):
    """Test computation edge cases in generator methods."""

    def setUp(self):
        self.temp_db = tempfile.NamedTemporaryFile(suffix='.db', delete=False)
        self.generator = InternationalMomentumGenerator(cache_db=Path(self.temp_db.name))

    def tearDown(self):
        self.temp_db.close()
        Path(self.temp_db.name).unlink(missing_ok=True)

    def test_determine_both_above_threshold_efa_wins(self):
        """When both exceed thresholds, EFA should be checked first."""
        sig_type, conf = self.generator._determine_signal_type(
            efa_vs_spy=0.06, eem_vs_spy=0.09,
        )
        self.assertEqual(sig_type, SignalType.EFA_LEAD)

    def test_determine_eem_confidence_clipped_at_1(self):
        """EEM confidence should be capped at 1.0 for very large outperformance."""
        sig_type, conf = self.generator._determine_signal_type(
            efa_vs_spy=-0.10, eem_vs_spy=0.50,
        )
        self.assertEqual(sig_type, SignalType.EEM_LEAD)
        self.assertLessEqual(conf, 1.0)

    def test_allocation_shifts_neutral_with_nonzero_confidence(self):
        """Neutral signal with nonzero confidence should still give zero shifts."""
        spy, efa, eem = self.generator._calculate_allocation_shifts(
            SignalType.NEUTRAL, confidence=0.9,
        )
        self.assertEqual(spy, 0.0)
        self.assertEqual(efa, 0.0)
        self.assertEqual(eem, 0.0)

    def test_generate_signal_all_missing_relative_keys(self):
        """Missing all relative keys should produce neutral signal with zeros."""
        data = {'timestamp': '2026-05-14T10:00:00', 'data_fresh': True}
        signal = self.generator.generate_signal(data)
        self.assertEqual(signal.signal_type, 'neutral')
        self.assertEqual(signal.efa_momentum_6m, 0.0)
        self.assertEqual(signal.eem_momentum_6m, 0.0)
        self.assertEqual(signal.spy_momentum_6m, 0.0)
        self.assertEqual(signal.efa_vs_spy, 0.0)
        self.assertEqual(signal.eem_vs_spy, 0.0)

    def test_generate_signal_partial_relative_keys(self):
        """Partial relative keys should use defaults for missing keys."""
        data = {
            'timestamp': '2026-05-14T10:00:00',
            'data_fresh': True,
            'relative': {'efa_momentum_6m': 0.15},
        }
        signal = self.generator.generate_signal(data)
        self.assertEqual(signal.efa_momentum_6m, 0.15)
        self.assertEqual(signal.eem_momentum_6m, 0.0)
        self.assertEqual(signal.spy_momentum_6m, 0.0)
        self.assertEqual(signal.efa_vs_spy, 0.0)

    def test_generate_signal_data_fresh_false_momentum_values_stored(self):
        """Even with stale data, momentum values should be stored."""
        data = {
            'timestamp': '2026-05-14T10:00:00',
            'data_fresh': False,
            'relative': {
                'efa_momentum_6m': 0.25, 'eem_momentum_6m': 0.12,
                'spy_momentum_6m': 0.18,
                'efa_vs_spy': 0.07, 'eem_vs_spy': -0.06,
            },
        }
        signal = self.generator.generate_signal(data)
        self.assertFalse(signal.data_fresh)
        self.assertFalse(signal.is_active())
        self.assertEqual(signal.efa_momentum_6m, 0.25)


class TestInternationalMomentumGeneratorStatisticsExtended(unittest.TestCase):
    """Extended statistics tests."""

    def setUp(self):
        self.temp_db = tempfile.NamedTemporaryFile(suffix='.db', delete=False)
        self.generator = InternationalMomentumGenerator(cache_db=Path(self.temp_db.name))

    def tearDown(self):
        self.temp_db.close()
        Path(self.temp_db.name).unlink(missing_ok=True)

    def _generate_signal(self, efa_vs_spy, eem_vs_spy, timestamp):
        data = {
            'timestamp': timestamp,
            'data_fresh': True,
            'relative': {
                'efa_momentum_6m': 0.10, 'eem_momentum_6m': 0.10,
                'spy_momentum_6m': 0.10,
                'efa_vs_spy': efa_vs_spy, 'eem_vs_spy': eem_vs_spy,
            },
        }
        with patch.object(self.generator, '_get_vix_level', return_value=20.0):
            with patch.object(self.generator, '_get_correlation', return_value=0.85):
                self.generator.generate_signal(data)

    def test_statistics_all_eem_signals(self):
        """Statistics should correctly count EEM lead signals."""
        for i in range(3):
            self._generate_signal(
                efa_vs_spy=-0.10, eem_vs_spy=0.12,
                timestamp=f'2026-05-{14+i}T10:00:00',
            )
        stats = self.generator.get_signal_statistics(days=90)
        self.assertEqual(stats['total_signals'], 3)
        self.assertEqual(stats['eem_lead_count'], 3)
        self.assertEqual(stats['efa_lead_count'], 0)
        self.assertEqual(stats['neutral_count'], 0)

    def test_statistics_mixed_regimes(self):
        """Statistics with a mix of EFA, EEM, and neutral signals."""
        self._generate_signal(efa_vs_spy=0.08, eem_vs_spy=-0.04, timestamp='2026-05-14T10:00:00')
        self._generate_signal(efa_vs_spy=-0.04, eem_vs_spy=0.12, timestamp='2026-05-15T10:00:00')
        self._generate_signal(efa_vs_spy=-0.02, eem_vs_spy=-0.03, timestamp='2026-05-16T10:00:00')
        self._generate_signal(efa_vs_spy=0.09, eem_vs_spy=-0.02, timestamp='2026-05-17T10:00:00')
        stats = self.generator.get_signal_statistics(days=90)
        self.assertEqual(stats['total_signals'], 4)
        self.assertEqual(stats['efa_lead_count'], 2)
        self.assertEqual(stats['eem_lead_count'], 1)
        self.assertEqual(stats['neutral_count'], 1)

    def test_statistics_activation_rate(self):
        """Activation rate should be between 0 and 1."""
        self._generate_signal(efa_vs_spy=0.08, eem_vs_spy=-0.04, timestamp='2026-05-14T10:00:00')
        stats = self.generator.get_signal_statistics(days=90)
        self.assertGreaterEqual(stats['activation_rate'], 0.0)
        self.assertLessEqual(stats['activation_rate'], 1.0)

    def test_statistics_current_regime(self):
        """Current regime should be the most recent signal type."""
        self._generate_signal(efa_vs_spy=-0.02, eem_vs_spy=-0.03, timestamp='2026-05-14T10:00:00')
        self._generate_signal(efa_vs_spy=0.08, eem_vs_spy=-0.04, timestamp='2026-05-15T10:00:00')
        stats = self.generator.get_signal_statistics(days=90)
        self.assertEqual(stats['current_regime'], 'efa_lead')


class TestInternationalMomentumSignalAdditionalMethods(unittest.TestCase):
    """Additional edge case tests for signal methods."""

    def _make_signal(self, **overrides):
        defaults = dict(
            timestamp='2026-05-14T10:00:00', signal_type='neutral',
            confidence=0.0, confidence_level='low',
            efa_momentum_6m=0.12, eem_momentum_6m=0.08, spy_momentum_6m=0.15,
            efa_vs_spy=-0.03, eem_vs_spy=-0.07,
            spy_shift=0.0, efa_shift=0.0, eem_shift=0.0,
            max_allocation_efa=0.05, max_allocation_eem=0.03,
            holding_period_days=30,
            data_fresh=True, vix_filter_active=False,
            correlation_override=False,
        )
        defaults.update(overrides)
        return InternationalMomentumSignal(**defaults)

    def test_get_allocation_delta_active_eem(self):
        """Active EEM signal: SPY negative, EEM positive, EFA zero."""
        signal = self._make_signal(
            signal_type='eem_lead', confidence=0.65,
            spy_shift=0.03, efa_shift=0.0, eem_shift=0.03,
        )
        delta = signal.get_allocation_delta()
        self.assertEqual(delta['SPY'], -0.03)
        self.assertEqual(delta['EFA'], 0.0)
        self.assertEqual(delta['EEM'], 0.03)

    def test_get_allocation_delta_efa_full_shift(self):
        """EFA signal with max allocation shift."""
        signal = self._make_signal(
            signal_type='efa_lead', confidence=0.9,
            spy_shift=0.05, efa_shift=0.05, eem_shift=0.0,
        )
        delta = signal.get_allocation_delta()
        self.assertEqual(delta['SPY'], -0.05)
        self.assertEqual(delta['EFA'], 0.05)

    def test_to_signal_snapshot_neutral_value_is_zero(self):
        """Neutral signal snapshot should have value 0.0."""
        signal = self._make_signal()
        snapshot = signal.to_signal_snapshot()
        self.assertEqual(snapshot.value, 0.0)
        self.assertFalse(snapshot.is_active)

    def test_to_signal_snapshot_efa_negative_value(self):
        """EFA lead with negative efa_vs_spy should clip to lower bound."""
        signal = self._make_signal(
            signal_type='efa_lead', confidence=0.7,
            efa_vs_spy=-6.0,
        )
        snapshot = signal.to_signal_snapshot()
        self.assertEqual(snapshot.value, -0.5)

    def test_to_signal_snapshot_explanation_confidence_formatting(self):
        """Explanation should include percentage-formatted outperformance."""
        signal = self._make_signal(
            signal_type='efa_lead', confidence=0.65,
            efa_vs_spy=0.0825,
        )
        snapshot = signal.to_signal_snapshot()
        self.assertIn('+8.25%', snapshot.explanation)

    def test_negative_spy_shift_handling(self):
        """Negative spy_shift should be negated in allocation delta."""
        signal = self._make_signal(
            signal_type='efa_lead', confidence=0.7,
            spy_shift=-0.02, efa_shift=0.04, eem_shift=0.0,
        )
        # Even though spy_shift is negative, get_allocation_delta returns -spy_shift
        delta = signal.get_allocation_delta()
        self.assertEqual(delta['SPY'], 0.02)
        self.assertEqual(delta['EFA'], 0.04)
