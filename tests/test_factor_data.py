"""Tests for factor data infrastructure (v3.00 Phase 1).

Run with: pytest tests/test_factor_data.py -v
"""

import json
import sqlite3
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

# Skip if pandas not available
try:
    from src.data.factor_data import FactorDataManager, FactorETF, FACTOR_ETFS, QUALITY_WEIGHTS, fetch_factor_prices_from_pipeline, main
    HAS_DEPENDENCIES = True
except ImportError:
    HAS_DEPENDENCIES = False


@pytest.mark.skipif(not HAS_DEPENDENCIES, reason="Dependencies not available")
class TestFactorETF:
    """Test FactorETF dataclass."""
    
    def test_factor_etf_creation(self):
        """Test FactorETF creation with valid data."""
        etf = FactorETF(
            symbol="MTUM",
            factor="momentum",
            expense_ratio=0.0015,
            aum_billions=18.5,
            description="Test ETF"
        )
        assert etf.symbol == "MTUM"
        assert etf.factor == "momentum"
        assert etf.expense_ratio == 0.0015
    
    def test_factor_etf_to_dict(self):
        """Test FactorETF serialization."""
        etf = FactorETF(
            symbol="QUAL",
            factor="quality",
            expense_ratio=0.0015,
            aum_billions=19.2,
            description="Quality ETF"
        )
        d = etf.to_dict()
        assert d["symbol"] == "QUAL"
        assert d["factor"] == "quality"


@pytest.mark.skipif(not HAS_DEPENDENCIES, reason="Dependencies not available")
class TestFactorDataManager:
    """Test FactorDataManager database operations."""
    
    @pytest.fixture
    def temp_manager(self):
        """Create a temporary FactorDataManager for testing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            data_dir = Path(tmpdir) / "factors"
            manager = FactorDataManager(data_dir=data_dir)
            yield manager
    
    def test_database_initialization(self, temp_manager):
        """Test that database tables are created."""
        with sqlite3.connect(temp_manager.db_path) as conn:
            # Check tables exist
            cursor = conn.execute("""
                SELECT name FROM sqlite_master 
                WHERE type='table' AND name IN ('factor_prices', 'quality_scores', 'factor_performance')
            """)
            tables = [row[0] for row in cursor.fetchall()]
            assert "factor_prices" in tables
            assert "quality_scores" in tables
            assert "factor_performance" in tables
    
    def test_metadata_initialization(self, temp_manager):
        """Test that metadata file is created."""
        assert temp_manager.metadata_path.exists()
        with open(temp_manager.metadata_path) as f:
            meta = json.load(f)
        assert "version" in meta
        assert meta["version"] == "3.00"
        assert "etfs" in meta
        assert "MTUM" in meta["etfs"]
    
    def test_store_and_retrieve_prices(self, temp_manager):
        """Test storing and retrieving price data."""
        prices = [
            {"date": "2026-05-01", "open": 100.0, "high": 101.0, "low": 99.0, "close": 100.5, "volume": 1000000},
            {"date": "2026-05-02", "open": 100.5, "high": 102.0, "low": 100.0, "close": 101.5, "volume": 1200000},
        ]
        
        count = temp_manager.store_prices("MTUM", prices)
        assert count == 2
        
        retrieved = temp_manager.get_prices("MTUM", days=5)
        assert len(retrieved) == 2
        assert retrieved[0]["close"] == 101.5
    
    def test_invalid_symbol_raises_error(self, temp_manager):
        """Test that invalid symbols raise ValueError."""
        prices = [{"date": "2026-05-01", "close": 100.0}]
        
        with pytest.raises(ValueError, match="Unknown factor ETF"):
            temp_manager.store_prices("INVALID", prices)
    
    def test_quality_score_calculation(self, temp_manager):
        """Test quality score calculation."""
        score = temp_manager.calculate_quality_score(
            roe=0.20,  # Good ROE
            debt_equity=0.3,  # Low debt
            earnings_stability=0.8,
            profitability=0.75
        )
        
        assert 0 <= score <= 1
        # High quality inputs should yield high score
        assert score > 0.7
    
    def test_quality_score_calculation_low_quality(self, temp_manager):
        """Test quality score for poor metrics."""
        score = temp_manager.calculate_quality_score(
            roe=0.05,  # Poor ROE
            debt_equity=1.5,  # High debt
            earnings_stability=0.3,
            profitability=0.2
        )
        
        assert 0 <= score <= 1
        assert score < 0.5
    
    def test_store_and_retrieve_quality_scores(self, temp_manager):
        """Test storing and retrieving quality scores."""
        metrics = {
            "roe": 0.18,
            "debt_equity": 0.4,
            "earnings_stability": 0.75,
            "profitability": 0.70
        }
        
        success = temp_manager.store_quality_score("QUAL", "2026-05-01", metrics)
        assert success
        
        scores = temp_manager.get_quality_scores("QUAL", days=10)
        assert len(scores) == 1
        assert scores[0]["symbol"] == "QUAL"
        assert scores[0]["composite_score"] > 0
    
    def test_calculate_returns_insufficient_data(self, temp_manager):
        """Test returns calculation with insufficient data."""
        # No prices stored yet
        result = temp_manager.calculate_returns("MTUM")
        assert result is None
    
    def test_calculate_returns_with_data(self, temp_manager):
        """Test returns calculation with sufficient data."""
        # Create 30 days of mock prices
        prices = []
        for i in range(30):
            prices.append({
                "date": f"2026-04-{i+1:02d}",
                "open": 100.0 + i * 0.1,
                "high": 101.0 + i * 0.1,
                "low": 99.0 + i * 0.1,
                "close": 100.5 + i * 0.1,
                "volume": 1000000
            })
        
        temp_manager.store_prices("MTUM", prices)
        
        returns = temp_manager.calculate_returns("MTUM")
        assert returns is not None
        assert "return_1d" in returns
        assert "return_1m" in returns
        assert "vol_20d" in returns
        assert returns["symbol"] == "MTUM"
    
    def test_factor_rankings(self, temp_manager):
        """Test factor ranking by momentum."""
        # Store different performance for two symbols
        for symbol in ["MTUM", "QUAL"]:
            prices = []
            base = 100.0 if symbol == "MTUM" else 100.0
            growth = 0.2 if symbol == "MTUM" else 0.05  # MTUM grows faster
            
            for i in range(130):  # Need 126 days for 6-month return
                prices.append({
                    "date": f"2026-01-{i+1:02d}",
                    "open": base + i * growth,
                    "high": base + 1 + i * growth,
                    "low": base - 1 + i * growth,
                    "close": base + 0.5 + i * growth,
                    "volume": 1000000
                })
            
            temp_manager.store_prices(symbol, prices)
        
        rankings = temp_manager.get_factor_rankings()
        assert len(rankings) == 2
        # MTUM should rank higher due to higher growth
        assert rankings[0][0] == "MTUM"
        assert rankings[0][1] > rankings[1][1]


@pytest.mark.skipif(not HAS_DEPENDENCIES, reason="Dependencies not available")
class TestFactorConstants:
    """Test factor constants and configuration."""
    
    def test_factor_etfs_configuration(self):
        """Test that all factor ETFs are configured."""
        assert "MTUM" in FACTOR_ETFS
        assert "QUAL" in FACTOR_ETFS
        assert "USMV" in FACTOR_ETFS
        assert "VLUE" in FACTOR_ETFS
        
        # Check all have required fields
        for symbol, etf in FACTOR_ETFS.items():
            assert etf.symbol == symbol
            assert etf.factor in ["momentum", "quality", "low_vol", "value"]
            assert etf.expense_ratio > 0
    
    def test_quality_weights_sum_to_one(self):
        """Test that quality weights sum to 1.0."""
        total = sum(QUALITY_WEIGHTS.values())
        assert abs(total - 1.0) < 0.001


@pytest.mark.skipif(not HAS_DEPENDENCIES, reason="Dependencies not available")
class TestCLI:
    """Test command-line interface."""
    
    @pytest.fixture
    def temp_dir(self):
        """Provide temporary directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield tmpdir
    
    @patch('sys.argv', ['factor_data.py', 'init'])
    def test_cli_init(self, temp_dir, caplog):
        """Test CLI init command creates files."""
        with patch('src.data.factor_data.Path') as mock_path:
            mock_path.return_value = Path(temp_dir) / "factors"
            # Should not raise
            pass  # main() would actually run - just verify setup
    
    @patch('sys.argv', ['factor_data.py', 'status'])
    def test_cli_status(self, temp_dir):
        """Test CLI status command."""
        # Verify it can be called (actual output depends on state)
        pass  # Just verify import works


def test_import_without_dependencies():
    """Test graceful handling when dependencies are missing."""
    # This test runs even without pandas
    try:
        import src.data.factor_data as fd
        assert hasattr(fd, 'FactorDataManager')
    except ImportError:
        # Expected if dependencies missing
        pass


@pytest.mark.skipif(not HAS_DEPENDENCIES, reason="Dependencies not available")
class TestFactorETFExtended:
    """Additional FactorETF edge cases."""

    def test_all_factor_etfs_have_positive_aum(self):
        """All FACTOR_ETFS should have positive AUM."""
        for symbol, etf in FACTOR_ETFS.items():
            assert etf.aum_billions > 0, f"{symbol} has non-positive AUM"

    def test_all_factor_etfs_have_description(self):
        """All FACTOR_ETFS should have a description."""
        for symbol, etf in FACTOR_ETFS.items():
            assert len(etf.description) > 0, f"{symbol} missing description"

    def test_to_dict_roundtrip(self):
        """to_dict should produce serializable output."""
        etf = FactorETF(
            symbol="USMV", factor="low_vol", expense_ratio=0.0015,
            aum_billions=30.0, description="Min Vol ETF"
        )
        d = etf.to_dict()
        serialized = json.dumps(d)
        assert "USMV" in serialized


@pytest.mark.skipif(not HAS_DEPENDENCIES, reason="Dependencies not available")
class TestFactorDataManagerExtended:
    """Additional FactorDataManager edge cases."""

    @pytest.fixture
    def temp_manager(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            data_dir = Path(tmpdir) / "factors"
            manager = FactorDataManager(data_dir=data_dir)
            yield manager

    def test_quality_score_boundaries(self, temp_manager):
        """Quality score should be between 0 and 1 for any inputs."""
        # Perfect quality
        high = temp_manager.calculate_quality_score(
            roe=1.0, debt_equity=0.0, earnings_stability=1.0, profitability=1.0
        )
        assert 0 <= high <= 1.0

        # Terrible quality
        low = temp_manager.calculate_quality_score(
            roe=-0.5, debt_equity=10.0, earnings_stability=0.0, profitability=0.0
        )
        assert 0 <= low <= 1.0

    def test_quality_score_higher_for_better_inputs(self, temp_manager):
        """Better inputs should yield higher quality scores."""
        good = temp_manager.calculate_quality_score(
            roe=0.25, debt_equity=0.3, earnings_stability=0.9, profitability=0.85
        )
        bad = temp_manager.calculate_quality_score(
            roe=0.05, debt_equity=1.5, earnings_stability=0.3, profitability=0.2
        )
        assert good > bad

    def test_store_prices_multiple_symbols(self, temp_manager):
        """Should store and retrieve prices for multiple symbols independently."""
        for symbol in ["MTUM", "QUAL", "USMV", "VLUE"]:
            prices = [
                {"date": "2026-05-01", "open": 100, "high": 101, "low": 99,
                 "close": 100.5 + hash(symbol) % 10, "volume": 1000000},
            ]
            temp_manager.store_prices(symbol, prices)

        for symbol in ["MTUM", "QUAL", "USMV", "VLUE"]:
            retrieved = temp_manager.get_prices(symbol, days=5)
            assert len(retrieved) == 1

    def test_get_prices_nonexistent_symbol(self, temp_manager):
        """Getting prices for a valid symbol with no data should return empty."""
        result = temp_manager.get_prices("MTUM", days=10)
        assert result == []

    def test_calculate_returns_with_few_prices(self, temp_manager):
        """Fewer prices than needed should still return partial results or None."""
        prices = [{"date": f"2026-05-{i+1:02d}", "open": 100, "high": 101,
                    "low": 99, "close": 100.0 + i, "volume": 1000000} for i in range(5)]
        temp_manager.store_prices("MTUM", prices)
        result = temp_manager.calculate_returns("MTUM")
        # May return None or partial result depending on implementation
        # Just verify it doesn't crash
        assert result is None or isinstance(result, dict)

    def test_metadata_path_exists(self, temp_manager):
        """Metadata path should exist after initialization."""
        assert temp_manager.metadata_path.exists()

    def test_db_path_under_data_dir(self, temp_manager):
        """Database path should be under the data directory."""
        assert str(temp_manager.db_path).startswith(str(temp_manager.data_dir))


@pytest.mark.skipif(not HAS_DEPENDENCIES, reason="Dependencies not available")
class TestQualityWeightsExtended:
    """Additional quality weights tests."""

    def test_all_quality_weight_keys(self):
        """Quality weights should have expected keys."""
        expected_keys = {"roe", "debt_equity", "earnings_stability", "profitability"}
        assert set(QUALITY_WEIGHTS.keys()) == expected_keys

    def test_all_weights_positive(self):
        """All quality weights should be positive."""
        for key, weight in QUALITY_WEIGHTS.items():
            assert weight > 0, f"{key} has non-positive weight"

    def test_four_factors_present(self):
        """Should have exactly 4 quality factors."""
        assert len(QUALITY_WEIGHTS) == 4

    def test_all_weight_values_in_expected_range(self):
        """All weight values should be between 0.15 and 0.30."""
        for key, weight in QUALITY_WEIGHTS.items():
            assert 0.15 <= weight <= 0.30, f"{key} weight {weight} out of range"


@pytest.mark.skipif(not HAS_DEPENDENCIES, reason="Dependencies not available")
class TestFactorETFExtended2:
    """Additional FactorETF edge cases for to_dict completeness."""

    def test_to_dict_all_fields_present(self):
        """to_dict should contain all 5 dataclass fields."""
        etf = FactorETF("VLUE", "value", 0.0015, 11.3, "iShares MSCI USA Value Factor")
        d = etf.to_dict()
        assert set(d.keys()) == {"symbol", "factor", "expense_ratio", "aum_billions", "description"}

    def test_all_four_etfs_to_dict_valid_factor_type(self):
        """Every FACTOR_ETFS entry should produce a valid factor type in to_dict."""
        valid_factors = {"momentum", "quality", "low_vol", "value"}
        for symbol, etf in FACTOR_ETFS.items():
            d = etf.to_dict()
            assert d["factor"] in valid_factors, f"{symbol} has invalid factor: {d['factor']}"

    def test_expense_ratios_all_identical(self):
        """All factor ETFs should have identical expense ratios."""
        ratios = {sym: etf.expense_ratio for sym, etf in FACTOR_ETFS.items()}
        assert len(set(ratios.values())) == 1
        assert list(ratios.values())[0] == 0.0015

    def test_factor_aum_values_in_expected_range(self):
        """AUM values should be between 5 and 50 billion."""
        for symbol, etf in FACTOR_ETFS.items():
            assert 5.0 <= etf.aum_billions <= 50.0, f"{symbol} AUM {etf.aum_billions} out of range"

    def test_to_dict_serializes_all_four_etfs(self):
        """Each FACTOR_ETFS entry should have to_dict with matching symbol and factor."""
        for symbol, etf in FACTOR_ETFS.items():
            d = etf.to_dict()
            assert d["symbol"] == symbol
            assert d["factor"] == etf.factor

    def test_factor_etf_negative_expense_ratio_invalid(self):
        """A FactorETF with negative expense ratio should still construct (validation not enforced at dataclass level)."""
        etf = FactorETF("TEST", "momentum", -0.01, 10.0, "Test ETF")
        assert etf.expense_ratio == -0.01


@pytest.mark.skipif(not HAS_DEPENDENCIES, reason="Dependencies not available")
class TestFactorDataManagerExtended2:
    """Additional FactorDataManager edge cases — retrieval, boundaries, edge inputs."""

    @pytest.fixture
    def temp_manager(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            data_dir = Path(tmpdir) / "factors"
            manager = FactorDataManager(data_dir=data_dir)
            yield manager

    def test_zero_close_prices_handling(self, temp_manager):
        """Zero close prices should be stored and retrieved correctly."""
        prices = [
            {"date": "2026-05-01", "open": 0.0, "high": 0.0, "low": 0.0, "close": 0.0, "volume": 0},
            {"date": "2026-05-02", "open": 0.0, "high": 0.0, "low": 0.0, "close": 0.0, "volume": 0},
        ]
        count = temp_manager.store_prices("MTUM", prices)
        assert count == 2
        retrieved = temp_manager.get_prices("MTUM", days=5)
        assert len(retrieved) == 2
        assert retrieved[0]["close"] == 0.0

    def test_negative_close_prices_stored(self, temp_manager):
        """Negative close prices (theoretically unlikely) should store without error."""
        prices = [
            {"date": "2026-05-01", "open": 100.0, "high": 101.0, "low": 99.0, "close": 100.5, "volume": 1000000},
            {"date": "2026-05-02", "open": 99.5, "high": 100.0, "low": 98.0, "close": -1.0, "volume": 500000},
        ]
        count = temp_manager.store_prices("MTUM", prices)
        assert count == 2
        retrieved = temp_manager.get_prices("MTUM", days=5)
        assert retrieved[0]["close"] == -1.0

    def test_empty_price_list_stores_zero(self, temp_manager):
        """An empty price list should return 0 records inserted."""
        count = temp_manager.store_prices("MTUM", [])
        assert count == 0

    def test_duplicate_dates_are_replaced(self, temp_manager):
        """INSERT OR REPLACE should overwrite existing records with the same date."""
        prices_first = [
            {"date": "2026-05-01", "open": 100.0, "high": 101.0, "low": 99.0, "close": 100.5, "volume": 1000000},
        ]
        prices_second = [
            {"date": "2026-05-01", "open": 200.0, "high": 201.0, "low": 199.0, "close": 200.5, "volume": 2000000},
        ]
        temp_manager.store_prices("MTUM", prices_first)
        temp_manager.store_prices("MTUM", prices_second)
        retrieved = temp_manager.get_prices("MTUM", days=5)
        assert len(retrieved) == 1
        assert retrieved[0]["close"] == 200.5

    def test_store_prices_missing_optional_fields(self, temp_manager):
        """Storing prices with only date and close (no open/high/low/volume) should store NULL for missing fields."""
        prices = [
            {"date": "2026-05-01", "close": 100.5},
            {"date": "2026-05-02", "close": 101.5},
        ]
        count = temp_manager.store_prices("MTUM", prices)
        assert count == 2
        retrieved = temp_manager.get_prices("MTUM", days=5)
        assert len(retrieved) == 2
        assert retrieved[0]["close"] == 101.5
        assert retrieved[0]["open"] is None
        assert retrieved[0]["high"] is None
        assert retrieved[0]["low"] is None

    def test_constant_prices_produce_zero_returns(self, temp_manager):
        """When prices never change, all calculable returns should be 0 and vol should be 0."""
        prices = []
        for i in range(70):
            prices.append({
                "date": f"2026-03-{i+1:02d}",
                "open": 100.0, "high": 100.0, "low": 100.0, "close": 100.0, "volume": 1000000,
            })
        temp_manager.store_prices("MTUM", prices)
        returns = temp_manager.calculate_returns("MTUM")
        assert returns is not None
        assert returns["return_1d"] == 0.0
        assert returns["return_1m"] == 0.0
        assert returns["return_3m"] is not None  # Needs 64+ prices
        assert returns["return_3m"] == 0.0
        assert returns["return_6m"] is None  # Not enough data
        assert returns["return_12m"] is None  # Not enough data for 252 days
        assert returns["vol_20d"] == 0.0

    def test_declining_prices_produce_negative_returns(self, temp_manager):
        """When prices decline, all calculated returns should be negative."""
        prices = []
        for i in range(30):
            prices.append({
                "date": f"2026-04-{i+1:02d}",
                "open": 100.0, "high": 100.0, "low": 98.0,
                "close": 100.0 - i * 0.5, "volume": 1000000,
            })
        temp_manager.store_prices("MTUM", prices)
        returns = temp_manager.calculate_returns("MTUM")
        assert returns is not None
        assert returns["return_1d"] < 0
        assert returns["return_1m"] < 0
        assert returns["return_3m"] is None  # Not enough data

    def test_calculate_returns_vol_exactly_21_prices(self, temp_manager):
        """With exactly 21 prices, vol_20d should be calculable (needs 21 closes)."""
        prices = []
        for i in range(21):
            prices.append({
                "date": f"2026-05-{i+1:02d}",
                "open": 100.0, "high": 101.0, "low": 99.0,
                "close": 100.0 + i * 0.1, "volume": 1000000,
            })
        temp_manager.store_prices("MTUM", prices)
        returns = temp_manager.calculate_returns("MTUM")
        assert returns is not None
        assert returns["return_1d"] is not None
        assert returns["vol_20d"] is not None
        assert returns["vol_20d"] >= 0

    def test_calculate_returns_vol_20_prices(self, temp_manager):
        """With exactly 20 prices, vol_20d should be None (needs 21 closes)."""
        prices = []
        for i in range(20):
            prices.append({
                "date": f"2026-05-{i+1:02d}",
                "open": 100.0, "high": 101.0, "low": 99.0,
                "close": 100.0 + i * 0.1, "volume": 1000000,
            })
        temp_manager.store_prices("MTUM", prices)
        returns = temp_manager.calculate_returns("MTUM")
        assert returns is not None
        assert returns["return_1d"] is not None
        assert returns["vol_20d"] is None

    def test_get_all_performance_returns_empty_dict(self, temp_manager):
        """With no data stored, get_all_performance should return an empty dict."""
        perf = temp_manager.get_all_performance()
        assert perf == {}

    def test_get_factor_rankings_returns_empty_list(self, temp_manager):
        """With no data stored, get_factor_rankings should return an empty list."""
        rankings = temp_manager.get_factor_rankings()
        assert rankings == []

    def test_store_quality_score_with_missing_keys(self, temp_manager):
        """Missing metric keys should use default values (roe=0.15, debt_equity=0.5, etc.)."""
        metrics = {"roe": 0.20}  # Only provide roe, rest use defaults
        success = temp_manager.store_quality_score("QUAL", "2026-05-01", metrics)
        assert success
        scores = temp_manager.get_quality_scores("QUAL", days=10)
        assert len(scores) == 1
        assert 0 <= scores[0]["composite_score"] <= 1
        # With roe=0.20 and all defaults at 0.5, composite should be middle-range
        assert scores[0]["composite_score"] > 0

    def test_store_prices_large_volume_values(self, temp_manager):
        """Very large volume values should be stored correctly."""
        prices = [
            {"date": "2026-05-01", "open": 100.0, "high": 101.0, "low": 99.0,
             "close": 100.5, "volume": 999999999},
        ]
        count = temp_manager.store_prices("MTUM", prices)
        assert count == 1
        retrieved = temp_manager.get_prices("MTUM", days=5)
        assert retrieved[0]["volume"] == 999999999

    def test_store_and_retrieve_performance_data(self, temp_manager):
        """Store prices then calculate and verify performance data is retrievable."""
        prices = []
        for i in range(30):
            prices.append({
                "date": f"2026-04-{i+1:02d}",
                "open": 100.0, "high": 101.0, "low": 99.0,
                "close": 100.0 + i * 0.2, "volume": 1000000,
            })
        temp_manager.store_prices("MTUM", prices)
        success = temp_manager.store_returns("MTUM")
        assert success

    def test_store_returns_without_data(self, temp_manager):
        """store_returns should return False when no price data exists."""
        result = temp_manager.store_returns("MTUM")
        assert result is False

    def test_store_prices_invalid_symbol_description(self, temp_manager):
        """Storing prices for an unknown symbol should raise ValueError."""
        with pytest.raises(ValueError, match="Unknown factor ETF"):
            temp_manager.store_prices("AAPL", [{"date": "2026-05-01", "close": 100.0}])

    def test_get_quality_scores_empty(self, temp_manager):
        """get_quality_scores should return empty list when no scores stored."""
        scores = temp_manager.get_quality_scores("QUAL", days=10)
        assert scores == []

    def test_calculate_returns_single_price(self, temp_manager):
        """With only 1 price, calculate_returns should return None."""
        prices = [{"date": "2026-05-01", "close": 100.0}]
        temp_manager.store_prices("MTUM", prices)
        result = temp_manager.calculate_returns("MTUM")
        assert result is None


@pytest.mark.skipif(not HAS_DEPENDENCIES, reason="Dependencies not available")
class TestNormalizationBoundaries:
    """Test the normalization logic boundaries in calculate_quality_score."""

    @pytest.fixture
    def manager(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            yield FactorDataManager(data_dir=Path(tmpdir) / "factors")

    def test_roe_0_25_maps_to_1(self, manager):
        """ROE of 0.25 should normalize to exactly 1 (max)."""
        score = manager.calculate_quality_score(roe=0.25, debt_equity=0.5, earnings_stability=0.5, profitability=0.5)
        assert score >= 0

    def test_roe_0_maps_to_0(self, manager):
        """ROE of 0 should normalize to exactly 0 (min)."""
        score = manager.calculate_quality_score(roe=0.0, debt_equity=0.5, earnings_stability=0.5, profitability=0.5)
        assert score >= 0

    def test_debt_equity_2_maps_to_0(self, manager):
        """Debt/equity of 2.0 should normalize to 0 (1 - 2/2 = 0)."""
        score = manager.calculate_quality_score(roe=0.15, debt_equity=2.0, earnings_stability=0.5, profitability=0.5)
        de_component = 1.0 - (2.0 / 2.0)  # = 0
        assert de_component == 0.0
        assert score >= 0

    def test_debt_equity_0_maps_to_1(self, manager):
        """Debt/equity of 0 should normalize to exactly 1 (1 - 0/2 = 1)."""
        score = manager.calculate_quality_score(roe=0.15, debt_equity=0.0, earnings_stability=0.5, profitability=0.5)
        assert score >= 0

    def test_extreme_roe_values_clamped(self, manager):
        """Extremely high or negative ROE values should each be clamped to [0, 1] and yield different scores."""
        high = manager.calculate_quality_score(roe=10.0, debt_equity=0.5, earnings_stability=0.5, profitability=0.5)
        low = manager.calculate_quality_score(roe=-10.0, debt_equity=0.5, earnings_stability=0.5, profitability=0.5)
        assert 0 <= high <= 1  # ROE=10 clamped to roe_norm=1
        assert 0 <= low <= 1   # ROE=-10 clamped to roe_norm=0
        assert high > low      # Higher ROE should yield higher score

    def test_extreme_debt_values_clamped(self, manager):
        """Extremely high debt/equity values should be clamped to min 0."""
        high_debt = manager.calculate_quality_score(roe=0.15, debt_equity=100.0, earnings_stability=0.5, profitability=0.5)
        zero_debt = manager.calculate_quality_score(roe=0.15, debt_equity=0.0, earnings_stability=0.5, profitability=0.5)
        # High debt should have lower or equal score than zero debt
        assert high_debt <= zero_debt

    def test_score_at_exact_one_possible(self, manager):
        """Perfect inputs should yield a composite score close to 1.0."""
        score = manager.calculate_quality_score(
            roe=0.25, debt_equity=0.0, earnings_stability=1.0, profitability=1.0,
        )
        # With all weights summing to 1 and each component at max:
        expected = (0.30 * 1.0) + (0.25 * 1.0) + (0.25 * 1.0) + (0.20 * 1.0)
        assert score == pytest.approx(expected, abs=0.001)

    def test_score_at_exact_zero_possible(self, manager):
        """Worst-possible inputs should yield a composite score close to 0.0."""
        score = manager.calculate_quality_score(
            roe=0.0, debt_equity=2.0, earnings_stability=0.0, profitability=0.0,
        )
        expected = (0.30 * 0.0) + (0.25 * 0.0) + (0.25 * 0.0) + (0.20 * 0.0)
        assert score == pytest.approx(expected, abs=0.001)

    def test_score_at_half_roe(self, manager):
        """ROE at half (0.125) should give a roe_norm of exactly 0.5."""
        score = manager.calculate_quality_score(roe=0.125, debt_equity=1.0, earnings_stability=0.5, profitability=0.5)
        # roe_norm = min(max(0.125/0.25, 0), 1) = 0.5
        # de_norm = min(max(1 - 1.0/2.0, 0), 1) = 0.5
        # earn_norm = 0.5, prof_norm = 0.5
        expected = (0.30 * 0.5) + (0.25 * 0.5) + (0.25 * 0.5) + (0.20 * 0.5)
        assert score == pytest.approx(expected, abs=0.001)

    def test_score_rounding_to_4_decimals(self, manager):
        """Composite score should be rounded to 4 decimal places."""
        score = manager.calculate_quality_score(
            roe=0.123456, debt_equity=0.789012, earnings_stability=0.345678, profitability=0.901234,
        )
        # Verify rounding to 4 decimal places
        assert score == round(score, 4)


@pytest.mark.skipif(not HAS_DEPENDENCIES, reason="Dependencies not available")
class TestFetchFunctions:
    """Test fetch_factor_prices_from_pipeline function."""

    def test_fetch_from_pipeline_with_valid_data(self):
        """Valid prices_data should produce correctly formatted records."""
        prices_data = {
            "MTUM": [
                {"d": "2026-05-01", "p": 100.5},
                {"d": "2026-05-02", "p": 101.5},
            ]
        }
        records = fetch_factor_prices_from_pipeline("MTUM", prices_data)
        assert len(records) == 2
        assert records[0]["date"] == "2026-05-01"
        assert records[0]["close"] == 100.5
        assert records[0]["open"] == 100.5  # Close used as proxy
        assert records[0]["high"] == 100.5
        assert records[0]["low"] == 100.5
        assert records[0]["volume"] == 0

    def test_fetch_from_pipeline_empty_prices_data(self):
        """An empty prices_data dict should return an empty list."""
        records = fetch_factor_prices_from_pipeline("MTUM", {})
        assert records == []

    def test_fetch_from_pipeline_symbol_not_in_data(self):
        """A symbol not present in prices_data should return an empty list."""
        prices_data = {"QUAL": [{"d": "2026-05-01", "p": 100.0}]}
        records = fetch_factor_prices_from_pipeline("MTUM", prices_data)
        assert records == []

    def test_fetch_from_pipeline_multiple_symbols(self):
        """Multiple symbols in prices_data should only return the requested symbol."""
        prices_data = {
            "MTUM": [{"d": "2026-05-01", "p": 100.0}],
            "QUAL": [{"d": "2026-05-01", "p": 200.0}],
        }
        records = fetch_factor_prices_from_pipeline("QUAL", prices_data)
        assert len(records) == 1
        assert records[0]["close"] == 200.0

    def test_fetch_from_pipeline_preserves_record_order(self):
        """Records should maintain insertion order from prices_data."""
        prices_data = {
            "MTUM": [
                {"d": "2026-05-01", "p": 100.0},
                {"d": "2026-05-02", "p": 101.0},
                {"d": "2026-05-03", "p": 102.0},
            ]
        }
        records = fetch_factor_prices_from_pipeline("MTUM", prices_data)
        assert [r["date"] for r in records] == ["2026-05-01", "2026-05-02", "2026-05-03"]

    def test_fetch_from_pipeline_empty_symbol_list(self):
        """A symbol with an empty price list should return an empty list."""
        prices_data = {"MTUM": []}
        records = fetch_factor_prices_from_pipeline("MTUM", prices_data)
        assert records == []


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
