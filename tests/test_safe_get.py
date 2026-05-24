"""Tests for src.utils.safe_get — safe nested-dict traversal."""

import pytest

from src.utils import safe_get


class TestSafeGetBasic:
    """Core safe_get behavior."""

    def test_simple_key_present(self):
        assert safe_get({"a": 1}, "a") == 1

    def test_simple_key_missing(self):
        assert safe_get({"a": 1}, "b") is None

    def test_simple_key_missing_with_default(self):
        assert safe_get({"a": 1}, "b", default=0) == 0

    def test_nested_two_levels(self):
        assert safe_get({"a": {"b": 2}}, "a", "b") == 2

    def test_nested_three_levels(self):
        assert safe_get({"a": {"b": {"c": 3}}}, "a", "b", "c") == 3

    def test_nested_missing_intermediate_key(self):
        assert safe_get({"a": {}}, "a", "b") is None

    def test_nested_missing_intermediate_key_with_default(self):
        assert safe_get({"a": {}}, "a", "b", default=-1) == -1


class TestSafeGetNoneIntermediate:
    """Crash-on-None is the primary bug safe_get fixes."""

    def test_none_intermediate_two_levels(self):
        assert safe_get({"a": None}, "a", "b") is None

    def test_none_intermediate_two_levels_with_default(self):
        assert safe_get({"a": None}, "a", "b", default=0) == 0

    def test_none_intermediate_three_levels(self):
        assert safe_get({"a": {"b": None}}, "a", "b", "c") is None

    def test_none_intermediate_three_levels_with_default(self):
        assert safe_get({"a": {"b": None}}, "a", "b", "c", default=-1) == -1

    def test_none_root_value(self):
        assert safe_get({"a": None}, "a") is None

    def test_none_root_value_with_default(self):
        assert safe_get({"a": None}, "a", default="fallback") == "fallback"

    def test_deeply_nested_none(self):
        data = {"a": {"b": {"c": {"d": None}}}}
        assert safe_get(data, "a", "b", "c", "d", "e") is None

    def test_deeply_nested_none_with_default(self):
        data = {"a": {"b": {"c": {"d": None}}}}
        assert safe_get(data, "a", "b", "c", "d", "e", default=42) == 42


class TestSafeGetEmptyAndFalsy:
    """Edge cases with empty dicts and falsy values."""

    def test_empty_dict_root(self):
        assert safe_get({}, "a") is None

    def test_empty_dict_root_with_default(self):
        assert safe_get({}, "a", default=0) == 0

    def test_empty_dict_intermediate(self):
        assert safe_get({"a": {}}, "a", "b") is None

    def test_zero_value_preserved(self):
        assert safe_get({"a": {"b": 0}}, "a", "b") == 0

    def test_false_value_preserved(self):
        assert safe_get({"a": {"b": False}}, "a", "b") is False

    def test_empty_string_preserved(self):
        assert safe_get({"a": {"b": ""}}, "a", "b") == ""

    def test_empty_list_preserved(self):
        assert safe_get({"a": {"b": []}}, "a", "b") == []


class TestSafeGetNonDictIntermediate:
    """When an intermediate value is a non-dict type (str, int, list)."""

    def test_string_intermediate(self):
        assert safe_get({"a": "hello"}, "a", "b") is None

    def test_string_intermediate_with_default(self):
        assert safe_get({"a": "hello"}, "a", "b", default="def") == "def"

    def test_int_intermediate(self):
        assert safe_get({"a": 42}, "a", "b") is None

    def test_list_intermediate(self):
        assert safe_get({"a": [1, 2, 3]}, "a", "b") is None

    def test_tuple_intermediate(self):
        assert safe_get({"a": (1, 2)}, "a", "b") is None


class TestSafeGetRealWorldPatterns:
    """Patterns matching actual portfolio-lab usage."""

    def test_generator_weights_pattern(self):
        """generator.py L349: alt_data_raw.get("raw_data", {}).get("weights", {}).get("earnings")"""
        alt_data_raw = {"raw_data": {"weights": {"earnings": 0.3}}}
        assert safe_get(alt_data_raw, "raw_data", "weights", "earnings") == 0.3

    def test_generator_weights_none_raw_data(self):
        alt_data_raw = {"raw_data": None}
        assert safe_get(alt_data_raw, "raw_data", "weights", "earnings", default=0.0) == 0.0

    def test_generator_weights_none_weights(self):
        alt_data_raw = {"raw_data": {"weights": None}}
        assert safe_get(alt_data_raw, "raw_data", "weights", "earnings", default=0.0) == 0.0

    def test_generator_garch_filtered_pattern(self):
        """generator.py L599: data.get("checks", {}).get("cvar_metrics", {}).get("garch_filtered")"""
        data = {"checks": {"cvar_metrics": {"garch_filtered": True}}}
        assert safe_get(data, "checks", "cvar_metrics", "garch_filtered") is True

    def test_generator_garch_filtered_none_checks(self):
        data = {"checks": None}
        assert safe_get(data, "checks", "cvar_metrics", "garch_filtered") is None

    def test_dashboard_summary_pattern(self):
        """unified_dashboard.py L69: report.get("summary", {}).get("passed", 0)"""
        report = {"summary": {"passed": 5, "total_checks": 10}}
        assert safe_get(report, "summary", "passed", default=0) == 5

    def test_dashboard_summary_none(self):
        report = {"summary": None}
        assert safe_get(report, "summary", "passed", default=0) == 0

    def test_options_greeks_pattern(self):
        """options_utils.py L295: quote_data.get("greeks", {}).get("delta")"""
        quote_data = {"greeks": {"delta": 0.5, "gamma": 0.02}}
        assert safe_get(quote_data, "greeks", "delta") == 0.5

    def test_options_greeks_none(self):
        quote_data = {"greeks": None}
        assert safe_get(quote_data, "greeks", "delta") is None

    def test_ensemble_history_pattern(self):
        """ensemble_voter.py L241: self._history.get(regime, {}).get(signal, [])"""
        history = {"NORMAL": {"MULTI_SPEED_MOM": [0.1, 0.2, 0.3]}}
        assert safe_get(history, "NORMAL", "MULTI_SPEED_MOM", default=[]) == [0.1, 0.2, 0.3]

    def test_ensemble_history_none_regime(self):
        history = {"NORMAL": None}
        assert safe_get(history, "NORMAL", "MULTI_SPEED_MOM", default=[]) == []

    def test_prices_pattern(self):
        """international_momentum_backtest.py: prices.get(date, {}).get('SPY')"""
        prices = {"2026-05-24": {"SPY": 450.0, "EFA": 75.0}}
        assert safe_get(prices, "2026-05-24", "SPY") == 450.0

    def test_prices_missing_date(self):
        prices = {}
        assert safe_get(prices, "2026-05-24", "SPY") is None

    def test_daily_brief_regime_pattern(self):
        """daily_brief.py L96: regime.get('classifier', {}).get('current_regime', 'unknown')"""
        regime = {"classifier": {"current_regime": "NORMAL"}}
        assert safe_get(regime, "classifier", "current_regime", default="unknown") == "NORMAL"

    def test_daily_brief_regime_none_classifier(self):
        regime = {"classifier": None}
        assert safe_get(regime, "classifier", "current_regime", default="unknown") == "unknown"


class TestSafeGetNoKeys:
    """Edge case: no keys provided."""

    def test_no_keys_returns_root(self):
        data = {"a": 1}
        assert safe_get(data) == data

    def test_no_keys_empty_dict(self):
        assert safe_get({}) == {}
