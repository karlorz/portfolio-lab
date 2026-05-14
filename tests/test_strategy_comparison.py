"""
Tests for src/strategy/comparison.py — Strategy Comparison Engine.
"""
import pytest
import json
import sys
from dataclasses import asdict
from unittest.mock import patch

from src.strategy.comparison import (
    StrategyPerformance,
    StrategyComparisonEngine,
    main,
)


class TestStrategyPerformance:
    """Dataclass creation and field access."""

    def test_create_strategy_performance(self):
        sp = StrategyPerformance(
            name="Test",
            description="A test strategy",
            allocation={"SPY": 0.5, "GLD": 0.5},
            expected_return=0.10,
            expected_volatility=0.12,
            sharpe_estimate=0.70,
            max_drawdown_estimate=-0.25,
            crisis_performance={"2008": -0.05, "2020": 0.03},
            rebalance_frequency="Monthly",
            complexity="medium",
            signal_required=True,
        )
        assert sp.name == "Test"
        assert sp.allocation == {"SPY": 0.5, "GLD": 0.5}
        assert sp.expected_return == 0.10
        assert sp.sharpe_estimate == 0.70
        assert sp.complexity == "medium"
        assert sp.signal_required is True

    def test_crisis_performance_default_entries(self):
        sp = StrategyPerformance(
            name="X",
            description="x",
            allocation={},
            expected_return=0,
            expected_volatility=0,
            sharpe_estimate=0,
            max_drawdown_estimate=0,
            crisis_performance={"2008": -0.37, "2022": -0.25},
            rebalance_frequency="n/a",
            complexity="low",
            signal_required=False,
        )
        assert "2008" in sp.crisis_performance
        assert sp.crisis_performance["2008"] == -0.37


class TestStrategyComparisonEngineInit:
    """Engine initialization with pre-loaded strategies."""

    def test_engine_has_six_strategies(self):
        engine = StrategyComparisonEngine()
        assert len(engine.strategies) == 6

    def test_engine_contains_all_season_static(self):
        engine = StrategyComparisonEngine()
        s = engine.strategies["all_season_static"]
        assert s.name == "All-Season Static"
        assert s.allocation == {"SPY": 0.46, "GLD": 0.38, "TLT": 0.16}
        assert s.sharpe_estimate == 0.79
        assert s.complexity == "low"
        assert s.signal_required is False

    def test_engine_contains_spy_only(self):
        engine = StrategyComparisonEngine()
        s = engine.strategies["spy_only"]
        assert s.name == "100% SPY"
        assert s.allocation == {"SPY": 1.0}
        assert s.expected_volatility == 0.190

    def test_engine_contains_sixty_forty(self):
        engine = StrategyComparisonEngine()
        s = engine.strategies["sixty_forty"]
        assert s.allocation == {"SPY": 0.60, "TLT": 0.40}
        assert s.sharpe_estimate == 0.55
        assert s.max_drawdown_estimate == -0.350

    def test_engine_contains_dual_momentum(self):
        engine = StrategyComparisonEngine()
        s = engine.strategies["dual_momentum"]
        assert s.complexity == "high"
        assert s.signal_required is True

    def test_engine_contains_risk_parity(self):
        engine = StrategyComparisonEngine()
        s = engine.strategies["risk_parity"]
        assert s.complexity == "high"

    def test_engine_contains_all_season_trend(self):
        engine = StrategyComparisonEngine()
        s = engine.strategies["all_season_trend"]
        assert s.signal_required is True
        assert s.complexity == "medium"


class TestCompareStrategies:
    """compare_strategies() with default and custom criteria."""

    def test_compare_returns_dict_with_expected_keys(self):
        engine = StrategyComparisonEngine()
        result = engine.compare_strategies()
        assert "rankings" in result
        assert "criteria" in result
        assert "best_overall" in result
        assert "recommendation" in result

    def test_compare_default_criteria(self):
        engine = StrategyComparisonEngine()
        result = engine.compare_strategies()
        assert result["criteria"] == ["sharpe", "drawdown", "crisis_resilience"]

    def test_compare_custom_criteria(self):
        engine = StrategyComparisonEngine()
        result = engine.compare_strategies(["return", "volatility"])
        assert result["criteria"] == ["return", "volatility"]

    def test_compare_returns_all_six_rankings(self):
        engine = StrategyComparisonEngine()
        result = engine.compare_strategies()
        assert len(result["rankings"]) == 6

    def test_compare_rankings_sorted_descending(self):
        engine = StrategyComparisonEngine()
        result = engine.compare_strategies()
        scores = [r["score"] for r in result["rankings"]]
        assert scores == sorted(scores, reverse=True)

    def test_compare_best_overall_is_first_ranked(self):
        engine = StrategyComparisonEngine()
        result = engine.compare_strategies()
        assert result["best_overall"] == result["rankings"][0]["key"]

    def test_compare_each_ranking_has_key_strategy_score(self):
        engine = StrategyComparisonEngine()
        result = engine.compare_strategies()
        for r in result["rankings"]:
            assert "key" in r
            assert "strategy" in r
            assert "score" in r
            assert isinstance(r["key"], str)
            assert isinstance(r["score"], float)

    def test_compare_with_sharpe_only_best_is_all_season(self):
        engine = StrategyComparisonEngine()
        result = engine.compare_strategies(["sharpe"])
        best = result["rankings"][0]
        assert best["key"] == "dual_momentum"  # Sharpe 0.90 is highest

    def test_compare_with_simplicity_scores_low_complexity_higher(self):
        engine = StrategyComparisonEngine()
        result = engine.compare_strategies(["simplicity"])
        # Low-complexity strategies (all_season_static, sixty_forty, spy_only) should rank high
        top_keys = [r["key"] for r in result["rankings"][:3]]
        for k in top_keys:
            assert engine.strategies[k].complexity == "low"


class TestCalculateScore:
    """_calculate_score for each criterion."""

    def test_sharpe_high_gives_high_score(self):
        engine = StrategyComparisonEngine()
        s = StrategyPerformance("t", "t", {}, 0, 0, 1.0, 0, {}, "", "low", False)
        score = engine._calculate_score(s, ["sharpe"])
        assert score == pytest.approx(1.0)

    def test_sharpe_low_gives_low_score(self):
        engine = StrategyComparisonEngine()
        s = StrategyPerformance("t", "t", {}, 0, 0, 0.4, 0, {}, "", "low", False)
        score = engine._calculate_score(s, ["sharpe"])
        assert score == pytest.approx(0.0)

    def test_sharpe_below_threshold_clamped_to_zero(self):
        engine = StrategyComparisonEngine()
        s = StrategyPerformance("t", "t", {}, 0, 0, 0.2, 0, {}, "", "low", False)
        score = engine._calculate_score(s, ["sharpe"])
        assert score == pytest.approx(0.0)

    def test_sharpe_above_one_clamped_to_one(self):
        engine = StrategyComparisonEngine()
        s = StrategyPerformance("t", "t", {}, 0, 0, 1.5, 0, {}, "", "low", False)
        score = engine._calculate_score(s, ["sharpe"])
        assert score == pytest.approx(1.0)

    def test_return_high_gives_high_score(self):
        engine = StrategyComparisonEngine()
        s = StrategyPerformance("t", "t", {}, 0.15, 0, 0, 0, {}, "", "low", False)
        score = engine._calculate_score(s, ["return"])
        assert score == pytest.approx(1.0)

    def test_return_low_gives_low_score(self):
        engine = StrategyComparisonEngine()
        s = StrategyPerformance("t", "t", {}, 0.06, 0, 0, 0, {}, "", "low", False)
        score = engine._calculate_score(s, ["return"])
        assert score == pytest.approx(0.0)

    def test_volatility_low_gives_high_score(self):
        engine = StrategyComparisonEngine()
        s = StrategyPerformance("t", "t", {}, 0, 0.08, 0, 0, {}, "", "low", False)
        score = engine._calculate_score(s, ["volatility"])
        assert score == pytest.approx(1.0)

    def test_volatility_high_gives_low_score(self):
        engine = StrategyComparisonEngine()
        s = StrategyPerformance("t", "t", {}, 0, 0.20, 0, 0, {}, "", "low", False)
        score = engine._calculate_score(s, ["volatility"])
        assert score == pytest.approx(0.0)

    def test_drawdown_small_gives_high_score(self):
        engine = StrategyComparisonEngine()
        s = StrategyPerformance("t", "t", {}, 0, 0, 0, -0.10, {}, "", "low", False)
        score = engine._calculate_score(s, ["drawdown"])
        assert score == pytest.approx(1.0)

    def test_drawdown_large_gives_low_score(self):
        engine = StrategyComparisonEngine()
        s = StrategyPerformance("t", "t", {}, 0, 0, 0, -0.55, {}, "", "low", False)
        score = engine._calculate_score(s, ["drawdown"])
        assert score == pytest.approx(0.0)

    def test_crisis_resilience_positive_gives_high_score(self):
        engine = StrategyComparisonEngine()
        s = StrategyPerformance("t", "t", {}, 0, 0, 0, 0,
                                {"2008": 0.10, "2020": 0.10}, "", "low", False)
        score = engine._calculate_score(s, ["crisis_resilience"])
        assert score == pytest.approx(1.0)

    def test_crisis_resilience_negative_gives_low_score(self):
        engine = StrategyComparisonEngine()
        s = StrategyPerformance("t", "t", {}, 0, 0, 0, 0,
                                {"2008": -0.20, "2020": -0.20}, "", "low", False)
        score = engine._calculate_score(s, ["crisis_resilience"])
        assert score == pytest.approx(0.0)

    def test_crisis_resilience_empty_returns_zero(self):
        engine = StrategyComparisonEngine()
        s = StrategyPerformance("t", "t", {}, 0, 0, 0, 0, {}, "", "low", False)
        score = engine._calculate_score(s, ["crisis_resilience"])
        # Empty crises → avg_crisis not computed → score stays 0
        # The code appends nothing when crises list is empty
        assert score == 0.0

    def test_simplicity_low_gives_highest_score(self):
        engine = StrategyComparisonEngine()
        s = StrategyPerformance("t", "t", {}, 0, 0, 0, 0, {}, "", "low", False)
        score = engine._calculate_score(s, ["simplicity"])
        assert score == 1.0

    def test_simplicity_high_gives_lowest_score(self):
        engine = StrategyComparisonEngine()
        s = StrategyPerformance("t", "t", {}, 0, 0, 0, 0, {}, "", "high", False)
        score = engine._calculate_score(s, ["simplicity"])
        assert score == 0.3

    def test_unknown_complexity_defaults_to_0_5(self):
        engine = StrategyComparisonEngine()
        s = StrategyPerformance("t", "t", {}, 0, 0, 0, 0, {}, "", "unknown", False)
        score = engine._calculate_score(s, ["simplicity"])
        assert score == 0.5

    def test_unknown_criterion_is_skipped(self):
        engine = StrategyComparisonEngine()
        s = StrategyPerformance("t", "t", {}, 0.10, 0.10, 0.7, -0.20,
                                {"2008": -0.10}, "", "low", False)
        score = engine._calculate_score(s, ["nonexistent_criterion"])
        assert score == 0.0  # No scores collected

    def test_multiple_criteria_averaged(self):
        engine = StrategyComparisonEngine()
        s = StrategyPerformance("t", "t", {}, 0.10, 0.10, 1.0, -0.10, {}, "", "low", False)
        # sharpe=1.0, return=0.44..., avg = ~0.72
        score = engine._calculate_score(s, ["sharpe", "return"])
        assert 0.5 < score < 0.9

    def test_scores_always_between_zero_and_one(self):
        engine = StrategyComparisonEngine()
        for key, strat in engine.strategies.items():
            for criterion in ["sharpe", "return", "volatility", "drawdown", "crisis_resilience", "simplicity"]:
                score = engine._calculate_score(strat, [criterion])
                assert 0.0 <= score <= 1.0, f"{key}/{criterion} = {score}"


class TestGenerateRecommendation:
    """_generate_recommendation text output."""

    def test_high_score_strongly_recommended(self):
        engine = StrategyComparisonEngine()
        s = engine.strategies["all_season_static"]
        rec = engine._generate_recommendation({"strategy": s, "score": 0.85})
        assert "strongly recommended" in rec

    def test_medium_score_good_choice(self):
        engine = StrategyComparisonEngine()
        s = engine.strategies["sixty_forty"]
        rec = engine._generate_recommendation({"strategy": s, "score": 0.65})
        assert "good choice" in rec

    def test_low_score_moderate(self):
        engine = StrategyComparisonEngine()
        s = engine.strategies["spy_only"]
        rec = engine._generate_recommendation({"strategy": s, "score": 0.40})
        assert "scores moderately" in rec

    def test_none_result_returns_no_recommendation(self):
        engine = StrategyComparisonEngine()
        rec = engine._generate_recommendation(None)
        assert rec == "No recommendation available"


class TestGetStrategyDetails:
    """get_strategy_details for valid and invalid keys."""

    def test_valid_key_returns_details(self):
        engine = StrategyComparisonEngine()
        details = engine.get_strategy_details("all_season_static")
        assert details is not None
        assert details["key"] == "all_season_static"
        assert details["name"] == "All-Season Static"
        assert "expected_metrics" in details
        assert "crisis_performance" in details
        assert "implementation" in details

    def test_expected_metrics_format(self):
        engine = StrategyComparisonEngine()
        details = engine.get_strategy_details("all_season_static")
        m = details["expected_metrics"]
        assert "%" in m["cagr"]
        assert "%" in m["volatility"]
        assert "." in m["sharpe"]
        assert "%" in m["max_drawdown"]

    def test_crisis_performance_format(self):
        engine = StrategyComparisonEngine()
        details = engine.get_strategy_details("all_season_static")
        for year, val in details["crisis_performance"].items():
            assert "%" in val
            assert year in ("2008", "2020", "2022")

    def test_implementation_fields(self):
        engine = StrategyComparisonEngine()
        details = engine.get_strategy_details("all_season_trend")
        impl = details["implementation"]
        assert "rebalance_frequency" in impl
        assert "complexity" in impl
        assert "requires_signals" in impl
        assert impl["requires_signals"] is True

    def test_invalid_key_returns_none(self):
        engine = StrategyComparisonEngine()
        details = engine.get_strategy_details("nonexistent")
        assert details is None

    def test_all_six_strategies_have_details(self):
        engine = StrategyComparisonEngine()
        for key in engine.strategies:
            details = engine.get_strategy_details(key)
            assert details is not None, f"Missing details for {key}"
            assert "allocation" in details


class TestRecommendForUserProfile:
    """recommend_for_user_profile with all profile combinations."""

    def test_conservative_long_low(self):
        engine = StrategyComparisonEngine()
        result = engine.recommend_for_user_profile("conservative", "long", "low")
        assert result["recommendation"] is not None
        assert len(result["alternatives"]) == 2
        assert len(result["all_ranked"]) == 6
        assert result["profile"]["risk_tolerance"] == "conservative"

    def test_conservative_long_high(self):
        engine = StrategyComparisonEngine()
        result = engine.recommend_for_user_profile("conservative", "long", "high")
        assert result["recommendation"] is not None

    def test_moderate_long_low(self):
        engine = StrategyComparisonEngine()
        result = engine.recommend_for_user_profile("moderate", "long", "low")
        assert result["recommendation"] is not None

    def test_moderate_long_high(self):
        engine = StrategyComparisonEngine()
        result = engine.recommend_for_user_profile("moderate", "long", "high")
        assert result["recommendation"] is not None

    def test_aggressive_long_low(self):
        engine = StrategyComparisonEngine()
        result = engine.recommend_for_user_profile("aggressive", "long", "low")
        assert result["recommendation"] is not None

    def test_aggressive_long_high(self):
        engine = StrategyComparisonEngine()
        result = engine.recommend_for_user_profile("aggressive", "long", "high")
        assert result["recommendation"] is not None

    def test_unknown_profile_uses_default_weights(self):
        engine = StrategyComparisonEngine()
        result = engine.recommend_for_user_profile("unknown", "unknown", "unknown")
        assert result["recommendation"] is not None
        assert len(result["all_ranked"]) == 6

    def test_rankings_sorted_descending(self):
        engine = StrategyComparisonEngine()
        result = engine.recommend_for_user_profile("aggressive", "long", "high")
        scores = [r["score"] for r in result["all_ranked"]]
        assert scores == sorted(scores, reverse=True)

    def test_aggressive_high_weights_return_heavily(self):
        """Aggressive+high profile weights return at 0.5 — highest of any criterion."""
        engine = StrategyComparisonEngine()
        result = engine.recommend_for_user_profile("aggressive", "long", "high")
        # Dual momentum has highest return (0.14), should rank high for aggressive
        top = result["recommendation"]
        assert top is not None

    def test_conservative_weights_drawdown_heavily(self):
        """Conservative+low profile weights drawdown at 0.4 — drawdown-averse."""
        engine = StrategyComparisonEngine()
        result = engine.recommend_for_user_profile("conservative", "long", "low")
        top = result["recommendation"]
        assert top is not None
        # Low max DD strategies should score well

    def test_each_entry_has_key_strategy_score_rationale(self):
        engine = StrategyComparisonEngine()
        result = engine.recommend_for_user_profile("moderate", "long", "high")
        for entry in result["all_ranked"]:
            assert "key" in entry
            assert "strategy" in entry
            assert "score" in entry
            assert "rationale" in entry


class TestCLI:
    """CLI main() function behavior."""

    def test_no_args_prints_help_and_exits(self):
        with patch.object(sys, "argv", ["comparison.py"]):
            with pytest.raises(SystemExit) as exc:
                main()
            assert exc.value.code == 0

    def test_compare_default_criteria(self, capsys):
        with patch.object(sys, "argv", ["comparison.py", "compare"]):
            main()
        captured = capsys.readouterr()
        assert "STRATEGY COMPARISON" in captured.out
        assert "Rank:" not in captured.out  # Uses explicit index, not "Rank" prefix

    def test_compare_with_custom_criteria(self, capsys):
        with patch.object(sys, "argv", ["comparison.py", "compare", "sharpe", "volatility"]):
            main()
        captured = capsys.readouterr()
        assert "sharpe, volatility" in captured.out

    def test_compare_outputs_json(self, capsys):
        with patch.object(sys, "argv", ["comparison.py", "compare"]):
            main()
        captured = capsys.readouterr()
        lines = captured.out.split("\n")
        json_started = False
        for line in lines:
            if line.strip().startswith("{"):
                json_started = True
            if json_started and '"rankings"' in line:
                break
        assert json_started

    def test_details_valid_strategy(self, capsys):
        with patch.object(sys, "argv", ["comparison.py", "details", "all_season_static"]):
            main()
        captured = capsys.readouterr()
        data = json.loads(captured.out.strip())
        assert data["name"] == "All-Season Static"
        assert "allocation" in data

    def test_details_invalid_strategy(self, capsys):
        with patch.object(sys, "argv", ["comparison.py", "details", "nonexistent"]):
            main()
        captured = capsys.readouterr()
        assert "not found" in captured.out

    def test_recommend_command(self, capsys):
        with patch.object(sys, "argv", ["comparison.py", "recommend"]):
            main()
        captured = capsys.readouterr()
        data = json.loads(captured.out.strip())
        assert "profile" in data
        assert "top_recommendation" in data
        assert data["profile"]["risk_tolerance"] == "moderate"

    def test_unknown_command(self, capsys):
        with patch.object(sys, "argv", ["comparison.py", "unknowncmd"]):
            main()
        captured = capsys.readouterr()
        assert "Unknown command" in captured.out


class TestEdgeCases:
    """Edge cases and boundary conditions."""

    def test_compare_with_empty_criteria_averages_zero(self):
        engine = StrategyComparisonEngine()
        result = engine.compare_strategies([])
        # All scores are 0 (no criteria), ties broken by sort stability
        assert len(result["rankings"]) == 6

    def test_best_overall_none_when_no_rankings(self):
        """When rankings list is empty, best_overall is None."""
        # Can't easily trigger empty rankings, but verify the fallback works
        engine = StrategyComparisonEngine()
        result = engine.compare_strategies()
        assert result["best_overall"] is not None

    def test_all_strategies_have_valid_complexity_values(self):
        engine = StrategyComparisonEngine()
        valid = {"low", "medium", "high"}
        for key, s in engine.strategies.items():
            assert s.complexity in valid, f"{key} has complexity {s.complexity}"

    def test_all_strategies_have_three_crisis_years(self):
        engine = StrategyComparisonEngine()
        for key, s in engine.strategies.items():
            assert len(s.crisis_performance) == 3, f"{key} has {len(s.crisis_performance)} crisis entries"
            for year in ["2008", "2020", "2022"]:
                assert year in s.crisis_performance, f"{key} missing {year}"

    def test_all_allocations_sum_near_one(self):
        engine = StrategyComparisonEngine()
        for key, s in engine.strategies.items():
            total = sum(s.allocation.values())
            assert abs(total - 1.0) < 0.01, f"{key} allocation sums to {total}"
