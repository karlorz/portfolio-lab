"""
Tests for v5.56 Agentic Prediction Evaluation Framework
"""
import json
import os
import sys
from pathlib import Path
from datetime import datetime
import numpy as np
import pytest

# Add project root
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.evaluation.agentic_evaluator import (
    AgenticEvaluator,
    AgentPrediction,
    EvaluationResult,
    DimensionScore,
    AgentEvaluationSummary,
    compute_composite_score,
    grade_composite,
    create_fallback_evaluation,
    record_prediction,
    evaluate_pending,
    get_current_scores,
    COMPOSITE_THRESHOLD_WARNING,
    COMPOSITE_THRESHOLD_CRITICAL,
    DEFAULT_WEIGHTS,
    AGENT_TYPES,
    STATE_PATH, EVAL_LOG_PATH,
)


class TestAgentPrediction:
    """Test AgentPrediction dataclass."""

    def test_default_fields(self):
        """Defaults for optional fields."""
        pred = AgentPrediction(
            timestamp="2025-01-01T00:00:00",
            agent_id="test_agent",
            agent_type="analyst",
            prediction_value=0.5,
            prediction_direction="bullish",
            confidence=0.7,
            rationale="Bullish on tech stocks",
        )
        assert pred.data_sources == []
        assert pred.context == {}
        assert pred.actual_outcome is None

    def test_all_fields(self):
        """All fields populated correctly."""
        pred = AgentPrediction(
            timestamp="2025-01-01T00:00:00",
            agent_id="analyst_1",
            agent_type="analyst",
            prediction_value=0.8,
            prediction_direction="bullish",
            confidence=0.75,
            rationale="Strong earnings growth expected",
            data_sources=["SEC filings", "earnings reports"],
            actual_outcome=0.05,
            context={"sector": "technology"},
        )
        assert pred.agent_id == "analyst_1"
        assert pred.actual_outcome == 0.05
        assert pred.context["sector"] == "technology"

    def test_json_serializable(self):
        """AgentPrediction should be JSON serializable."""
        pred = AgentPrediction(
            timestamp="2025-01-01T00:00:00",
            agent_id="test",
            agent_type="sentiment",
            prediction_value=-0.3,
            prediction_direction="bearish",
            confidence=0.6,
            rationale="Negative sentiment in news",
            data_sources=["Twitter", "Reddit"],
        )
        import dataclasses
        d = dataclasses.asdict(pred)
        json_str = json.dumps(d, default=str)
        assert json_str
        parsed = json.loads(json_str)
        assert parsed["agent_id"] == "test"


class TestDimensionScore:
    """Test DimensionScore dataclass."""

    def test_defaults(self):
        """Default strengths/weaknesses are empty."""
        ds = DimensionScore(dimension="reasoning", score=4.0, explanation="Good reasoning")
        assert ds.strengths == []
        assert ds.weaknesses == []

    def test_all_fields(self):
        """All fields populated."""
        ds = DimensionScore(
            dimension="reasoning",
            score=4.5,
            explanation="Clear logical chain",
            strengths=["Causal reasoning", "Evidence-based"],
            weaknesses=["Minor gap in data sourcing"],
        )
        assert len(ds.strengths) == 2
        assert len(ds.weaknesses) == 1


class TestEvaluationResult:
    """Test EvaluationResult dataclass."""

    def test_default_fields(self):
        """Defaults are reasonable."""
        dims = {
            "reasoning": DimensionScore("reasoning", 4.0, "Good"),
            "data_sourcing": DimensionScore("data_sourcing", 3.0, "Average"),
        }
        result = EvaluationResult(
            timestamp="2025-01-01T00:00:00",
            prediction_timestamp="2025-01-01T00:00:00",
            agent_id="test",
            agent_type="analyst",
            dimensions=dims,
            composite_score=0.75,
            weighted_weights=DEFAULT_WEIGHTS,
            overall_grade="good",
            recommendation="Continue monitoring",
            llm_available=False,
        )
        assert result.num_dimensions == 4  # Default
        assert result.threshold_warning == COMPOSITE_THRESHOLD_WARNING
        assert result.threshold_critical == COMPOSITE_THRESHOLD_CRITICAL

    def test_json_serializable(self):
        """EvaluationResult should be JSON serializable."""
        dims = {"reasoning": DimensionScore("reasoning", 4.0, "Good")}
        result = EvaluationResult(
            timestamp="2025-01-01T00:00:00",
            prediction_timestamp="2025-01-01T00:00:00",
            agent_id="test",
            agent_type="analyst",
            dimensions=dims,
            composite_score=0.75,
            weighted_weights={"reasoning": 1.0},
            overall_grade="good",
            recommendation="OK",
            llm_available=False,
        )
        import dataclasses
        d = dataclasses.asdict(result)
        json_str = json.dumps(d, default=str)
        assert json_str


class TestComputeComposite:
    """Test composite score computation."""

    def test_perfect_scores(self):
        """All 5s (max) should give composite of 1.0."""
        composite, weights = compute_composite_score(
            {"reasoning": 5.0, "data_sourcing": 5.0, "calibration": 5.0, "decision_impact": 5.0}
        )
        assert composite == pytest.approx(1.0, abs=0.01)

    def test_minimum_scores(self):
        """All 1s (min) should give composite of 0.0."""
        composite, weights = compute_composite_score(
            {"reasoning": 1.0, "data_sourcing": 1.0, "calibration": 1.0, "decision_impact": 1.0}
        )
        assert composite == pytest.approx(0.0, abs=0.01)

    def test_mid_scores(self):
        """All 3s should give composite of 0.5."""
        composite, weights = compute_composite_score(
            {"reasoning": 3.0, "data_sourcing": 3.0, "calibration": 3.0, "decision_impact": 3.0}
        )
        assert composite == pytest.approx(0.5, abs=0.01)

    def test_custom_weights(self):
        """Custom weights affect composite."""
        custom_weights = {"reasoning": 1.0, "data_sourcing": 0.0, "calibration": 0.0, "decision_impact": 0.0}
        composite, weights = compute_composite_score(
            {"reasoning": 5.0, "data_sourcing": 1.0, "calibration": 1.0, "decision_impact": 1.0},
            custom_weights,
        )
        # Only reasoning matters, which is max = 5.0 → normalized 1.0
        assert composite == pytest.approx(1.0, abs=0.01)

    def test_default_weights_sum(self):
        """Default weights should sum close to 1.0."""
        total = sum(DEFAULT_WEIGHTS.values())
        assert total == pytest.approx(1.0, abs=0.01)

    def test_missing_dimension_ignored(self):
        """Missing dimension should not cause error."""
        composite, weights = compute_composite_score(
            {"reasoning": 4.0, "data_sourcing": 3.0}  # Missing calibration and impact
        )
        assert 0.0 <= composite <= 1.0


class TestGradeComposite:
    """Test composite score grading."""

    def test_excellent(self):
        grade, rec = grade_composite(0.90)
        assert grade == "excellent"

    def test_good(self):
        grade, rec = grade_composite(0.75)
        assert grade == "good"

    def test_fair(self):
        grade, rec = grade_composite(0.60)
        assert grade == "fair"

    def test_poor(self):
        grade, rec = grade_composite(0.40)
        assert grade == "poor"

    def test_critical(self):
        grade, rec = grade_composite(0.20)
        assert grade == "critical"

    def test_boundaries(self):
        """Boundary values map correctly."""
        assert grade_composite(0.85)[0] in ("excellent", "good")
        assert grade_composite(0.70)[0] in ("good", "fair")
        assert grade_composite(0.50)[0] in ("fair", "poor")
        assert grade_composite(0.30)[0] in ("poor", "critical")


class TestFallbackEvaluation:
    """Test rule-based fallback evaluation."""

    def test_creates_valid_result(self):
        """Fallback produces valid EvaluationResult."""
        pred = AgentPrediction(
            timestamp="2025-01-01T00:00:00",
            agent_id="test_agent",
            agent_type="analyst",
            prediction_value=0.5,
            prediction_direction="bullish",
            confidence=0.7,
            rationale="Detailed reasoning with multiple factors considered",
            data_sources=["SEC filings", "economic data"],
        )
        result = create_fallback_evaluation(pred, "LLM unavailable")
        assert isinstance(result, EvaluationResult)
        assert result.llm_available is False
        assert 0.0 <= result.composite_score <= 1.0
        assert len(result.dimensions) == 4

    def test_fallback_known_directions(self):
        """All 4 dimensions should be present in fallback."""
        pred = AgentPrediction(
            timestamp="2025-01-01T00:00:00",
            agent_id="test",
            agent_type="sentiment",
            prediction_value=0.0,
            prediction_direction="neutral",
            confidence=0.5,
            rationale="Test",
        )
        result = create_fallback_evaluation(pred, "Fallback")
        for dim in ["reasoning", "data_sourcing", "calibration", "decision_impact"]:
            assert dim in result.dimensions

    def test_fallback_scoring_outcome_correct(self):
        """Correct prediction gets higher impact score."""
        pred_correct = AgentPrediction(
            timestamp="2025-01-01T00:00:00",
            agent_id="test",
            agent_type="analyst",
            prediction_value=0.5,
            prediction_direction="bullish",
            confidence=0.7,
            rationale="Good analysis",
            actual_outcome=0.03,  # Positive → correct bullish
        )
        result_correct = create_fallback_evaluation(pred_correct, "Fallback")
        
        pred_wrong = AgentPrediction(
            timestamp="2025-01-01T00:00:01",
            agent_id="test",
            agent_type="analyst",
            prediction_value=-0.5,
            prediction_direction="bearish",
            confidence=0.7,
            rationale="Wrong analysis",
            actual_outcome=0.03,  # Positive → wrong bearish
        )
        result_wrong = create_fallback_evaluation(pred_wrong, "Fallback")
        
        assert result_correct.dimensions["decision_impact"].score >= result_wrong.dimensions["decision_impact"].score


class TestAgenticEvaluator:
    """Test AgenticEvaluator class."""

    @pytest.fixture
    def evaluator(self, tmp_path):
        """Create evaluator with temporary paths."""
        # Monkey-patch paths to use tmp
        original_log = EVAL_LOG_PATH
        original_state = STATE_PATH
        
        import src.evaluation.agentic_evaluator as ae
        ae.EVAL_LOG_PATH = tmp_path / "agent_evaluation.jsonl"
        ae.STATE_PATH = tmp_path / "agent_evaluation_state.json"
        
        evaluator = AgenticEvaluator()
        yield evaluator
        
        # Restore
        ae.EVAL_LOG_PATH = original_log
        ae.STATE_PATH = original_state

    def test_record_prediction(self, evaluator):
        """Recording a prediction works."""
        pred = evaluator.record_prediction(
            agent_id="analyst_1",
            agent_type="analyst",
            prediction_value=0.5,
            prediction_direction="bullish",
            confidence=0.7,
            rationale="Strong earnings expected",
            data_sources=["SEC filings"],
        )
        assert pred.agent_id == "analyst_1"
        assert len(evaluator.predictions) == 1

    def test_record_outcome_found(self, evaluator):
        """Recording an outcome for existing prediction works."""
        pred = evaluator.record_prediction(
            agent_id="test",
            agent_type="analyst",
            prediction_value=0.5,
            prediction_direction="bullish",
            confidence=0.7,
            rationale="Test",
        )
        success = evaluator.record_outcome("test", pred.timestamp, 0.03)
        assert success
        assert evaluator.predictions[0].actual_outcome == 0.03

    def test_record_outcome_not_found(self, evaluator):
        """Recording outcome for non-existent prediction fails."""
        success = evaluator.record_outcome("ghost", "2025-01-01", 0.1)
        assert not success

    def test_evaluate_prediction_fallback(self, evaluator):
        """Evaluate with fallback produces valid result."""
        pred = evaluator.record_prediction(
            agent_id="analyst_1",
            agent_type="analyst",
            prediction_value=0.5,
            prediction_direction="bullish",
            confidence=0.7,
            rationale="Detailed reasoning with comprehensive analysis",
            data_sources=["SEC filings", "economic data", "industry reports"],
        )
        result = evaluator.evaluate_prediction(pred, use_llm=False)
        assert isinstance(result, EvaluationResult)
        assert result.llm_available is False
        assert result.agent_id == "analyst_1"
        assert 0.0 <= result.composite_score <= 1.0

    def test_evaluate_all_pending(self, evaluator):
        """Evaluate all unevaluated predictions."""
        evaluator.record_prediction("agent_a", "analyst", 0.5, "bullish", 0.7, "Reason A")
        evaluator.record_prediction("agent_b", "sentiment", -0.3, "bearish", 0.6, "Reason B")
        
        results = evaluator.evaluate_all_pending(use_llm=False)
        assert len(results) == 2
        assert len(evaluator.evaluations) == 2

    def test_evaluate_only_unevaluated(self, evaluator):
        """Only unevaluated predictions are evaluated."""
        pred = evaluator.record_prediction("agent_c", "risk", 0.2, "neutral", 0.5, "Reason")
        evaluator.evaluate_prediction(pred, use_llm=False)
        
        # Add another prediction
        evaluator.record_prediction("agent_d", "execution", 0.1, "bullish", 0.4, "Reason 2")
        
        pending = evaluator.get_un_evaluated_predictions()
        assert len(pending) == 1
        assert pending[0].agent_id == "agent_d"

    def test_get_agent_summary(self, evaluator):
        """Summary for agent with evaluations works."""
        pred = evaluator.record_prediction("analyst_1", "analyst", 0.5, "bullish", 0.7, "Reason")
        evaluator.evaluate_prediction(pred, use_llm=False)
        
        summary = evaluator.get_agent_summary("analyst_1")
        assert summary is not None
        assert summary.agent_id == "analyst_1"
        assert summary.num_evaluations == 1
        assert 0.0 <= summary.avg_composite <= 1.0
        assert summary.trend == "insufficient"  # Only 1 eval

    def test_get_agent_summary_missing(self, evaluator):
        """Missing agent returns None."""
        summary = evaluator.get_agent_summary("nonexistent")
        assert summary is None

    def test_get_all_summaries(self, evaluator):
        """Multiple agents return multiple summaries."""
        pred_a = evaluator.record_prediction("agent_a", "analyst", 0.5, "bullish", 0.7, "Reason A")
        pred_b = evaluator.record_prediction("agent_b", "sentiment", -0.3, "bearish", 0.6, "Reason B")
        evaluator.evaluate_prediction(pred_a, use_llm=False)
        evaluator.evaluate_prediction(pred_b, use_llm=False)
        
        summaries = evaluator.get_all_summaries()
        assert len(summaries) == 2

    def test_check_alerts_no_alerts(self, evaluator):
        """No alerts when scores are good."""
        for i in range(5):
            pred = evaluator.record_prediction(f"agent_{i}", "analyst", 0.5, "bullish", 0.7, "Good reasoning")
            evaluator.evaluate_prediction(pred, use_llm=False)
        
        alerts = evaluator.check_alerts()
        # With good composite scores (should be >0.5), no critical/warning alerts
        non_info = [a for a in alerts if a["severity"] != "info"]
        assert len(non_info) == 0

    def test_check_alerts_critical(self, evaluator):
        """Very low scores trigger critical alerts."""
        # Create predictions with very low confidence to trigger low scores
        pred = evaluator.record_prediction(
            "bad_agent", "analyst", 0.0, "neutral", 0.05, "", data_sources=[]
        )
        evaluator.evaluate_prediction(pred, use_llm=False)
        
        # With 0 data sources, empty rationale, and very low confidence,
        # composite should be low enough for alerts after enough evals
        for _ in range(4):
            evaluator.record_prediction("bad_agent", "analyst", 0.0, "neutral", 0.05, "")
            evaluator.evaluate_prediction(evaluator.predictions[-1], use_llm=False)
        
        alerts = evaluator.check_alerts()
        assert len(alerts) > 0

    def test_persistence(self, tmp_path):
        """Evaluations persist to disk."""
        import src.evaluation.agentic_evaluator as ae
        ae.EVAL_LOG_PATH = tmp_path / "agent_evaluation.jsonl"
        ae.STATE_PATH = tmp_path / "agent_evaluation_state.json"
        
        # Write
        eval1 = AgenticEvaluator()
        pred = eval1.record_prediction("persist_test", "analyst", 0.5, "bullish", 0.7, "Test")
        eval1.evaluate_prediction(pred, use_llm=False)
        
        # Read back
        eval2 = AgenticEvaluator()
        assert len(eval2.predictions) >= 1
        assert len(eval2.evaluations) >= 1


class TestModuleFunctions:
    """Test module-level convenience functions."""

    def test_record_prediction(self):
        """Module-level record_prediction works."""
        pred = record_prediction(
            agent_id="func_test",
            agent_type="analyst",
            prediction_value=0.5,
            prediction_direction="bullish",
            confidence=0.7,
            rationale="Test",
        )
        assert pred.agent_id == "func_test"

    def test_evaluate_pending_empty(self):
        """evaluate_pending with no predictions returns empty."""
        results = evaluate_pending(use_llm=False)
        assert isinstance(results, list)

    def test_get_current_scores(self):
        """get_current_scores returns dict with expected keys."""
        scores = get_current_scores()
        assert "timestamp" in scores
        assert "num_agents_evaluated" in scores
        assert "average_composite_score" in scores
        assert "thresholds" in scores


class TestAGENT_TYPES:
    """Test AGENT_TYPES constant."""

    def test_all_agent_types_present(self):
        """All 5 agent types should be present."""
        assert "analyst" in AGENT_TYPES
        assert "sentiment" in AGENT_TYPES
        assert "risk" in AGENT_TYPES
        assert "execution" in AGENT_TYPES
        assert "controller" in AGENT_TYPES
        assert len(AGENT_TYPES) == 5

    def test_no_duplicates(self):
        """No duplicate agent types."""
        assert len(AGENT_TYPES) == len(set(AGENT_TYPES))


class TestDefaultWeights:
    """Test DEFAULT_WEIGHTS constant."""

    def test_all_dimensions_present(self):
        """All 4 dimensions should have weights."""
        assert "reasoning" in DEFAULT_WEIGHTS
        assert "data_sourcing" in DEFAULT_WEIGHTS
        assert "calibration" in DEFAULT_WEIGHTS
        assert "decision_impact" in DEFAULT_WEIGHTS

    def test_weights_sum_to_one(self):
        """Weights should sum to ~1.0."""
        assert sum(DEFAULT_WEIGHTS.values()) == pytest.approx(1.0, abs=0.01)

    def test_no_zero_weights(self):
        """No dimension should have zero weight."""
        for w in DEFAULT_WEIGHTS.values():
            assert w > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
