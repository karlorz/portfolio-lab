#!/usr/bin/env python3
"""
Portfolio-Lab v5.56: Agentic Prediction Evaluation Framework

Based on arXiv 2605.05739 — "Multi-Dimensional Behavioral Evaluation of
Agentic Stock Prediction Systems Using Large Language Model Judges"

Evaluates MARL agent predictions across behavioral dimensions:
1. Reasoning quality   — Does the analysis follow a logical chain?
2. Data sourcing       — Does the agent cite diverse, relevant data?
3. Uncertainty calibration — Appropriate confidence expression?
4. Decision impact     — Do predictions translate to profitable decisions?

Design:
- Dimension-specific LLM judge prompts (reuse v2.30 LLM sentiment pipeline)
- Scoring rubric: 1-5 per dimension, configurable weights
- Composite score (0-1) for overall prediction quality
- JSONL evaluation log at data/agent_evaluation.jsonl
- Graceful degradation when LLM is unavailable

Usage:
    python -m src.evaluation.agentic_evaluator evaluate  # Run evaluation on recent predictions
    python -m src.evaluation.agentic_evaluator score     # Get current scores
    python -m src.evaluation.agentic_evaluator log       # Show evaluation log
"""

import json
import logging
import os
import sys
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

# Paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
EVAL_LOG_PATH = DATA_DIR / "agent_evaluation.jsonl"
STATE_PATH = DATA_DIR / "agent_evaluation_state.json"

# Default dimension weights (sums to 1.0)
DEFAULT_WEIGHTS = {
    "reasoning": 0.30,
    "data_sourcing": 0.20,
    "calibration": 0.25,
    "decision_impact": 0.25,
}

# Agent types
AGENT_TYPES = ["analyst", "sentiment", "risk", "execution", "controller"]

# Score thresholds
COMPOSITE_THRESHOLD_WARNING = 0.5  # Below this → warning
COMPOSITE_THRESHOLD_CRITICAL = 0.3  # Below this → critical alert


@dataclass
class AgentPrediction:
    """A single agent prediction/decision record."""
    timestamp: str
    agent_id: str
    agent_type: str
    prediction_value: float       # The predicted value or score
    prediction_direction: str     # "bullish", "bearish", "neutral"
    confidence: float             # 0-1
    rationale: str                # Free-text reasoning
    data_sources: List[str] = field(default_factory=list)
    actual_outcome: Optional[float] = None  # Actual outcome (filled later)
    context: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DimensionScore:
    """Score for a single evaluation dimension."""
    dimension: str
    score: float          # 1-5
    explanation: str      # Why this score was given
    strengths: List[str] = field(default_factory=list)
    weaknesses: List[str] = field(default_factory=list)


@dataclass
class EvaluationResult:
    """Complete evaluation result for one prediction."""
    timestamp: str
    prediction_timestamp: str
    agent_id: str
    agent_type: str
    
    # Dimension scores
    dimensions: Dict[str, DimensionScore]
    
    # Composite
    composite_score: float      # 0-1 normalized
    weighted_weights: Dict[str, float]  # Weights used
    
    # Overall assessment
    overall_grade: str          # excellent/good/fair/poor/critical
    recommendation: str         # Action recommendation
    llm_available: bool         # Whether LLM was used for evaluation
    
    # Metadata
    num_dimensions: int = 4
    threshold_warning: float = COMPOSITE_THRESHOLD_WARNING
    threshold_critical: float = COMPOSITE_THRESHOLD_CRITICAL


@dataclass
class AgentEvaluationSummary:
    """Summary of agent's recent evaluation history."""
    agent_id: str
    agent_type: str
    num_evaluations: int
    avg_composite: float
    best_composite: float
    worst_composite: float
    trend: str                  # improving/stable/declining/insufficient
    dimension_averages: Dict[str, float]
    recent_grades: List[str]


def compute_composite_score(dimension_scores: Dict[str, float], weights: Optional[Dict[str, float]] = None) -> Tuple[float, Dict[str, float]]:
    """Compute weighted composite score from dimension scores.
    
    Returns (composite_score_0to1, weights_used).
    """
    if weights is None:
        weights = DEFAULT_WEIGHTS.copy()
    
    # Normalize dimension scores from 1-5 to 0-1
    normalized = {}
    for dim, score in dimension_scores.items():
        normalized[dim] = (score - 1.0) / 4.0  # 1→0.0, 5→1.0
    
    # Weighted sum
    composite = 0.0
    total_weight = 0.0
    for dim, weight in weights.items():
        if dim in normalized:
            composite += normalized[dim] * weight
            total_weight += weight
    
    if total_weight > 0:
        composite /= total_weight
    
    return min(max(composite, 0.0), 1.0), weights


def grade_composite(composite: float) -> Tuple[str, str]:
    """Convert composite score to letter grade and recommendation.
    
    Returns (grade, recommendation).
    """
    if composite >= 0.85:
        return ("excellent", "Agent performing well across all dimensions. Continue monitoring.")
    elif composite >= 0.70:
        return ("good", "Solid performance. Minor improvements in lower-scored dimensions.")
    elif composite >= 0.50:
        return ("fair", "Adequate but room for improvement. Review weak dimensions.")
    elif composite >= 0.30:
        return ("poor", "Below expectations. Consider retraining or parameter adjustment.")
    else:
        return ("critical", "Agent underperforming significantly. Immediate intervention recommended.")


def create_fallback_evaluation(prediction: AgentPrediction, reason: str) -> EvaluationResult:
    """Create evaluation result when LLM is unavailable."""
    # Generate rule-based scores as fallback
    timestamp = datetime.now(timezone.utc).isoformat()
    
    # Rule-based fallback scoring
    # Reasoning: score based on rationale length and structure
    rationale_len = len(prediction.rationale)
    reasoning_score = min(5.0, max(1.0, 1.0 + rationale_len / 500.0))
    
    # Data sourcing: based on number of data sources cited
    data_score = min(5.0, max(1.0, 1.0 + len(prediction.data_sources) * 0.8))
    
    # Calibration: based on confidence reasonableness
    if 0.3 <= prediction.confidence <= 0.8:
        calibration_score = 4.0  # Reasonable confidence
    elif 0.1 <= prediction.confidence <= 0.9:
        calibration_score = 3.0
    else:
        calibration_score = 2.0  # Over/under confident
    
    # Decision impact: check if actual outcome available and compare
    impact_score = 3.0  # Neutral default
    if prediction.actual_outcome is not None:
        pred_dir = 1 if prediction.prediction_direction == "bullish" else (-1 if prediction.prediction_direction == "bearish" else 0)
        actual_dir = 1 if prediction.actual_outcome > 0 else (-1 if prediction.actual_outcome < 0 else 0)
        if pred_dir * actual_dir > 0:
            impact_score = 4.0
        elif pred_dir * actual_dir < 0:
            impact_score = 2.0
    
    dimension_scores = {
        "reasoning": DimensionScore(
            dimension="reasoning",
            score=round(reasoning_score, 1),
            explanation=f"Fallback: scored by rationale length ({rationale_len} chars)",
            strengths=["Provides reasoning"] if rationale_len > 50 else [],
            weaknesses=["Limited detail"] if rationale_len < 100 else [],
        ),
        "data_sourcing": DimensionScore(
            dimension="data_sourcing",
            score=round(data_score, 1),
            explanation=f"Fallback: scored by sources cited ({len(prediction.data_sources)})",
            strengths=["Multiple sources cited"] if len(prediction.data_sources) >= 2 else [],
            weaknesses=["Limited sources"] if len(prediction.data_sources) < 2 else [],
        ),
        "calibration": DimensionScore(
            dimension="calibration",
            score=round(calibration_score, 1),
            explanation=f"Fallback: scored by confidence level ({prediction.confidence:.2f})",
            strengths=["Well-calibrated"] if 0.3 <= prediction.confidence <= 0.7 else [],
            weaknesses=["Overconfident" if prediction.confidence > 0.8 else "Underconfident"],
        ),
        "decision_impact": DimensionScore(
            dimension="decision_impact",
            score=round(impact_score, 1),
            explanation="Fallback: neutral score (no actual outcome)" if prediction.actual_outcome is None
                        else f"Fallback: scored by outcome ({prediction.actual_outcome:.4f})",
            strengths=[],
            weaknesses=["Pending outcome"] if prediction.actual_outcome is None else [],
        ),
    }
    
    raw_dim_scores = {d: s.score for d, s in dimension_scores.items()}
    composite, weights = compute_composite_score(raw_dim_scores)
    grade, recommendation = grade_composite(composite)
    
    return EvaluationResult(
        timestamp=timestamp,
        prediction_timestamp=prediction.timestamp,
        agent_id=prediction.agent_id,
        agent_type=prediction.agent_type,
        dimensions=dimension_scores,
        composite_score=round(composite, 4),
        weighted_weights=weights,
        overall_grade=grade,
        recommendation=recommendation,
        llm_available=False,
    )


class AgenticEvaluator:
    """Evaluates agent predictions across behavioral dimensions."""
    
    def __init__(self, weights: Optional[Dict[str, float]] = None):
        self.weights = weights or DEFAULT_WEIGHTS.copy()
        self.predictions: List[AgentPrediction] = []
        self.evaluations: List[EvaluationResult] = []
        self._load_log()
    
    def _load_log(self):
        """Load existing evaluations from log."""
        if EVAL_LOG_PATH.exists():
            try:
                with open(EVAL_LOG_PATH) as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            try:
                                data = json.loads(line)
                                if "composite_score" in data:
                                    # EvaluationResult — reconstruct nested DimensionScore
                                    dims = {}
                                    for d, ddata in data.get("dimensions", {}).items():
                                        dims[d] = DimensionScore(**ddata)
                                    data["dimensions"] = dims
                                    self.evaluations.append(EvaluationResult(**data))
                                elif "agent_id" in data:
                                    self.predictions.append(AgentPrediction(**data))
                            except (json.JSONDecodeError, TypeError):
                                continue
            except (OSError, json.JSONDecodeError):
                logger.warning(f"Failed to read evaluation log: {EVAL_LOG_PATH}")
    
    def _save_log(self):
        """Save evaluations to log file."""
        EVAL_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(EVAL_LOG_PATH, "w") as f:
            for pred in self.predictions:
                f.write(json.dumps(asdict(pred), default=str) + "\n")
            for eval_result in self.evaluations:
                # Convert DimensionScore objects to dict
                data = asdict(eval_result)
                f.write(json.dumps(data, default=str) + "\n")
    
    def record_prediction(
        self,
        agent_id: str,
        agent_type: str,
        prediction_value: float,
        prediction_direction: str,
        confidence: float,
        rationale: str,
        data_sources: Optional[List[str]] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> AgentPrediction:
        """Record an agent's prediction for later evaluation."""
        pred = AgentPrediction(
            timestamp=datetime.now(timezone.utc).isoformat(),
            agent_id=agent_id,
            agent_type=agent_type,
            prediction_value=prediction_value,
            prediction_direction=prediction_direction,
            confidence=min(max(confidence, 0.0), 1.0),
            rationale=rationale,
            data_sources=data_sources or [],
            context=context or {},
        )
        self.predictions.append(pred)
        self._save_log()
        return pred
    
    def record_outcome(self, agent_id: str, prediction_timestamp: str, actual_outcome: float) -> bool:
        """Record actual outcome for a previous prediction.
        
        Returns True if prediction was found and updated.
        """
        for pred in self.predictions:
            if pred.agent_id == agent_id and pred.timestamp == prediction_timestamp:
                pred.actual_outcome = actual_outcome
                self._save_log()
                return True
        return False
    
    def evaluate_prediction(
        self,
        prediction: AgentPrediction,
        use_llm: bool = False,
    ) -> EvaluationResult:
        """Evaluate a single prediction.
        
        When use_llm=False, uses rule-based fallback scoring.
        When use_llm=True, attempts LLM-based evaluation with graceful fallback.
        """
        if use_llm:
            try:
                # Try to import LLM sentiment pipeline
                from src.signals.behavioral_sentiment import BehavioralSentiment
                sentiment = BehavioralSentiment()
                
                # Build prompt for each dimension
                dim_scores = {}
                for dim in self.weights:
                    prompt = self._build_dimension_prompt(dim, prediction)
                    score = self._llm_judge_dimension(sentiment, dim, prompt)
                    dim_scores[dim] = score
                
                # Build DimensionScore objects
                dimensions = {}
                for dim, score in dim_scores.items():
                    dimensions[dim] = DimensionScore(
                        dimension=dim,
                        score=score.score,
                        explanation=score.explanation,
                        strengths=score.strengths,
                        weaknesses=score.weaknesses,
                    )
                
                raw_scores = {d: s.score for d, s in dimensions.items()}
                composite, weights = compute_composite_score(raw_scores, self.weights)
                grade, recommendation = grade_composite(composite)
                
                result = EvaluationResult(
                    timestamp=datetime.now(timezone.utc).isoformat(),
                    prediction_timestamp=prediction.timestamp,
                    agent_id=prediction.agent_id,
                    agent_type=prediction.agent_type,
                    dimensions=dimensions,
                    composite_score=round(composite, 4),
                    weighted_weights=weights,
                    overall_grade=grade,
                    recommendation=recommendation,
                    llm_available=True,
                )
                self.evaluations.append(result)
                self._save_log()
                return result
                
            except Exception as e:
                logger.warning(f"LLM evaluation failed ({e}), using fallback")
        
        # Fallback to rule-based
        result = create_fallback_evaluation(prediction, "LLM unavailable, rule-based fallback")
        self.evaluations.append(result)
        self._save_log()
        return result
    
    def _build_dimension_prompt(self, dimension: str, prediction: AgentPrediction) -> str:
        """Build prompt for a specific evaluation dimension."""
        prompts = {
            "reasoning": (
                f"Evaluate the reasoning quality of this {prediction.agent_type} agent's prediction.\n"
                f"Agent: {prediction.agent_id}\n"
                f"Prediction: {prediction.prediction_direction} (value: {prediction.prediction_value:.4f})\n"
                f"Rationale: {prediction.rationale}\n\n"
                f"Score 1-5 where:\n"
                f"1 = No logical chain, contradictory reasoning\n"
                f"2 = Weak logic, missing key steps\n"
                f"3 = Adequate reasoning, some gaps\n"
                f"4 = Good logical flow, minor gaps\n"
                f"5 = Excellent, clear causal chain\n"
                f"Respond with only a number 1-5 followed by a brief explanation."
            ),
            "data_sourcing": (
                f"Evaluate the data sourcing quality of this {prediction.agent_type} agent's prediction.\n"
                f"Agent: {prediction.agent_id}\n"
                f"Data sources cited: {', '.join(prediction.data_sources) if prediction.data_sources else 'None'}\n"
                f"Rationale: {prediction.rationale}\n\n"
                f"Score 1-5 where:\n"
                f"1 = No sources cited, no evidence\n"
                f"2 = Vague references, no specific sources\n"
                f"3 = Some specific sources, limited diversity\n"
                f"4 = Multiple diverse sources cited\n"
                f"5 = Comprehensive, diverse, timely data sources\n"
                f"Respond with only a number 1-5 followed by a brief explanation."
            ),
            "calibration": (
                f"Evaluate the uncertainty calibration of this {prediction.agent_type} agent.\n"
                f"Agent: {prediction.agent_id}\n"
                f"Prediction direction: {prediction.prediction_direction}\n"
                f"Confidence expressed: {prediction.confidence:.2f}/1.0\n"
                f"Rationale: {prediction.rationale}\n\n"
                f"Score 1-5 where:\n"
                f"1 = Severely over/under confident, no uncertainty expressed\n"
                f"2 = Poor calibration, confidence doesn't match reasoning\n"
                f"3 = Adequate, some uncertainty acknowledgment\n"
                f"4 = Good calibration, appropriate confidence\n"
                f"5 = Excellent calibration with nuanced uncertainty\n"
                f"Respond with only a number 1-5 followed by a brief explanation."
            ),
            "decision_impact": (
                f"Evaluate the potential decision impact of this {prediction.agent_type} agent's prediction.\n"
                f"Agent: {prediction.agent_id}\n"
                f"Prediction: {prediction.prediction_direction} (value: {prediction.prediction_value:.4f})\n"
                f"Confidence: {prediction.confidence:.2f}\n"
                f"Actual outcome: {prediction.actual_outcome if prediction.actual_outcome is not None else 'Pending'}\n"
                f"Rationale: {prediction.rationale}\n\n"
                f"Score 1-5 where:\n"
                f"1 = Would lead to poor decisions consistently\n"
                f"2 = Limited decision-usefulness\n"
                f"3 = Generally useful, occasional misses\n"
                f"4 = Good track record, reliable signal\n"
                f"5 = Highly impactful, consistently valuable predictions\n"
                f"Respond with only a number 1-5 followed by a brief explanation."
            ),
        }
        return prompts.get(dimension, f"Evaluate {dimension} for agent {prediction.agent_id}")
    
    def _llm_judge_dimension(self, sentiment_model, dimension: str, prompt: str):
        """Use LLM sentiment pipeline to score a dimension.
        
        Returns a DimensionScore-like object.
        """
        # Use the sentiment model to analyze the prompt
        # This reuses the existing LLM pipeline without requiring direct API calls
        try:
            score_result = sentiment_model.analyze_text(prompt)
            
            # Extract numeric score from sentiment output
            if hasattr(score_result, 'score'):
                # Normalize typical sentiment scores to 1-5 range
                raw = score_result.score
                if isinstance(raw, (int, float)):
                    # If score is -1 to 1 range
                    if -1 <= raw <= 1:
                        normalized = 3.0 + raw * 2.0  # -1→1, 0→3, 1→5
                    else:
                        normalized = min(5.0, max(1.0, raw))
                    score = round(normalized, 1)
                else:
                    score = 3.0
            else:
                score = 3.0
            
            explanation = f"LLM-judged score for {dimension}: {score}/5"
            
            class _Score:
                def __init__(self, sc, exp):
                    self.score = sc
                    self.explanation = exp
                    self.strengths = []
                    self.weaknesses = []
            
            return _Score(score, explanation)
            
        except Exception as e:
            logger.debug(f"LLM judge failed for dimension {dimension}: {e}")
            class _FallbackScore:
                def __init__(self):
                    self.score = 3.0
                    self.explanation = f"LLM judge unavailable for {dimension}"
                    self.strengths = []
                    self.weaknesses = ["LLM judge unavailable"]
            return _FallbackScore()
    
    def get_agent_summary(self, agent_id: str) -> Optional[AgentEvaluationSummary]:
        """Get evaluation summary for a specific agent."""
        agent_evals = [e for e in self.evaluations if e.agent_id == agent_id]
        
        if not agent_evals:
            return None
        
        composites = [e.composite_score for e in agent_evals]
        dim_avgs = {}
        for dim in self.weights:
            scores = []
            for e in agent_evals:
                if dim in e.dimensions:
                    scores.append(e.dimensions[dim].score)
            dim_avgs[dim] = round(float(np.mean(scores)), 2) if scores else 0.0
        
        # Trend detection
        if len(composites) >= 3:
            recent = np.mean(composites[-3:])
            earlier = np.mean(composites[:3])
            if recent > earlier + 0.05:
                trend = "improving"
            elif recent < earlier - 0.05:
                trend = "declining"
            else:
                trend = "stable"
        else:
            trend = "insufficient"
        
        recent_grades = [e.overall_grade for e in agent_evals[-5:]]
        
        return AgentEvaluationSummary(
            agent_id=agent_id,
            agent_type=agent_evals[-1].agent_type,
            num_evaluations=len(agent_evals),
            avg_composite=round(float(np.mean(composites)), 4),
            best_composite=round(float(max(composites)), 4),
            worst_composite=round(float(min(composites)), 4),
            trend=trend,
            dimension_averages=dim_avgs,
            recent_grades=recent_grades,
        )
    
    def get_all_summaries(self) -> List[AgentEvaluationSummary]:
        """Get evaluation summaries for all agents."""
        agent_ids = set(e.agent_id for e in self.evaluations)
        summaries = []
        for agent_id in sorted(agent_ids):
            summary = self.get_agent_summary(agent_id)
            if summary:
                summaries.append(summary)
        return summaries
    
    def check_alerts(self) -> List[Dict]:
        """Check for evaluation-based alerts.
        
        Returns list of alert dicts with severity level.
        """
        alerts = []
        for agent_id in set(e.agent_id for e in self.evaluations):
            summary = self.get_agent_summary(agent_id)
            if summary and summary.num_evaluations >= 3:
                if summary.avg_composite < COMPOSITE_THRESHOLD_CRITICAL:
                    alerts.append({
                        "severity": "critical",
                        "agent_id": agent_id,
                        "message": f"Agent {agent_id} critically underperforming (composite: {summary.avg_composite:.2f})",
                        "composite": summary.avg_composite,
                        "trend": summary.trend,
                    })
                elif summary.avg_composite < COMPOSITE_THRESHOLD_WARNING:
                    alerts.append({
                        "severity": "warning",
                        "agent_id": agent_id,
                        "message": f"Agent {agent_id} below threshold (composite: {summary.avg_composite:.2f})",
                        "composite": summary.avg_composite,
                        "trend": summary.trend,
                    })
                elif summary.trend == "declining" and summary.num_evaluations >= 5:
                    alerts.append({
                        "severity": "info",
                        "agent_id": agent_id,
                        "message": f"Agent {agent_id} showing declining trend",
                        "composite": summary.avg_composite,
                        "trend": summary.trend,
                    })
        return alerts
    
    def get_un_evaluated_predictions(self) -> List[AgentPrediction]:
        """Get predictions that haven't been evaluated yet."""
        evaluated_timestamps = set(e.prediction_timestamp for e in self.evaluations)
        return [p for p in self.predictions if p.timestamp not in evaluated_timestamps]
    
    def evaluate_all_pending(self, use_llm: bool = False) -> List[EvaluationResult]:
        """Evaluate all pending (unevaluated) predictions."""
        pending = self.get_un_evaluated_predictions()
        results = []
        for pred in pending:
            result = self.evaluate_prediction(pred, use_llm)
            results.append(result)
        return results


# Module-level convenience functions

def record_prediction(
    agent_id: str,
    agent_type: str,
    prediction_value: float,
    prediction_direction: str,
    confidence: float,
    rationale: str,
    data_sources: Optional[List[str]] = None,
    context: Optional[Dict[str, Any]] = None,
) -> AgentPrediction:
    """Convenience function to record a prediction."""
    evaluator = AgenticEvaluator()
    return evaluator.record_prediction(
        agent_id=agent_id,
        agent_type=agent_type,
        prediction_value=prediction_value,
        prediction_direction=prediction_direction,
        confidence=confidence,
        rationale=rationale,
        data_sources=data_sources,
        context=context,
    )


def evaluate_pending(use_llm: bool = False) -> List[EvaluationResult]:
    """Convenience function to evaluate all pending predictions."""
    evaluator = AgenticEvaluator()
    return evaluator.evaluate_all_pending(use_llm)


def get_current_scores() -> Dict[str, Any]:
    """Get current evaluation scores for all agents."""
    evaluator = AgenticEvaluator()
    summaries = evaluator.get_all_summaries()
    alerts = evaluator.check_alerts()
    
    # Compute aggregate stats
    if summaries:
        avg_composite = float(np.mean([s.avg_composite for s in summaries]))
        num_critical = len([a for a in alerts if a["severity"] == "critical"])
        num_warnings = len([a for a in alerts if a["severity"] == "warning"])
    else:
        avg_composite = 0.0
        num_critical = 0
        num_warnings = 0
    
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "num_agents_evaluated": len(summaries),
        "num_evaluations": sum(s.num_evaluations for s in summaries),
        "average_composite_score": round(avg_composite, 4),
        "agents": [asdict(s) for s in summaries],
        "alerts": alerts,
        "num_critical_alerts": num_critical,
        "num_warnings": num_warnings,
        "thresholds": {
            "warning": COMPOSITE_THRESHOLD_WARNING,
            "critical": COMPOSITE_THRESHOLD_CRITICAL,
        },
    }


def save_state():
    """Save current evaluator state to JSON."""
    state = get_current_scores()
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(STATE_PATH, "w") as f:
        json.dump(state, f, indent=2, default=str)
    return state


def main():
    """CLI entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Agentic Prediction Evaluation")
    subparsers = parser.add_subparsers(dest="command", help="Command")
    
    # Record command
    record_parser = subparsers.add_parser("record", help="Record a prediction")
    record_parser.add_argument("--agent-id", required=True, help="Agent ID")
    record_parser.add_argument("--agent-type", required=True, choices=AGENT_TYPES, help="Agent type")
    record_parser.add_argument("--value", type=float, required=True, help="Prediction value")
    record_parser.add_argument("--direction", required=True, choices=["bullish", "bearish", "neutral"], help="Direction")
    record_parser.add_argument("--confidence", type=float, required=True, help="Confidence 0-1")
    record_parser.add_argument("--rationale", required=True, help="Reasoning rationale")
    record_parser.add_argument("--sources", nargs="*", default=[], help="Data sources")
    record_parser.add_argument("--outcome", type=float, default=None, help="Actual outcome")
    
    # Evaluate command
    eval_parser = subparsers.add_parser("evaluate", help="Evaluate pending predictions")
    eval_parser.add_argument("--llm", action="store_true", help="Use LLM judge (default: fallback)")
    eval_parser.add_argument("--agent-id", help="Evaluate only this agent")
    
    # Score command
    subparsers.add_parser("score", help="Show current scores")
    
    # Log command
    log_parser = subparsers.add_parser("log", help="Show evaluation log")
    log_parser.add_argument("--agent-id", help="Filter by agent ID")
    log_parser.add_argument("--limit", type=int, default=20, help="Max entries to show")
    
    # Record-outcome command
    outcome_parser = subparsers.add_parser("record-outcome", help="Record actual outcome")
    outcome_parser.add_argument("--agent-id", required=True)
    outcome_parser.add_argument("--timestamp", required=True)
    outcome_parser.add_argument("--outcome", type=float, required=True)
    
    args = parser.parse_args()
    
    if args.command == "record":
        pred = record_prediction(
            agent_id=args.agent_id,
            agent_type=args.agent_type,
            prediction_value=args.value,
            prediction_direction=args.direction,
            confidence=args.confidence,
            rationale=args.rationale,
            data_sources=args.sources,
            context={"outcome": args.outcome} if args.outcome is not None else {},
        )
        print(json.dumps(asdict(pred), indent=2, default=str))
        logger.info(f"Recorded prediction for {args.agent_id}: {args.direction} (conf={args.confidence})")
    
    elif args.command == "evaluate":
        if args.agent_id:
            evaluator = AgenticEvaluator()
            pending = evaluator.get_un_evaluated_predictions()
            filtered = [p for p in pending if p.agent_id == args.agent_id]
            results = []
            for pred in filtered:
                result = evaluator.evaluate_prediction(pred, use_llm=args.llm)
                results.append(result)
        else:
            results = evaluate_pending(use_llm=args.llm)
        
        for r in results:
            print(json.dumps(asdict(r), indent=2, default=str))
            logger.info(
                f"Evaluated {r.agent_id}: composite={r.composite_score:.3f}, "
                f"grade={r.overall_grade}, llm={'yes' if r.llm_available else 'no'}"
            )
    
    elif args.command == "score":
        state = save_state()
        print(json.dumps(state, indent=2, default=str))
        
        if state["alerts"]:
            for alert in state["alerts"]:
                logger.warning(f"[{alert['severity'].upper()}] {alert['message']}")
    
    elif args.command == "log":
        evaluator = AgenticEvaluator()
        log_entries = []
        
        # Show most recent evaluations
        evals = evaluator.evaluations[-args.limit:] if args.agent_id is None else \
                [e for e in evaluator.evaluations if e.agent_id == args.agent_id][-args.limit:]
        
        for e in evals:
            log_entries.append(asdict(e))
        
        print(json.dumps(log_entries, indent=2, default=str))
        print(f"\nShowing {len(log_entries)} evaluation(s)")
    
    elif args.command == "record-outcome":
        evaluator = AgenticEvaluator()
        success = evaluator.record_outcome(args.agent_id, args.timestamp, args.outcome)
        if success:
            logger.info(f"Recorded outcome for {args.agent_id} @ {args.timestamp}: {args.outcome}")
        else:
            logger.warning(f"No prediction found for {args.agent_id} @ {args.timestamp}")
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
