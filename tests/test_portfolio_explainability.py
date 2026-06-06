"""Tests for portfolio explainability data generation."""

from src.dashboard.explainability import build_portfolio_explainability


def test_builds_top_drivers_and_opposers_from_ensemble_breakdown():
    ensemble = {
        "regime": "normal",
        "regime_confidence": 0.74,
        "weighted_consensus": 0.18,
        "agreement_ratio": 0.66,
        "action": "increase_equity",
        "confidence": 0.62,
        "num_sources": 3,
        "n_eff": 2.4,
        "source_breakdown": [
            {
                "source": "alternative_data",
                "direction": "bullish",
                "strength": 0.5,
                "confidence": 0.8,
                "weight": 0.4,
            },
            {
                "source": "unified_overlay",
                "direction": "bearish",
                "strength": 0.3,
                "confidence": 0.7,
                "weight": 0.2,
            },
            {
                "source": "vix_term_structure",
                "direction": "bullish",
                "strength": 0.2,
                "confidence": 0.9,
                "weight": 0.1,
            },
        ],
    }

    result = build_portfolio_explainability(
        ensemble,
        analysis_date="2026-06-07",
        timestamp="2026-06-07T12:00:00",
    )

    latest = result["latest_decision"]
    assert latest["regime"] == "normal"
    assert latest["consensus_direction"] == "bullish"
    assert latest["total_signals"] == 3
    assert latest["signals"][0]["source"] == "alternative_data"
    assert latest["signals"][0]["contribution"] == 0.2
    assert latest["top_drivers"][0] == {
        "source": "alternative_data",
        "contribution": 0.2,
        "direction": "bullish",
    }
    assert latest["top_opposers"][0] == {
        "source": "unified_overlay",
        "contribution": -0.06,
        "direction": "bearish",
    }
    assert result["top_sources_today"] == [
        "alternative_data",
        "unified_overlay",
        "vix_term_structure",
    ]


def test_empty_or_missing_ensemble_returns_no_decision():
    result = build_portfolio_explainability(None, analysis_date="2026-06-07")

    assert result["latest_decision"] is None
    assert result["top_sources_today"] == []
    assert result["decision_quality"]["status"] == "no_ensemble_data"


def test_limits_to_top_five_signals():
    ensemble = {
        "regime": "normal",
        "weighted_consensus": -0.1,
        "source_breakdown": [
            {
                "source": f"signal_{i}",
                "direction": "bearish" if i % 2 else "bullish",
                "strength": 0.1 + i / 100,
                "confidence": 0.5,
                "weight": 0.1,
            }
            for i in range(8)
        ],
    }

    result = build_portfolio_explainability(ensemble, analysis_date="2026-06-07")

    assert len(result["latest_decision"]["signals"]) == 5
    assert len(result["top_sources_today"]) == 5
