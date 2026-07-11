"""CLI output visibility tests for ``python -m src.signals.behavioral_sentiment``."""

import runpy
import sys
import warnings

from src.data.behavioral_sentiment_fetcher import (
    BehavioralSentimentSnapshot,
    OptionsSentiment,
    RetailFlow,
    SocialIntensity,
)


def _snapshot() -> BehavioralSentimentSnapshot:
    return BehavioralSentimentSnapshot(
        timestamp="2026-07-05T00:00:00",
        options=OptionsSentiment(
            timestamp="2026-07-05T00:00:00",
            skew_index=102.0,
            vix=16.0,
            vix9d=14.4,
            vix9d_ratio=0.9,
            put_call_ratio=0.65,
            fear_greed_score=0.0,
        ),
        retail=RetailFlow(
            timestamp="2026-07-05T00:00:00",
            retail_call_put_ratio=1.0,
            retail_buy_sell_imbalance=0.0,
            retail_top_100_correlation=-0.15,
            small_lot_premium_ratio=0.85,
        ),
        social=SocialIntensity(
            timestamp="2026-07-05T00:00:00",
            mention_velocity_7d=1.0,
            sentiment_divergence=0.0,
            bot_activity_flag=False,
            influencer_concentration=0.15,
        ),
        composite_score=0.0,
        signal_type="neutral",
        confidence=0.7,
        data_fresh=True,
    )


def _run_behavioral_sentiment_module(monkeypatch, tmp_path, *args) -> None:
    import src.data.behavioral_sentiment_fetcher as fetcher_module
    import src.paths as paths

    monkeypatch.setattr(paths, "MARKET_DB", tmp_path / "market.db")
    monkeypatch.setattr(
        fetcher_module.BehavioralSentimentFetcher,
        "fetch_snapshot",
        lambda self: _snapshot(),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        ["python -m src.signals.behavioral_sentiment", *args],
    )

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            category=RuntimeWarning,
            message=".*found in sys.modules.*",
        )
        runpy.run_module("src.signals.behavioral_sentiment", run_name="__main__")


def test_status_command_emits_visible_output(monkeypatch, tmp_path, capsys):
    _run_behavioral_sentiment_module(monkeypatch, tmp_path, "--status")

    captured = capsys.readouterr()
    combined = captured.out + captured.err

    assert "Signal Generator Status" in combined
    assert "paused" in combined


def test_signal_command_emits_visible_output_without_live_network(
    monkeypatch, tmp_path, capsys
):
    _run_behavioral_sentiment_module(monkeypatch, tmp_path, "--signal")

    captured = capsys.readouterr()
    combined = captured.out + captured.err

    assert "Behavioral Sentiment Signal" in combined
    assert "Signal Type: neutral" in combined
