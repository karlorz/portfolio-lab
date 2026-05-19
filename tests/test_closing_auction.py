"""Tests for src/signals/closing_auction.py — closing auction signal logic."""

from datetime import datetime, timedelta
import pytest
from unittest.mock import MagicMock, patch
from src.signals.closing_auction import (
    SignalDirection,
    SignalConfidence,
    ClosingAuctionSignal,
    HistoricalValidator,
)


# ── Fake MOCImbalance for test dataclass ──────────────────────────────────

class FakeMOCImbalance:
    def __init__(self, symbol="SPY", direction_score=2):
        self.symbol = symbol
        self.direction_score = direction_score
        self.imbalance_ratio = 0.3
        self.imbalance_shares = 50000
        self.reference_price = 500.0
        self.auction_price = None
        self.timestamp = datetime.now()
        self.confidence = 0.8

    def to_dict(self):
        return {"symbol": self.symbol, "direction_score": self.direction_score}


def make_signal(symbol="SPY", direction=SignalDirection.BUY,
                confidence=SignalConfidence.HIGH, minutes_ago=0,
                win_rate=0.70, count=50):
    """Helper: create a ClosingAuctionSignal for testing."""
    imb = FakeMOCImbalance(symbol, direction.value)
    ts = datetime.now() - timedelta(minutes=minutes_ago)
    return ClosingAuctionSignal(
        symbol=symbol,
        timestamp=ts,
        direction=direction,
        confidence=confidence,
        imbalance=imb,
        entry_price=500.0,
        target_exit_price=502.0,
        stop_loss_price=498.0,
        historical_win_rate=win_rate,
        historical_count=count,
        max_position_pct=0.05,
        urgency="high",
    )


# ── SignalDirection ───────────────────────────────────────────────────────

class TestSignalDirection:
    def test_values(self):
        assert SignalDirection.STRONG_BUY.value == 3
        assert SignalDirection.BUY.value == 2
        assert SignalDirection.WEAK_BUY.value == 1
        assert SignalDirection.NEUTRAL.value == 0
        assert SignalDirection.WEAK_SELL.value == -1
        assert SignalDirection.SELL.value == -2
        assert SignalDirection.STRONG_SELL.value == -3

    def test_buy_sell_symmetry(self):
        assert SignalDirection.STRONG_BUY.value == -SignalDirection.STRONG_SELL.value
        assert SignalDirection.BUY.value == -SignalDirection.SELL.value
        assert SignalDirection.WEAK_BUY.value == -SignalDirection.WEAK_SELL.value


# ── SignalConfidence ──────────────────────────────────────────────────────

class TestSignalConfidence:
    def test_values(self):
        assert SignalConfidence.HIGH.value == "high"
        assert SignalConfidence.MEDIUM.value == "medium"
        assert SignalConfidence.LOW.value == "low"
        assert SignalConfidence.INSUFFICIENT_DATA.value == "insufficient_data"


# ── ClosingAuctionSignal ──────────────────────────────────────────────────

class TestClosingAuctionSignal:
    def test_to_dict(self):
        signal = make_signal()
        d = signal.to_dict()
        assert d["symbol"] == "SPY"
        assert d["direction"] == "BUY"
        assert d["direction_score"] == 2
        assert d["confidence"] == "high"
        assert d["entry_price"] == 500.0
        assert d["historical_win_rate"] == 0.70

    def test_side_buy(self):
        signal = make_signal(direction=SignalDirection.STRONG_BUY)
        assert signal.side == "buy"

    def test_side_sell(self):
        signal = make_signal(direction=SignalDirection.SELL)
        assert signal.side == "sell"

    def test_side_neutral(self):
        signal = make_signal(direction=SignalDirection.NEUTRAL)
        assert signal.side == "neutral"

    def test_should_trade_high_confidence_buy(self):
        signal = make_signal(
            direction=SignalDirection.BUY, confidence=SignalConfidence.HIGH
        )
        assert signal.should_trade is True

    def test_should_not_trade_low_confidence(self):
        signal = make_signal(
            direction=SignalDirection.BUY, confidence=SignalConfidence.LOW
        )
        assert signal.should_trade is False

    def test_should_not_trade_neutral(self):
        signal = make_signal(
            direction=SignalDirection.NEUTRAL, confidence=SignalConfidence.HIGH
        )
        assert signal.should_trade is False

    def test_should_not_trade_stale_signal(self):
        signal = make_signal(minutes_ago=10)
        assert signal.should_trade is False

    def test_should_not_trade_insufficient_data(self):
        signal = make_signal(
            direction=SignalDirection.BUY,
            confidence=SignalConfidence.INSUFFICIENT_DATA,
        )
        assert signal.should_trade is False

    def test_weak_buy_is_not_neutral(self):
        signal = make_signal(direction=SignalDirection.WEAK_BUY)
        assert signal.side == "buy"
        assert signal.direction != SignalDirection.NEUTRAL

    def test_strong_sell_side(self):
        signal = make_signal(direction=SignalDirection.STRONG_SELL)
        assert signal.side == "sell"
