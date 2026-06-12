"""Synthetic compact-price payload builders for data-quality audit tests."""

from __future__ import annotations

from copy import deepcopy
from typing import Final, Literal, TypeAlias, TypedDict


class CompactPriceRow(TypedDict):
    d: str
    p: float


CompactPricePayload: TypeAlias = dict[str, list[CompactPriceRow]]
CompactPriceFixtureName: TypeAlias = Literal[
    "clean",
    "adjusted_close_proxy",
    "duplicate_date",
    "internal_gap",
    "stale_latest",
    "non_monotonic",
    "zero_price",
    "negative_price",
    "split_like_return",
    "extreme_return",
]

COMPACT_PRICE_FIXTURE_NAMES: Final[tuple[CompactPriceFixtureName, ...]] = (
    "clean",
    "adjusted_close_proxy",
    "duplicate_date",
    "internal_gap",
    "stale_latest",
    "non_monotonic",
    "zero_price",
    "negative_price",
    "split_like_return",
    "extreme_return",
)

_FIXTURES: Final[dict[CompactPriceFixtureName, CompactPricePayload]] = {
    "clean": {
        "SPY": [
            {"d": "2026-06-10", "p": 612.34},
            {"d": "2026-06-11", "p": 614.25},
        ],
        "GLD": [{"d": "2026-06-11", "p": 318.12}],
    },
    "adjusted_close_proxy": {
        "SPY": [{"d": "2026-06-10", "p": 612.34}],
        "TLT": [{"d": "2026-06-10", "p": 88.75}],
    },
    "duplicate_date": {
        "SPY": [
            {"d": "2026-06-10", "p": 612.34},
            {"d": "2026-06-10", "p": 612.35},
        ],
    },
    "internal_gap": {
        "SPY": [
            {"d": "2026-06-10", "p": 612.34},
            {"d": "2026-06-11", "p": 614.25},
            {"d": "2026-06-12", "p": 615.10},
        ],
        "TLT": [
            {"d": "2026-06-10", "p": 88.75},
            {"d": "2026-06-12", "p": 89.01},
        ],
    },
    "stale_latest": {
        "SPY": [
            {"d": "2026-06-10", "p": 612.34},
            {"d": "2026-06-11", "p": 614.25},
            {"d": "2026-06-12", "p": 615.10},
        ],
        "GLD": [
            {"d": "2026-06-10", "p": 318.12},
            {"d": "2026-06-11", "p": 319.20},
        ],
    },
    "non_monotonic": {
        "SPY": [
            {"d": "2026-06-11", "p": 612.34},
            {"d": "2026-06-10", "p": 613.50},
        ],
    },
    "zero_price": {
        "SPY": [{"d": "2026-06-10", "p": 0}],
    },
    "negative_price": {
        "SPY": [{"d": "2026-06-10", "p": -1}],
    },
    "split_like_return": {
        "SPY": [
            {"d": "2026-06-10", "p": 100},
            {"d": "2026-06-11", "p": 45},
        ],
    },
    "extreme_return": {
        "SPY": [
            {"d": "2026-06-10", "p": 100},
            {"d": "2026-06-11", "p": 250},
        ],
    },
}


def build_compact_price_fixture(name: CompactPriceFixtureName) -> CompactPricePayload:
    return deepcopy(_FIXTURES[name])


def clean_compact_prices() -> CompactPricePayload:
    return build_compact_price_fixture("clean")


def adjusted_close_proxy_prices() -> CompactPricePayload:
    return build_compact_price_fixture("adjusted_close_proxy")


def duplicate_date_prices() -> CompactPricePayload:
    return build_compact_price_fixture("duplicate_date")


def internal_gap_prices() -> CompactPricePayload:
    return build_compact_price_fixture("internal_gap")


def stale_latest_prices() -> CompactPricePayload:
    return build_compact_price_fixture("stale_latest")


def non_monotonic_prices() -> CompactPricePayload:
    return build_compact_price_fixture("non_monotonic")


def zero_price_prices() -> CompactPricePayload:
    return build_compact_price_fixture("zero_price")


def negative_price_prices() -> CompactPricePayload:
    return build_compact_price_fixture("negative_price")


def split_like_return_prices() -> CompactPricePayload:
    return build_compact_price_fixture("split_like_return")


def extreme_return_prices() -> CompactPricePayload:
    return build_compact_price_fixture("extreme_return")
