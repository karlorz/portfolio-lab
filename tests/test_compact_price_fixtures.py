from tests.fixtures.compact_prices import (
    COMPACT_PRICE_FIXTURE_NAMES,
    adjusted_close_proxy_prices,
    clean_compact_prices,
    duplicate_date_prices,
    extreme_return_prices,
    internal_gap_prices,
    negative_price_prices,
    non_monotonic_prices,
    split_like_return_prices,
    stale_latest_prices,
    zero_price_prices,
)


def test_compact_price_fixture_builders_cover_expected_issue_shapes() -> None:
    assert set(COMPACT_PRICE_FIXTURE_NAMES) == {
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
    }

    assert clean_compact_prices() == {
        "SPY": [
            {"d": "2026-06-10", "p": 612.34},
            {"d": "2026-06-11", "p": 614.25},
        ],
        "GLD": [{"d": "2026-06-11", "p": 318.12}],
    }
    assert adjusted_close_proxy_prices() == {
        "SPY": [{"d": "2026-06-10", "p": 612.34}],
        "TLT": [{"d": "2026-06-10", "p": 88.75}],
    }
    assert duplicate_date_prices()["SPY"] == [
        {"d": "2026-06-10", "p": 612.34},
        {"d": "2026-06-10", "p": 612.35},
    ]
    assert internal_gap_prices()["TLT"] == [
        {"d": "2026-06-10", "p": 88.75},
        {"d": "2026-06-12", "p": 89.01},
    ]
    assert stale_latest_prices()["GLD"][-1]["d"] == "2026-06-11"
    assert non_monotonic_prices()["SPY"][1]["d"] == "2026-06-10"
    assert zero_price_prices()["SPY"][0]["p"] == 0
    assert negative_price_prices()["SPY"][0]["p"] == -1
    assert split_like_return_prices()["SPY"] == [
        {"d": "2026-06-10", "p": 100},
        {"d": "2026-06-11", "p": 45},
    ]
    assert extreme_return_prices()["SPY"] == [
        {"d": "2026-06-10", "p": 100},
        {"d": "2026-06-11", "p": 250},
    ]


def test_compact_price_fixture_builders_return_fresh_payloads() -> None:
    first = clean_compact_prices()
    second = clean_compact_prices()

    first["SPY"][0]["p"] = 1

    assert second["SPY"][0]["p"] == 612.34
