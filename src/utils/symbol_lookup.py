"""Symbol-to-category lookup for portfolio asset classification."""

SYMBOL_TO_CATEGORY: dict[str, str] = {
    "SPY": "equity",
    "QQQ": "equity",
    "GLD": "commodity",
    "TLT": "bond",
    "IEF": "bond",
    "SHY": "bond",
    "BTC": "crypto",
    "ETH": "crypto",
}


def get_symbol_category(symbol: str) -> str:
    """Return the asset category for a given symbol.

    Returns "unknown" for symbols not in the lookup table.
    """
    if not symbol:
        return "unknown"
    return SYMBOL_TO_CATEGORY.get(symbol.upper(), "unknown")


if __name__ == "__main__":
    expected = {
        "SPY": "equity",
        "QQQ": "equity",
        "GLD": "commodity",
        "TLT": "bond",
        "IEF": "bond",
        "SHY": "bond",
        "BTC": "crypto",
        "ETH": "crypto",
    }
    for sym, cat in expected.items():
        result = get_symbol_category(sym)
        assert result == cat, f"{sym}: expected {cat}, got {result}"
    print("All 8 symbol lookups passed.")
