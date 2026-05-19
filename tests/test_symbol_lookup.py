"""Tests for src/utils/symbol_lookup.py — Symbol-to-category lookup."""

import pytest
from src.utils.symbol_lookup import get_symbol_category, SYMBOL_TO_CATEGORY


class TestGetSymbolCategory:
    def test_equity_symbols(self):
        assert get_symbol_category("SPY") == "equity"
        assert get_symbol_category("QQQ") == "equity"

    def test_commodity_symbols(self):
        assert get_symbol_category("GLD") == "commodity"

    def test_bond_symbols(self):
        assert get_symbol_category("TLT") == "bond"
        assert get_symbol_category("IEF") == "bond"
        assert get_symbol_category("SHY") == "bond"

    def test_crypto_symbols(self):
        assert get_symbol_category("BTC") == "crypto"
        assert get_symbol_category("ETH") == "crypto"

    def test_unknown_symbol(self):
        assert get_symbol_category("UNKNOWN") == "unknown"
        assert get_symbol_category("AAPL") == "unknown"

    def test_case_insensitive(self):
        assert get_symbol_category("spy") == "equity"
        assert get_symbol_category("gld") == "commodity"
        assert get_symbol_category("btc") == "crypto"

    def test_all_known_symbols_have_category(self):
        for symbol in SYMBOL_TO_CATEGORY:
            assert get_symbol_category(symbol) != "unknown"
            assert get_symbol_category(symbol) == SYMBOL_TO_CATEGORY[symbol]

    def test_empty_string(self):
        assert get_symbol_category("") == "unknown"
