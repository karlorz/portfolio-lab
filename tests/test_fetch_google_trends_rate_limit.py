"""Batch CG: Google Trends 429 → exit 3 (rate_limited), not generic 1."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import scripts.fetch_google_trends as ft


def test_is_rate_limit_error():
    assert ft._is_rate_limit_error(Exception("The request failed: code 429"))
    assert ft._is_rate_limit_error(Exception("Too Many Requests"))
    assert not ft._is_rate_limit_error(Exception("timeout"))


def test_fetch_trends_returns_rate_limited_flag_on_429():
    mock_req = MagicMock()
    mock_req.return_value.build_payload.side_effect = Exception("response with code 429")
    with patch.dict("sys.modules", {"pytrends": MagicMock(), "pytrends.request": MagicMock(TrendReq=mock_req)}):
        # re-import path uses TrendReq from pytrends.request inside function
        with patch("scripts.fetch_google_trends.time.sleep"):
            # Patch TrendReq at import site inside fetch_trends
            with patch.object(ft, "fetch_trends", wraps=None):
                pass
    # Direct: inject TrendReq via patching the import inside function
    with patch("pytrends.request.TrendReq", mock_req), patch.object(ft.time, "sleep"):
        results, rate_limited = ft.fetch_trends(["recession"], days=7, max_retries=1)
    assert results == {}
    assert rate_limited is True
