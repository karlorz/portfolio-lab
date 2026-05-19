#!/usr/bin/env python3
"""
v7.10: Natural Language Portfolio Query

Reads unified_dashboard.json and answers natural language questions about
the portfolio using Claude Haiku (with structured fallback for no API).

Usage:
    python -m src.chat.portfolio_query "What is my equity exposure?"
    python -m src.chat.portfolio_query "Which signals are bearish?"
    make ask "How diversified am I?"
"""

import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Guarded anthropic import
try:
    from anthropic import Anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    Anthropic = None  # type: ignore
    ANTHROPIC_AVAILABLE = False

SUPPORTED_QUERY_TYPES = {
    "portfolio": [
        "equity exposure", "position", "allocation", "weight",
        "how much", "portfolio value", "holdings",
    ],
    "risk": [
        "drawdown", "VaR", "CVaR", "volatility", "risk", "diversified",
        "diversification", "concentration",
    ],
    "signals": [
        "signal", "bearish", "bullish", "driving", "momentum",
        "factor", "trend", "attribution",
    ],
    "overlays": [
        "overlay", "collar", "crypto", "bond duration", "hedge",
    ],
    "costs": [
        "slippage", "cost", "fee", "trading cost", "TCA", "execution",
    ],
}


def build_query_context(dashboard: Dict[str, Any]) -> str:
    """Build a structured text context from the unified dashboard for the LLM."""
    parts = []

    portfolio = dashboard.get("portfolio", {})
    if portfolio.get("available"):
        total = portfolio.get("total_value", 0)
        positions = portfolio.get("positions", [])
        pos_str = "; ".join(
            f"{p['symbol']}: {p['weight']:.1f}% (${p['value']:,.0f})"
            for p in positions
        )
        parts.append(f"Portfolio: ${total:,.0f} total. {pos_str}")

    risk = dashboard.get("risk", {})
    if risk.get("available"):
        parts.append(
            f"Risk: DD {risk.get('current_drawdown', 0):.1f}%, "
            f"VaR95 {risk.get('var_95_daily')}%, "
            f"CVaR95 {risk.get('cvar_95_daily')}%, "
            f"Vol(ann) {risk.get('volatility_annual')}%"
        )

    overlays = dashboard.get("overlays", {})
    active = [n for n, d in overlays.items() if d.get("active") and not n.startswith("_")]
    parts.append(f"Active overlays: {', '.join(active) if active else 'none'}")

    attribution = dashboard.get("attribution", {})
    if attribution.get("available"):
        sources = attribution.get("sources", [])
        src_str = "; ".join(
            f"{s['name']}: {s.get('total_return_bps', 0):+.1f}bps (hit={s.get('hit_rate', 0):.0%})"
            for s in sources[:10]
        )
        parts.append(f"Signal sources: {src_str}")

    tca = dashboard.get("tca", {})
    if tca.get("available"):
        sc = tca.get("scorecard", {})
        parts.append(
            f"TCA: avg slippage {sc.get('avg_slippage_bps', 0):.1f}bps, "
            f"quality {sc.get('avg_quality_score', 0):.0f}/100, "
            f"{sc.get('total_orders', 0)} orders"
        )

    regime = dashboard.get("regime", {})
    if regime.get("available"):
        clf = regime.get("classifier", {})
        parts.append(f"Regime: {clf.get('current_regime', 'unknown')}")

    health = dashboard.get("health", {})
    if health.get("available"):
        alerts = health.get("alerts", [])
        if alerts:
            parts.append(f"Alerts: {'; '.join(str(a) for a in alerts)}")

    for k in list(dashboard.keys()):
        v = dashboard[k]
        if isinstance(v, dict) and not v.get("available", True):
            parts.append(f"{k}: not available")

    return "\n".join(parts)


def format_system_prompt(context: str) -> str:
    """Format the system prompt with portfolio context."""
    return (
        "You are a personal portfolio management assistant. "
        "Answer questions concisely using the portfolio data provided. "
        "Include specific numbers when available. "
        "If the data doesn't contain the answer, say so clearly.\n\n"
        f"Current portfolio state:\n{context}"
    )


def fallback_answer(question: str, dashboard: Dict[str, Any]) -> str:
    """Generate a structured answer without LLM, using template matching."""
    q = question.lower()
    portfolio = dashboard.get("portfolio", {})
    risk = dashboard.get("risk", {})
    overlays = dashboard.get("overlays", {})
    attribution = dashboard.get("attribution", {})
    tca = dashboard.get("tca", {})

    # Portfolio queries
    if any(kw in q for kw in SUPPORTED_QUERY_TYPES["portfolio"]):
        if "exposure" in q or "allocation" in q or "weight" in q:
            pos_list = portfolio.get("positions", [])
            lines = [f"{p['symbol']}: {p['weight']:.1f}% (${p['value']:,.0f})" for p in pos_list]
            return "Current allocation:\n" + "\n".join(lines)
        if "value" in q or "how much" in q:
            return f"Total portfolio value: ${portfolio.get('total_value', 0):,.0f}"

    # Risk queries
    if any(kw in q for kw in SUPPORTED_QUERY_TYPES["risk"]):
        return (
            f"Current drawdown: {risk.get('current_drawdown', 'N/A')}%\n"
            f"VaR (95% daily): {risk.get('var_95_daily', 'N/A')}%\n"
            f"CVaR (95% daily): {risk.get('cvar_95_daily', 'N/A')}%\n"
            f"Annual volatility: {risk.get('volatility_annual', 'N/A')}%"
        )

    # Overlay queries
    if any(kw in q for kw in SUPPORTED_QUERY_TYPES["overlays"]):
        lines = []
        for name, data in overlays.items():
            if not name.startswith("_"):
                status = "active" if data.get("active") else "inactive"
                lines.append(f"  {name}: {status}")
        return "Overlay status:\n" + "\n".join(lines) if lines else "No overlays configured."

    # Cost queries
    if any(kw in q for kw in SUPPORTED_QUERY_TYPES["costs"]):
        sc = tca.get("scorecard", {})
        return (
            f"Avg slippage: {sc.get('avg_slippage_bps', 'N/A')} bps\n"
            f"Avg quality score: {sc.get('avg_quality_score', 'N/A')}/100\n"
            f"Total orders: {sc.get('total_orders', 'N/A')}"
        )

    # Signal queries
    if any(kw in q for kw in SUPPORTED_QUERY_TYPES["signals"]):
        sources = attribution.get("sources", [])
        if not sources:
            return "Signal attribution data not available. Run performance attribution first."
        bullish = [s for s in sources if s.get("total_return_bps", 0) > 0]
        bearish = [s for s in sources if s.get("total_return_bps", 0) < 0]
        lines = [
            f"Bullish ({len(bullish)}): " + ", ".join(s["name"] for s in bullish[:5]),
            f"Bearish ({len(bearish)}): " + ", ".join(s["name"] for s in bearish[:5]),
        ]
        return "\n".join(lines)

    return (
        "I'm not sure how to answer that. Try asking about:\n"
        "- Portfolio allocation or exposure\n"
        "- Current risk metrics or drawdown\n"
        "- Active overlays\n"
        "- Signal sources (bullish/bearish)\n"
        "- Trading costs or slippage"
    )


def answer_query(question: str, dashboard: Optional[Dict[str, Any]] = None) -> str:
    """Answer a natural language query about the portfolio.

    Uses Claude Haiku if available, falls back to template matching.
    """
    if dashboard is None:
        try:
            from src.monitor.unified_dashboard import generate_unified_dashboard
            dashboard = generate_unified_dashboard()
        except Exception:
            dashboard = {}

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key or not ANTHROPIC_AVAILABLE:
        return fallback_answer(question, dashboard)

    try:
        client = Anthropic(api_key=api_key)
        context = build_query_context(dashboard)
        system_prompt = format_system_prompt(context)

        response = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=300,
            system=system_prompt,
            messages=[{"role": "user", "content": question}],
        )
        return response.content[0].text.strip()
    except Exception as e:
        logger.warning("LLM query failed: %s, using fallback", e)
        return fallback_answer(question, dashboard)


def main():
    if len(sys.argv) < 2:
        print("Usage: python -m src.chat.portfolio_query <question>")
        print('Example: python -m src.chat.portfolio_query "What is my equity exposure?"')
        sys.exit(1)

    question = " ".join(sys.argv[1:])
    print(f"Q: {question}\n")

    answer = answer_query(question)
    print(answer)


if __name__ == "__main__":
    main()
