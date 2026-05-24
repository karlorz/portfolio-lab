#!/usr/bin/env python3
"""
v7.10: Daily Personal CFO Brief Generator

Reads unified_dashboard.json and generates a structured daily portfolio brief
with template-driven data sections and optional LLM narrative (Claude Haiku).

Usage:
    python -m src.monitor.daily_brief              # Console brief
    python -m src.monitor.daily_brief --save       # Save JSON + print
    python -m src.monitor.daily_brief --no-narrative  # Skip LLM call
"""

import json
import logging
import os
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from src.paths import DATA_DIR


__all__ = ['SEVERITY_THRESHOLDS', 'BriefSection', 'generate_brief_sections', 'render_brief_text', 'generate_narrative', 'generate_daily_brief']

logger = logging.getLogger(__name__)

# Guarded anthropic import (follows src/llm/sentiment_client.py pattern)
try:
    from anthropic import Anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    Anthropic = None  # type: ignore
    ANTHROPIC_AVAILABLE = False

SEVERITY_THRESHOLDS = {
    "drawdown_warning": -10.0,
    "drawdown_alert": -20.0,
    "slippage_warning_bps": 8.0,
    "slippage_alert_bps": 15.0,
}


@dataclass
class BriefSection:
    name: str
    title: str
    severity: str  # normal, warning, alert
    data_text: str
    recommendation: str = ""


def generate_brief_sections(dashboard: Dict[str, Any]) -> List[BriefSection]:
    """Generate all brief sections from unified dashboard data."""
    sections = []
    portfolio = dashboard.get("portfolio", {})
    risk = dashboard.get("risk", {})
    overlays = dashboard.get("overlays", {})
    regime = dashboard.get("regime", {})
    tca = dashboard.get("tca", {})
    attribution = dashboard.get("attribution", {})
    health = dashboard.get("health", {})

    # ── Portfolio Snapshot ──
    total_val = portfolio.get("total_value", 0)
    positions = portfolio.get("positions", [])
    pos_text = ", ".join(
        f"{p['symbol']} {p['weight']:.0f}%" for p in positions[:5]
    ) if positions else "No positions"
    sections.append(BriefSection(
        name="portfolio_snapshot",
        title="Portfolio Snapshot",
        severity="normal",
        data_text=f"Total value: ${total_val:,.0f}. Allocation: {pos_text}.",
    ))

    # ── Risk Check ──
    dd = risk.get("current_drawdown", 0) or 0
    var95 = risk.get("var_95_daily")
    vol = risk.get("volatility_annual")
    if abs(dd) >= abs(SEVERITY_THRESHOLDS["drawdown_alert"]):
        risk_severity = "alert"
        risk_rec = "Review hedges. Consider reducing equity exposure."
    elif abs(dd) >= abs(SEVERITY_THRESHOLDS["drawdown_warning"]):
        risk_severity = "warning"
        risk_rec = "Monitor drawdown. Check circuit breaker status."
    else:
        risk_severity = "normal"
        risk_rec = "Risk within normal parameters."
    var_text = f"{var95}" if var95 is not None else "N/A"
    vol_text = f"{vol}" if vol is not None else "N/A"
    sections.append(BriefSection(
        name="risk_check",
        title="Risk Check",
        severity=risk_severity,
        data_text=f"Current DD: {dd:.1f}%. VaR(95): {var_text}%. Vol(ann): {vol_text}%. Regime: {regime.get('classifier', {}).get('current_regime', 'unknown')}.",
        recommendation=risk_rec,
    ))

    # ── Signal Roundup ──
    sources = attribution.get("sources", [])
    bullish = [s for s in sources if s.get("total_return_bps", 0) > 0]
    bearish = [s for s in sources if s.get("total_return_bps", 0) < 0]
    top_bull = bullish[0]["name"] if bullish else "none"
    top_bear = bearish[0]["name"] if bearish else "none"
    signal_severity = "warning" if len(bearish) > len(bullish) else "normal"
    sections.append(BriefSection(
        name="signal_roundup",
        title="Signal Roundup",
        severity=signal_severity,
        data_text=f"{len(bullish)} bullish, {len(bearish)} bearish. Top bull: {top_bull}. Top bear: {top_bear}.",
    ))

    # ── Overlay Status ──
    meta = overlays.get("_meta", {})
    active_count = meta.get("active_count", 0)
    active_names = [name for name, data in overlays.items()
                    if isinstance(data, dict) and data.get("active") and not name.startswith("_")]
    sections.append(BriefSection(
        name="overlay_status",
        title="Overlay Status",
        severity="normal",
        data_text=f"{active_count} active: {', '.join(active_names) if active_names else 'none'}.",
    ))

    # ── TCA Watch ──
    scorecard = tca.get("scorecard", {})
    avg_slip = scorecard.get("avg_slippage_bps", 0) or 0
    if avg_slip >= SEVERITY_THRESHOLDS["slippage_alert_bps"]:
        tca_severity = "alert"
        tca_rec = "Slippage high. Review order routing and limit order settings."
    elif avg_slip >= SEVERITY_THRESHOLDS["slippage_warning_bps"]:
        tca_severity = "warning"
        tca_rec = "Slippage above threshold. Consider optimal execution windows."
    else:
        tca_severity = "normal"
        tca_rec = "Execution quality within acceptable range."
    sections.append(BriefSection(
        name="tca_watch",
        title="TCA Watch",
        severity=tca_severity,
        data_text=f"Avg slippage: {avg_slip:.1f} bps. Orders: {scorecard.get('total_orders', 0)}.",
        recommendation=tca_rec,
    ))

    # ── Model Validation (DSR + BL) ──
    model_severity = "normal"
    model_text_parts = []

    # Deflated Sharpe Ratio — validates champion against multiple testing
    try:
        from src.backtest.metrics import compute_deflated_sharpe_ratio
        dsr = compute_deflated_sharpe_ratio(
            sharpe_ratio=0.79, n_trials=94, n_observations=5371,
        )
        model_text_parts.append(f"DSR={dsr:.2f} (94 configs)")
        if dsr < 0.50:
            model_severity = "warning"
    except (ImportError, ValueError, ZeroDivisionError, OverflowError):
        model_text_parts.append("DSR: unavailable")

    # Black-Litterman posterior — shows BL weight perspective
    bl_weights = dashboard.get("bl_weights")
    if bl_weights:
        bl_parts = [f"{k} {v:.0%}" for k, v in bl_weights.items()]
        model_text_parts.append(f"BL: {', '.join(bl_parts)}")

    model_text = ". ".join(model_text_parts) if model_text_parts else "No validation data."
    sections.append(BriefSection(
        name="model_validation",
        title="Model Validation",
        severity=model_severity,
        data_text=model_text,
    ))

    # ── Action Items ──
    warnings = [s for s in sections if s.severity in ("warning", "alert")]
    alerts = health.get("alerts", [])
    if warnings or alerts:
        action_severity = "alert" if any(s.severity == "alert" for s in warnings) else "warning"
        action_items = []
        for w in warnings:
            if w.recommendation:
                action_items.append(w.recommendation)
        for a in alerts[:3]:
            action_items.append(str(a))
        action_text = "; ".join(action_items)
    else:
        action_severity = "normal"
        action_text = "No action needed. Portfolio is on track."
    sections.append(BriefSection(
        name="action_items",
        title="Action Items",
        severity=action_severity,
        data_text=action_text,
    ))

    return sections


def render_brief_text(sections: List[BriefSection], narrative: Optional[str] = None) -> str:
    """Render brief sections into formatted text output."""
    now = datetime.now()
    lines = [
        "=" * 60,
        "  PORTFOLIO-LAB DAILY BRIEF",
        f"  {now.strftime('%Y-%m-%d %H:%M')}",
        "=" * 60,
        "",
    ]

    if narrative:
        lines.append(f"  {narrative}")
        lines.append("")

    severity_icons = {"normal": "   ", "warning": " W ", "alert": " A "}
    for section in sections:
        icon = severity_icons.get(section.severity, "   ")
        lines.append(f"{icon} {section.title.upper()}")
        lines.append(f"     {section.data_text}")
        if section.recommendation:
            lines.append(f"     -> {section.recommendation}")
        lines.append("")

    lines.append("=" * 60)
    return "\n".join(lines)


def generate_narrative(dashboard: Dict[str, Any]) -> Optional[str]:
    """Generate natural-language narrative using Claude Haiku.

    Returns None if API is unavailable or fails -- caller uses template fallback.
    """
    if not ANTHROPIC_AVAILABLE:
        return None

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        return None

    try:
        client = Anthropic(api_key=api_key)

        portfolio = dashboard.get("portfolio", {})
        risk = dashboard.get("risk", {})
        overlays = dashboard.get("overlays", {})

        total_val = portfolio.get("total_value", 0)
        dd = risk.get("current_drawdown", 0) or 0
        meta = overlays.get("_meta", {})
        active_count = meta.get("active_count", 0)

        prompt = (
            f"Write a 2-3 sentence morning brief for a portfolio worth ${total_val:,.0f}. "
            f"Current drawdown is {dd:.1f}%. "
            f"{active_count} tactical overlays are active. "
            f"Be concise and professional. Include one actionable observation if relevant."
        )

        response = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=150,
            system="You are a personal CFO assistant. Be concise, professional, and data-driven.",
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text.strip()
    except Exception as e:
        logger.warning("Narrative generation failed: %s", e)
        return None


def generate_daily_brief() -> Dict[str, Any]:
    """Generate the complete daily brief, returning a dict with all data."""
    try:
        from src.monitor.unified_dashboard import generate_unified_dashboard
        dashboard = generate_unified_dashboard()
    except Exception as e:
        logger.exception("Failed to generate unified dashboard: %s", e)
        dashboard = {}

    sections = generate_brief_sections(dashboard)

    overall_severity = "normal"
    for s in sections:
        if s.severity == "alert":
            overall_severity = "alert"
            break
        if s.severity == "warning":
            overall_severity = "warning"

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "sections": [asdict(s) for s in sections],
        "full_text": render_brief_text(sections),
        "severity": overall_severity,
        "has_narrative": False,
    }


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Daily Personal CFO Brief")
    parser.add_argument("--save", action="store_true", help="Save brief to JSON")
    parser.add_argument("--no-narrative", action="store_true", help="Skip LLM narrative")
    args = parser.parse_args()

    brief = generate_daily_brief()

    # Try narrative unless disabled
    if not args.no_narrative:
        try:
            from src.monitor.unified_dashboard import generate_unified_dashboard
            dashboard = generate_unified_dashboard()
            narrative = generate_narrative(dashboard)
            if narrative:
                brief["full_text"] = render_brief_text(
                    [BriefSection(**s) for s in brief["sections"]], narrative
                )
                brief["has_narrative"] = True
        except Exception as e:
            logger.warning("Failed to generate unified dashboard narrative: %s", e)

    print(brief["full_text"])

    if args.save:
        out_path = DATA_DIR / "daily_brief.json"
        with open(out_path, "w") as f:
            json.dump(brief, f, indent=2, default=str)
        print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
