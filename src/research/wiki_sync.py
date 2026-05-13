#!/usr/bin/env python3
"""
Portfolio-Lab Alpha: Wiki Sync
Crystallizes research findings to ~/wiki/projects/portfolio-lab/ compound pages.
Follows wiki schema: frontmatter, citations, typed knowledge.
"""

import os
import json
import sqlite3
import hashlib
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

DATA_DIR = Path("~/projects/portfolio-lab/data").expanduser()
WIKI_DIR = Path("~/wiki/projects/portfolio-lab").expanduser()
RAW_DIR = Path("~/projects/portfolio-lab/raw").expanduser() / "market"
DB_PATH = DATA_DIR / "market.db"

class WikiSync:
    def __init__(self):
        RAW_DIR.mkdir(parents=True, exist_ok=True)
        (WIKI_DIR / "compound").mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(DB_PATH)
        self.conn.row_factory = sqlite3.Row
    
    def hash_file(self, content: str) -> str:
        """Generate SHA256 hash for raw provenance."""
        return hashlib.sha256(content.encode()).hexdigest()[:16]
    
    def save_raw_source(self, data: Dict, name: str) -> Path:
        """Save data as raw source file with provenance."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        content = json.dumps(data, indent=2, default=str)
        hash_val = self.hash_file(content)
        
        raw_path = RAW_DIR / f"{name}_{timestamp}_{hash_val}.json"
        
        frontmatter = f"""---
type: raw
source_type: market_data
sha256: {hash_val}
created: {datetime.now().isoformat()}
---

"""
        with open(raw_path, 'w') as f:
            f.write(frontmatter + content)
        
        return raw_path
    
    def sync_regime_analysis(self) -> Optional[Path]:
        """Sync regime log to wiki compound page."""
        cursor = self.conn.cursor()
        
        cursor.execute("""
            SELECT * FROM regime_log 
            WHERE detected_at >= datetime('now', '-7 days')
            ORDER BY detected_at DESC
        """)
        
        regimes = [dict(row) for row in cursor.fetchall()]
        if not regimes:
            return None
        
        # Save raw source
        raw_path = self.save_raw_source(regimes, "regime_log")
        raw_citation = f"raw/market/{raw_path.name}"
        
        # Generate compound page
        timestamp = datetime.now().strftime("%Y-%m-%d")
        page_path = WIKI_DIR / "compound" / f"regime-changes-{timestamp}.md"
        
        # Build table
        rows = "\n".join(
            f"| {r['detected_at'][:10]} | {r['regime']} | {r['vix_level']:.2f if r['vix_level'] else 'N/A'} | {r['trend_strength']:.3f if r['trend_strength'] else 'N/A'} |"
            for r in regimes[:10]
        )
        
        content = f"""---
type: query
tags: [regime, analysis, market-data, portfolio-lab]
provenance: project
provenance_projects: [[portfolio-lab]]
confidence: high
created: {timestamp}
updated: {datetime.now().isoformat()}
---

# Market Regime Changes: Weekly Analysis

**Generated:** {datetime.now().isoformat()}
**Source:** {raw_citation}

## Recent Regime Detections

| Date | Regime | VIX | Trend Strength |
|------|--------|-----|----------------|
{rows}

## Regime Distribution (Last 7 Days)

```
{self._regime_distribution(regimes)}
```

## Implications for Strategy

Based on recent regime patterns:

{self._regime_implications(regimes)}

## Data Quality

- Source: Yahoo Finance v8 API
- Detection method: VIX threshold + trend strength
- Update frequency: Hourly via data pipeline

## Sources

- ^{[raw/market/{raw_path.name}]}
"""
        
        with open(page_path, 'w') as f:
            f.write(content)
        
        return page_path
    
    def sync_performance_summary(self) -> Optional[Path]:
        """Sync paper trading performance to wiki."""
        perf_log = DATA_DIR / "performance.jsonl"
        if not perf_log.exists():
            return None
        
        # Load recent performance
        entries = []
        with open(perf_log) as f:
            for line in f:
                try:
                    entries.append(json.loads(line))
                except (json.JSONDecodeError, OSError):
                    pass
        
        if len(entries) < 10:
            return None
        
        # Calculate metrics
        recent = entries[-63:]  # Last 63 entries
        values = [e.get("total_value", 0) for e in recent if e.get("total_value")]
        returns = [e.get("daily_return", 0) for e in recent if e.get("daily_return") is not None]
        
        if not values or len(values) < 10:
            return None
        
        total_return = (values[-1] - values[0]) / values[0] if values[0] > 0 else 0
        
        # Sharpe ratio calculation with variance check to avoid division by zero
        if returns and len(returns) > 1:
            mean_return = sum(returns) / len(returns)
            variance = sum((r - mean_return) ** 2 for r in returns) / len(returns)
            if variance > 0:
                std_dev = variance ** 0.5
                sharpe = (mean_return / std_dev) * (252 ** 0.5)
            else:
                sharpe = 0  # All returns identical, undefined Sharpe
        else:
            sharpe = 0
        max_dd = 0
        peak = values[0]
        for v in values:
            if v > peak:
                peak = v
            dd = (peak - v) / peak
            if dd > max_dd:
                max_dd = dd
        
        # Save raw
        raw_data = {
            "performance_summary": {
                "total_return": total_return,
                "sharpe": sharpe,
                "max_drawdown": max_dd,
                "days_tracked": len(values),
                "start_value": values[0],
                "current_value": values[-1]
            },
            "raw_entries_count": len(entries)
        }
        raw_path = self.save_raw_source(raw_data, "performance_summary")
        
        timestamp = datetime.now().strftime("%Y-%m-%d")
        page_path = WIKI_DIR / "compound" / f"paper-trading-performance-{timestamp}.md"
        
        # Build citation separately to avoid f-string brace conflicts
        raw_citation = f"raw/market/{raw_path.name}"
        
        content = f"""---
type: query
tags: [performance, paper-trading, portfolio-lab, metrics]
provenance: project
provenance_projects: [[portfolio-lab]]
confidence: high
created: {timestamp}
---

# Paper Trading Performance: Live Summary

**Period:** {len(values)} days tracked
**Strategy:** SPY/GLD/TLT 46/38/16 with regime overrides
**Mode:** Paper (simulated with slippage)

## Performance Metrics

| Metric | Value |
|--------|-------|
| Total Return | {total_return:.2%} |
| Sharpe Ratio | {sharpe:.2f} |
| Max Drawdown | {max_dd:.2%} |
| Start Value | ${values[0]:,.2f} |
| Current Value | ${values[-1]:,.2f} |

## Graduation Status

{self._graduation_status(total_return, sharpe, max_dd, len(values))}

## Daily Returns Distribution

- Positive days: {sum(1 for r in returns if r > 0)}
- Negative days: {sum(1 for r in returns if r < 0)}
- Win rate: {sum(1 for r in returns if r > 0) / len(returns):.1%}

## Notes

- Slippage model: 0.1% per trade
- Rebalance threshold: 10% drift
- Volatility target: 12% annual

## Sources

- ^{[{raw_citation}]}
- [[compound/grid-search-results]] — original backtest validation
- [[compound/decision-framework]] — allocation rationale
"""
        
        with open(page_path, 'w') as f:
            f.write(content)
        
        return page_path
    
    def sync_order_history(self) -> Optional[Path]:
        """Sync recent orders to wiki."""
        orders_log = DATA_DIR / "orders.jsonl"
        if not orders_log.exists():
            return None
        
        orders = []
        with open(orders_log) as f:
            for line in f:
                try:
                    orders.append(json.loads(line))
                except (json.JSONDecodeError, OSError):
                    pass
        
        if not orders:
            return None
        
        # Recent orders only
        recent = orders[-20:]  # Last 20 orders
        
        # Save raw
        raw_path = self.save_raw_source(recent, "order_history")
        
        timestamp = datetime.now().strftime("%Y-%m-%d")
        page_path = WIKI_DIR / "compound" / f"order-history-{timestamp}.md"
        
        # Build table
        rows = "\n".join(
            f"| {o.get('timestamp', 'N/A')[:10] if o.get('timestamp') else 'N/A'} | {o.get('symbol')} | {o.get('side')} | {o.get('fill_shares', 0):.2f} | ${o.get('fill_value', 0):,.2f} | {o.get('reason', 'rebalance')} |"
            for o in recent
        )
        
        content = f"""---
type: query
tags: [orders, execution, portfolio-lab]
provenance: project
provenance_projects: [[portfolio-lab]]
confidence: high
created: {timestamp}
---

# Order History: Recent Executions

**Total Orders:** {len(orders)}
**Recent Shown:** 20

## Recent Orders

| Date | Symbol | Side | Shares | Value | Reason |
|------|--------|------|--------|-------|--------|
{rows}

## Order Statistics

- Total buy orders: {sum(1 for o in orders if o.get('side') == 'buy')}
- Total sell orders: {sum(1 for o in orders if o.get('side') == 'sell')}
- Total volume: ${sum(o.get('fill_value', 0) for o in orders):,.2f}

## Sources

- ^[raw/market/{raw_path.name}]
"""
        
        with open(page_path, 'w') as f:
            f.write(content)
        
        return page_path
    
    def update_knowledge_md(self):
        """Update knowledge.md to link new pages."""
        knowledge_path = WIKI_DIR / "knowledge.md"
        
        # Find all compound pages
        compound_pages = sorted((WIKI_DIR / "compound").glob("*.md"))
        
        content = f"""---
slug: portfolio-lab
updated: {datetime.now().isoformat()}
---

# Portfolio-Lab: Auto-Generated Knowledge Index

This file bridges Layer 2 (global knowledge) and Layer 3 (project workspace).
Generated by wiki-sync agent.

## Compound Pages (Auto-Updated)

"""
        
        for page in compound_pages:
            name = page.stem
            content += f"- [[compound/{name}]]\n"
        
        content += f"""
## Raw Sources

Market data snapshots saved to `raw/market/` with SHA256 provenance.

## External Links

- Code: `~/projects/portfolio-lab/`
- Data: `~/projects/portfolio-lab/data/`
"""
        
        with open(knowledge_path, 'w') as f:
            f.write(content)
        
        return knowledge_path
    
    def _regime_distribution(self, regimes: List[Dict]) -> str:
        """Generate text distribution of regimes."""
        counts = {}
        for r in regimes:
            reg = r.get('regime', 'unknown')
            counts[reg] = counts.get(reg, 0) + 1
        
        total = sum(counts.values())
        lines = []
        for reg, count in sorted(counts.items(), key=lambda x: -x[1]):
            pct = count / total * 100
            bar = "█" * int(pct / 5)
            lines.append(f"{reg:12} {bar} {pct:.0f}%")
        
        return "\n".join(lines)
    
    def _regime_implications(self, regimes: List[Dict]) -> str:
        """Generate implications text."""
        latest = regimes[0] if regimes else {}
        regime = latest.get('regime', 'unknown')
        
        implications = {
            'crisis': """
- **Action:** Risk-off allocation (SPY 20%, GLD 50%, TLT 30%)
- **Rationale:** High volatility regime, protect capital
- **Next Check:** Monitor VIX for normalization (<25)
""",
            'vol_spike': """
- **Action:** Defensive shift (SPY 30%, GLD 45%, TLT 25%)
- **Rationale:** Elevated volatility, reduce equity exposure
- **Next Check:** Watch for trend stabilization
""",
            'low_vol': """
- **Action:** Risk-on allocation (SPY 55%, GLD 30%, TLT 15%)
- **Rationale:** Calm markets, increase equity for growth
- **Next Check:** Monitor VIX floor breach (>15)
""",
            'normal': """
- **Action:** Base allocation (SPY 46%, GLD 38%, TLT 16%)
- **Rationale:** Stable regime, standard risk parity
- **Next Check:** Standard hourly monitoring
"""
        }
        
        return implications.get(regime, implications['normal'])
    
    def _graduation_status(self, total_return: float, sharpe: float, max_dd: float, days: int) -> str:
        """Generate graduation status text."""
        MIN_DAYS = 63
        MIN_SHARPE = 0.5
        MAX_DD = 0.15
        
        if days < MIN_DAYS:
            return f"⏳ **Not Ready** — Need {MIN_DAYS - days} more days of history"
        
        checks = []
        if sharpe >= MIN_SHARPE:
            checks.append(f"✓ Sharpe {sharpe:.2f} >= {MIN_SHARPE}")
        else:
            checks.append(f"✗ Sharpe {sharpe:.2f} < {MIN_SHARPE}")
        
        if max_dd <= MAX_DD:
            checks.append(f"✓ Max DD {max_dd:.1%} <= {MAX_DD:.0%}")
        else:
            checks.append(f"✗ Max DD {max_dd:.1%} > {MAX_DD:.0%}")
        
        if sharpe >= MIN_SHARPE and max_dd <= MAX_DD:
            return f"🎓 **GRADUATION CANDIDATE**\n\n" + "\n".join(f"- {c}" for c in checks) + "\n\nReady for live promotion approval."
        else:
            return f"📊 **Tracking** — Not yet meeting graduation criteria\n\n" + "\n".join(f"- {c}" for c in checks)
    
    def run(self):
        """Run full wiki sync."""
        print(f"[{datetime.now()}] Wiki Sync Starting")
        
        pages = []
        
        if result := self.sync_regime_analysis():
            pages.append(f"Regime: {result.name}")
        
        if result := self.sync_performance_summary():
            pages.append(f"Performance: {result.name}")
        
        if result := self.sync_order_history():
            pages.append(f"Orders: {result.name}")
        
        if pages:
            knowledge = self.update_knowledge_md()
            print(f"Updated {knowledge.name}")
        
        for p in pages:
            print(f"  Synced: {p}")
        
        self.conn.close()
        print(f"[{datetime.now()}] Wiki Sync Complete ({len(pages)} pages)")

if __name__ == "__main__":
    sync = WikiSync()
    sync.run()
