#!/usr/bin/env python3
"""
Portfolio-Lab Alpha: Wiki Sync
Crystallizes research findings to ~/wiki/projects/portfolio-lab/ compound pages.
Follows wiki schema: frontmatter, citations, typed knowledge.
"""

import json
import sqlite3
import hashlib
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from src.paths import DATA_DIR as _DATA_DIR, WIKI_DIR as _WIKI_DIR, sqlite_connect

logger = logging.getLogger(__name__)

DATA_DIR = _DATA_DIR
WIKI_DIR = _WIKI_DIR / "projects" / "portfolio-lab"
RAW_DIR = _DATA_DIR.parent / "raw" / "market"
DB_PATH = DATA_DIR / "market.db"

class WikiSync:
    def __init__(self):
        RAW_DIR.mkdir(parents=True, exist_ok=True)
        (WIKI_DIR / "compound").mkdir(parents=True, exist_ok=True)
        self.conn = sqlite_connect(DB_PATH)
        self.conn.row_factory = sqlite3.Row
    
    def hash_file(self, content: str) -> str:
        """Generate SHA256 hash for raw provenance."""
        return hashlib.sha256(content.encode()).hexdigest()[:16]
    
    def save_raw_source(self, data: Dict, name: str) -> Path:
        """Save data as raw source file with provenance.

        Uses a stable filename (name.json) so repeated runs with
        unchanged data don't create duplicate files. Only writes
        when the content hash differs from the existing file.
        """
        content = json.dumps(data, indent=2, default=str)
        hash_val = self.hash_file(content)

        raw_path = RAW_DIR / f"{name}.json"

        # Skip write if file exists with same content hash
        if raw_path.exists():
            try:
                existing = raw_path.read_text()
                if f"sha256: {hash_val}" in existing:
                    return raw_path  # Unchanged — skip write
            except Exception as e:
                logger.warning("Failed to read existing raw file %s: %s", raw_path, e)

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
        def _fmt_regime_row(r):
            vix = f"{r['vix_level']:.2f}" if r.get('vix_level') else 'N/A'
            ts = f"{r['trend_strength']:.3f}" if r.get('trend_strength') else 'N/A'
            return f"| {r['detected_at'][:10]} | {r['regime']} | {vix} | {ts} |"
        rows = "\n".join(_fmt_regime_row(r) for r in regimes[:10])
        
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

- {{[raw/market/{raw_path.name}]}}
"""
        
        with open(page_path, 'w') as f:
            f.write(content)
        
        return page_path
    
    def sync_performance_summary(self) -> Optional[Path]:
        """Sync paper trading performance to app-level data directory.

        Paper trading P&L is personal app state, not research knowledge.
        Written to DATA_DIR (not wiki vault) to avoid polluting shared
        knowledge base with user-specific runtime data.
        """
        perf_log = DATA_DIR / "performance.jsonl"
        if not perf_log.exists():
            return None

        # Load recent performance
        entries = []
        with open(perf_log) as f:
            for line in f:
                try:
                    entries.append(json.loads(line))
                except (json.JSONDecodeError, OSError) as e:
                    logger.debug("Skipping malformed line: %s", e)

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

        timestamp = datetime.now().strftime("%Y-%m-%d")
        # Write to app-level DATA_DIR, not wiki vault
        page_path = DATA_DIR / f"paper-trading-performance-{timestamp}.json"

        summary = {
            "date": timestamp,
            "performance": {
                "total_return": total_return,
                "sharpe": sharpe,
                "max_drawdown": max_dd,
                "days_tracked": len(values),
                "start_value": values[0],
                "current_value": values[-1],
            },
            "daily_returns_distribution": {
                "positive_days": sum(1 for r in returns if r > 0),
                "negative_days": sum(1 for r in returns if r < 0),
                "win_rate": sum(1 for r in returns if r > 0) / len(returns) if returns else 0,
            },
            "graduation": self._graduation_status_dict(total_return, sharpe, max_dd, len(values)),
            "raw_entries_count": len(entries),
        }

        with open(page_path, 'w') as f:
            json.dump(summary, f, indent=2, default=str)

        return page_path
    
    def sync_order_history(self) -> Optional[Path]:
        """Sync recent orders to app-level data directory.

        Order fills are personal app state, not research knowledge.
        Written to DATA_DIR (not wiki vault) to avoid polluting shared
        knowledge base with user-specific runtime data.
        """
        orders_log = DATA_DIR / "orders.jsonl"
        if not orders_log.exists():
            return None

        orders = []
        with open(orders_log) as f:
            for line in f:
                try:
                    orders.append(json.loads(line))
                except (json.JSONDecodeError, OSError) as e:
                    logger.debug("Skipping malformed order line: %s", e)

        if not orders:
            return None

        # Recent orders only
        recent = orders[-20:]  # Last 20 orders

        timestamp = datetime.now().strftime("%Y-%m-%d")
        # Write to app-level DATA_DIR, not wiki vault
        page_path = DATA_DIR / f"order-history-{timestamp}.json"

        summary = {
            "date": timestamp,
            "total_orders": len(orders),
            "recent_shown": len(recent),
            "recent_orders": recent,
            "statistics": {
                "total_buy_orders": sum(1 for o in orders if o.get('side') == 'buy'),
                "total_sell_orders": sum(1 for o in orders if o.get('side') == 'sell'),
                "total_volume": sum(o.get('fill_value', 0) for o in orders),
            },
        }

        with open(page_path, 'w') as f:
            json.dump(summary, f, indent=2, default=str)

        return page_path
    
    def update_knowledge_md(self):
        """Update knowledge.md to link new pages."""
        knowledge_path = WIKI_DIR / "knowledge.md"
        
        # Find all compound pages
        compound_pages = sorted((WIKI_DIR / "compound").glob("*.md"))
        
        compound_links = "\n".join(
            f"- [[compound/{page.stem}]]" for page in compound_pages
        )

        content = f"""---
slug: portfolio-lab
updated: {datetime.now().isoformat()}
---

# Portfolio-Lab: Auto-Generated Knowledge Index

This file bridges Layer 2 (global knowledge) and Layer 3 (project workspace).
Generated by wiki-sync agent.

## Compound Pages (Auto-Updated)

{compound_links}
"""
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
        """Generate graduation status text (markdown, for regime analysis page)."""
        MIN_DAYS = 63
        MIN_SHARPE = 0.5
        MAX_DD = 0.15

        if days < MIN_DAYS:
            return f"Not Ready — Need {MIN_DAYS - days} more days of history"

        checks = []
        if sharpe >= MIN_SHARPE:
            checks.append(f"Sharpe {sharpe:.2f} >= {MIN_SHARPE}")
        else:
            checks.append(f"Sharpe {sharpe:.2f} < {MIN_SHARPE}")

        if max_dd <= MAX_DD:
            checks.append(f"Max DD {max_dd:.1%} <= {MAX_DD:.0%}")
        else:
            checks.append(f"Max DD {max_dd:.1%} > {MAX_DD:.0%}")

        if sharpe >= MIN_SHARPE and max_dd <= MAX_DD:
            return f"GRADUATION CANDIDATE: " + "; ".join(checks)
        else:
            return f"Tracking — Not yet meeting graduation criteria: " + "; ".join(checks)

    def _graduation_status_dict(self, total_return: float, sharpe: float, max_dd: float, days: int) -> dict:
        """Generate graduation status as dict (for JSON output)."""
        MIN_DAYS = 63
        MIN_SHARPE = 0.5
        MAX_DD = 0.15

        ready = days >= MIN_DAYS and sharpe >= MIN_SHARPE and max_dd <= MAX_DD
        return {
            "status": "candidate" if ready else "tracking",
            "days_tracked": days,
            "min_days_required": MIN_DAYS,
            "sharpe_met": sharpe >= MIN_SHARPE,
            "max_dd_met": max_dd <= MAX_DD,
            "sharpe": sharpe,
            "max_drawdown": max_dd,
        }
    
    def run(self):
        """Run full wiki sync."""
        print(f"[{datetime.now()}] Wiki Sync Starting")

        wiki_pages = []   # Pages written to wiki vault
        app_data = []     # Data written to app-level DATA_DIR

        if result := self.sync_regime_analysis():
            wiki_pages.append(f"Regime: {result.name}")

        if result := self.sync_performance_summary():
            app_data.append(f"Performance: {result.name}")

        if result := self.sync_order_history():
            app_data.append(f"Orders: {result.name}")

        # Only update knowledge.md when wiki vault pages change
        if wiki_pages:
            knowledge = self.update_knowledge_md()
            print(f"Updated {knowledge.name}")

        for p in wiki_pages:
            print(f"  Synced (wiki): {p}")
        for p in app_data:
            print(f"  Synced (app): {p}")

        self.conn.close()
        print(f"[{datetime.now()}] Wiki Sync Complete ({len(wiki_pages)} wiki, {len(app_data)} app)")

if __name__ == "__main__":
    sync = WikiSync()
    sync.run()
