#!/usr/bin/env python3
"""
Portfolio-Lab Alpha: Wiki Sync
Crystallizes research findings to the active SkillWiki portfolio-lab project.
Follows wiki schema: frontmatter, citations, typed knowledge.
"""

import json
import sqlite3
import hashlib
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from src.paths import (
    DATA_DIR as _DATA_DIR,
    PROJECT_WIKI_DIR as _PROJECT_WIKI_DIR,
    require_project_wiki_dir as _require_project_wiki_dir,
    sqlite_connect,
    MARKET_DB,
)
from src.backtest.metrics import save_results_json

logger = logging.getLogger(__name__)

DATA_DIR = _DATA_DIR
WIKI_DIR = _PROJECT_WIKI_DIR
RAW_DIR = _DATA_DIR.parent / "raw" / "market"
DB_PATH = MARKET_DB


def _ensure_default_wiki_dir_has_vault() -> None:
    """Default SkillWiki-backed wiki dir must resolve before write operations."""
    if WIKI_DIR == _PROJECT_WIKI_DIR:
        _require_project_wiki_dir()


class WikiSync:
    def __init__(self):
        RAW_DIR.mkdir(parents=True, exist_ok=True)
        _ensure_default_wiki_dir_has_vault()
        (WIKI_DIR / "compound").mkdir(parents=True, exist_ok=True)
        self._conn = None

    @property
    def conn(self):
        """Lazy connection with row factory."""
        if self._conn is None:
            self._conn = sqlite_connect(DB_PATH)
            self._conn.row_factory = sqlite3.Row
        return self._conn

    @conn.setter
    def conn(self, value):
        """Allow tests to inject a connection."""
        self._conn = value

    def close(self):
        """Close the database connection."""
        if self._conn is not None:
            try:
                self._conn.close()
            except (OSError, sqlite3.Error):
                pass
            self._conn = None
    
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
            except (OSError, ValueError, UnicodeDecodeError) as e:
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
    
    @staticmethod
    def _filter_phantom_cash_days(daily_entries: List[dict]) -> List[dict]:
        """Drop trailing cash-only days after a history that already held positions.

        Phantom rows (positions_count==0, initial capital) from test isolation
        leaks must not become last-of-day equity for graduation summaries.
        """
        if not daily_entries:
            return daily_entries

        def positions_count(entry: dict) -> int | None:
            if "positions_count" in entry and entry.get("positions_count") is not None:
                try:
                    return int(entry.get("positions_count"))
                except (TypeError, ValueError):
                    return None
            positions = entry.get("positions")
            if isinstance(positions, dict):
                return len(positions)
            if isinstance(positions, list):
                return len(positions)
            return None

        ever_held = False
        for entry in daily_entries:
            n = positions_count(entry)
            if n is not None and n > 0:
                ever_held = True
                break
        if not ever_held:
            return daily_entries

        trimmed = list(daily_entries)
        while trimmed:
            n = positions_count(trimmed[-1])
            # Only strip clear empty-portfolio tails (missing count stays).
            if n == 0:
                trimmed.pop()
                continue
            break
        return trimmed or daily_entries

    @staticmethod
    def _portfolio_paper_mark(data_dir: Path | None = None) -> Optional[dict]:
        """Load portfolio_paper mark when positions exist."""
        root = Path(data_dir) if data_dir is not None else Path(DATA_DIR)
        path = root / "portfolio_paper.json"
        if not path.exists():
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError, UnicodeDecodeError):
            return None
        if not isinstance(payload, dict):
            return None
        positions = payload.get("positions") or {}
        if not isinstance(positions, dict) or not positions:
            return None
        cash = float(payload.get("cash") or 0.0)
        position_value = 0.0
        for pos in positions.values():
            if not isinstance(pos, dict):
                continue
            if pos.get("value") is not None:
                try:
                    position_value += float(pos.get("value") or 0.0)
                    continue
                except (TypeError, ValueError):
                    pass
            try:
                shares = float(pos.get("shares") or 0.0)
                price = float(pos.get("current_price") or pos.get("avg_price") or 0.0)
            except (TypeError, ValueError):
                continue
            position_value += shares * price
        total = cash + position_value
        return {
            "total_value": round(total, 2),
            "positions_count": len(positions),
            "cash": cash,
            "source": "portfolio_paper",
        }

    @staticmethod
    def _kill_blocks_graduation(data_dir: Path | None = None) -> tuple[bool, Optional[dict]]:
        """Return (blocked, payload) when kill authority blocks candidacy."""
        root = Path(data_dir) if data_dir is not None else Path(DATA_DIR)
        try:
            from src.dashboard.kill_authority import (
                is_kill_execution_blocked,
                load_kill_switch_payload,
            )

            payload = load_kill_switch_payload(root)
            return is_kill_execution_blocked(payload), payload
        except ImportError:
            kill_file = root / "kill_switch.json"
            if not kill_file.exists():
                return False, None
            try:
                payload = json.loads(kill_file.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError, UnicodeDecodeError):
                return True, None
            if isinstance(payload, dict) and payload.get("enabled"):
                return True, payload
            return False, None

    def sync_performance_summary(self) -> Optional[Path]:
        """Sync paper trading performance to app-level data directory.

        Paper trading P&L is personal app state, not research knowledge.
        Written to DATA_DIR (not wiki vault) to avoid polluting shared
        knowledge base with user-specific runtime data.

        Prefer daily_pnl.jsonl session SSOT (c358) when available so
        current_value cannot lag the capture path. Fall back to the legacy
        performance.jsonl path when daily_pnl is thin.
        """
        # c358: prefer write-SSOT series when capture has enough session history.
        try:
            from src.monitor.paper_return_ssot import (
                load_daily_pnl_sessions,
                write_paper_trading_performance_from_ssot,
            )

            sessions = load_daily_pnl_sessions(DATA_DIR)
            if len(sessions) >= 5:
                paper_mark = self._portfolio_paper_mark(DATA_DIR)
                cv = paper_mark["total_value"] if paper_mark is not None else None
                path = write_paper_trading_performance_from_ssot(
                    DATA_DIR, current_value=cv
                )
                if path is not None:
                    # Enrich with graduation block for wiki_sync consumers.
                    try:
                        summary = json.loads(path.read_text(encoding="utf-8"))
                        perf = summary.get("performance") or {}
                        summary["graduation"] = self._graduation_status_dict(
                            float(perf.get("total_return") or 0),
                            float(perf.get("sharpe") or 0),
                            float(perf.get("max_drawdown") or 0),
                            int(perf.get("days_tracked") or 0),
                            data_dir=DATA_DIR,
                        )
                        summary["schema_version"] = "paper-trading-performance/v3-ssot"
                        generator_git_sha = None
                        try:
                            from src.monitor.decision_registry import _git_sha_short

                            generator_git_sha = _git_sha_short()
                        except Exception:
                            generator_git_sha = None
                        summary["generator_git_sha"] = generator_git_sha
                        save_results_json(summary, output_path=str(path))
                    except (OSError, json.JSONDecodeError, TypeError, ValueError) as exc:
                        logger.debug("SSOT snapshot enrich failed: %s", exc)
                    return path
        except Exception as exc:  # noqa: BLE001
            logger.debug("daily_pnl SSOT snapshot path skipped: %s", exc)

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

        # Deduplicate to daily: keep last entry per calendar date.
        # performance.jsonl contains intraday entries; raw count overstates
        # trading days and inflates Sharpe (many zero-return intraday rows).
        daily_map: dict[str, dict] = {}
        for idx, entry in enumerate(entries):
            ts = entry.get("timestamp", "")
            date_key = ts[:10] if len(ts) >= 10 else ""
            if not date_key:
                # Fallback: entries without timestamps are treated as
                # separate days (preserves legacy test behavior)
                date_key = f"__no_ts_{idx}__"
            daily_map[date_key] = entry
        daily_entries = [daily_map[d] for d in sorted(daily_map)]
        filtered_entries = self._filter_phantom_cash_days(daily_entries)
        phantom_days_dropped = max(0, len(daily_entries) - len(filtered_entries))

        # Calculate metrics from deduplicated daily entries
        values = [e.get("total_value", 0) for e in filtered_entries if e.get("total_value")]
        returns = [e.get("daily_return", 0) for e in filtered_entries if e.get("daily_return") is not None]

        if not values or len(values) < 5:
            return None

        paper_mark = self._portfolio_paper_mark(DATA_DIR)
        current_value = values[-1]
        current_value_source = "performance_jsonl"
        if paper_mark is not None:
            current_value = paper_mark["total_value"]
            current_value_source = "portfolio_paper"

        total_return = (current_value - values[0]) / values[0] if values[0] > 0 else 0

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
        series_for_dd = list(values)
        if paper_mark is not None:
            series_for_dd = list(values[:-1]) + [current_value] if values else [current_value]
        for v in series_for_dd:
            if v > peak:
                peak = v
            dd = (peak - v) / peak if peak else 0
            if dd > max_dd:
                max_dd = dd

        timestamp = datetime.now().strftime("%Y-%m-%d")
        # Write to app-level DATA_DIR, not wiki vault
        page_path = DATA_DIR / f"paper-trading-performance-{timestamp}.json"

        generator_git_sha = None
        try:
            from src.monitor.decision_registry import _git_sha_short

            generator_git_sha = _git_sha_short()
        except Exception:
            generator_git_sha = None

        summary = {
            "date": timestamp,
            "performance": {
                "total_return": total_return,
                "sharpe": sharpe,
                "max_drawdown": max_dd,
                "days_tracked": len(values),
                "start_value": values[0],
                "current_value": current_value,
                "current_value_source": current_value_source,
            },
            "daily_returns_distribution": {
                "positive_days": sum(1 for r in returns if r > 0),
                "negative_days": sum(1 for r in returns if r < 0),
                "win_rate": sum(1 for r in returns if r > 0) / len(returns) if returns else 0,
            },
            "graduation": self._graduation_status_dict(
                total_return, sharpe, max_dd, len(values), data_dir=DATA_DIR
            ),
            "raw_entries_count": len(entries),
            "phantom_cash_days_dropped": phantom_days_dropped,
            "generator_git_sha": generator_git_sha,
            "schema_version": "paper-trading-performance/v2",
            "return_source": "performance.jsonl_fallback",
        }

        save_results_json(summary, output_path=str(page_path))

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

        # Batch DS: file date is write day; also stamp last real fill event so
        # rebalance_health does not treat daily log snapshots as new executions.
        write_day = datetime.now().strftime("%Y-%m-%d")
        last_event = None
        for o in orders:
            ts = o.get("timestamp")
            if not ts:
                continue
            try:
                t = datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
            except (TypeError, ValueError):
                continue
            if last_event is None or t > last_event:
                last_event = t
        last_event_at = last_event.isoformat() if last_event else None
        last_event_day = (
            last_event.strftime("%Y-%m-%d") if last_event else write_day
        )

        page_path = DATA_DIR / f"order-history-{write_day}.json"

        summary = {
            "date": write_day,
            "write_day": write_day,
            "last_order_event_at": last_event_at,
            "last_order_event_day": last_event_day,
            "snapshot_kind": "orders_jsonl_tail",
            "total_orders": len(orders),
            "recent_shown": len(recent),
            "recent_orders": recent,
            "statistics": {
                "total_buy_orders": sum(1 for o in orders if o.get('side') == 'buy'),
                "total_sell_orders": sum(1 for o in orders if o.get('side') == 'sell'),
                "total_volume": sum(o.get('fill_value', 0) for o in orders),
            },
            "provenance_note": (
                "Daily file is a rolling snapshot of orders.jsonl; schedule "
                "freshness must use last_order_event_at / order timestamps, "
                "not write_day alone (Batch DS)."
            ),
        }

        save_results_json(summary, output_path=str(page_path))

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
        """Generate graduation status text (markdown, for regime analysis page).

        Narrative uses advisory performance metrics for readability. When the
        GraduationChecklist SSOT disagrees with an advisory candidacy claim,
        append a conflict note so operators do not treat wiki prose as promote
        authority.
        """
        MIN_DAYS = 63
        MIN_SHARPE = 0.5
        MAX_DD = 0.15

        if days < MIN_DAYS:
            base = f"Not Ready — Need {MIN_DAYS - days} more days of history"
        else:
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
                base = f"GRADUATION CANDIDATE: " + "; ".join(checks)
            else:
                base = f"Tracking — Not yet meeting graduation criteria: " + "; ".join(checks)

        # Annotate when checklist SSOT would block while advisory text claims candidate
        try:
            from src.strategy.graduation_checklist import GraduationChecklist

            checklist = GraduationChecklist()
            results = checklist.check()
            if (
                "GRADUATION CANDIDATE" in base
                and not checklist.is_graduation_ready(results)
            ):
                return (
                    f"{base} — BLOCKED by checklist SSOT "
                    f"(readiness={checklist.readiness_score(results)}%; "
                    f"graduation_conflict=true; promote marker not authoritative)"
                )
        except (ImportError, OSError, TypeError, ValueError):
            pass
        return base

    def _graduation_status_dict(
        self,
        total_return: float,
        sharpe: float,
        max_dd: float,
        days: int,
        *,
        data_dir: Path | None = None,
    ) -> dict:
        """Generate graduation status as dict (for JSON output).

        ``sharpe_met`` / candidacy follow GraduationChecklist when available so
        performance files cannot claim candidate while checklist fails.
        Kill halt always forces tracking (never candidate).
        """
        MIN_DAYS = 63
        MIN_SHARPE = 0.5
        MAX_DD = 0.15
        advisory_sharpe_met = sharpe >= MIN_SHARPE
        advisory_max_dd_met = max_dd <= MAX_DD
        advisory_ready = days >= MIN_DAYS and advisory_sharpe_met and advisory_max_dd_met

        kill_blocked, kill_payload = self._kill_blocks_graduation(data_dir)
        kill_level = None
        kill_reason = None
        if isinstance(kill_payload, dict):
            kill_level = kill_payload.get("level")
            kill_reason = kill_payload.get("reason")

        checklist_ready = None
        readiness_score = None
        graduation_conflict = False
        try:
            from src.strategy.graduation_checklist import GraduationChecklist

            checklist = GraduationChecklist()
            results = checklist.check()
            checklist_ready = checklist.is_graduation_ready(results)
            readiness_score = checklist.readiness_score(results)
            graduation_conflict = bool(advisory_ready and not checklist_ready)
        except (ImportError, OSError, TypeError, ValueError):
            checklist_ready = None

        # Authoritative candidacy = checklist when available; never claim
        # candidate from advisory metrics alone; never under kill halt.
        if kill_blocked:
            status = "tracking"
            sharpe_met = False
            max_dd_met = False
            graduation_conflict = bool(graduation_conflict or advisory_ready or checklist_ready)
        elif checklist_ready is True:
            status = "candidate"
            sharpe_met = True
            max_dd_met = True
        elif checklist_ready is False:
            status = "tracking"
            sharpe_met = False
            max_dd_met = False
        else:
            # Checklist unavailable: fail closed — tracking only
            status = "tracking"
            sharpe_met = False
            max_dd_met = False

        return {
            "status": status,
            "days_tracked": days,
            "min_days_required": MIN_DAYS,
            "sharpe_met": sharpe_met,
            "max_dd_met": max_dd_met,
            "sharpe": sharpe,
            "max_drawdown": max_dd,
            "source": "graduation_checklist" if checklist_ready is not None else "advisory_metrics_fail_closed",
            "checklist_ready": checklist_ready,
            "readiness_score": readiness_score,
            "graduation_conflict": graduation_conflict,
            "advisory_sharpe_met": advisory_sharpe_met,
            "advisory_max_dd_met": advisory_max_dd_met,
            "advisory_ready": advisory_ready,
            "kill_blocked": kill_blocked,
            "kill_level": kill_level,
            "kill_reason": kill_reason,
        }
    
    def run(self):
        """Run full wiki sync."""
        logger.info("Wiki Sync Starting")

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
            logger.info("Updated %s", knowledge.name)

        for p in wiki_pages:
            logger.info("  Synced (wiki): %s", p)
        for p in app_data:
            logger.info("  Synced (app): %s", p)

        self.close()
        logger.info("Wiki Sync Complete (%d wiki, %d app)", len(wiki_pages), len(app_data))

if __name__ == "__main__":
    # Cron/Makefile tee captures stdout; enable StreamHandler so tasker logs
    # and data/wiki_sync.log receive structured INFO lines on each run.
    from src.utils.log_config import configure_logging

    configure_logging()
    sync = WikiSync()
    sync.run()
