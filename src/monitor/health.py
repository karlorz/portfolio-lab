#!/usr/bin/env python3
"""
Portfolio-Lab Alpha: Health Monitor
Monitors cron job results and reports status. Can trigger alerts or Claude Code review.
"""

import json
import sqlite3
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.market_calendar import MarketCalendar, get_stale_status

DATA_DIR = Path("~/projects/portfolio-lab/data").expanduser()
LOG_DIR = DATA_DIR
REPORT_PATH = DATA_DIR / ".health_report.json"


class HealthMonitor:
    def __init__(self):
        self.checks = []
        self.status = "healthy"
        self.alerts = []
    
    def check_data_freshness(self) -> bool:
        """Check if market data is fresh, accounting for market calendar."""
        db_path = DATA_DIR / "market.db"
        if not db_path.exists():
            self.checks.append({"name": "database", "status": "missing", "ok": False})
            return False
        
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT symbol, MAX(date) as last_date, COUNT(*) as count
            FROM prices GROUP BY symbol
        """)
        
        stale_symbols = []
        calendar = MarketCalendar()
        
        for row in cursor.fetchall():
            symbol, last_date, count = row
            if last_date:
                last = datetime.strptime(last_date, "%Y-%m-%d")
                
                # Use market calendar for stale detection
                if not calendar.is_trading_day():
                    # Market closed today - no expectation of new data
                    continue
                
                trading_days_stale = calendar.trading_days_since(last)
                
                # Alert if more than 1 trading day stale
                if trading_days_stale > 1:
                    stale_symbols.append(f"{symbol} ({trading_days_stale} trading days)")
        
        conn.close()
        
        ok = len(stale_symbols) == 0
        self.checks.append({
            "name": "data_freshness",
            "status": "ok" if ok else f"stale: {', '.join(stale_symbols)}",
            "ok": ok
        })
        
        if not ok:
            self.alerts.append(f"Stale data detected: {stale_symbols}")
        
        return ok
    
    def check_cron_execution(self) -> bool:
        """Check if cron jobs are running."""
        log_files = {
            "data_pipeline": LOG_DIR / "cron.log",
            "strategy_eval": LOG_DIR / "eval.log",
            "research": LOG_DIR / "research.log",
            "dashboard": LOG_DIR / "dashboard.log",
            "wiki_sync": LOG_DIR / "wiki_sync.log"
        }
        
        stale_jobs = []
        for job, log_file in log_files.items():
            if log_file.exists():
                # Check modification time
                mtime = datetime.fromtimestamp(log_file.stat().st_mtime)
                hours_since = (datetime.now() - mtime).total_seconds() / 3600
                
                # Different thresholds per job frequency
                thresholds = {
                    "data_pipeline": 2,      # Hourly
                    "strategy_eval": 1,      # Every 30 min
                    "research": 6,           # Every 4 hours
                    "dashboard": 0.5,      # Every 10 min
                    "wiki_sync": 8          # Every 6 hours
                }
                
                if hours_since > thresholds.get(job, 24):
                    stale_jobs.append(f"{job} ({hours_since:.1f}h)")
            else:
                stale_jobs.append(f"{job} (no log)")
        
        ok = len(stale_jobs) == 0
        self.checks.append({
            "name": "cron_execution",
            "status": "ok" if ok else f"stale: {', '.join(stale_jobs)}",
            "ok": ok
        })
        
        if not ok:
            self.alerts.append(f"Cron jobs need attention: {stale_jobs}")
        
        return ok
    
    def check_portfolio_health(self) -> bool:
        """Check paper portfolio metrics."""
        portfolio_file = DATA_DIR / "portfolio_paper.json"
        if not portfolio_file.exists():
            self.checks.append({"name": "portfolio", "status": "not_initialized", "ok": True})
            return True  # Not an error, just not started
        
        with open(portfolio_file) as f:
            portfolio = json.load(f)
        
        # Check for issues
        issues = []
        
        # Large drawdown
        if len(portfolio.get("history", [])) > 20:
            values = [h.get("total_value", 0) for h in portfolio["history"]]
            peak = max(values[-252:]) if len(values) >= 252 else max(values)
            current = values[-1] if values else 0
            if peak > 0:
                dd = (peak - current) / peak
                if dd > 0.15:
                    issues.append(f"drawdown {dd:.1%}")
        
        # Cash too high (not invested)
        cash = portfolio.get("cash", 0)
        positions_value = sum(
            p.get("value", 0) for p in portfolio.get("positions", {}).values()
        )
        total = cash + positions_value
        if total > 0 and cash / total > 0.5:
            issues.append(f"high cash {cash/total:.1%}")
        
        ok = len(issues) == 0
        self.checks.append({
            "name": "portfolio",
            "status": "ok" if ok else f"issues: {', '.join(issues)}",
            "ok": ok,
            "value": f"${total:,.2f}" if total > 0 else "N/A"
        })
        
        if not ok:
            self.alerts.append(f"Portfolio issues: {issues}")
        
        return ok
    
    def check_graduation_candidate(self) -> bool:
        """Check for promotion to live."""
        trigger_file = DATA_DIR / ".promote_to_live"
        if not trigger_file.exists():
            self.checks.append({"name": "graduation", "status": "no_candidate", "ok": True})
            return True
        
        with open(trigger_file) as f:
            trigger = json.load(f)
        
        metrics = trigger.get("metrics", {})
        
        self.checks.append({
            "name": "graduation",
            "status": "candidate_ready",
            "ok": True,
            "metrics": {
                "sharpe": metrics.get("sharpe"),
                "max_dd": metrics.get("max_drawdown"),
                "win_rate": metrics.get("win_rate")
            }
        })
        
        self.alerts.append("PROMOTION CANDIDATE: Ready for live trading approval")
        
        return True
    
    def check_kill_switches(self) -> bool:
        """Check if any kill switches are active."""
        active_switches = []
        
        for mode in ["paper", "live"]:
            switch_file = DATA_DIR / f".kill_switch_{mode}"
            if switch_file.exists():
                with open(switch_file) as f:
                    data = json.load(f)
                active_switches.append(f"{mode}: {data.get('reason', 'unknown')}")
        
        ok = len(active_switches) == 0
        self.checks.append({
            "name": "kill_switches",
            "status": "ok" if ok else f"ACTIVE: {', '.join(active_switches)}",
            "ok": ok
        })
        
        if not ok:
            self.status = "critical"
            self.alerts.append(f"KILL SWITCHES ACTIVE: {active_switches}")
        
        return ok
    
    def check_circuit_breaker(self) -> bool:
        """Check drawdown circuit breaker status."""
        try:
            sys.path.insert(0, str(Path(__file__).parent.parent / "strategy"))
            from circuit_breaker import DrawdownCircuitBreaker
            
            cb = DrawdownCircuitBreaker()
            result = cb.check_and_update()
            
            status = result.get("status", "unknown")
            drawdown = result.get("drawdown_pct", 0)
            
            # Determine severity
            if status in ["black", "red"]:
                severity = "critical"
            elif status in ["orange"]:
                severity = "warning"
            elif status in ["yellow"]:
                severity = "caution"
            else:
                severity = "ok"
            
            ok = severity == "ok"
            
            self.checks.append({
                "name": "circuit_breaker",
                "status": f"{status} ({drawdown}% dd)" if drawdown else status,
                "ok": ok,
                "severity": severity,
                "drawdown": drawdown
            })
            
            if not ok:
                self.alerts.append(f"Circuit breaker {status}: {result.get('message', '')}")
            
            return ok
            
        except Exception as e:
            self.checks.append({
                "name": "circuit_breaker",
                "status": f"error: {str(e)}",
                "ok": True  # Don't fail health check for circuit breaker errors
            })
            return True
    
    def check_wiki_sync(self) -> bool:
        """Check if wiki is being synced."""
        wiki_dir = Path("~/wiki/projects/portfolio-lab/compound").expanduser()
        
        if not wiki_dir.exists():
            self.checks.append({"name": "wiki_sync", "status": "not_configured", "ok": True})
            return True
        
        # Check latest page
        pages = list(wiki_dir.glob("*.md"))
        if not pages:
            self.checks.append({"name": "wiki_sync", "status": "no_pages", "ok": True})
            return True
        
        latest = max(pages, key=lambda p: p.stat().st_mtime)
        hours_since = (datetime.now() - datetime.fromtimestamp(latest.stat().st_mtime)).total_seconds() / 3600
        
        ok = hours_since < 12  # Should sync every 6 hours
        self.checks.append({
            "name": "wiki_sync",
            "status": "ok" if ok else f"stale ({hours_since:.1f}h)",
            "ok": ok
        })
        
        if not ok:
            self.alerts.append(f"Wiki sync stale: last update {hours_since:.1f}h ago")
        
        return ok
    
    def run(self):
        """Run all health checks."""
        print(f"[{datetime.now()}] Health Monitor Starting")
        
        checks = [
            self.check_data_freshness(),
            self.check_cron_execution(),
            self.check_portfolio_health(),
            self.check_graduation_candidate(),
            self.check_kill_switches(),
            self.check_circuit_breaker(),
            self.check_wiki_sync()
        ]
        
        # Determine overall status
        critical = any(not c.get("ok", True) for c in self.checks 
                      if c["name"] in ["kill_switches"])
        warnings = sum(1 for c in self.checks if not c.get("ok", True))
        
        if critical:
            self.status = "critical"
        elif warnings > 0:
            self.status = "warning"
        else:
            self.status = "healthy"
        
        # Generate report
        report = {
            "timestamp": datetime.now().isoformat(),
            "status": self.status,
            "checks": {c["name"]: c for c in self.checks},
            "alerts": self.alerts,
            "summary": {
                "total_checks": len(self.checks),
                "passed": sum(1 for c in self.checks if c.get("ok", True)),
                "failed": sum(1 for c in self.checks if not c.get("ok", True))
            }
        }
        
        with open(REPORT_PATH, 'w') as f:
            json.dump(report, f, indent=2)
        
        # Print summary
        print(f"Status: {self.status.upper()}")
        for check in self.checks:
            status_icon = "✓" if check.get("ok") else "✗"
            print(f"  {status_icon} {check['name']}: {check['status']}")
        
        if self.alerts:
            print("\nAlerts:")
            for alert in self.alerts:
                print(f"  ! {alert}")
        
        # Trigger escalation if needed
        if self.status == "critical":
            self._escalate_critical()
        elif self.status == "warning" and len(self.alerts) > 2:
            self._escalate_warning()
        
        print(f"[{datetime.now()}] Health Monitor Complete")
        return report
    
    def _escalate_critical(self):
        """Escalate critical issues to Claude Code work item."""
        work_dir = Path("~/projects/portfolio-lab/work").expanduser()
        work_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        work_file = work_dir / f"critical_health_{timestamp}.md"
        
        content = f"""# CRITICAL: Portfolio-Lab Health Issues

**Detected:** {datetime.now().isoformat()}
**Status:** {self.status}

## Active Alerts

{chr(10).join(f"- {a}" for a in self.alerts)}

## Check Results

{chr(10).join(f"- {c['name']}: {c['status']}" for c in self.checks)}

## Immediate Actions Needed

1. Review kill switches in `~/projects/portfolio-lab/data/`
2. Check cron job logs in `~/projects/portfolio-lab/data/*.log`
3. Verify database connectivity
4. Review recent orders for anomalies

## Auto-Generated by Health Monitor

This requires immediate attention and possible manual intervention.
"""
        
        with open(work_file, 'w') as f:
            f.write(content)
        
        print(f"  🚨 Critical escalation created: {work_file}")
    
    def _escalate_warning(self):
        """Escalate warnings to notification."""
        # Could send notification here
        print(f"  ⚠️ Multiple warnings detected ({len(self.alerts)} alerts)")

if __name__ == "__main__":
    monitor = HealthMonitor()
    monitor.run()
