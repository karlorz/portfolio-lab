#!/usr/bin/env python3
"""
Portfolio-Lab Alpha: Research Agent (Hermes + Claude Code Bridge)
Triggered by regime changes or manual request. Uses Claude Code for complex analysis.
"""

import os
import json
import sqlite3
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

DATA_DIR = Path("~/projects/portfolio-lab/data").expanduser()
WIKI_DIR = Path("~/wiki/projects/portfolio-lab").expanduser()
WORK_DIR = Path("~/projects/portfolio-lab/work").expanduser()
DB_PATH = DATA_DIR / "market.db"

class ResearchAgent:
    def __init__(self):
        self.conn = sqlite3.connect(DB_PATH)
        self.conn.row_factory = sqlite3.Row
        WORK_DIR.mkdir(parents=True, exist_ok=True)
    
    def check_triggers(self) -> List[Dict]:
        """Check for research triggers."""
        triggers = []
        
        # Check regime trigger file
        regime_trigger = DATA_DIR / ".regime_trigger"
        if regime_trigger.exists():
            with open(regime_trigger) as f:
                data = json.load(f)
                data["type"] = "regime_change"
                triggers.append(data)
            regime_trigger.unlink()
        
        # Check manual work items
        for work_file in WORK_DIR.glob("*.json"):
            if work_file.name.startswith("pending_"):
                with open(work_file) as f:
                    triggers.append(json.load(f))
                work_file.rename(work_file.with_name(work_file.name.replace("pending_", "in_progress_")))
        
        return triggers
    
    def analyze_regime(self, trigger: Dict) -> Dict:
        """Analyze regime change and determine strategy adjustments."""
        cursor = self.conn.cursor()
        
        # Fetch recent data for analysis
        cursor.execute("""
            SELECT symbol, date, close FROM prices 
            WHERE date >= date('now', '-90 days')
            AND symbol IN ('SPY', 'GLD', 'TLT', 'VIX', 'QQQ')
            ORDER BY symbol, date
        """)
        
        rows = cursor.fetchall()
        data = {}
        for row in rows:
            sym = row[0]
            if sym not in data:
                data[sym] = []
            data[sym].append({"date": row[1], "close": row[2]})
        
        analysis = {
            "regime": trigger.get("regime"),
            "vix": trigger.get("vix"),
            "data_summary": {sym: len(pts) for sym, pts in data.items()},
            "recommended_action": None,
            "confidence": "low"
        }
        
        # Simple rule-based analysis
        if trigger.get("regime") == "crisis":
            analysis["recommended_action"] = "risk_off"
            analysis["suggested_allocation"] = {"SPY": 0.20, "GLD": 0.50, "TLT": 0.30}
            analysis["confidence"] = "medium"
        elif trigger.get("regime") == "vol_spike":
            analysis["recommended_action"] = "defensive_shift"
            analysis["suggested_allocation"] = {"SPY": 0.30, "GLD": 0.45, "TLT": 0.25}
            analysis["confidence"] = "medium"
        elif trigger.get("regime") == "low_vol":
            analysis["recommended_action"] = "risk_on"
            analysis["suggested_allocation"] = {"SPY": 0.55, "GLD": 0.30, "TLT": 0.15}
            analysis["confidence"] = "medium"
        
        return analysis
    
    def should_delegate_claude(self, analysis: Dict) -> bool:
        """Determine if this requires Claude Code intervention."""
        # Delegate if:
        # - Regime is "crisis" (complex risk management)
        # - Confidence is low
        # - Requires strategy code changes
        if analysis.get("regime") == "crisis":
            return True
        if analysis.get("confidence") == "low":
            return True
        if analysis.get("requires_implementation"):
            return True
        return False
    
    def delegate_to_claude(self, analysis: Dict, trigger: Dict) -> str:
        """Create a work item for Claude Code to handle."""
        work_item = {
            "id": f"regime_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "type": "claude_code_delegate",
            "trigger": trigger,
            "analysis": analysis,
            "context": {
                "data_summary": analysis.get("data_summary"),
                "current_regime": analysis.get("regime"),
                "recommended_action": analysis.get("recommended_action")
            },
            "tasks": [],
            "created_at": datetime.now().isoformat(),
            "status": "pending_delegate"
        }
        
        # Build Claude Code prompt
        if analysis.get("regime") == "crisis":
            work_item["tasks"].append({
                "type": "code_review",
                "description": "Review risk management code for crisis regime handling",
                "files": ["src/strategy/evaluator.py", "config/paper.yaml"]
            })
            work_item["tasks"].append({
                "type": "implement",
                "description": "Add kill switch for tail risk events (VIX > 40)",
                "spec": "Implement automatic risk-off liquidation when VIX exceeds 40"
            })
        
        if analysis.get("confidence") == "low":
            work_item["tasks"].append({
                "type": "research",
                "description": "Analyze correlation breakdown patterns in current regime",
                "data_request": "SPY-GLD-TLT correlation over last 30 days"
            })
        
        # Save work item for Claude Code
        work_file = WORK_DIR / f"claude_{work_item['id']}.json"
        with open(work_file, 'w') as f:
            json.dump(work_item, f, indent=2)
        
        return str(work_file)
    
    def crystallize_to_wiki(self, analysis: Dict) -> Path:
        """Save research findings to wiki compound page."""
        timestamp = datetime.now().strftime("%Y-%m-%d")
        page_path = WIKI_DIR / "compound" / f"regime-analysis-{timestamp}.md"
        
        content = f"""---
type: query
tags: [regime, analysis, portfolio-lab]
provenance: project
provenance_projects: [[portfolio-lab]]
created: {timestamp}
---

# Regime Analysis: {analysis.get('regime', 'unknown').upper()}

**Detected:** {datetime.now().isoformat()}
**VIX Level:** {analysis.get('vix', 'N/A')}
**Confidence:** {analysis.get('confidence')}

## Recommended Action

{analysis.get('recommended_action', 'No action')}

## Suggested Allocation

```json
{json.dumps(analysis.get('suggested_allocation', {}), indent=2)}
```

## Data Summary

| Symbol | Data Points |
|--------|-------------|
{chr(10).join(f"| {sym} | {count} |" for sym, count in analysis.get('data_summary', {}).items())}

## Next Steps

- [ ] Review strategy allocation
- [ ] Validate with out-of-sample data
- [ ] Update wiki if new pattern discovered

## Sources

- Market data from Yahoo Finance v8 API
- Regime detection via volatility and trend analysis
- ^[raw/market/regime-log-{timestamp}]
"""
        
        page_path.parent.mkdir(parents=True, exist_ok=True)
        with open(page_path, 'w') as f:
            f.write(content)
        
        return page_path
    
    def run(self):
        """Main research agent loop."""
        print(f"[{datetime.now()}] Research Agent Starting")
        
        triggers = self.check_triggers()
        
        if not triggers:
            print("No triggers found, checking for scheduled analysis...")
            # Run daily summary analysis
            return self.run_daily_summary()
        
        for trigger in triggers:
            print(f"Processing trigger: {trigger.get('type')}")
            
            analysis = self.analyze_regime(trigger)
            print(f"Analysis complete: {analysis.get('recommended_action')}")
            
            if self.should_delegate_claude(analysis):
                work_file = self.delegate_to_claude(analysis, trigger)
                print(f"Delegated to Claude Code: {work_file}")
                
                # In a real system, this would spawn Claude Code
                # For now, we create a human-readable task file
                self.create_claude_prompt(work_file, analysis)
            
            wiki_page = self.crystallize_to_wiki(analysis)
            print(f"Crystallized to wiki: {wiki_page}")
        
        self.conn.close()
        print(f"[{datetime.now()}] Research Agent Complete")
    
    def create_claude_prompt(self, work_file: Path, analysis: Dict):
        """Create a human/Claude readable prompt from work item."""
        prompt_file = work_file.with_suffix('.md')
        
        prompt = f"""# Claude Code Task: Portfolio-Lab Regime Analysis

## Situation

A market regime change has been detected requiring code-level intervention.

**Regime:** {analysis.get('regime')}
**VIX:** {analysis.get('vix')}
**Recommended Action:** {analysis.get('recommended_action')}

## Your Task

Review the current strategy implementation and make necessary adjustments.

### Files to Review

- `~/projects/portfolio-lab/src/strategy/evaluator.py` - Main strategy logic
- `~/projects/portfolio-lab/config/paper.yaml` - Paper trading config
- `~/projects/portfolio-lab/src/data/pipeline.py` - Data pipeline

### Suggested Changes

Based on regime `{analysis.get('regime')}`:

```
{json.dumps(analysis.get('suggested_allocation', {}), indent=2)}
```

### Steps

1. Review current implementation
2. Identify where regime overrides are applied
3. Adjust parameters if needed
4. Add any missing risk checks
5. Run tests: `cd ~/projects/portfolio-lab && python3 -m pytest src/`
6. Update configuration if allocation changes

### Deliverables

- Modified code with clear comments
- Updated config files
- Brief summary of changes in work/claude_summary_{datetime.now().strftime('%Y%m%d')}.md

## Context

This task was auto-generated by the Portfolio-Lab Research Agent due to regime
change detection. The goal is bounded: adjust strategy for current market
conditions, not full redesign.

Work item: {work_file}
"""
        
        with open(prompt_file, 'w') as f:
            f.write(prompt)
        
        print(f"Created Claude prompt: {prompt_file}")
    
    def run_daily_summary(self) -> Dict:
        """Run daily summary analysis."""
        cursor = self.conn.cursor()
        
        # Get performance summary
        cursor.execute("""
            SELECT COUNT(*) as days,
                   AVG(daily_return) as avg_return,
                   MAX(total_value) as peak,
                   MIN(total_value) as trough
            FROM (
                SELECT date, close as daily_return, close as total_value
                FROM prices WHERE symbol = 'SPY' 
                AND date >= date('now', '-30 days')
            )
        """)
        
        row = cursor.fetchone()
        summary = {
            "days": row[0],
            "avg_return": row[1],
            "peak": row[2],
            "trough": row[3]
        }
        
        print(f"Daily summary: {summary}")
        
        self.conn.close()
        return summary

if __name__ == "__main__":
    agent = ResearchAgent()
    agent.run()
