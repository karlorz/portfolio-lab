#!/usr/bin/env python3
"""
Portfolio-Lab Alpha: Notification System
Telegram integration for critical alerts and notifications.
"""

import os
import json
import asyncio
import aiohttp
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, List, Dict
from dataclasses import dataclass, asdict
import hashlib

DATA_DIR = Path("~/projects/portfolio-lab/data").expanduser()
ALERT_LOG = DATA_DIR / "alerts.jsonl"
NOTIFIER_STATE = DATA_DIR / ".notifier_state.json"

# Telegram configuration from environment
TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID")

# Rate limiting
MIN_INTERVAL_SECONDS = 300  # 5 minutes between similar alerts
MAX_ALERTS_PER_HOUR = 10

@dataclass
class Alert:
    level: str  # 'error', 'warning', 'success', 'info'
    type: str   # 'kill_switch', 'graduation_candidate', 'regime_change', 'stale_data', 'system'
    title: str
    message: str
    timestamp: str
    requires_action: bool = False
    metadata: Optional[Dict] = None
    
    def to_dict(self) -> Dict:
        return {
            **asdict(self),
            'metadata': self.metadata or {}
        }
    
    def fingerprint(self) -> str:
        """Generate unique fingerprint for deduplication."""
        content = f"{self.level}:{self.type}:{self.title}:{self.message[:50]}"
        return hashlib.md5(content.encode()).hexdigest()[:12]

class NotificationManager:
    """Manages alert notifications with rate limiting and deduplication."""
    
    def __init__(self):
        self.recent_alerts: Dict[str, datetime] = {}
        self.hourly_count = 0
        self.hourly_reset = datetime.now()
        self._load_state()
        DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    def _load_state(self):
        """Load notification state for deduplication."""
        if NOTIFIER_STATE.exists():
            try:
                with open(NOTIFIER_STATE) as f:
                    state = json.load(f)
                    # Convert string timestamps back to datetime
                    self.recent_alerts = {
                        k: datetime.fromisoformat(v) 
                        for k, v in state.get('recent_alerts', {}).items()
                    }
                    self.hourly_count = state.get('hourly_count', 0)
                    self.hourly_reset = datetime.fromisoformat(
                        state.get('hourly_reset', datetime.now().isoformat())
                    )
            except Exception:
                self.recent_alerts = {}
                self.hourly_count = 0
                self.hourly_reset = datetime.now()
    
    def _save_state(self):
        """Save notification state."""
        state = {
            'recent_alerts': {
                k: v.isoformat() for k, v in self.recent_alerts.items()
            },
            'hourly_count': self.hourly_count,
            'hourly_reset': self.hourly_reset.isoformat(),
            'updated': datetime.now().isoformat()
        }
        with open(NOTIFIER_STATE, 'w') as f:
            json.dump(state, f, indent=2)
    
    def _check_rate_limit(self, alert: Alert) -> bool:
        """Check if alert can be sent based on rate limits."""
        now = datetime.now()
        
        # Reset hourly counter if needed
        if now - self.hourly_reset > timedelta(hours=1):
            self.hourly_count = 0
            self.hourly_reset = now
        
        # Check hourly limit
        if self.hourly_count >= MAX_ALERTS_PER_HOUR:
            return False
        
        # Check deduplication window
        fingerprint = alert.fingerprint()
        if fingerprint in self.recent_alerts:
            last_sent = self.recent_alerts[fingerprint]
            if now - last_sent < timedelta(seconds=MIN_INTERVAL_SECONDS):
                return False
        
        return True
    
    def _record_sent(self, alert: Alert):
        """Record that an alert was sent."""
        self.recent_alerts[alert.fingerprint()] = datetime.now()
        self.hourly_count += 1
        self._save_state()
    
    def _log_alert(self, alert: Alert, sent: bool = False):
        """Persist alert to log file."""
        entry = {
            **alert.to_dict(),
            'sent': sent,
            'logged_at': datetime.now().isoformat()
        }
        
        # Append to log
        with open(ALERT_LOG, 'a') as f:
            f.write(json.dumps(entry) + '\n')
        
        # Rotate if too large (>1000 entries)
        self._maybe_rotate_log()
    
    def _maybe_rotate_log(self):
        """Rotate log file if it exceeds 1000 lines."""
        if not ALERT_LOG.exists():
            return
        
        with open(ALERT_LOG) as f:
            lines = f.readlines()
        
        if len(lines) > 1000:
            # Keep last 500 entries
            with open(ALERT_LOG, 'w') as f:
                f.writelines(lines[-500:])
    
    async def send_telegram(self, alert: Alert) -> bool:
        """Send alert via Telegram."""
        if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
            print(f"[Notifier] Telegram not configured, logging only: {alert.title}")
            return False
        
        if not self._check_rate_limit(alert):
            print(f"[Notifier] Rate limited: {alert.title}")
            return False
        
        # Format message
        emoji_map = {
            'error': '🚨',
            'warning': '⚠️',
            'success': '✅',
            'info': 'ℹ️'
        }
        
        emoji = emoji_map.get(alert.level, '🔔')
        
        message = f"""{emoji} <b>{alert.title}</b>

<b>Level:</b> {alert.level.upper()}
<b>Type:</b> {alert.type}
<b>Time:</b> {alert.timestamp}

{alert.message}"""
        
        if alert.requires_action:
            message += "\n\n‼️ <b>Action Required</b>"
        
        # Send via Telegram API
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
        payload = {
            'chat_id': TELEGRAM_CHAT_ID,
            'text': message,
            'parse_mode': 'HTML',
            'disable_notification': alert.level in ['info', 'success']
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=payload, timeout=aiohttp.ClientTimeout(total=10)) as resp:
                    if resp.status == 200:
                        self._record_sent(alert)
                        print(f"[Notifier] Telegram sent: {alert.title}")
                        return True
                    else:
                        error_text = await resp.text()
                        print(f"[Notifier] Telegram failed ({resp.status}): {error_text}")
                        return False
        except Exception as e:
            print(f"[Notifier] Telegram error: {e}")
            return False
    
    async def notify(self, alert: Alert, channels: Optional[List[str]] = None) -> Dict[str, bool]:
        """Send alert through specified channels (default: all configured)."""
        channels = channels or ['telegram', 'log']
        results = {}
        
        # Always log
        self._log_alert(alert, sent=False)
        results['log'] = True
        
        # Send to Telegram if configured and requested
        if 'telegram' in channels:
            results['telegram'] = await self.send_telegram(alert)
            if results['telegram']:
                # Update log entry to mark as sent
                self._log_alert(alert, sent=True)
        
        return results
    
    def get_recent_alerts(self, hours: int = 24, level: Optional[str] = None) -> List[Dict]:
        """Retrieve recent alerts from log."""
        if not ALERT_LOG.exists():
            return []
        
        cutoff = datetime.now() - timedelta(hours=hours)
        alerts = []
        
        with open(ALERT_LOG) as f:
            for line in f:
                try:
                    entry = json.loads(line)
                    entry_time = datetime.fromisoformat(entry.get('timestamp', '2000-01-01'))
                    if entry_time >= cutoff:
                        if level is None or entry.get('level') == level:
                            alerts.append(entry)
                except Exception:
                    continue
        
        return sorted(alerts, key=lambda x: x.get('timestamp', ''), reverse=True)

# Singleton instance
_notifier: Optional[NotificationManager] = None

def get_notifier() -> NotificationManager:
    """Get or create notification manager singleton."""
    global _notifier
    if _notifier is None:
        _notifier = NotificationManager()
    return _notifier

async def send_alert(
    level: str,
    type: str,
    title: str,
    message: str,
    requires_action: bool = False,
    metadata: Optional[Dict] = None
) -> Dict[str, bool]:
    """Convenience function to send an alert."""
    alert = Alert(
        level=level,
        type=type,
        title=title,
        message=message,
        timestamp=datetime.now().isoformat(),
        requires_action=requires_action,
        metadata=metadata
    )
    
    notifier = get_notifier()
    return await notifier.notify(alert)

# Severity-based routing helpers
async def notify_critical(title: str, message: str, type: str = "system", metadata: Optional[Dict] = None):
    """Send critical alert (immediate Telegram + log)."""
    return await send_alert('error', type, title, message, requires_action=True, metadata=metadata)

async def notify_warning(title: str, message: str, type: str = "system", metadata: Optional[Dict] = None):
    """Send warning alert (Telegram digest + log)."""
    return await send_alert('warning', type, title, message, metadata=metadata)

async def notify_success(title: str, message: str, type: str = "system", metadata: Optional[Dict] = None):
    """Send success alert (log only, no Telegram)."""
    return await send_alert('success', type, title, message, metadata=metadata)

async def notify_info(title: str, message: str, type: str = "system", metadata: Optional[Dict] = None):
    """Send info alert (log only)."""
    return await send_alert('info', type, title, message, metadata=metadata)

# Example/test
if __name__ == "__main__":
    async def test():
        print("Testing notification system...")
        
        # Test without Telegram (log only)
        result = await notify_info(
            "Test Notification",
            "This is a test of the notification system."
        )
        print(f"Info notification: {result}")
        
        # Test critical (would send to Telegram if configured)
        result = await notify_critical(
            "Test Critical Alert",
            "This would be sent immediately to Telegram if configured."
        )
        print(f"Critical notification: {result}")
        
        # Show recent alerts
        notifier = get_notifier()
        recent = notifier.get_recent_alerts(hours=1)
        print(f"\nRecent alerts (last hour): {len(recent)}")
        for alert in recent[:3]:
            print(f"  - [{alert['level']}] {alert['title']}")
    
    asyncio.run(test())
