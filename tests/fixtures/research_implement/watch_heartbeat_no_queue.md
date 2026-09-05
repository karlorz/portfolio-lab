---
title: beat20-no-queue
status: living
---

# Beat 20 plan — Watch + Heartbeat only (no Queue section)

Front matter and non-Queue sections must survive Session A append /
write_queue_section when the Queue heading is created from scratch.

## Watch

BEAT20_WATCH_MARKER
| Row | Why not queued |
|---|---|
| **beat20-watch-row** | Watch unless PROMOTE |

## Heartbeat

BEAT20_HEARTBEAT_MARKER
- Queue **0/10 OPEN**; B pick = STANDBY.
