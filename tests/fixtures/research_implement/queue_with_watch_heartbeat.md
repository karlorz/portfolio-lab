# Session A/B plan — Watch + Queue + Heartbeat preserve (beat 19)

Living plan used to prove write_queue_section / append / ship keep non-Queue
sections. Watch is intentionally *before* Queue so rewrite offsets cannot
assume Queue is the first heading.

## Watch

| Row | Why not queued |
|---|---|
| **beat19-watch-marker** | Watch-class; must survive append / write / ship |

## Queue

**GROUP CHECK**: 0 OPEN items.

## Project Work

| Path | Why filed |
|---|---|
| `raw/transcripts/beat19-preserve.md` | Not a Queue item; must survive append / write / ship |

## Heartbeat

- Queue **0/10 OPEN**; B pick = STANDBY.
- beat19-heartbeat-marker
