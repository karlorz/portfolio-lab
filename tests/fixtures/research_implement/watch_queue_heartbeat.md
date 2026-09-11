# Beat 19 plan — Watch + Queue + Project Work + Heartbeat preserve

Living producer plan with distinctive non-Queue bodies. Rewriting ## Queue
(append / ship / serialize) must keep these sections intact.

## Queue

### Q1. Beat19 shippable preserve item
1. **title**: Beat19 shippable preserve item
2. **acceptance**: SHIPPED appears; Watch/Project Work/Heartbeat markers remain
3. **risks**: do not drop non-Queue markdown sections
4. **file_touch**: tests/test_research_implement_loop.py; src/research_implement/queue.py
5. **breaking_change**: false
6. **redeploy_notes**: none
status: OPEN
ready-for-implement: yes

## Watch

BEAT19_WATCH_MARKER
| Row | Why not queued |
|---|---|
| **beat19-watch-row** | Watch unless PROMOTE |

## Project Work

BEAT19_PROJECT_MARKER
| Path | Why filed |
|---|---|
| `raw/transcripts/beat19-watch-only.md` | Ignore unless PROMOTE |

## Heartbeat

BEAT19_HEARTBEAT_MARKER
- Queue **1/10 OPEN**; B pick = Beat19 shippable preserve item.
