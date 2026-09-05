---
title: beat21-two-queue
status: living
---

# Beat 21 plan — duplicate Queue section (fail-closed)

Deliberately ambiguous living plan with TWO Queue section headings.
Parse / write / append / Session A / Session B must refuse without destroying
Watch / Heartbeat / either Queue body.

## Watch

BEAT21_WATCH_MARKER
| Row | Why not queued |
|---|---|
| **beat21-watch-row** | Watch unless PROMOTE |

## Queue

### Q1. Beat21 first queue item
1. **title**: Beat21 first queue item
2. **acceptance**: ambiguous Queue refuses; content preserved
3. **risks**: do not silently merge Queue sections
4. **file_touch**: tests/test_research_implement_loop.py; src/research_implement/queue.py
5. **breaking_change**: false
6. **redeploy_notes**: none
status: OPEN
ready-for-implement: yes

## Heartbeat

BEAT21_HEARTBEAT_MARKER
- Intentionally malformed: a second Queue section follows.

## Queue

### Q9. Beat21 second queue item
1. **title**: Beat21 second queue item
2. **acceptance**: must not be silently merged into first Queue
3. **risks**: silent pick-first corrupts plan
4. **file_touch**: tests/test_research_implement_loop.py
5. **breaking_change**: false
6. **redeploy_notes**: none
status: OPEN
ready-for-implement: yes
