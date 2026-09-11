# Session A plan — Watch lookalike must not be picked

## Queue

### Q3. Real ready Queue item
1. **title**: Real ready Queue item
2. **acceptance**: pytest EXIT=0
3. **risks**: do not touch order_router
4. **file_touch**: write `tests/test_example.py`
5. **breaking_change**: false
6. **redeploy_notes**: none
status: OPEN
ready-for-implement: yes

## Watch

### Fake. Looks like a task
1. **title**: Watch lookalike must never be B-picked
2. **acceptance**: should be ignored
3. **risks**: n/a
4. **file_touch**: n/a
5. **breaking_change**: false
6. **redeploy_notes**: none
status: OPEN
ready-for-implement: yes

## Project Work

| Path | Why |
|---|---|
| example | ignore |

## Heartbeat

- B pick from Queue only
