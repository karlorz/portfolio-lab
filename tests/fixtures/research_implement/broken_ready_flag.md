# Session A plan — broken ready-for-implement flag (not pickable)

## Queue

### Q1. Complete fields but ready flag garbage
1. **title**: Complete fields but ready flag garbage
2. **acceptance**: should not be picked with broken ready
3. **risks**: do not touch kill_switch
4. **file_touch**: write tests/test_research_implement_loop.py
5. **breaking_change**: false
6. **redeploy_notes**: none
status: OPEN
ready-for-implement: maybe

### Q2. Ready flag uppercase YES-ish but not yes
1. **title**: Ready flag uppercase YES-ish but not yes
2. **acceptance**: still not pickable
3. **risks**: do not touch order_router
4. **file_touch**: write tests/test_example.py
5. **breaking_change**: false
6. **redeploy_notes**: none
status: OPEN
ready-for-implement: READY

## Watch

### Fake. Watch lookalike with ready yes
1. **title**: Watch lookalike must never be B-picked
2. **acceptance**: should be ignored
3. **risks**: n/a
4. **file_touch**: n/a
5. **breaking_change**: false
6. **redeploy_notes**: none
status: OPEN
ready-for-implement: yes

## Heartbeat

- Queue **0/10 OPEN** for B-pick; broken ready flags do not count.
