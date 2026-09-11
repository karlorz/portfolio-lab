# Session A plan — fixture two OPEN ready

## Queue

### Q1. First ready complete item
1. **title**: First ready complete item
2. **acceptance**: pytest first item EXIT=0
3. **risks**: do not touch kill_switch
4. **file_touch**: write tests/test_research_implement_loop.py
5. **breaking_change**: false
6. **redeploy_notes**: none
status: OPEN
ready-for-implement: yes

### Q2. Second ready complete item
1. **title**: Second ready complete item
2. **acceptance**: pytest second item EXIT=0
3. **risks**: do not touch order_router
4. **file_touch**: read src/research_implement/queue.py
5. **breaking_change**: false
6. **redeploy_notes**: none
status: OPEN
ready-for-implement: yes

## Watch

| Row | Why not queued |
|---|---|
| leftover | ignore |

## Heartbeat

- Queue **2/10 OPEN**; B pick = First ready complete item.
