# Session A plan — fixture one OPEN

## Queue

### Q1. Add fixture unit test for queue parser
1. **title**: Add fixture unit test for queue parser
2. **acceptance**: `PORTFOLIO_LAB_ENABLE_ML=0 uv run pytest tests/test_research_implement_loop.py -q` EXIT=0
3. **risks**: do not touch live authority / allocations / order_router / kill_switch
4. **file_touch**: write `tests/test_research_implement_loop.py`; read `src/research_implement/queue.py`
5. **breaking_change**: false
6. **redeploy_notes**: none
status: OPEN
ready-for-implement: yes

## Watch

| Row | Why not queued |
|---|---|
| **ops-followup-waitress** | Watch unless PROMOTE |

## Project Work

| Path | Why filed |
|---|---|
| `raw/transcripts/watch-only.md` | Ignore unless PROMOTE |

## Heartbeat

- Queue **1/10 OPEN**; B pick = Add fixture unit test for queue parser.
