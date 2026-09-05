# Research-implement A/B side-dev

Side-dev Session A/B loop only. No Tasker, no LLM, no prod ports (8000/8001), no wiki.

## CLI

```text
python -m src.research_implement session-a|session-b|idle-decode [flags]
```

| Command | Role |
| --- | --- |
| `session-a` | Queue producer (stub / `--candidate-json` when OPEN=0; recount-only when OPEN≥1) |
| `session-b` | Decode-only pick or idle; optional `--dry-run` |
| `idle-decode` | Same decode/idle path as B without shipping |

Common flags: `--plan` / `--log`, `--json`, `--dry-run`, `--stub` (A), `--candidate-json <path>` (A).

## Contracts

- Empty `## Queue` → idle fire (`queue 0/10`); keep schedule.
- Never `scheduler_delete` (idle, dry-run, or decode-only).
- `--dry-run` never marks SHIPPED; second B can re-pick the same OPEN.
- `--json` emits shared `SessionResult` / `SessionAResult` shapes.

## Make targets

- `make test-research-implement` — full A/B + contract fixture suite
- `make research-implement-e2e-dry-run` — echo recipe for A stub → B dry-run
- `make research-implement-e2e-pipeline` — echo Beat 11 pipeline / pytest hint

Proof: `PORTFOLIO_LAB_ENABLE_ML=0 pytest tests/test_research_implement_loop.py -q -k 'beat21 or beat22'`

## Errors

- Missing `--plan`/`--log` path → non-zero exit (`file not found`); the CLI does not create the file.
- Pass only one of `--plan` / `--log` (alias); both together → non-zero exit.
