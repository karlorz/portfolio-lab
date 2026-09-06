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

Proof: `PORTFOLIO_LAB_ENABLE_ML=0 pytest tests/test_research_implement_loop.py -q -k 'beat22 or beat23'`

## Errors

- Missing `--plan`/`--log` path → non-zero exit (`file not found`); the CLI does not create the file.
- Pass only one of `--plan` / `--log` (alias); both together → non-zero exit.
- Pass only one of `--stub` / `--no-stub`; both together → non-zero exit.
- `--candidate-json` overrides `--stub` when OPEN is 0.
- `idle-decode` rejects `--dry-run` (decode-only; use `session-b --dry-run`).
- Session A `--dry-run` leaves the plan file unchanged.
- Producer-only flags (`--stub` / `--no-stub` / `--candidate-json`) are rejected on `session-b` / `idle-decode`.
- `session-a --candidate-json` with `[]` fails closed (no stub fallback); plan unchanged.
- Unknown subcommand → non-zero exit.
- `--candidate-json` list with no dict elements fails closed (no stub fallback).
- CLI `--json` keys match `SESSION_A_RESULT_JSON_KEYS` / `SESSION_RESULT_JSON_KEYS`.
- `--candidate-json` list skips leading non-dicts and uses the first dict.
- Relative `--plan` / `--log` paths resolve when the file exists in cwd.
- `--plan` / `--log` that exists but is not a file (e.g. directory) → non-zero exit.
- Relative `--log` alias works like relative `--plan`.
- `--candidate-json` that exists but is not a file (e.g. directory) → non-zero exit.
- Top-level JSON `null` for `--candidate-json` fails closed.
- Empty / whitespace-only `--plan` / `--log` → non-zero exit.
- Idle `--json` keeps `keep_schedule=true` and `scheduler_delete_called=false`.
- Empty / whitespace `--log` fails like `--plan`.
- `session-b --dry-run --json` keeps `keep_schedule=true` / `scheduler_delete_called=false` and does not mutate the plan.
- `session-b` decode-only `--json` keeps schedule / never `scheduler_delete` / plan unchanged.
- Session A OPEN>=1 recount-only `--json` has `wrote_item=false`.
- `idle-decode --json` on ready OPEN matches `session-b` decode-only JSON shape.
- Session A `--stub` on empty Queue `--json` → `wrote_item=true` / `verdict=queued`.
- Empty / whitespace `--candidate-json` → non-zero exit (no Path('.') coerce).
- Session A `--stub --dry-run --json` reports queued/`wrote_item=true` but does not write the plan.
- Relative `--candidate-json` resolves when the file exists in cwd.
- Top-level JSON `true`/`false` for `--candidate-json` fails closed.
- `--candidate-json` `{}` fails closed as incomplete candidate (rc=1; plan unchanged).
- `session-a --help` documents `--stub` / `--no-stub` / `--candidate-json`.
- Top-level `--help` lists `session-a` / `session-b` / `idle-decode`.
- `idle-decode --help` has no `--dry-run`; `session-b --help` does.
- Top-level `--help` states side-dev / no Tasker / no live LLM.
- No CLI `--implement` / stub-ship flag — SHIPPED stays callback-only.
- `session-b --json` emits shared `SessionResult.to_dict` (idle | picked | dry_run); never `scheduler_delete`.
- Session B dry-run text includes the same `decode pick` six-field report as decode-only.
- `--json` help: session-a → `SessionAResult.to_dict`; session-b/idle-decode → `SessionResult.to_dict`.
- Decode-only and dry-run messages both include `decode pick` six-field report.
- OPEN complete but not ready-for-implement → idle (`queue 0/10`); plan unchanged.
- Incomplete OPEN / SHIPPED-only → idle (`queue 0/10`); mixed-priority picks first ready OPEN.
- Watch lookalike rows are never B-picked; watch-only / no-Queue plans idle.
- Broken / non-yes `ready-for-implement` → idle (`queue 0/10`); never B-picked.
- `ready-for-implement` aliases: `yes` / `y` / `true` / `1` (case-insensitive); anything else is not B-pickable.
- Ambiguous dual `## Queue` plans: CLI exits non-zero (`verdict=failed`); never mutates the plan.
- `watch_queue_heartbeat_empty`: idle B-fire leaves Watch/Project/Heartbeat markers; Session A stub append keeps them.
- `watch_queue_heartbeat` (OPEN ready): idle-decode pick / `session-b --dry-run` keep Watch/Project/Heartbeat markers.
- `watch_heartbeat_no_queue`: B idle; Session A stub creates `## Queue` and keeps Watch/Heartbeat markers.
- CLI `--json` for `light` / `dry_run` / `failed` uses the same Session A/B key sets as queued/idle/picked.
- Dual-Queue Session A failed CLI uses `SESSION_A_RESULT_JSON_KEYS`; `AmbiguousQueueError` is a public export.
- `session-b --dry-run` on empty Queue → `idle` (no implement_result); helpers `is_b_pickable` / `is_ready_yes` / `count_open` stay public.
- `two_open_ready` + `session-b --dry-run`: picks first OPEN; both remain OPEN on disk.
- `watch_lookalike`: dry-run/idle-decode pick Queue item only; Watch lookalike never selected.
- Non-pickable plans (`shipped_only` / `incomplete_open` / `watch_only_lookalike`): `session-b --dry-run` → `idle`.
- `open_complete_not_ready` / `broken_ready_flag` + `session-b --dry-run` → `idle`.
