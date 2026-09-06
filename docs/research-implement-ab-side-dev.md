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
- `mixed_priority` + `session-b --dry-run` / idle-decode: picks first ready OPEN; plan unchanged.
- `one_open_ready` + `session-b --dry-run` / idle-decode: picks Q1; plan unchanged.
- Dual-Queue + `--dry-run` still fail-closed (`verdict=failed`); never becomes a dry_run pick.
- `session-a --no-stub`: OPEN>=1 → light recount; empty Queue → failed (no stub fallback).
- `session-a --no-stub --dry-run`: same light/failed as `--no-stub`; plan unchanged.
- `first_b_pick`: first B-pickable OPEN, else None; idle-decode messages include `decode pick`.
- CLI `--json` matches `session_a_result_dict` / `session_b_result_dict` key sets.
- Public path runners: `run_session_a_path` / `run_session_b_path`; Session A light/failed `--json` uses `SESSION_A_RESULT_JSON_KEYS`.
- Session B `--json` for `dry_run` / `idle` / `failed` uses `SESSION_RESULT_JSON_KEYS`; `to_dict` ≡ `to_json_dict`.
- Session A `to_dict` ≡ `to_json_dict` ≡ `session_a_result_dict` for queued/light/failed.
- Public dataclasses: `SessionAResult`, `SessionResult`/`SessionBResult`, `QueueItem` (all expose `to_dict` where applicable).
- `format_queue_item` → `parse_queue_items` round-trip keeps B-pickable fields.
- `incomplete_open` / `shipped_only`: not B-pickable; `count_open` is 0.
- `open_complete_not_ready` / `broken_ready_flag`: not B-pickable; `count_open` is 0.
- `mixed_priority`: `first_b_pick` skips incomplete; `count_open` is pickable-only.
- `two_open_ready`: `first_b_pick` is the first of two pickable OPENs; `count_open` is 2.
- `one_open_ready`: `first_b_pick` and `count_open` are 1; `empty_queue` is 0 / None.
- Milestone beat80: CLI subcommands + public runners/helpers remain stable; Makefile suite note includes beat80.
- Beat 81: watch_lookalike first_b_pick=Q3; watch_only idle; mark_item_shipped remains public.
- Beat 82: subcommand --help lists --plan/--json; render_queue_count + QUEUE_CAPACITY remain public.
- Beat 83: non-pickable fixtures first_b_pick=None; serialize_queue_item(s) remain public.
- Beat 84: pickable fixtures first_b_pick ids; is_complete_six_field + is_open_status remain public.
- Beat 85: session-a producer help flags; session-b --decode-only; idle omits --dry-run; REQUIRED_FIELDS public.
- Beat 86: idle queue 0/10 + scheduler_delete guard; SchedulerDeleteForbidden public; decode pick report.
- Beat 87: two_queue AmbiguousQueueError + heading count=2; format_decode_report/decode_fields remain public.
- Beat 88: Watch/Heartbeat fixtures idle or pick Q1; SESSION_A/B_RESULT_KEYS remain public.
- Beat 89: incomplete_candidate_reasons contract; SESSION_*_JSON_KEYS remain public.
- Milestone beat90: CLI subcommands + public runners/helpers/guards remain stable; Makefile suite note includes beat90.
- Beat 91: empty Queue append next Q1 pickable; append/format/write helpers remain public.
- Beat 92: mark_item_shipped clears pick; stub_brainstorm ready yes; stub/dry-run helpers remain public.
- Beat 93: is_ready_yes aliases; default/stub_search_plan alias stub; session_*_result_dict remain public.
- Beat 94: session_*_result_dict / to_dict match JSON keys; path runners remain public.
- Beat 95: serialize_queue_item round-trip stays pickable; count_queue_headings fixture contract.
- Beat 96: decode_fields covers six fields + status/ready; format_decode_report lines 1–6.
- Beat 97: next_queue_id progression (empty→Q1 … lookalike→Q4); require_unique fail-closed on dual Queue.
- Beat 98: is_b_pickable fixture matrix; is_complete_six_field / is_open_status / is_ready_yes remain public.
- Beat 99: count_open fixture matrix aligns with render_queue_count; QUEUE_CAPACITY remains public.
- Milestone beat100: CLI subcommands + public runners/helpers/guards/decode/serialize remain stable; Makefile suite note includes beat100.
- Beat 101: format_queue_item round-trip stays pickable; dry_run_implement / make_fixture_ship_implement remain public.
