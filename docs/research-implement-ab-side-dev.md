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
- Beat 102: write_queue_section transplants two_open onto empty; default_implement aliases dry_run_implement.
- Beat 103: serialize_queue_items two_open round-trip stays dual-pickable.
- Beat 104: QueueItem.field_map covers REQUIRED_FIELDS; QueueItem remains public.
- Beat 105: to_json_dict aliases to_dict; SessionBResult is SessionResult.
- Beat 106: idle/picked keep_schedule True; never scheduler_delete_called.
- Beat 107: Session A queued/light/failed to_dict matrix; run_session_a remains public.
- Beat 108: Session B idle/picked/failed to_dict matrix; run_session_b remains public.
- Beat 109: first_b_pick/count_open consistent across fixtures; __all__ has unique names.
- Milestone beat110: CLI subcommands + public A/B runners/helpers/aliases/write/serialize remain stable; Makefile suite note includes beat110.
- Beat 111: idle fixtures queue 0/10 + decode_report None; picked decode_report == format_decode_report.
- Beat 112: path runners — Session A empty writes stub; Session B idle leaves plan untouched.
- Beat 113: idle-decode CLI --json idle/picked/failed matrix (mixed_priority→Q2).
- Beat 114: session-b --decode-only --json idle/picked/failed matrix; plan untouched.
- Beat 115: session-a --stub|--no-stub --json CLI matrix via tmp_path (queued/light/failed).
- Beat 116: session-b --dry-run --json CLI matrix via tmp_path (idle/dry_run/failed); plan untouched.
- Beat 117: session-a --stub|--no-stub --dry-run --json CLI matrix via tmp_path (queued wrote_item=True but plan unchanged / light / failed); contrast Beat 115.
- Beat 118: idle-decode --json Watch-fixture CLI matrix via tmp_path (watch_only→idle; watch_lookalike / watch_queue_heartbeat→picked; empty→idle; two_queue→failed); plan untouched; contrast Beat 113.
- Beat 119: session-b --dry-run --json Watch-fixture CLI matrix via tmp_path (watch_only→idle; watch_lookalike / watch_queue_heartbeat→dry_run; two_queue→failed); plan untouched; keep_schedule / no scheduler_delete; contrast Beat 116 + Beat 118.
- Milestone beat120: CLI subcommands + public A/B runners/helpers/guards/decode/serialize remain stable; Makefile suite note includes beat120.
- Beat 121: session-a --stub|--no-stub --json Watch-fixture CLI matrix via tmp_path (watch_only/heartbeat_no_queue stub→queued keeps Watch/Heartbeat markers; watch_lookalike stub→light unchanged; watch_only --no-stub→failed unchanged); contrast Beats 115/118/119.
- Beat 122: session-a --stub|--no-stub --dry-run --json Watch-fixture CLI matrix via tmp_path (watch_only/heartbeat_no_queue stub dry-run→queued wrote_item=True but plan unchanged; watch_lookalike stub dry-run→light; watch_only --no-stub dry-run→failed); contrast Beat 117 + Beat 121.
- Beat 123: session-b --decode-only --json Watch-fixture CLI matrix via tmp_path (watch_only→idle; watch_lookalike / watch_queue_heartbeat→picked; two_queue→failed); plan untouched; keep_schedule / no scheduler_delete; completes Watch trilogy with Beat 118 + Beat 119.
- Beat 124: idle-decode --json non-pickable OPEN CLI matrix via tmp_path (shipped_only / incomplete_open / broken_ready_flag / open_complete_not_ready→idle); plan untouched; keep_schedule / no scheduler_delete; distinct from Beat 113 (ready picks) + Beat 118 (Watch).
- Beat 125: session-b --dry-run --json non-pickable OPEN CLI matrix via tmp_path (shipped_only / incomplete_open / broken_ready_flag / open_complete_not_ready→idle, never dry_run); plan untouched; keep_schedule / no scheduler_delete; pairs with Beat 124 (idle-decode same fixtures).
- Beat 126: session-b --json non-pickable OPEN CLI matrix via tmp_path (shipped_only / incomplete_open / broken_ready_flag / open_complete_not_ready→idle, never dry_run); plan untouched; keep_schedule / no scheduler_delete; completes non-pickable trilogy with Beat 124 + Beat 125.
- Beat 127: session-a --no-stub|--stub --json non-pickable OPEN CLI matrix via tmp_path (shipped_only / incomplete_open / broken_ready_flag / open_complete_not_ready count as OPEN=0: --no-stub→failed plan unchanged; --stub→queued wrote_item=True open_count=1 under tmp only); contrast Beats 115 + 124–126.
- Beat 128: session-a --no-stub|--stub --dry-run --json non-pickable OPEN CLI matrix via tmp_path (shipped_only / incomplete_open --no-stub dry-run→failed wrote_item=False plan unchanged; broken_ready_flag / open_complete_not_ready --stub dry-run→queued wrote_item=True open_count=1 but plan unchanged); contrast Beat 127 (same fixtures without dry-run): dry-run may report wrote_item=True while plan bytes stay unchanged.
- Beat 129: mixed_priority cross-CLI --json matrix via tmp_path (first ready OPEN is Q2; skips incomplete): idle-decode + session-b --decode-only both pick Q2 (keep_schedule / no scheduler_delete); session-b --dry-run→dry_run Q2; session-a --stub→light b_pick_title Second ready complete item; plans unchanged on B/idle; Session A light recount only.
- Milestone beat130: CLI subcommands + public A/B runners/helpers/guards/decode/serialize remain stable; Makefile suite note includes beat130.
- Beat 131: two_open_ready cross-CLI --json matrix via tmp_path (first ready OPEN is Q1; open_count stays 2; second remains OPEN): idle-decode + session-b --decode-only both pick Q1 (keep_schedule / no scheduler_delete); session-b --dry-run→dry_run Q1; session-a --stub→light b_pick_title First ready complete item; plans unchanged. Parallel to Beat 129 (mixed_priority→Q2).
- Beat 132: one_open_ready cross-CLI --json matrix via tmp_path (single ready OPEN is Q1; open_count=1): idle-decode + session-b --decode-only both pick Q1 (keep_schedule / no scheduler_delete); session-b --dry-run→dry_run Q1; session-a --stub→light b_pick_title Add fixture unit test for queue parser; plans unchanged. Completes cross-CLI pick trilogy with Beat 129 (mixed→Q2) + Beat 131 (two_open→Q1 open_count=2).
- Beat 133: watch_lookalike cross-CLI --json matrix via tmp_path (picks real Queue OPEN Q3; never Watch lookalike; open_count=1): idle-decode + session-b --decode-only both pick Q3 (keep_schedule / no scheduler_delete); session-b --dry-run→dry_run Q3; session-a --stub→light b_pick_title Real ready Queue item; plans unchanged. Distinct from Beat 118 (idle-decode Watch-only matrix) and Beats 129/131/132 (non-Watch pick trilogy).
- Beat 134: watch_queue_heartbeat cross-CLI --json matrix via tmp_path (picks Queue OPEN Q1 while preserving Watch/Heartbeat markers; open_count=1): idle-decode + session-b --decode-only both pick Q1 (keep_schedule / no scheduler_delete; ## Watch + ## Heartbeat intact); session-b --dry-run→dry_run Q1; session-a --stub→light b_pick_title Beat19 shippable preserve item; plans unchanged. Distinct from Beat 133 (watch_lookalike→Q3) and Beats 118/119/123 (Watch CLI matrices without full cross-CLI).
- Beat 135: watch_only_lookalike cross-CLI --json matrix via tmp_path (empty Queue; Watch lookalike Q99 never B-picked; open_count=0 idle): idle-decode + session-b --decode-only both idle (keep_schedule / no scheduler_delete; ## Watch + ## Heartbeat intact); session-b --dry-run→idle (NOT dry_run); session-a --stub→queued wrote_item=True open_count=1 with ## Watch + ## Heartbeat still present after write. Idle counterpart to Beat 133 (watch_lookalike→Q3) + Beat 134 (watch_queue_heartbeat→Q1).
- Beat 136: watch_heartbeat_no_queue cross-CLI --json matrix via tmp_path (no Queue section initially; B paths idle): idle-decode + session-b --decode-only both idle (keep_schedule / no scheduler_delete; ## Watch + ## Heartbeat present; no ## Queue); session-b --dry-run→idle (NOT dry_run); session-a --stub→queued wrote_item=True open_count=1 creating ## Queue while keeping ## Watch + ## Heartbeat. Completes Watch idle trilogy with Beat 135 (watch_only_lookalike).
- Beat 137: empty_queue cross-CLI --json matrix via tmp_path (foundational baseline after Watch idle trilogy 135/136): idle-decode + session-b --decode-only both idle (keep_schedule / no scheduler_delete; open_count=0 queue 0/10; plan unchanged); session-b --dry-run→idle (NOT dry_run); session-a --stub→queued wrote_item=True open_count=1 queue 1/10 (tmp write); session-a --no-stub→rc=1 failed wrote_item=False plan unchanged.
- Beat 138: watch_queue_heartbeat_empty cross-CLI --json matrix via tmp_path (empty Queue WITH Watch/Heartbeat markers; after plain empty_queue Beat 137): idle-decode + session-b --decode-only both idle (keep_schedule / no scheduler_delete; open_count=0 queue 0/10; plan unchanged; ## Watch + ## Heartbeat kept); session-b --dry-run→idle (NOT dry_run); session-a --stub→queued wrote_item=True open_count=1 queue 1/10 with Watch/Heartbeat still present (tmp write); session-a --no-stub→rc=1 failed wrote_item=False plan unchanged markers intact.
- Beat 139: two_queue_sections fail-closed cross-CLI --json matrix via tmp_path (ambiguous dual Queue sections; every path fails; plan unchanged): idle-decode + session-b --decode-only + session-b --dry-run all failed (rc=1; open_count=0 queue 0/10; keep_schedule / no scheduler_delete; NOT dry_run); session-a --stub / --no-stub / --stub --dry-run all failed wrote_item=False; plans unchanged.
- Milestone beat140: CLI subcommands + public A/B runners/helpers/guards/decode/serialize remain stable; Makefile suite note includes beat140.
- Beat 141: shipped_only cross-CLI --json matrix via tmp_path (foundational post-milestone baseline; only SHIPPED items; open_count=0): idle-decode + session-b --decode-only both idle (keep_schedule / no scheduler_delete; open_count=0 queue 0/10; plan unchanged); session-b --dry-run→idle (NOT dry_run); session-a --stub→queued wrote_item=True open_count=1 queue 1/10 title Stub shippable change (tmp write); session-a --no-stub→rc=1 failed wrote_item=False plan unchanged.
- Beat 142: incomplete_open cross-CLI --json matrix via tmp_path (foundational non-pickable OPEN after shipped_only Beat 141; incomplete OPEN missing fields; open_count=0): idle-decode + session-b --decode-only both idle (keep_schedule / no scheduler_delete; open_count=0 queue 0/10; plan unchanged); session-b --dry-run→idle (NOT dry_run); session-a --stub→queued wrote_item=True open_count=1 queue 1/10 title Stub shippable change Heartbeat kept (tmp write); session-a --no-stub→rc=1 failed wrote_item=False plan unchanged.
- Beat 143: broken_ready_flag cross-CLI --json matrix via tmp_path (non-pickable OPEN with broken ready flag after incomplete_open Beat 142; open_count=0): idle-decode + session-b --decode-only both idle (keep_schedule / no scheduler_delete; open_count=0 queue 0/10; plan unchanged); session-b --dry-run→idle (NOT dry_run); session-a --stub→queued wrote_item=True open_count=1 queue 1/10 title Stub shippable change (tmp write); session-a --no-stub→rc=1 failed wrote_item=False plan unchanged.
- Beat 144: open_complete_not_ready cross-CLI --json matrix via tmp_path (completes non-pickable foundational quartet after shipped_only/incomplete_open/broken_ready_flag Beats 141–143; complete six-field OPEN but Ready≠yes; open_count=0): idle-decode + session-b --decode-only both idle (keep_schedule / no scheduler_delete; open_count=0 queue 0/10; plan unchanged); session-b --dry-run→idle (NOT dry_run); session-a --stub→queued wrote_item=True open_count=1 queue 1/10 title Stub shippable change (tmp write); session-a --no-stub→rc=1 failed wrote_item=False plan unchanged.
- Beat 145: queue_with_watch_heartbeat cross-CLI --json matrix via tmp_path (Watch *before* Queue + Heartbeat; open_count=0 idle baseline; stub queues while keeping Watch/Heartbeat markers): idle-decode + session-b --decode-only both idle (keep_schedule / no scheduler_delete; open_count=0 queue 0/10; plan unchanged; ## Watch + ## Heartbeat present; Watch before Queue); session-b --dry-run→idle (NOT dry_run); session-a --stub→queued wrote_item=True open_count=1 queue 1/10 title Stub shippable change with Watch/Heartbeat still present (tmp write); session-a --no-stub→rc=1 failed wrote_item=False plan unchanged markers intact.
