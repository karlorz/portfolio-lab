"""Session A/B research-implement loop (Queue producer/consumer).

Distinct from ``src.research.agent`` / ``make research`` (regime crystallizer).
Session A accepts a pluggable search/plan callback (default: ``stub_brainstorm``).
Session B accepts a pluggable implement callback (default: ``dry_run_implement``).
Empty Queue is an idle fire (``queue 0/10``); never ``scheduler_delete``.

Dry-run contract: ``dry_run_implement`` is non-mutating — it never marks
SHIPPED, never deletes Queue rows, and never calls ``scheduler_delete``.
A second Session B fire therefore picks the same OPEN again (not idle).

Optional ship path (test double only): pass ``fixture_ship_implement`` /
``make_fixture_ship_implement`` explicitly — never the default. Live prod
implement stays unwired.

CLI: ``python -m src.research_implement {session-a|session-b|idle-decode}``.
E2E dry-run (tmp plan): see module ``__main__`` examples / ``make research-implement-e2e-dry-run``.
Full pipeline (Beat 11): A stub append → B dry_run (JSON) → B fixture_ship → then
A light recount (while OPEN>=1, mid-pipeline) and/or B idle — on tmp_path only;
``make research-implement-e2e-pipeline`` / pytest ``-k beat11``.

Beat 12: CLI ``--help`` smoke (session-a / session-b / idle-decode mention
JSON / dry-run / idle fire as appropriate) and OPEN>=1 brainstorm/search_plan
spy (recount-only; custom callback never called). Proof: pytest ``-k beat12``.

Beat 13: Queue markdown round-trip (parse → serialize/write → parse preserves
six fields + ready-for-implement for OPEN items) and Session A stub append id
stability (empty→Q1; after ship/clear→new id no collide; two_open first-OPEN
unchanged). Proof: pytest ``-k beat13``.

Beat 14: Session A fail-closed on incomplete brainstorm/search_plan candidate
(missing required six fields or ready-for-implement not yes) → verdict failed,
wrote_item=False, plan unchanged, no partial OPEN append. Complete stub on
empty Queue still queues. Proof: pytest ``-k beat14``.

Beat 15: CLI ``session-a --candidate-json`` loads dict/list candidate for
brainstorm when OPEN=0; incomplete fail-closes; complete queues one OPEN;
OPEN>=1 recount-only ignores candidate-json. Proof: pytest ``-k beat15``.

Beat 16: CLI ``--candidate-json`` error paths (missing / invalid JSON / wrong
type) → clear non-zero failure, no plan mutation; sequential double-OPEN ship
on tmp_path (ship → ship → idle) never ``scheduler_delete``. Proof: pytest
``-k beat16``.

Beat 17: Session B decode_only default / CLI session-b never calls implement;
``decode_only=False`` dry_run path invokes ``dry_run_implement``; fixture ship
only when opted in (``decode_only=False`` + ``fixture_ship_implement``).
Proof: pytest ``-k beat17``.

Beat 18: Public API export smoke (``import src.research_implement`` + key
``__all__`` / getattr names: SessionAResult, SessionResult, queue helpers,
dry_run_implement, fixture_ship, incomplete_candidate_reasons, …) and Makefile
help/echo listing ``test-research-implement``, ``e2e-dry-run``, ``e2e-pipeline``.
Proof: pytest ``-k beat18``.

Beat 19: When rewriting ``## Queue`` (append / ship / serialize via
``write_queue_section``), preserve other markdown sections (Watch / Project
Work / Heartbeat). Fixture ``watch_queue_heartbeat*.md`` + pytest ``-k beat19``:
A append or B ship updates Queue while non-Queue markers stay present.

Beat 20: If markdown has no ``## Queue`` section, Session A append /
``write_queue_section`` creates one without destroying Watch / Heartbeat /
YAML front matter. Fixture ``watch_heartbeat_no_queue.md`` + empty file:
A stub append → ``## Queue`` with one OPEN; prior beat19 stays green.
Proof: pytest ``-k beat20``.

Beat 21: More than one ``## Queue`` heading is fail-closed
(``AmbiguousQueueError``): parse / write / append raise; Session A/B return
failed results without mutating the plan (no silent merge). Fixture
``two_queue_sections.md`` + pytest ``-k beat21``. Prior beat20 stays green.

JSON: ``SessionResult.to_dict()`` (aliases ``to_json_dict`` / ``session_b_result_dict``)
is the shared Session B ``--json`` / test contract for idle / decode_only / dry_run /
shipped. Session A: ``SessionAResult.to_dict()`` (aliases ``to_json_dict`` /
``session_a_result_dict``) covers append/queued vs recount-only/light for
``session-a --json``.


Beat 22: see ``docs/research-implement-ab-side-dev.md`` for side-dev CLI/make notes.

Beat 23: CLI ``session-a`` / ``session-b`` / ``idle-decode`` with missing
``--plan`` / ``--log`` path → clear non-zero ``SystemExit`` (``file not found``);
does not create the plan file. Existing plan path still works. Proof: pytest
``-k beat23``. Prior beat22 stays green.

Beat 40: CLI help contracts — top-level lists session-a/session-b/idle-decode; idle-decode has no --dry-run; session-b does. Proof: pytest ``-k beat40``.

Beat 41–42: side-dev help wording / no CLI ``--implement``; Makefile lists ``test-research-implement`` + e2e targets; B/idle help never ``scheduler_delete``. Proof: pytest ``-k beat41`` / ``beat42``.

Beat 43: side-dev guide pins make/ship contracts; package docs mention Beat 40 help. Proof: pytest ``-k beat43``.

Beat 44: Session B dry-run message includes decode-pick six-field report; ``session-b --json`` help matches ``SessionResult.to_dict``. Proof: pytest ``-k beat44``.

Beat 50: broken ready-for-implement flags idle via CLI; public API exports remain stable through A/B side-dev. Proof: pytest ``-k beat50``.

Beat 51: ready-for-implement aliases (YES/Y/TRUE/1) remain B-pickable via CLI; non-aliases still idle. Proof: pytest ``-k beat51``.

Beat 52: ambiguous dual-Queue plans fail closed via CLI with plan unchanged. Proof: pytest ``-k beat52``.

Beat 53: empty Queue + Watch/Project/Heartbeat CLI idle/append preserves non-Queue markers. Proof: pytest ``-k beat53``.

Beat 54: OPEN Watch/Heartbeat plan CLI pick/dry-run preserves non-Queue markers. Proof: pytest ``-k beat54``.

Beat 55: no-Queue Watch/Heartbeat plans: B idle CLI; A stub CLI creates Queue without dropping markers. Proof: pytest ``-k beat55``.

Beat 56: light/dry_run/failed CLI JSON key contracts stay locked; Watch-before-Queue A→B CLI preserve. Proof: pytest ``-k beat56``.

Beat 57: Session A failed JSON key contract + AmbiguousQueueError export; dry-run A no write. Proof: pytest ``-k beat57``.

Beat 58: empty-queue ``session-b --dry-run`` stays idle; queue helper exports remain public. Proof: pytest ``-k beat58``.

Beat 59: two-OPEN dry-run picks first only; plan stays dual-OPEN. Proof: pytest ``-k beat59``.

Beat 60: Watch lookalike dry-run/pick stays Queue-only. Proof: pytest ``-k beat60``.

Beat 61: non-pickable plans stay idle under ``session-b --dry-run``. Proof: pytest ``-k beat61``.

Beat 62: not-ready / broken-ready dry-run idle; next_queue_id export. Proof: pytest ``-k beat62``.

Beat 63: mixed_priority dry-run/pick first ready; count_queue_headings export. Proof: pytest ``-k beat63``.

Beat 64: one_open dry-run/pick; format_queue_item + require_unique_queue exports. Proof: pytest ``-k beat64``.

Beat 65: dual-Queue dry-run still fail-closed; write_queue_section export. Proof: pytest ``-k beat65``.

Beat 66: --no-stub light/failed CLI paths; implement helper exports. Proof: pytest ``-k beat66``.

Beat 67: --no-stub --dry-run light/failed; decode helper exports. Proof: pytest ``-k beat67``.

Beat 68: first_b_pick / incomplete_candidate_reasons contracts; idle-decode decode-pick message. Proof: pytest ``-k beat68``.

Beat 69: session_*_result_dict helpers align with CLI --json. Proof: pytest ``-k beat69``.

Beat 70: path runners exported; light/failed Session A JSON keys locked. Proof: pytest ``-k beat70``.

Beat 71: Session B dry_run/idle/failed JSON keys; to_dict aliases. Proof: pytest ``-k beat71``.

Beat 72: Session A to_dict aliases align across queued/light/failed. Proof: pytest ``-k beat72``.

Beat 73: result/queue dataclasses remain public API. Proof: pytest ``-k beat73``.

Beat 74: QueueItem pickability + format/parse round-trip. Proof: pytest ``-k beat74``.

Beat 75: incomplete/shipped fixtures never B-pickable; count_open=0. Proof: pytest ``-k beat75``.

Beat 76: not-ready / broken-ready fixtures never pickable. Proof: pytest ``-k beat76``.

Beat 77: mixed_priority first_b_pick skips incomplete. Proof: pytest ``-k beat77``.

Beat 78: two_open_ready first_b_pick is first of two. Proof: pytest ``-k beat78``.

Beat 79: one_open vs empty_queue pick/count contract. Proof: pytest ``-k beat79``.

Beat 80: milestone — CLI help + public API export smoke through beat80. Proof: pytest ``-k beat80``.
Beat 81: watch_lookalike / watch_only first_b_pick + mark_item_shipped export. Proof: pytest ``-k beat81``.
Beat 82: subcommand help plan/json + render_queue_count/QUEUE_CAPACITY export. Proof: pytest ``-k beat82``.
Beat 83: non-pickable fixtures first_b_pick=None; serialize_queue_item(s) export. Proof: pytest ``-k beat83``.
Beat 84: pickable fixtures first_b_pick ids; is_complete_six_field/is_open_status export. Proof: pytest ``-k beat84``.
Beat 85: producer/decode help flags + REQUIRED_FIELDS export. Proof: pytest ``-k beat85``.
Beat 86: idle ``queue 0/10`` + ``scheduler_delete`` guard; ``SchedulerDeleteForbidden`` export;
decode report starts with ``decode pick``. Proof: pytest ``-k beat86``.
Beat 87: two_queue AmbiguousQueueError + heading count; decode helpers export. Proof: pytest ``-k beat87``.
Beat 88: Watch/Heartbeat fixture idle/pick; SESSION_A/B_RESULT_KEYS export. Proof: pytest ``-k beat88``.
Beat 89: incomplete_candidate_reasons contract; SESSION_*_JSON_KEYS export. Proof: pytest ``-k beat89``.
Beat 90: milestone — CLI help + public API export smoke through beat90. Proof: pytest ``-k beat90``.
Beat 91: empty append next Q1 pickable; append/format/write exports. Proof: pytest ``-k beat91``.
Beat 92: mark_item_shipped clears pick; stub_brainstorm + implement helpers export. Proof: pytest ``-k beat92``.
Beat 93: is_ready_yes aliases; search_plan aliases; session_*_result_dict export. Proof: pytest ``-k beat93``.
Beat 94: result_dict/to_dict JSON key align; path runners export. Proof: pytest ``-k beat94``.
Beat 95: serialize round-trip pickable; count_queue_headings fixtures. Proof: pytest ``-k beat95``.
Beat 96: decode_fields + format_decode_report line contract. Proof: pytest ``-k beat96``.
Beat 97: next_queue_id progression; require_unique fail-closed. Proof: pytest ``-k beat97``.
Beat 98: is_b_pickable fixture matrix; complete/open/ready exports. Proof: pytest ``-k beat98``.
Beat 99: count_open fixture matrix + render_queue_count align. Proof: pytest ``-k beat99``.
Beat 100: milestone — CLI help + public API export smoke through beat100. Proof: pytest ``-k beat100``.
Beat 101: format_queue_item round-trip; dry_run/fixture-ship exports. Proof: pytest ``-k beat101``.
Beat 102: write_queue_section transplant; default_implement alias dry-run. Proof: pytest ``-k beat102``.
Beat 103: serialize_queue_items two_open round-trip. Proof: pytest ``-k beat103``.
Beat 104: QueueItem.field_map covers REQUIRED_FIELDS. Proof: pytest ``-k beat104``.
Beat 105: to_json_dict aliases to_dict; SessionBResult is SessionResult. Proof: pytest ``-k beat105``.
Beat 106: keep_schedule True; never scheduler_delete_called. Proof: pytest ``-k beat106``.
Beat 107: Session A queued/light/failed to_dict matrix. Proof: pytest ``-k beat107``.
Beat 108: Session B idle/picked/failed to_dict matrix. Proof: pytest ``-k beat108``.
Beat 109: first_b_pick/count_open fixture consistency; __all__ unique. Proof: pytest ``-k beat109``.
Beat 110: milestone — CLI help + public API export smoke through beat110. Proof: pytest ``-k beat110``.
Beat 111: idle fixtures queue 0/10 + decode_report None; picked decode_report == format_decode_report. Proof: pytest ``-k beat111``.
Beat 112: path runners A write / B idle no-touch. Proof: pytest ``-k beat112``.
Beat 113: idle-decode CLI --json idle/picked/failed matrix. Proof: pytest ``-k beat113``.
Beat 114: session-b --decode-only --json idle/picked/failed matrix. Proof: pytest ``-k beat114``.
Beat 115: session-a --stub|--no-stub --json CLI matrix (queued/light/failed) via tmp_path. Proof: pytest ``-k beat115``.
Beat 116: session-b --dry-run --json CLI matrix (idle/dry_run/failed) via tmp_path; plan untouched. Proof: pytest ``-k beat116``.
Beat 117: session-a --stub|--no-stub --dry-run --json CLI matrix via tmp_path (queued wrote_item=True but plan unchanged / light / failed). Contrast Beat 115. Proof: pytest ``-k beat117``.
Beat 118: idle-decode --json Watch-fixture CLI matrix via tmp_path (watch_only→idle; watch_lookalike/heartbeat→picked; two_queue→failed). Contrast Beat 113. Proof: pytest ``-k beat118``.
Beat 119: session-b --dry-run --json Watch-fixture CLI matrix via tmp_path (watch_only→idle; lookalike/heartbeat→dry_run; two_queue→failed). Contrast Beat 116 + Beat 118. Proof: pytest ``-k beat119``.
Beat 120: milestone — CLI help + public API export smoke through beat120. Proof: pytest ``-k beat120``.
Beat 121: session-a --stub|--no-stub --json Watch-fixture CLI matrix via tmp_path (queued keeps markers / light / failed). Contrast Beats 115/118/119. Proof: pytest ``-k beat121``.
Beat 122: session-a --stub|--no-stub --dry-run --json Watch-fixture CLI matrix via tmp_path (queued wrote_item=True but plan unchanged / light / failed). Contrast Beat 117 + Beat 121. Proof: pytest ``-k beat122``.
Beat 123: session-b --decode-only --json Watch-fixture CLI matrix via tmp_path (watch_only→idle; lookalike/heartbeat→picked; two_queue→failed). Completes Watch trilogy with Beat 118 + Beat 119. Proof: pytest ``-k beat123``.
Beat 124: idle-decode --json non-pickable OPEN CLI matrix via tmp_path (shipped_only/incomplete_open/broken_ready_flag/open_complete_not_ready→idle); plan untouched; keep_schedule / no scheduler_delete. Distinct from Beat 113 + Beat 118. Proof: pytest ``-k beat124``.
Beat 125: session-b --dry-run --json non-pickable OPEN CLI matrix via tmp_path (shipped_only/incomplete_open/broken_ready_flag/open_complete_not_ready→idle, never dry_run); plan untouched; keep_schedule / no scheduler_delete. Pairs with Beat 124. Proof: pytest ``-k beat125``.
Beat 126: session-b --json non-pickable OPEN CLI matrix via tmp_path (shipped/incomplete/broken/not_ready→idle, never dry_run). Completes non-pickable trilogy with Beat 124 + Beat 125. Proof: pytest ``-k beat126``.
Beat 127: session-a --no-stub|--stub --json non-pickable OPEN CLI matrix via tmp_path (non-pickable=OPEN=0: --no-stub→failed; --stub→queued under tmp). Contrast Beats 115 + 124–126. Proof: pytest ``-k beat127``.
Beat 128: session-a --no-stub|--stub --dry-run --json non-pickable OPEN CLI matrix via tmp_path (--no-stub dry-run→failed; --stub dry-run→queued wrote_item=True but plan unchanged). Contrast Beat 127. Proof: pytest ``-k beat128``.
Beat 129: mixed_priority cross-CLI --json matrix via tmp_path (first ready OPEN=Q2): idle-decode + session-b --decode-only pick Q2; session-b --dry-run→dry_run Q2; session-a --stub→light b_pick_title Second ready complete item; plans unchanged. Proof: pytest ``-k beat129``.
Beat 130: milestone — CLI help + public API export smoke through beat130. Proof: pytest ``-k beat130``.
Beat 131: two_open_ready cross-CLI --json matrix via tmp_path (first ready OPEN=Q1; open_count=2): idle-decode + session-b --decode-only pick Q1; session-b --dry-run→dry_run Q1; session-a --stub→light b_pick_title First ready complete item; plans unchanged. Parallel to Beat 129. Proof: pytest ``-k beat131``.
Beat 132: one_open_ready cross-CLI --json matrix via tmp_path (single ready OPEN=Q1; open_count=1): idle-decode + session-b --decode-only pick Q1; session-b --dry-run→dry_run Q1; session-a --stub→light b_pick_title Add fixture unit test for queue parser; plans unchanged. Completes cross-CLI pick trilogy with Beat 129 + Beat 131. Proof: pytest ``-k beat132``.
Beat 133: watch_lookalike cross-CLI --json matrix via tmp_path (real Queue OPEN=Q3; never Watch lookalike; open_count=1): idle-decode + session-b --decode-only pick Q3; session-b --dry-run→dry_run Q3; session-a --stub→light b_pick_title Real ready Queue item; plans unchanged. Distinct from Beat 118 + Beats 129/131/132. Proof: pytest ``-k beat133``.
Beat 134: watch_queue_heartbeat cross-CLI --json matrix via tmp_path (Queue OPEN=Q1; preserve Watch/Heartbeat markers; open_count=1): idle-decode + session-b --decode-only pick Q1; session-b --dry-run→dry_run Q1; session-a --stub→light b_pick_title Beat19 shippable preserve item; plans unchanged. Distinct from Beat 133 + Beats 118/119/123. Proof: pytest ``-k beat134``.
Beat 135: watch_only_lookalike cross-CLI --json matrix via tmp_path (empty Queue; Watch lookalike never B-picked): B paths idle; session-a --stub queues while keeping Watch/Heartbeat. Idle counterpart to Beat 133 + Beat 134. Proof: pytest ``-k beat135``.
Beat 136: watch_heartbeat_no_queue cross-CLI --json matrix via tmp_path (no Queue initially; B idle; session-a --stub creates Queue keeping Watch/Heartbeat). Completes Watch idle trilogy with Beat 135. Proof: pytest ``-k beat136``.
Beat 137: empty_queue cross-CLI --json matrix via tmp_path (foundational baseline after Watch idle trilogy 135/136): B/idle paths idle keep_schedule; session-a --stub→queued write; --no-stub→failed plan unchanged. Proof: pytest ``-k beat137``.
Beat 138: watch_queue_heartbeat_empty cross-CLI --json matrix via tmp_path (empty Queue WITH Watch/Heartbeat markers; after plain empty_queue Beat 137): B/idle paths idle keep_schedule Watch/Heartbeat kept; session-a --stub→queued write keep markers; --no-stub→failed plan unchanged markers intact. Proof: pytest ``-k beat138``.
Beat 139: two_queue_sections fail-closed cross-CLI --json matrix via tmp_path (ambiguous dual Queue sections; every path fails; plan unchanged): idle-decode + session-b --decode-only + session-b --dry-run all failed keep_schedule; session-a --stub/--no-stub/--stub --dry-run all failed wrote_item=False; plans unchanged. Proof: pytest ``-k beat139``.
Beat 140: milestone — CLI help + public API export smoke through beat140. Proof: pytest ``-k beat140``.
Beat 141: shipped_only cross-CLI --json matrix via tmp_path (foundational post-milestone baseline; only SHIPPED; open_count=0): B/idle paths idle keep_schedule; session-a --stub→queued write title Stub shippable change; --no-stub→failed plan unchanged. Proof: pytest ``-k beat141``.
Beat 142: incomplete_open cross-CLI --json matrix via tmp_path (foundational non-pickable OPEN after shipped_only Beat 141; incomplete OPEN missing fields; open_count=0): B/idle paths idle keep_schedule; session-a --stub→queued write title Stub shippable change Heartbeat kept; --no-stub→failed plan unchanged. Proof: pytest ``-k beat142``.
Beat 143: broken_ready_flag cross-CLI --json matrix via tmp_path (non-pickable OPEN with broken ready flag after incomplete_open Beat 142; open_count=0): B/idle paths idle keep_schedule; session-a --stub→queued write title Stub shippable change; --no-stub→failed plan unchanged. Proof: pytest ``-k beat143``.
Beat 144: open_complete_not_ready cross-CLI --json matrix via tmp_path (completes non-pickable foundational quartet after shipped_only/incomplete_open/broken_ready_flag Beats 141–143; complete six-field OPEN but Ready≠yes; open_count=0): B/idle paths idle keep_schedule; session-a --stub→queued write title Stub shippable change; --no-stub→failed plan unchanged. Proof: pytest ``-k beat144``.
Beat 145: queue_with_watch_heartbeat cross-CLI --json matrix via tmp_path (Watch *before* Queue + Heartbeat; open_count=0 idle baseline; stub queues while keeping Watch/Heartbeat markers): B/idle paths idle keep_schedule Watch/Heartbeat present Watch before Queue; session-a --stub→queued write title Stub shippable change keep markers; --no-stub→failed plan unchanged markers intact. Proof: pytest ``-k beat145``.
Beat 146: contract_spec cross-CLI --json matrix via tmp_path (foundational contract-plan baseline; open_count=0 idle; stub queues; no-stub fails): B/idle paths idle keep_schedule; session-a --stub→queued write title Stub shippable change; --no-stub→failed plan unchanged. Proof: pytest ``-k beat146``.
Beat 147: shipped_only session-a --stub|--no-stub --dry-run --json matrix via tmp_path (dry-run extension of Beat 141; stub wrote_item=True but plan unchanged; no-stub fails): idle+B dry-run idle keep_schedule; session-a --stub --dry-run→queued wrote_item=True title Stub shippable change plan UNCHANGED; --no-stub --dry-run→failed wrote_item=False plan unchanged. Contrast Beat 141. Proof: pytest ``-k beat147``.
Beat 148: incomplete_open session-a --stub|--no-stub --dry-run --json matrix via tmp_path (dry-run extension of Beat 142; mirrors Beat 147 shipped_only dry-run shape; stub wrote_item=True but plan unchanged; no-stub fails): idle+B dry-run idle keep_schedule; session-a --stub --dry-run→queued wrote_item=True title Stub shippable change plan UNCHANGED; --no-stub --dry-run→failed wrote_item=False plan unchanged. Contrast Beat 142. Proof: pytest ``-k beat148``.
Beat 149: broken_ready_flag session-a --stub|--no-stub --dry-run --json matrix via tmp_path (dry-run extension of Beat 143; mirrors Beats 147–148 dry-run shape; stub wrote_item=True but plan unchanged; no-stub fails): idle+B dry-run idle keep_schedule; session-a --stub --dry-run→queued wrote_item=True title Stub shippable change plan UNCHANGED; --no-stub --dry-run→failed wrote_item=False plan unchanged. Contrast Beat 143. Proof: pytest ``-k beat149``.
Beat 150: milestone — CLI help + public API export smoke through beat150. Proof: pytest ``-k beat150``.
Beat 151: open_complete_not_ready session-a --stub|--no-stub --dry-run --json matrix via tmp_path (completes non-pickable dry-run quartet after shipped_only/incomplete_open/broken_ready_flag Beats 147–149; stub wrote_item=True but plan unchanged; no-stub fails): idle+B dry-run idle keep_schedule; session-a --stub --dry-run→queued wrote_item=True title Stub shippable change plan UNCHANGED; --no-stub --dry-run→failed wrote_item=False plan unchanged. Contrast Beat 144. Proof: pytest ``-k beat151``.
Beat 152: contract_spec session-a --stub|--no-stub --dry-run --json matrix via tmp_path (dry-run extension of Beat 146; mirrors Beats 147–151 dry-run shape; stub wrote_item=True but plan unchanged; no-stub fails): idle+B dry-run idle keep_schedule; session-a --stub --dry-run→queued wrote_item=True title Stub shippable change plan UNCHANGED; --no-stub --dry-run→failed wrote_item=False plan unchanged. Contrast Beat 146. Proof: pytest ``-k beat152``.
Beat 153: queue_with_watch_heartbeat session-a --stub|--no-stub --dry-run --json matrix via tmp_path (dry-run extension of Beat 145; Watch *before* Queue + Heartbeat; stub wrote_item=True but plan unchanged markers intact; no-stub fails): idle+B dry-run idle keep_schedule Watch/Heartbeat present Watch before Queue; session-a --stub --dry-run→queued wrote_item=True title Stub shippable change plan UNCHANGED markers intact; --no-stub --dry-run→failed wrote_item=False plan unchanged markers intact. Contrast Beat 145. Proof: pytest ``-k beat153``.
Beat 154: empty_queue session-a --stub|--no-stub --dry-run --json matrix via tmp_path (dry-run extension of Beat 137; mirrors Beats 147–153 dry-run shape; stub wrote_item=True but plan unchanged; no-stub fails): idle+B dry-run idle keep_schedule; session-a --stub --dry-run→queued wrote_item=True title Stub shippable change plan UNCHANGED; --no-stub --dry-run→failed wrote_item=False plan unchanged. Contrast Beat 137. Proof: pytest ``-k beat154``.
Beat 155: watch_queue_heartbeat_empty session-a --stub|--no-stub --dry-run --json matrix via tmp_path (dry-run extension of Beat 138; mirrors Beat 154 empty_queue dry-run + marker keep from 153; stub wrote_item=True but plan unchanged markers intact; no-stub fails): idle+B dry-run idle keep_schedule Watch/Heartbeat present; session-a --stub --dry-run→queued wrote_item=True title Stub shippable change plan UNCHANGED markers intact; --no-stub --dry-run→failed wrote_item=False plan unchanged markers intact. Contrast Beat 138. Proof: pytest ``-k beat155``.
Beat 156: watch_heartbeat_no_queue session-a --stub|--no-stub --dry-run --json matrix via tmp_path (dry-run extension of Beat 136; no ## Queue initially; stub wrote_item=True but plan UNCHANGED still no Queue; no-stub fails): idle+B dry-run idle keep_schedule Watch/Heartbeat present no Queue; session-a --stub --dry-run→queued wrote_item=True title Stub shippable change plan UNCHANGED still no Queue markers intact; --no-stub --dry-run→failed wrote_item=False plan unchanged. Contrast Beat 136 (stub creates ## Queue). Proof: pytest ``-k beat156``.
Beat 157: watch_only_lookalike session-a --stub|--no-stub --dry-run --json matrix via tmp_path (dry-run extension of Beat 135; Watch lookalike never B-picked; stub wrote_item=True but plan UNCHANGED keep markers; no-stub fails): idle+B dry-run idle keep_schedule Watch/Heartbeat present; session-a --stub --dry-run→queued wrote_item=True title Stub shippable change plan UNCHANGED markers intact; --no-stub --dry-run→failed wrote_item=False plan unchanged. Contrast Beat 135 (stub writes). Proof: pytest ``-k beat157``.
Beat 158: two_queue_sections session-a --stub|--no-stub --dry-run --json fail-closed matrix via tmp_path (dry-run extension of Beat 139; every path still fails; wrote_item=False; plan unchanged): idle+B dry-run failed keep_schedule (B NOT dry_run); session-a --stub|--no-stub --dry-run both failed wrote_item=False plan unchanged. Contrast Beat 139 (broader cross-CLI; Beat 158 focuses dry-run fail-closed). Proof: pytest ``-k beat158``.
Beat 159: watch_lookalike dry-run / light --json matrix via tmp_path (pre-milestone pickable contrast after fail-closed Beat 158; B dry-run picks real Queue OPEN Q3 not Watch lookalike; Session A --stub|--no-stub --dry-run both light): idle→picked open_count=1 keep_schedule; B --dry-run→dry_run open_count=1 keep_schedule; session-a --stub|--no-stub --dry-run both light wrote_item=False open_count=1 b_pick_title Real ready Queue item; plans unchanged. Contrast Beat 158 (fail-closed) + Beat 133 (broader cross-CLI; Beat 159 focuses dry-run / light). Proof: pytest ``-k beat159``.
Beat 160: milestone — CLI help + public API export smoke through beat160. Proof: pytest ``-k beat160``.
Beat 161: one_open_ready dry-run / light --json matrix via tmp_path (post-milestone pickable baseline; mirrors Beat 159 watch_lookalike dry-run/light shape; B dry-run picks Q1; Session A --stub|--no-stub --dry-run both light): idle→picked open_count=1 keep_schedule; B --dry-run→dry_run open_count=1 keep_schedule; session-a --stub|--no-stub --dry-run both light wrote_item=False open_count=1 b_pick_title Add fixture unit test for queue parser; plans unchanged. Contrast Beat 159 (watch_lookalike→Q3) + Beat 160 (milestone). Proof: pytest ``-k beat161``.
Beat 162: two_open_ready dry-run / light --json matrix via tmp_path (mirrors Beat 161 one_open_ready dry-run/light shape with open_count=2 / queue 2/10; B dry-run picks Q1; Session A --stub|--no-stub --dry-run both light): idle→picked open_count=2 keep_schedule; B --dry-run→dry_run open_count=2 keep_schedule; session-a --stub|--no-stub --dry-run both light wrote_item=False open_count=2 b_pick_title First ready complete item; plans unchanged. Contrast Beat 161 (one_open_ready open_count=1). Proof: pytest ``-k beat162``.
Beat 163: mixed_priority dry-run / light --json matrix via tmp_path (mirrors Beat 161/162 dry-run/light shape; incomplete Q1 skipped; B picks first complete six-field ready OPEN Q2; open_count=1 / queue 1/10; Session A --stub|--no-stub --dry-run both light): idle→picked open_count=1 keep_schedule; B --dry-run→dry_run open_count=1 keep_schedule; session-a --stub|--no-stub --dry-run both light wrote_item=False open_count=1 b_pick_title Second ready complete item; plans unchanged. Contrast Beat 162 (two_open_ready→Q1) + Beat 129 (same fixture broader). Proof: pytest ``-k beat163``.
Beat 164: watch_queue_heartbeat dry-run / light --json matrix via tmp_path (mirrors Beat 161–163 dry-run/light shape with Watch/Heartbeat markers kept; B dry-run picks Q1; Session A --stub|--no-stub --dry-run both light): idle→picked open_count=1 keep_schedule markers intact; B --dry-run→dry_run open_count=1 keep_schedule markers intact; session-a --stub|--no-stub --dry-run both light wrote_item=False open_count=1 b_pick_title Beat19 shippable preserve item; plans unchanged. Contrast Beat 163 (mixed_priority→Q2 no markers) + Beat 134 (same fixture broader). Proof: pytest ``-k beat164``.
Beat 165: multi-fixture pickable dry-run / light --json smoke via tmp_path (closes pickable dry-run/light series across watch_lookalike / one_open_ready / two_open_ready / mixed_priority / watch_queue_heartbeat): for each idle→picked correct open_count/queue/item_id keep_schedule; B --dry-run→dry_run same pick ids/counts keep_schedule; session-a --stub|--no-stub --dry-run both light wrote_item=False correct open_count/queue/b_pick_title; watch_queue_heartbeat keeps Watch/Heartbeat/BEAT19 markers; plans unchanged. Contrast Beat 164 (single watch_queue_heartbeat) + Beats 159/161–163 (single-fixture series); Beat 165 consolidates. Proof: pytest ``-k beat165``.
Beat 166: multi-fixture idle dry-run --json smoke via tmp_path (idle counterpart to Beat 165 pickable smoke; consolidates Beats 154–157 across empty_queue / watch_queue_heartbeat_empty / watch_heartbeat_no_queue / watch_only_lookalike): for each idle-decode + B --dry-run→both idle open_count=0 queue 0/10 keep_schedule no scheduler_delete plan unchanged (B NOT dry_run); session-a --stub --dry-run→queued wrote_item=True open_count=1 queue 1/10 title Stub shippable change plan UNCHANGED; --no-stub --dry-run→failed wrote_item=False plan unchanged; Watch fixtures keep Watch/Heartbeat; watch_heartbeat_no_queue still no ## Queue. Contrast Beat 165 (pickable multi-fixture light) + Beats 154–157 (single-fixture idle dry-run); Beat 166 consolidates idle series. Proof: pytest ``-k beat166``.
Beat 167: multi-fixture non-pickable dry-run --json smoke via tmp_path (consolidates Beats 147–153 across shipped_only / incomplete_open / broken_ready_flag / open_complete_not_ready / contract_spec / queue_with_watch_heartbeat; third consolidation after 165 pickable + 166 idle): for each idle-decode + B --dry-run→both idle open_count=0 queue 0/10 keep_schedule no scheduler_delete plan unchanged (B NOT dry_run); session-a --stub --dry-run→queued wrote_item=True open_count=1 queue 1/10 title Stub shippable change plan UNCHANGED; --no-stub --dry-run→failed wrote_item=False plan unchanged; queue_with_watch_heartbeat keeps Watch/Heartbeat with Watch before Queue. Contrast Beat 166 (idle Watch/empty smoke) + Beats 147–153 (single-fixture non-pickable dry-run); Beat 167 consolidates non-pickable series. Proof: pytest ``-k beat167``.
Beat 168: multi-fixture verdict-spectrum dry-run / decode --json smoke via tmp_path (crosses pickable / idle / fail-closed after Beats 165–167 class consolidations + Beat 158 fail-closed alone): one_open_ready idle-decode + session-b --decode-only→picked Q1 keep_schedule; B --dry-run→dry_run Q1 keep_schedule; session-a --stub|--no-stub --dry-run both light wrote_item=False open_count=1 b_pick_title Add fixture unit test for queue parser; empty_queue idle-decode + B --dry-run→both idle open_count=0 queue 0/10 keep_schedule (B NOT dry_run); session-a --stub --dry-run→queued wrote_item=True title Stub shippable change plan UNCHANGED; --no-stub --dry-run→failed wrote_item=False; two_queue_sections idle-decode + B --dry-run→both failed (rc=1; keep_schedule; B NOT dry_run); session-a --stub|--no-stub --dry-run both failed wrote_item=False; plans unchanged. Contrast Beats 165–167 (class consolidations) + Beat 158 (fail-closed alone); Beat 168 crosses pickable/idle/fail-closed. Proof: pytest ``-k beat168``.
Beat 169: multi-fixture idle-decode ≡ session-b --decode-only --json smoke via tmp_path (closes idle-decode ≡ session-b --decode-only multi-fixture gap after Beat 165 pickable dry-run/light + Beat 168 spectrum with one decode-only): pickables (watch_lookalike / one_open_ready / two_open_ready / mixed_priority / watch_queue_heartbeat) idle-decode + session-b --decode-only both →picked same open_count/queue/item_id keep_schedule no scheduler_delete plan unchanged; watch_queue_heartbeat keeps Watch/Heartbeat/BEAT19 markers; idle/fail (empty_queue / watch_only_lookalike idle; two_queue_sections failed) same verdict/rc/open_count/queue keep_schedule (decode-only NOT dry_run). Contrast Beat 165 (pickable dry-run/light) + Beat 168 (spectrum with one decode-only); Beat 169 is pre-milestone decode-only equivalence smoke. Next beat 170 = milestone. Proof: pytest ``-k beat169``.
Beat 170: milestone — CLI help + public API export smoke through beat170. Proof: pytest ``-k beat170``.
Beat 171: multi-fixture Session A --stub write vs --stub --dry-run no-write --json contrast via tmp_path (post-milestone; dry-run reports wrote_item without mutating plan bytes; empty_queue / watch_queue_heartbeat_empty / watch_heartbeat_no_queue / watch_only_lookalike): --stub --json→queued wrote_item=True plan CHANGED ## Queue present (Watch keep markers; watch_heartbeat_no_queue creates Queue); --stub --dry-run→queued wrote_item=True plan UNCHANGED (watch_heartbeat_no_queue still no Queue). Contrast Beat 170 (milestone) + Beats 154–157/166 (dry-run-only) + Beat 136 (stub creates Queue); Beat 171 pairs live write vs dry-run no-write. Proof: pytest ``-k beat171``.
Beat 172: multi-fixture non-pickable Session A --stub write vs --stub --dry-run no-write --json contrast via tmp_path (parallel to Beat 171 idle contrast; shipped_only / incomplete_open / broken_ready_flag / open_complete_not_ready / contract_spec / queue_with_watch_heartbeat): --stub --json→queued wrote_item=True open_count=1 queue 1/10 title Stub shippable change plan CHANGED; --stub --dry-run→queued wrote_item=True plan UNCHANGED; queue_with_watch_heartbeat keeps Watch/Heartbeat with Watch before Queue. Contrast Beat 171 (idle stub write vs dry-run) + Beat 167 (non-pickable dry-run-only smoke); Beat 172 pairs live write vs dry-run for non-pickables. Proof: pytest ``-k beat172``.
Beat 173: multi-fixture pickable Session A light no-write --json contrast via tmp_path (completes 171/172/173 trilogy: idle write / non-pickable write / pickable never-write; watch_lookalike / one_open_ready / two_open_ready / mixed_priority / watch_queue_heartbeat): --stub|--no-stub --json (NO dry-run) both light wrote_item=False correct open_count/queue/b_pick_title plan UNCHANGED; --stub|--no-stub --dry-run --json both light same shape plan UNCHANGED; watch_queue_heartbeat keeps Watch/Heartbeat/BEAT19 markers. Contrast Beat 171 (idle stub mutates) + Beat 172 (non-pickable stub mutates) + Beat 165 (pickable dry-run light only); Beat 173 proves pickable OPEN>=1 never appends for stub|no-stub × dry-run|live. Proof: pytest ``-k beat173``.
Beat 174: multi-fixture idle/non-pickable/fail-closed Session A --no-stub live vs --no-stub --dry-run fail --json contrast via tmp_path (counterpart to Beats 171–173 stub write / light trilogy; empty_queue / watch_only_lookalike / shipped_only / incomplete_open / broken_ready_flag / open_complete_not_ready / contract_spec / queue_with_watch_heartbeat / two_queue_sections): --no-stub --json (NO dry-run)→rc=1 failed wrote_item=False plan UNCHANGED; --no-stub --dry-run --json→rc=1 failed wrote_item=False plan UNCHANGED; Watch fixtures keep Watch/Heartbeat; queue_with_watch_heartbeat keeps Watch before Queue. Contrast Beats 171–173 (stub write / light) + prior dry-run fail matrices; Beat 174 proves --no-stub fails live and dry-run without mutating plans. Proof: pytest ``-k beat174``.
Beat 175: multi-fixture Session A --candidate-json write vs --candidate-json --dry-run no-write --json contrast via tmp_path (same idle fixtures as Beat 171; candidate complete_candidate.json copied to tmp_path): --candidate-json --json→queued wrote_item=True open_count=1 queue 1/10 title Complete candidate fixture plan CHANGED ## Queue present (Watch keep markers; watch_heartbeat_no_queue creates Queue); --candidate-json --dry-run→queued wrote_item=True plan UNCHANGED (watch_heartbeat_no_queue still no Queue). Contrast Beat 171 (stub write vs dry-run) + Beat 174 (--no-stub fail); Beat 175 is candidate-json write vs dry-run on idle fixtures. Proof: pytest ``-k beat175``.
Beat 176: multi-fixture idle Session A --candidate-json **incomplete** fail live vs --candidate-json --dry-run fail --json contrast via tmp_path (same idle fixtures as Beat 171/175; candidate incomplete_candidate.json copied to tmp_path): --candidate-json --json (NO dry-run)→rc=1 ok=False verdict=failed wrote_item=False open_count=0 queue 0/10 title Incomplete candidate fixture plan UNCHANGED; --candidate-json --dry-run --json→same fail shape plan UNCHANGED; Watch fixtures keep Watch/Heartbeat; watch_heartbeat_no_queue still no Queue. Contrast Beat 175 (complete candidate write vs dry-run success) + Beat 174 (--no-stub fail); Beat 176 is incomplete-candidate fail live+dry-run on idle fixtures. Proof: pytest ``-k beat176``.
Beat 177: multi-fixture idle Session A --candidate-json **list** write vs --candidate-json --dry-run no-write --json contrast via tmp_path (same idle fixtures as Beat 171/175/176; candidate complete_candidate_list.json copied to tmp_path): --candidate-json --json (NO dry-run)→rc=0 ok=True verdict=queued wrote_item=True open_count=1 queue 1/10 title Complete list candidate plan CHANGED ## Queue present (Watch keep markers; watch_heartbeat_no_queue creates Queue); --candidate-json --dry-run→queued wrote_item=True plan UNCHANGED (watch_heartbeat_no_queue still no Queue). Contrast Beat 175 (complete **dict** write vs dry-run) + Beat 176 (incomplete fail); Beat 177 is complete **list** candidate-json write vs dry-run on idle fixtures. Proof: pytest ``-k beat177``.
Beat 178: multi-fixture idle Session A --candidate-json **skip-nondict list** write vs --candidate-json --dry-run no-write --json contrast via tmp_path (same idle fixtures as Beat 171/175–177; candidate complete_candidate_list_skip_nondict.json copied to tmp_path): --candidate-json --json (NO dry-run)→rc=0 ok=True verdict=queued wrote_item=True open_count=1 queue 1/10 title Skip-nondict first dict plan CHANGED ## Queue present (Watch keep markers; watch_heartbeat_no_queue creates Queue); --candidate-json --dry-run→queued wrote_item=True plan UNCHANGED (watch_heartbeat_no_queue still no Queue). Contrast Beat 177 (complete list write vs dry-run) + Beat 30 (skip-nondict unit); Beat 178 is skip-nondict list candidate-json write vs dry-run on idle fixtures. Proof: pytest ``-k beat178``.
Beat 179: multi-fixture non-pickable Session A --candidate-json complete **dict** write vs --candidate-json --dry-run no-write --json contrast via tmp_path (parallel to Beat 172 stub nonpick + Beat 175 idle candidate; candidate complete_candidate.json copied to tmp_path): --candidate-json --json→queued wrote_item=True open_count=1 queue 1/10 title Complete candidate fixture plan CHANGED; --candidate-json --dry-run→queued wrote_item=True plan UNCHANGED; queue_with_watch_heartbeat keeps Watch/Heartbeat with Watch before Queue. Contrast Beat 175 (idle complete dict write vs dry-run) + Beat 172 (nonpick stub write vs dry-run); Beat 179 is complete candidate-json write vs dry-run on non-pickables. Proof: pytest ``-k beat179``.
Beat 180: milestone — CLI help + public API export smoke through beat180. Proof: pytest ``-k beat180``.
Beat 181: multi-fixture non-pickable Session A --candidate-json **incomplete** fail live vs --candidate-json --dry-run fail --json contrast via tmp_path (post-milestone; parallel Beat 176 idle incomplete fail + Beat 179 nonpick complete write; candidate incomplete_candidate.json copied to tmp_path): --candidate-json --json (NO dry-run)→rc=1 ok=False verdict=failed wrote_item=False open_count=0 queue 0/10 title Incomplete candidate fixture plan UNCHANGED; --candidate-json --dry-run --json→same fail shape plan UNCHANGED; queue_with_watch_heartbeat keeps Watch/Heartbeat with Watch before Queue. Contrast Beat 176 (idle incomplete fail) + Beat 179 (nonpick complete write); Beat 181 is incomplete-candidate fail live+dry-run on non-pickables. Proof: pytest ``-k beat181``.
Beat 182: multi-fixture non-pickable Session A --candidate-json **list** write vs --candidate-json --dry-run no-write --json contrast via tmp_path (parallel Beat 177 idle list + Beat 179 nonpick dict; candidate complete_candidate_list.json copied to tmp_path): --candidate-json --json (NO dry-run)→rc=0 ok=True verdict=queued wrote_item=True open_count=1 queue 1/10 title Complete list candidate plan CHANGED; --candidate-json --dry-run→queued wrote_item=True plan UNCHANGED; queue_with_watch_heartbeat keeps Watch/Heartbeat with Watch before Queue. Contrast Beat 177 (idle list write vs dry-run) + Beat 179 (nonpick dict write) + Beat 181 (nonpick incomplete fail); Beat 182 is complete **list** candidate-json write vs dry-run on non-pickables. Proof: pytest ``-k beat182``.
Beat 183: multi-fixture non-pickable Session A --candidate-json **skip-nondict list** write vs --candidate-json --dry-run no-write --json contrast via tmp_path (parallel Beat 178 idle skip-nondict + Beat 182 nonpick list; candidate complete_candidate_list_skip_nondict.json copied to tmp_path): --candidate-json --json (NO dry-run)→rc=0 ok=True verdict=queued wrote_item=True open_count=1 queue 1/10 title Skip-nondict first dict plan CHANGED; --candidate-json --dry-run→queued wrote_item=True plan UNCHANGED; queue_with_watch_heartbeat keeps Watch/Heartbeat with Watch before Queue. Contrast Beat 178 (idle skip-nondict) + Beat 182 (nonpick complete list); Beat 183 is skip-nondict list candidate-json write vs dry-run on non-pickables. Proof: pytest ``-k beat183``.
Beat 184: multi-fixture pickable Session A --candidate-json light no-write --json contrast via tmp_path (completes candidate trilogy like Beat 173 stub/no-stub: idle write / nonpick write / pickable never-write; candidate complete_candidate.json copied to tmp_path; OPEN>=1 recount-only ignores candidate-json): --candidate-json --json (NO dry-run)→rc=0 ok=True verdict=light wrote_item=False correct open_count/queue/b_pick_title plan UNCHANGED; --candidate-json --dry-run→same light shape plan UNCHANGED; watch_queue_heartbeat keeps Watch/Heartbeat + BEAT19 markers. Contrast Beat 173 (stub|no-stub pickable light) + Beat 179/182/183 (nonpick candidate write); Beat 184 proves pickable OPEN>=1 never appends for --candidate-json × live|dry-run. Proof: pytest ``-k beat184``.
Beat 185: multi-fixture idle Session A --candidate-json **empty list []** fail live vs --candidate-json --dry-run fail --json contrast via tmp_path (post–Beat 184 pickable light; empty list written to tmp_path each test; no stub fallback; same idle fixtures as Beat 171/175/176): --candidate-json --json (NO dry-run)→rc=1 ok=False verdict=failed wrote_item=False open_count=0 queue 0/10 title is None plan UNCHANGED; --candidate-json --dry-run --json→same fail shape plan UNCHANGED; Watch fixtures keep Watch/Heartbeat; watch_heartbeat_no_queue still no Queue. Contrast Beat 176 (incomplete candidate fail idle) + Beat 28/37 (empty list unit) + Beat 184 (pickable light); Beat 185 is empty-list candidate-json fail live+dry-run on idle fixtures. Proof: pytest ``-k beat185``.
Beat 186: multi-fixture non-pickable Session A --candidate-json **empty list []** fail live vs --candidate-json --dry-run fail --json contrast via tmp_path (parallel Beat 185 idle empty-list fail; empty list written to tmp_path each test; no stub fallback; same non-pickable fixtures as Beat 172/179/181): --candidate-json --json (NO dry-run)→rc=1 ok=False verdict=failed wrote_item=False open_count=0 queue 0/10 title is None plan UNCHANGED; --candidate-json --dry-run --json→same fail shape plan UNCHANGED; queue_with_watch_heartbeat keeps Watch/Heartbeat with Watch before Queue. Contrast Beat 185 (idle empty-list fail) + Beat 181 (nonpick incomplete fail); Beat 186 is empty-list candidate-json fail live+dry-run on non-pickables. Proof: pytest ``-k beat186``.
Beat 187: multi-fixture idle Session A --candidate-json **no-dict list** (e.g. [1, "x", true]) fail live vs --candidate-json --dry-run fail --json contrast via tmp_path (parallel Beat 185 empty-list fail; candidate JSON written to tmp_path each test; no stub fallback; Beat 29 unit → multi-fixture; same idle fixtures as Beat 171/175/185): --candidate-json --json (NO dry-run)→rc=1 ok=False verdict=failed wrote_item=False open_count=0 queue 0/10 title is None plan UNCHANGED; --candidate-json --dry-run --json→same fail shape plan UNCHANGED; Watch fixtures keep Watch/Heartbeat; watch_heartbeat_no_queue still no Queue. Contrast Beat 185 (empty list [] fail idle) + Beat 29 (no-dict list unit); Beat 187 is no-dict-list candidate-json fail live+dry-run on idle fixtures. Proof: pytest ``-k beat187``.
Beat 188: multi-fixture non-pickable Session A --candidate-json **no-dict list** (e.g. [1, "x", true]) fail live vs --candidate-json --dry-run fail --json contrast via tmp_path (parallel Beat 187 idle no-dict fail; candidate JSON written to tmp_path each test; no stub fallback; same non-pickable fixtures as Beat 172/179/186): --candidate-json --json (NO dry-run)→rc=1 ok=False verdict=failed wrote_item=False open_count=0 queue 0/10 title is None plan UNCHANGED; --candidate-json --dry-run --json→same fail shape plan UNCHANGED; queue_with_watch_heartbeat keeps Watch/Heartbeat with Watch before Queue. Contrast Beat 187 (idle no-dict fail) + Beat 186 (nonpick empty-list fail); Beat 188 is no-dict-list candidate-json fail live+dry-run on non-pickables. Proof: pytest ``-k beat188``.
Beat 189: multi-fixture pickable Session A --candidate-json **no-dict list** [1, "x", true] light no-write --json contrast via tmp_path (OPEN>=1 recount-only ignores candidate-json; parallel Beat 184 complete-candidate pickable light + Beat 187/188 no-dict fails on idle/nonpick): --candidate-json --json (NO dry-run)→rc=0 ok=True verdict=light wrote_item=False correct open_count/queue/b_pick_title plan UNCHANGED; --candidate-json --dry-run→same light shape plan UNCHANGED; watch_queue_heartbeat keeps Watch/Heartbeat + BEAT19 markers. Contrast Beat 184 (pickable complete candidate light) + Beat 187/188 (no-dict fail idle/nonpick); Beat 189 proves pickable OPEN>=1 ignores no-dict-list candidate too. Proof: pytest ``-k beat189``.
Beat 190: milestone — CLI help + public API export smoke through beat190. Proof: pytest ``-k beat190``.
Beat 191: multi-fixture pickable Session A --candidate-json **empty list []** light no-write --json contrast via tmp_path (post-milestone; OPEN>=1 recount-only ignores candidate-json; completes empty-list trilogy after Beat 185 idle fail + Beat 186 nonpick fail; parallel Beat 189 no-dict pickable light): --candidate-json --json (NO dry-run)→rc=0 ok=True verdict=light wrote_item=False correct open_count/queue/b_pick_title plan UNCHANGED; --candidate-json --dry-run→same light shape plan UNCHANGED; watch_queue_heartbeat keeps Watch/Heartbeat + BEAT19 markers. Contrast Beat 185/186 (empty list fails idle/nonpick) + Beat 189 (no-dict pickable light); Beat 191 proves pickable OPEN>=1 ignores empty-list candidate too. Proof: pytest ``-k beat191``.
Beat 192: multi-fixture pickable Session A --candidate-json **incomplete** light no-write --json contrast via tmp_path (OPEN>=1 recount-only ignores candidate-json; completes incomplete trilogy after Beat 176 idle fail + Beat 181 nonpick fail; parallel Beat 184/189/191 pickable light for other candidate shapes; candidate incomplete_candidate.json copied to tmp_path): --candidate-json --json (NO dry-run)→rc=0 ok=True verdict=light wrote_item=False correct open_count/queue/b_pick_title plan UNCHANGED; --candidate-json --dry-run→same light shape plan UNCHANGED; watch_queue_heartbeat keeps Watch/Heartbeat + BEAT19 markers. Contrast Beat 176/181 (incomplete fails idle/nonpick) + Beat 191 (empty-list pickable light); Beat 192 proves pickable OPEN>=1 ignores incomplete candidate too. Proof: pytest ``-k beat192``.
Beat 193: multi-fixture pickable Session A --candidate-json **complete list** light no-write --json contrast via tmp_path (OPEN>=1 recount-only ignores candidate-json; completes complete-list trilogy after Beat 177 idle write + Beat 182 nonpick write; parallel Beat 184 dict / Beat 191 empty-list / Beat 192 incomplete pickable light; candidate complete_candidate_list.json copied to tmp_path): --candidate-json --json (NO dry-run)→rc=0 ok=True verdict=light wrote_item=False correct open_count/queue/b_pick_title plan UNCHANGED; --candidate-json --dry-run→same light shape plan UNCHANGED; watch_queue_heartbeat keeps Watch/Heartbeat + BEAT19 markers. Contrast Beat 177/182 (list write idle/nonpick) + Beat 184 (dict pickable light); Beat 193 proves pickable OPEN>=1 ignores complete list candidate too. Proof: pytest ``-k beat193``.
Beat 194: multi-fixture pickable Session A --candidate-json **skip-nondict list** light no-write --json contrast via tmp_path (OPEN>=1 recount-only ignores candidate-json; completes skip-nondict trilogy after Beat 178 idle write + Beat 183 nonpick write; parallel Beat 184 dict / Beat 191 empty-list / Beat 192 incomplete / Beat 193 complete-list / Beat 189 no-dict pickable light; candidate complete_candidate_list_skip_nondict.json copied to tmp_path): --candidate-json --json (NO dry-run)→rc=0 ok=True verdict=light wrote_item=False correct open_count/queue/b_pick_title plan UNCHANGED; --candidate-json --dry-run→same light shape plan UNCHANGED; watch_queue_heartbeat keeps Watch/Heartbeat + BEAT19 markers. Contrast Beat 178/183 (skip-nondict write idle/nonpick) + Beat 193 (complete-list pickable light); Beat 194 proves pickable OPEN>=1 ignores skip-nondict list candidate too. Proof: pytest ``-k beat194``.
Beat 195: multi-fixture idle Session A --candidate-json **empty object {}** fail live vs --candidate-json --dry-run fail --json contrast via tmp_path (parallel Beat 185 empty-list fail / Beat 176 incomplete fail / Beat 187 no-dict fail; empty object written to tmp_path each test; no stub fallback; Beat 39 unit → multi-fixture; same idle fixtures as Beat 171/175/185/187): --candidate-json --json (NO dry-run)→rc=1 ok=False verdict=failed wrote_item=False open_count=0 queue 0/10 title is None plan UNCHANGED; --candidate-json --dry-run --json→same fail shape plan UNCHANGED; Watch fixtures keep Watch/Heartbeat; watch_heartbeat_no_queue still no Queue. Contrast Beat 185 ([] fail idle) + Beat 176 (incomplete fail idle) + Beat 39 ({} unit); Beat 195 is empty-object {} candidate-json fail live+dry-run on idle fixtures. Proof: pytest ``-k beat195``.
Beat 196: multi-fixture non-pickable Session A --candidate-json **empty object {}** fail live vs --candidate-json --dry-run fail --json contrast via tmp_path (parallel Beat 195 idle empty-object fail + Beat 186 nonpick empty-list fail; empty object written to tmp_path each test; no stub fallback; Beat 39 unit → Beat 195 idle → Beat 196 nonpick; same non-pickable fixtures as Beat 172/179/181/186): --candidate-json --json (NO dry-run)→rc=1 ok=False verdict=failed wrote_item=False open_count=0 queue 0/10 title is None plan UNCHANGED; --candidate-json --dry-run --json→same fail shape plan UNCHANGED; queue_with_watch_heartbeat keeps Watch/Heartbeat with Watch before Queue. Contrast Beat 195 ({} fail idle) + Beat 186 ([] fail nonpick) + Beat 39 ({} unit); Beat 196 is empty-object {} candidate-json fail live+dry-run on non-pickables. Proof: pytest ``-k beat196``.
Beat 197: multi-fixture pickable Session A --candidate-json **empty object {}** light no-write --json contrast via tmp_path (OPEN>=1 recount-only ignores candidate-json; completes empty-object trilogy after Beat 195 idle fail + Beat 196 nonpick fail; parallel Beat 184 dict / Beat 189 no-dict / Beat 191 empty-list / Beat 192 incomplete / Beat 193 complete-list / Beat 194 skip-nondict pickable light; empty object written to tmp_path each test): --candidate-json --json (NO dry-run)→rc=0 ok=True verdict=light wrote_item=False correct open_count/queue/b_pick_title plan UNCHANGED; --candidate-json --dry-run→same light shape plan UNCHANGED; watch_queue_heartbeat keeps Watch/Heartbeat + BEAT19 markers. Contrast Beat 195/196 ({} fail idle/nonpick) + Beat 191 ([] pickable light); Beat 197 proves pickable OPEN>=1 ignores empty-object candidate too. Proof: pytest ``-k beat197``.
Beat 198: multi-fixture idle Session A --candidate-json **missing file** fail live vs --candidate-json --dry-run fail (SystemExit, not verdict=failed) via tmp_path (promotes Beat 16 unit → multi-fixture idle contrast; parallel Beat 195 idle {} fail; same idle fixtures as Beat 171/175/185/187/195; path NEVER created): --candidate-json missing --json (NO dry-run)→SystemExit nonzero "--candidate-json"+"not found" plan UNCHANGED no success JSON; --candidate-json missing --dry-run --json→same SystemExit plan UNCHANGED (dry-run does not skip missing-file load); Watch fixtures keep Watch/Heartbeat; watch_heartbeat_no_queue still no Queue. Contrast Beat 195 ({} fail idle JSON rc=1) + Beat 16 (missing-file unit); Beat 198 is missing-path candidate-json CLI load-fail live+dry-run on idle fixtures. Proof: pytest ``-k beat198``.
Beat 199: multi-fixture non-pickable Session A --candidate-json **missing file** fail live vs --candidate-json --dry-run fail (SystemExit, not verdict=failed) via tmp_path (parallel Beat 198 idle missing-file SystemExit + Beat 196 nonpick {} fail; continues missing-file trilogy after Beat 16 unit → Beat 198 idle → Beat 199 nonpick; same non-pickable fixtures as Beat 172/179/181/186/196; path NEVER created): --candidate-json missing --json (NO dry-run)→SystemExit nonzero "--candidate-json"+"not found" plan UNCHANGED no success JSON; --candidate-json missing --dry-run --json→same SystemExit plan UNCHANGED (dry-run does not skip missing-file load); queue_with_watch_heartbeat keeps Watch/Heartbeat with Watch before Queue. Contrast Beat 198 (missing-file SystemExit idle) + Beat 196 ({} fail nonpick) + Beat 16 (missing-file unit); Beat 199 is missing-path candidate-json CLI load-fail live+dry-run on non-pickables. Proof: pytest ``-k beat199``.
Beat 201: multi-fixture pickable Session A --candidate-json **missing file** still light / no-write --json contrast via tmp_path (OPEN>=1 recount-only; ``_load_candidate`` deferred inside brainstorm callback so missing path does **NOT** SystemExit on pickables — contrast Beat 198 idle SystemExit + Beat 199 nonpick SystemExit; completes missing-file trilogy after Beat 198/199; same pickable fixtures as Beat 197; path NEVER created): --candidate-json missing --json (NO dry-run)→rc=0 ok=True verdict=light wrote_item=False correct open_count/queue/b_pick_title plan UNCHANGED Must NOT raise SystemExit; --candidate-json missing --dry-run→same light shape plan UNCHANGED no SystemExit; watch_queue_heartbeat keeps Watch/Heartbeat + BEAT19 markers. Contrast Beat 198/199 (missing-file SystemExit idle/nonpick) + Beat 197 ({} pickable light); Beat 201 proves pickable OPEN>=1 never loads missing candidate-json (deferred load). Proof: pytest ``-k beat201``.
Beat 202: multi-fixture idle Session A --candidate-json **invalid JSON** fail live vs --candidate-json --dry-run fail (SystemExit, not verdict=failed) via tmp_path (promotes Beat 16 unit → multi-fixture idle contrast; parallel Beat 198 idle missing-file SystemExit; same idle fixtures as Beat 171/175/185/187/195/198; writes `{not valid json` to tmp_path invalid_candidate.json each test; NEVER mutate fixtures): --candidate-json invalid --json (NO dry-run)→SystemExit nonzero "--candidate-json"+"invalid JSON" plan UNCHANGED no success JSON; --candidate-json invalid --dry-run --json→same SystemExit plan UNCHANGED (dry-run does not skip invalid-JSON load); Watch fixtures keep Watch/Heartbeat; watch_heartbeat_no_queue still no Queue. Contrast Beat 198 (missing-file SystemExit idle) + Beat 16 (invalid-JSON unit); Beat 202 is invalid-JSON candidate-json CLI load-fail live+dry-run on idle fixtures. Proof: pytest ``-k beat202``.
Beat 203: multi-fixture non-pickable Session A --candidate-json **invalid JSON** fail live vs --candidate-json --dry-run fail (SystemExit, not verdict=failed) via tmp_path (parallel Beat 202 idle invalid-JSON SystemExit + Beat 199 nonpick missing-file SystemExit; continues invalid-JSON trilogy after Beat 16 unit → Beat 202 idle → Beat 203 nonpick; same non-pickable fixtures as Beat 172/179/181/186/196/199; writes `{not valid json` to tmp_path invalid_candidate.json each test; NEVER mutate fixtures): --candidate-json invalid --json (NO dry-run)→SystemExit nonzero "--candidate-json"+"invalid JSON" plan UNCHANGED no success JSON; --candidate-json invalid --dry-run --json→same SystemExit plan UNCHANGED (dry-run does not skip invalid-JSON load); queue_with_watch_heartbeat keeps Watch/Heartbeat with Watch before Queue. Contrast Beat 202 (invalid-JSON SystemExit idle) + Beat 199 (missing-file SystemExit nonpick) + Beat 16 (invalid-JSON unit); Beat 203 is invalid-JSON candidate-json CLI load-fail live+dry-run on non-pickables. Proof: pytest ``-k beat203``.
Beat 200: milestone — CLI help + public API export smoke through beat200. Proof: pytest ``-k beat200``.
"""

from __future__ import annotations

from src.research_implement.queue import (
    QUEUE_CAPACITY,
    REQUIRED_FIELDS,
    AmbiguousQueueError,
    QueueItem,
    append_queue_item,
    count_open,
    count_queue_headings,
    first_b_pick,
    format_queue_item,
    is_b_pickable,
    is_ready_yes,
    is_complete_six_field,
    is_open_status,
    next_queue_id,
    parse_queue_items,
    render_queue_count,
    require_unique_queue_section,
    serialize_queue_item,
    serialize_queue_items,
    write_queue_section,
    mark_item_shipped,
)
from src.research_implement.session_a import (
    SESSION_A_RESULT_JSON_KEYS,
    SESSION_A_RESULT_KEYS,
    SessionAResult,
    default_search_plan,
    incomplete_candidate_reasons,
    run_session_a,
    run_session_a_path,
    session_a_result_dict,
    stub_brainstorm,
    stub_search_plan,
)
from src.research_implement.session_b import (
    SESSION_B_RESULT_KEYS,
    SESSION_RESULT_JSON_KEYS,
    SchedulerDeleteForbidden,
    SessionBResult,
    SessionResult,
    decode_fields,
    default_implement,
    dry_run_implement,
    fixture_ship_implement,
    format_decode_report,
    make_fixture_ship_implement,
    run_session_b,
    run_session_b_path,
    scheduler_delete,
    session_b_result_dict,
)

__all__ = [
    "QUEUE_CAPACITY",
    "REQUIRED_FIELDS",
    "AmbiguousQueueError",
    "SchedulerDeleteForbidden",
    "QueueItem",
    "SessionAResult",
    "SessionBResult",
    "SessionResult",
    "SESSION_A_RESULT_JSON_KEYS",
    "SESSION_A_RESULT_KEYS",
    "SESSION_RESULT_JSON_KEYS",
    "SESSION_B_RESULT_KEYS",
    "session_a_result_dict",
    "session_b_result_dict",
    "append_queue_item",
    "count_open",
    "count_queue_headings",
    "first_b_pick",
    "format_queue_item",
    "next_queue_id",
    "serialize_queue_item",
    "serialize_queue_items",
    "write_queue_section",
    "mark_item_shipped",
    "is_b_pickable",
    "is_ready_yes",
    "is_complete_six_field",
    "is_open_status",
    "parse_queue_items",
    "require_unique_queue_section",
    "render_queue_count",
    "default_search_plan",
    "run_session_a",
    "run_session_a_path",
    "run_session_b",
    "run_session_b_path",
    "incomplete_candidate_reasons",
    "stub_brainstorm",
    "stub_search_plan",
    "decode_fields",
    "format_decode_report",
    "scheduler_delete",
    "dry_run_implement",
    "default_implement",
    "fixture_ship_implement",
    "make_fixture_ship_implement",
]
