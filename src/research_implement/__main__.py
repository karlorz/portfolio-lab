"""CLI: python -m src.research_implement {session-a|session-b|idle-decode}

Side-dev entrypoints only. Does not start Tasker, does not load Tasker yaml on
prod ports (8000/8001/18000), and does not call external LLMs / Hermes /
``src.research.agent``.

Examples (fixture or --log path)::

    python -m src.research_implement session-a --plan tests/fixtures/research_implement/empty_queue.md --stub --dry-run --json
    python -m src.research_implement session-b --log logs/research-implement.md --json
    python -m src.research_implement session-b --plan tests/fixtures/research_implement/one_open_ready.md --dry-run --json
    python -m src.research_implement idle-decode --plan tests/fixtures/research_implement/empty_queue.md

E2E dry-run on a temp plan (A stub → B dry-run → B again; OPEN stays OPEN)::

    cp tests/fixtures/research_implement/empty_queue.md /tmp/ri-plan.md
    python -m src.research_implement session-a --plan /tmp/ri-plan.md --stub
    python -m src.research_implement session-b --plan /tmp/ri-plan.md --dry-run --json
    python -m src.research_implement session-b --plan /tmp/ri-plan.md --dry-run --json

Dry-run contract: non-mutating — never marks SHIPPED, never deletes Queue
rows, never ``scheduler_delete``. Second B therefore picks the same OPEN
again (not idle ``queue 0/10``). Or: ``make research-implement-e2e-dry-run``.

Full pipeline (Beat 11, tmp_path / test double only)::

    A stub append → B dry_run --json → B fixture_ship → A light (OPEN>=1) and/or B idle

Light recount runs mid-pipeline while OPEN>=1 (after dry_run, before ship). Dry-run
never ships; ``fixture_ship_implement`` ships on tmp_path only (never CLI default).
Proof: ``make research-implement-e2e-pipeline`` / pytest ``-k beat11``.

Beat 12: CLI ``--help`` smoke (session-a / session-b / idle-decode) + OPEN>=1
brainstorm/search_plan spy (recount-only; callback never called). Proof:
pytest ``-k beat12``.

Beat 13: Queue markdown round-trip (parse → serialize/write → parse) preserves
six fields + ready-for-implement for OPEN; Session A stub append id stability
(empty→Q1; after ship/clear→new id no collide; two_open first-OPEN unchanged).
Proof: pytest ``-k beat13``.

Beat 14: Session A fail-closed on incomplete brainstorm/search_plan candidate
(missing six fields or ready flag) → failed, wrote_item=False, plan unchanged,
no partial OPEN append; empty + complete stub still queues. Proof: pytest
``-k beat14``.

Beat 15: CLI ``session-a --candidate-json <path>`` loads a candidate dict (or
first dict in a JSON list) as the brainstorm callback when OPEN=0 (instead of
stub). Incomplete JSON still fail-closes (Beat 14); complete candidate queues
one OPEN; OPEN>=1 remains recount-only and ignores candidate-json. Proof:
pytest ``-k beat15``.

Beat 16: CLI ``--candidate-json`` error paths (missing file, invalid JSON, wrong
type not object/list) → non-zero exit / clear failure; plan unchanged. Sequential
double-OPEN ship on tmp_path: two_open_ready → ship first → ship second → idle;
never ``scheduler_delete``; JSON shapes ok. Proof: pytest ``-k beat16``.

Beat 17: Session B default/CLI decode-only never invokes implement callback
(spy); ``--dry-run`` may call ``dry_run_implement``; ``fixture_ship_implement``
only when ``decode_only=False`` + ``implement=`` passed (never CLI default).
Proof: pytest ``-k beat17``.

Beat 19: Queue rewrite (append/ship/serialize) preserves Watch / Project
Work / Heartbeat. Proof: pytest ``-k beat19``.

Beat 20: Missing ``## Queue`` → Session A append / ``write_queue_section``
creates the section without destroying Watch / Heartbeat / front matter;
empty file creates Queue. Proof: pytest ``-k beat20``.

Beat 21: More than one ``## Queue`` → fail-closed (``AmbiguousQueueError`` /
Session A/B failed); plan unchanged. Proof: pytest ``-k beat21``.

Beat 23: Missing ``--plan`` / ``--log`` path (or omitted flag) → clear
non-zero failure (``SystemExit``); CLI does not create random plan files.
Existing plan still works. Proof: pytest ``-k beat23``.

Beat 24: Passing both ``--plan`` and ``--log`` → argparse mutually exclusive exit 2; ``--log`` alone aliases ``--plan``. Proof: pytest ``-k beat24``.

Beat 25: Passing both ``--stub`` and ``--no-stub`` → clear ``SystemExit``; plan unchanged. ``--candidate-json`` still overrides stub when OPEN=0. Proof: pytest ``-k beat25``.

Beat 26: ``idle-decode`` rejects ``--dry-run`` (decode-only alias; use ``session-b --dry-run``). Session A ``--dry-run`` leaves the plan file unchanged. Proof: pytest ``-k beat26``.

Beat 27: Producer-only flags (``--stub`` / ``--no-stub`` / ``--candidate-json``) are rejected on ``session-b`` / ``idle-decode``; plan unchanged. Proof: pytest ``-k beat27``.

Beat 28: ``session-a --candidate-json`` with an empty JSON list ``[]`` → failed fire (no stub fallback), plan unchanged; unknown CLI subcommand → non-zero exit. Proof: pytest ``-k beat28``.

Beat 29: ``--candidate-json`` list with no dict elements → failed fire (no stub), plan unchanged; CLI ``--json`` emits exactly ``SESSION_A_RESULT_JSON_KEYS`` / ``SESSION_RESULT_JSON_KEYS``. Proof: pytest ``-k beat29``.

Beat 30: ``--candidate-json`` list skips leading non-dicts and uses the first dict; relative ``--plan`` path resolves when the file exists. Proof: pytest ``-k beat30``.

Beat 31: ``--plan`` / ``--log`` path that exists but is not a file (e.g. directory) → clear non-zero exit; relative ``--log`` alias works like relative ``--plan``. Proof: pytest ``-k beat31``.

Beat 32: ``--candidate-json`` path that exists but is not a file (e.g. directory) → clear non-zero exit; top-level JSON ``null`` also fails closed. Proof: pytest ``-k beat32``.

Beat 33: empty / whitespace-only ``--plan`` / ``--log`` → clear non-zero exit; idle-decode ``--json`` keeps ``keep_schedule=true`` and ``scheduler_delete_called=false``. Proof: pytest ``-k beat33``.

Beat 34: empty / whitespace ``--log`` alias fails like ``--plan``; ``session-b --dry-run --json`` keeps ``keep_schedule=true`` and ``scheduler_delete_called=false``. Proof: pytest ``-k beat34``.

Beat 35: ``session-b`` decode-only ``--json`` (default) keeps ``keep_schedule=true`` / ``scheduler_delete_called=false`` and leaves the plan unchanged; Session A recount-only (OPEN>=1) ``--json`` has ``wrote_item=false``. Proof: pytest ``-k beat35``.

Beat 36: ``idle-decode --json`` on a ready OPEN matches session-b decode-only (picked + keep_schedule); Session A stub on empty Queue ``--json`` has ``wrote_item=true`` / ``verdict=queued``. Proof: pytest ``-k beat36``.

Beat 37: empty / whitespace ``--candidate-json`` → clear non-zero exit (no Path(".") coerce); Session A ``--stub --dry-run --json`` reports ``wrote_item=true`` / ``verdict=queued`` but leaves the plan unchanged. Proof: pytest ``-k beat37``.

Beat 38: relative ``--candidate-json`` resolves when the file exists in cwd; top-level JSON boolean ``true``/``false`` fails closed (wrong type). Proof: pytest ``-k beat38``.

Beat 39: ``--candidate-json`` empty object ``{}`` fail-closes via Session A incomplete candidate (rc=1, plan unchanged); ``session-a --help`` mentions ``--stub`` / ``--no-stub`` / ``--candidate-json``. Proof: pytest ``-k beat39``.

Beat 40: top-level ``--help`` lists ``session-a`` / ``session-b`` / ``idle-decode``; ``idle-decode --help`` usage has no ``--dry-run``; ``session-b --help`` usage includes ``--dry-run``. Proof: pytest ``-k beat40``.

Beat 41: top-level ``--help`` mentions side-dev / no Tasker / no live LLM; CLI rejects ``--implement`` / stub-ship style flags (SHIPPED remains callback-only). Proof: pytest ``-k beat41``.

Beat 42: Makefile lists ``test-research-implement`` / ``research-implement-e2e-dry-run`` / ``research-implement-e2e-pipeline``; ``session-b`` / ``idle-decode --help`` mention never ``scheduler_delete``. Proof: pytest ``-k beat42``.

Beat 43: side-dev guide documents make targets + no CLI ship flag + never ``scheduler_delete``; ``__init__`` mentions Beats 40–42. Proof: pytest ``-k beat43``.

Beat 44: ``session-b --json`` help matches shared ``SessionResult.to_dict`` (idle | picked | dry_run); dry-run message includes ``decode pick`` six-field report like decode-only. Proof: pytest ``-k beat44``.

Beat 45: ``idle-decode --json`` / ``session-a --json`` help mention shared ``SessionResult.to_dict`` / ``SessionAResult.to_dict``; decode-only ``session-b`` message includes ``decode pick``. Proof: pytest ``-k beat45``.

Beat 46: OPEN complete but not ready-for-implement → idle fire (``queue 0/10``, keep_schedule) via ``idle-decode`` / ``session-b`` CLI; plan unchanged. Proof: pytest ``-k beat46``.

Beat 47: incomplete OPEN / SHIPPED-only → idle CLI JSON; ``mixed_priority`` picks first ready OPEN via ``session-b --json`` (plan unchanged on decode-only). Proof: pytest ``-k beat47``.

Beat 48: Watch lookalike rows are never B-picked; ``watch_lookalike`` CLI picks real Queue OPEN; ``watch_only_lookalike`` / no-Queue watch plans idle. Proof: pytest ``-k beat48``.

Beat 49: ``two_open_ready`` CLI picks first ready OPEN (second stays); Session A on OPEN>=1 is recount-only; Queue+Watch+Heartbeat plans keep non-Queue sections on stub append. Proof: pytest ``-k beat49``.

Beat 50: ``broken_ready_flag`` → idle CLI JSON (never pick garbage ready); public ``research_implement`` exports still include Session A/B result helpers; Makefile echo mentions beat50. Proof: pytest ``-k beat50``.

Beat 51: ``ready-for-implement`` accepts case-insensitive aliases (``yes``/``y``/``true``/``1``) for B-pick CLI; rejects non-alias garbage. Proof: pytest ``-k beat51``.

Beat 52: ``two_queue_sections`` CLI (session-a / session-b / idle-decode) → non-zero exit, ``verdict=failed``, plan untouched (ambiguous Queue fail-closed). Proof: pytest ``-k beat52``.

Beat 53: ``watch_queue_heartbeat_empty`` idle-decode/session-b → idle with Watch/Project/Heartbeat markers intact; session-a stub append keeps those sections. Proof: pytest ``-k beat53``.

Beat 54: ``watch_queue_heartbeat`` (OPEN ready) idle-decode pick + ``session-b --dry-run`` leave Watch/Project/Heartbeat markers intact. Proof: pytest ``-k beat54``.

Beat 55: ``watch_heartbeat_no_queue`` idle-decode stays idle with markers intact; ``session-a --stub`` creates ``## Queue`` while keeping Watch/Heartbeat markers. Proof: pytest ``-k beat55``.

Beat 56: CLI ``--json`` for light / dry_run / failed (dual-Queue) still matches ``SESSION_A_RESULT_JSON_KEYS`` / ``SESSION_RESULT_JSON_KEYS``; ``queue_with_watch_heartbeat`` A-stub then idle-decode pick keeps Watch markers. Proof: pytest ``-k beat56``.

Beat 57: ``session-a`` failed (dual-Queue) ``--json`` keys match ``SESSION_A_RESULT_JSON_KEYS``; ``AmbiguousQueueError`` stays public; ``session-a --dry-run`` on empty Queue leaves plan unchanged. Proof: pytest ``-k beat57``.

Beat 58: ``session-b --dry-run`` on empty Queue is idle (not dry_run); public exports still include ``is_b_pickable`` / ``is_ready_yes`` / ``count_open``. Proof: pytest ``-k beat58``.

Beat 59: ``two_open_ready`` ``session-b --dry-run`` picks first OPEN (``open_count`` stays 2); plan unchanged with both still OPEN. Proof: pytest ``-k beat59``.

Beat 60: ``watch_lookalike`` ``session-b --dry-run`` / idle-decode pick Queue ``Q3`` only (never Watch lookalike); plan unchanged. Proof: pytest ``-k beat60``.

Beat 61: ``session-b --dry-run`` on ``shipped_only`` / ``incomplete_open`` / ``watch_only_lookalike`` stays idle (never dry_run); plan unchanged. Proof: pytest ``-k beat61``.

Beat 62: ``session-b --dry-run`` on ``open_complete_not_ready`` / ``broken_ready_flag`` stays idle; ``next_queue_id`` remains public. Proof: pytest ``-k beat62``.

Beat 63: ``mixed_priority`` ``session-b --dry-run`` / idle-decode pick first ready OPEN; plan unchanged; ``count_queue_headings`` stays public. Proof: pytest ``-k beat63``.

Beat 64: ``one_open_ready`` ``session-b --dry-run`` / idle-decode pick Q1; plan unchanged; ``format_queue_item`` / ``require_unique_queue_section`` stay public. Proof: pytest ``-k beat64``.

Beat 65: ``two_queue_sections`` ``session-b --dry-run`` / ``session-a --dry-run`` still fail-closed (``verdict=failed``, non-zero); ``write_queue_section`` stays public. Proof: pytest ``-k beat65``.

Beat 66: ``session-a --no-stub`` on OPEN>=1 is light recount; on empty Queue fails closed (no stub fallback); ``stub_brainstorm`` / ``dry_run_implement`` / ``make_fixture_ship_implement`` stay public. Proof: pytest ``-k beat66``.

Beat 67: ``session-a --no-stub --dry-run`` keeps the same light/failed paths without mutating the plan; ``decode_fields`` / ``format_decode_report`` / ``first_b_pick`` stay public. Proof: pytest ``-k beat67``.

Beat 68: ``first_b_pick`` returns the first ready OPEN (None on empty/shipped-only); idle-decode ``--json`` message still includes ``decode pick``; ``incomplete_candidate_reasons({{}})`` lists all six fields + ready. Proof: pytest ``-k beat68``.

Beat 69: CLI ``--json`` matches ``session_a_result_dict`` / ``session_b_result_dict`` for queued/picked; those helpers stay public with ``SESSION_*_RESULT_JSON_KEYS``. Proof: pytest ``-k beat69``.

Beat 70: ``session-a --json`` light/failed keys match ``SESSION_A_RESULT_JSON_KEYS``; ``run_session_a_path`` / ``run_session_b_path`` are public. Proof: pytest ``-k beat70``.

Beat 71: ``session-b --json`` dry_run/idle/failed keys match ``SESSION_RESULT_JSON_KEYS``; ``to_dict`` / ``to_json_dict`` / ``session_b_result_dict`` stay aligned. Proof: pytest ``-k beat71``.

Beat 72: Session A ``to_dict`` / ``to_json_dict`` / ``session_a_result_dict`` align for queued/light/failed; CLI ``--json`` light already keys-locked. Proof: pytest ``-k beat72``.

Beat 73: ``SessionAResult`` / ``SessionResult`` / ``SessionBResult`` / ``QueueItem`` stay public; result classes still expose ``to_dict``. Proof: pytest ``-k beat73``.

Beat 74: ``QueueItem`` from ``one_open_ready`` is B-pickable; ``format_queue_item`` round-trips via ``parse_queue_items``; ``is_b_pickable`` stays public. Proof: pytest ``-k beat74``.

Beat 75: ``incomplete_open`` / ``shipped_only`` items are not B-pickable; ``count_open`` stays public and returns 0 for those fixtures. Proof: pytest ``-k beat75``.

Beat 76: ``open_complete_not_ready`` is not B-pickable (``count_open`` 0); ``broken_ready_flag`` likewise; ``is_ready_yes`` rejects READY/maybe. Proof: pytest ``-k beat76``.

Beat 77: ``mixed_priority`` ``first_b_pick`` skips incomplete rows and returns the first ready OPEN; ``count_open`` counts only pickable items. Proof: pytest ``-k beat77``.

Beat 78: ``two_open_ready`` ``first_b_pick`` returns the first OPEN; ``count_open`` is 2; second stays OPEN. Proof: pytest ``-k beat78``.

Beat 79: ``one_open_ready`` ``first_b_pick`` / ``count_open`` are 1; ``empty_queue`` stays 0 / None. Proof: pytest ``-k beat79``.

Beat 80: milestone — top-level ``--help`` still lists session-a/session-b/idle-decode; public API still exports Session A/B runners + queue helpers; Makefile echo reaches beat80. Proof: pytest ``-k beat80``.
Beat 81: watch_lookalike ``first_b_pick`` is Q3 (not Watch); watch_only idle; ``mark_item_shipped`` stays public. Proof: pytest ``-k beat81``.
Beat 82: subcommand ``--help`` still lists ``--plan``/``--json``; ``render_queue_count`` + ``QUEUE_CAPACITY`` stay public. Proof: pytest ``-k beat82``.
Beat 83: incomplete/shipped/not-ready/broken/empty ``first_b_pick`` is None; ``serialize_queue_item(s)`` stay public. Proof: pytest ``-k beat83``.
Beat 84: pickable fixtures ``first_b_pick`` ids (Q1/Q1/Q3/Q2); ``is_complete_six_field`` + ``is_open_status`` stay public. Proof: pytest ``-k beat84``.
Beat 85: session-a producer help flags; session-b ``--decode-only``; idle-decode omits ``--dry-run``; ``REQUIRED_FIELDS`` public. Proof: pytest ``-k beat85``.
Beat 86: idle ``queue 0/10`` + ``scheduler_delete`` guard; ``SchedulerDeleteForbidden`` public; decode report ``decode pick``. Proof: pytest ``-k beat86``.
Beat 87: two_queue ``AmbiguousQueueError`` + heading count=2; ``format_decode_report``/``decode_fields`` stay public. Proof: pytest ``-k beat87``.
Beat 88: Watch/Heartbeat fixtures idle or pick Q1; ``SESSION_A_RESULT_KEYS`` / ``SESSION_B_RESULT_KEYS`` stay public. Proof: pytest ``-k beat88``.
Beat 89: ``incomplete_candidate_reasons`` contract; ``SESSION_*_JSON_KEYS`` stay public. Proof: pytest ``-k beat89``.
Beat 90: milestone — top-level ``--help`` + public API through SchedulerDeleteForbidden/REQUIRED_FIELDS/JSON keys; Makefile echo reaches beat90. Proof: pytest ``-k beat90``.
Beat 91: empty Queue append ``next_queue_id`` Q1 becomes pickable; append/format/write helpers stay public. Proof: pytest ``-k beat91``.
Beat 92: ``mark_item_shipped`` clears pick; ``stub_brainstorm`` ready yes; stub/dry-run helpers stay public. Proof: pytest ``-k beat92``.
Beat 93: ``is_ready_yes`` aliases; ``default_search_plan``/``stub_search_plan`` alias stub; result-dict helpers public. Proof: pytest ``-k beat93``.
Beat 94: session_*_result_dict / to_dict match JSON keys; path runners stay public. Proof: pytest ``-k beat94``.
Beat 95: serialize_queue_item round-trip stays pickable; count_queue_headings fixture contract. Proof: pytest ``-k beat95``.
Beat 96: ``decode_fields`` covers six fields + status/ready; decode report lines 1–6. Proof: pytest ``-k beat96``.
Beat 97: ``next_queue_id`` progression (empty→Q1, one→Q2, two→Q3, lookalike→Q4); require_unique fail-closed. Proof: pytest ``-k beat97``.
Beat 98: ``is_b_pickable`` fixture matrix; complete/open/ready helpers stay public. Proof: pytest ``-k beat98``.
Beat 99: ``count_open`` fixture matrix aligns with ``render_queue_count``; capacity helpers public. Proof: pytest ``-k beat99``.
Beat 100: milestone — top-level ``--help`` + public API through beat100 (guards/decode/serialize/count helpers); Makefile echo reaches beat100. Proof: pytest ``-k beat100``.
Beat 101: ``format_queue_item`` round-trip stays pickable; dry-run/fixture-ship helpers public. Proof: pytest ``-k beat101``.
Beat 102: ``write_queue_section`` transplants two_open; ``default_implement`` aliases dry-run. Proof: pytest ``-k beat102``.
Beat 103: ``serialize_queue_items`` two_open round-trip stays dual-pickable. Proof: pytest ``-k beat103``.
Beat 104: ``QueueItem.field_map`` covers ``REQUIRED_FIELDS``; QueueItem stays public. Proof: pytest ``-k beat104``.
Beat 105: ``to_json_dict`` aliases ``to_dict``; ``SessionBResult`` is ``SessionResult``. Proof: pytest ``-k beat105``.
Beat 106: idle/picked ``keep_schedule`` True; never ``scheduler_delete_called``. Proof: pytest ``-k beat106``.
Beat 107: Session A queued/light/failed ``to_dict`` matrix; ``run_session_a`` stays public. Proof: pytest ``-k beat107``.
Beat 108: Session B idle/picked/failed ``to_dict`` matrix; ``run_session_b`` stays public. Proof: pytest ``-k beat108``.
Beat 109: ``first_b_pick``/``count_open`` consistent across fixtures; ``__all__`` unique. Proof: pytest ``-k beat109``.
Beat 110: milestone — top-level ``--help`` + public API through beat110 (A/B verdicts, aliases, write/serialize); Makefile echo reaches beat110. Proof: pytest ``-k beat110``.
Beat 111: idle ``queue 0/10`` decode path (decode_report None); picked report matches ``format_decode_report``. Proof: pytest ``-k beat111``.
Beat 112: path runners — A empty writes stub; B idle leaves plan untouched. Proof: pytest ``-k beat112``.
Beat 113: idle-decode CLI ``--json`` idle/picked/failed matrix (mixed→Q2). Proof: pytest ``-k beat113``.
Beat 114: session-b ``--decode-only --json`` idle/picked/failed matrix; plan untouched. Proof: pytest ``-k beat114``.
Beat 115: session-a ``--stub|--no-stub --json`` CLI matrix via tmp_path (empty→queued write; one_open→light no-write; two_queue / empty --no-stub→failed). Proof: pytest ``-k beat115``.
Beat 116: session-b ``--dry-run --json`` CLI matrix via tmp_path (empty/shipped→idle; one_open/mixed→dry_run open_count=1; two_open→dry_run open_count=2; two_queue→failed); plan untouched. Proof: pytest ``-k beat116``.
Beat 117: session-a ``--stub|--no-stub --dry-run --json`` CLI matrix via tmp_path (empty stub→queued wrote_item=True but plan unchanged; one_open→light; two_queue / empty --no-stub→failed; one_open --no-stub→light). Contrast Beat 115: dry-run may report wrote_item=True while plan bytes stay unchanged. Proof: pytest ``-k beat117``.
Beat 118: idle-decode ``--json`` Watch-fixture CLI matrix via tmp_path (watch_only→idle; watch_lookalike / watch_queue_heartbeat→picked; empty→idle; two_queue→failed); plan untouched. Distinct from Beat 113 general matrix. Proof: pytest ``-k beat118``.
Beat 119: session-b ``--dry-run --json`` Watch-fixture CLI matrix via tmp_path (watch_only→idle; watch_lookalike / watch_queue_heartbeat→dry_run; two_queue→failed); plan untouched; keep_schedule / no scheduler_delete. Contrast Beat 116 + Beat 118. Proof: pytest ``-k beat119``.
Beat 120: milestone — top-level ``--help`` + public API through beat120 (A/B runners + queue/decode/serialize helpers + guards); Makefile echo reaches beat120. Proof: pytest ``-k beat120``.
Beat 121: session-a ``--stub|--no-stub --json`` Watch-fixture CLI matrix via tmp_path (watch_only / watch_heartbeat_no_queue stub→queued write keeps Watch/Heartbeat markers; watch_lookalike stub→light plan unchanged; watch_only --no-stub→failed plan unchanged). Contrast Beats 115/118/119. Proof: pytest ``-k beat121``.
Beat 122: session-a ``--stub|--no-stub --dry-run --json`` Watch-fixture CLI matrix via tmp_path (watch_only / watch_heartbeat_no_queue stub dry-run→queued wrote_item=True but plan unchanged; watch_lookalike stub dry-run→light; watch_only --no-stub dry-run→failed). Contrast Beat 117 (general dry-run) + Beat 121 (Watch without dry-run): dry-run may report wrote_item=True while plan bytes stay unchanged. Proof: pytest ``-k beat122``.
Beat 123: session-b ``--decode-only --json`` Watch-fixture CLI matrix via tmp_path (watch_only→idle; watch_lookalike / watch_queue_heartbeat→picked; two_queue→failed); plan untouched; keep_schedule / no scheduler_delete. Completes Watch trilogy with Beat 118 (idle-decode) + Beat 119 (dry-run). Proof: pytest ``-k beat123``.
Beat 124: idle-decode ``--json`` non-pickable OPEN CLI matrix via tmp_path (shipped_only / incomplete_open / broken_ready_flag / open_complete_not_ready→idle); plan untouched; keep_schedule / no scheduler_delete. Distinct from Beat 113 (ready picks) + Beat 118 (Watch). Proof: pytest ``-k beat124``.
Beat 125: session-b ``--dry-run --json`` non-pickable OPEN CLI matrix via tmp_path (shipped_only / incomplete_open / broken_ready_flag / open_complete_not_ready→idle, never dry_run); plan untouched; keep_schedule / no scheduler_delete. Pairs with Beat 124 (idle-decode same fixtures). Proof: pytest ``-k beat125``.
Beat 126: session-b ``--json`` (default decode-only path) non-pickable OPEN CLI matrix via tmp_path (shipped_only / incomplete_open / broken_ready_flag / open_complete_not_ready→idle, never dry_run); plan untouched; keep_schedule / no scheduler_delete. Completes non-pickable trilogy with Beat 124 (idle-decode) + Beat 125 (dry-run). Proof: pytest ``-k beat126``.
Beat 127: session-a ``--no-stub|--stub --json`` non-pickable OPEN CLI matrix via tmp_path (shipped_only / incomplete_open / broken_ready_flag / open_complete_not_ready count as OPEN=0: --no-stub→failed plan unchanged; --stub→queued write open_count=1 under tmp only). Contrast Beats 115 (general) + 124–126 (Session B / idle non-pickable). Proof: pytest ``-k beat127``.
Beat 128: session-a ``--no-stub|--stub --dry-run --json`` non-pickable OPEN CLI matrix via tmp_path (shipped_only / incomplete_open --no-stub dry-run→failed wrote_item=False plan unchanged; broken_ready_flag / open_complete_not_ready --stub dry-run→queued wrote_item=True open_count=1 but plan bytes unchanged). Contrast Beat 127 (same fixtures without dry-run): dry-run may report wrote_item=True while plan bytes stay unchanged. Proof: pytest ``-k beat128``.
Beat 129: ``mixed_priority`` cross-CLI ``--json`` matrix via tmp_path (first ready OPEN is Q2; skips incomplete rows): idle-decode + session-b ``--decode-only`` both pick ``item_id=Q2`` (keep_schedule / no scheduler_delete); session-b ``--dry-run``→dry_run Q2; session-a ``--stub``→light recount ``b_pick_title=Second ready complete item``; plans unchanged on B/idle; Session A light only. Proof: pytest ``-k beat129``.
Beat 130: milestone — top-level ``--help`` + public API through beat130 (A/B runners + queue/decode/serialize helpers + guards); Makefile echo reaches beat130. Proof: pytest ``-k beat130``.
Beat 131: ``two_open_ready`` cross-CLI ``--json`` matrix via tmp_path (first ready OPEN is Q1; open_count stays 2): idle-decode + session-b ``--decode-only`` both pick ``item_id=Q1`` (keep_schedule / no scheduler_delete); session-b ``--dry-run``→dry_run Q1; session-a ``--stub``→light recount ``b_pick_title=First ready complete item``; plans unchanged. Parallel to Beat 129 (mixed_priority→Q2). Proof: pytest ``-k beat131``.
Beat 132: ``one_open_ready`` cross-CLI ``--json`` matrix via tmp_path (single ready OPEN is Q1; open_count=1): idle-decode + session-b ``--decode-only`` both pick ``item_id=Q1`` (keep_schedule / no scheduler_delete); session-b ``--dry-run``→dry_run Q1; session-a ``--stub``→light recount ``b_pick_title=Add fixture unit test for queue parser``; plans unchanged. Completes cross-CLI pick trilogy with Beat 129 (mixed→Q2) + Beat 131 (two_open→Q1 open_count=2). Proof: pytest ``-k beat132``.
Beat 133: ``watch_lookalike`` cross-CLI ``--json`` matrix via tmp_path (picks real Queue OPEN Q3; never Watch lookalike; open_count=1): idle-decode + session-b ``--decode-only`` both pick ``item_id=Q3`` (keep_schedule / no scheduler_delete); session-b ``--dry-run``→dry_run Q3; session-a ``--stub``→light recount ``b_pick_title=Real ready Queue item``; plans unchanged. Distinct from Beat 118 (idle-decode Watch-only matrix) and Beats 129/131/132 (non-Watch pick trilogy). Proof: pytest ``-k beat133``.
Beat 134: ``watch_queue_heartbeat`` cross-CLI ``--json`` matrix via tmp_path (picks Queue OPEN Q1 while preserving Watch/Heartbeat markers; open_count=1): idle-decode + session-b ``--decode-only`` both pick ``item_id=Q1`` (keep_schedule / no scheduler_delete; ``## Watch`` + ``## Heartbeat`` intact); session-b ``--dry-run``→dry_run Q1; session-a ``--stub``→light recount ``b_pick_title=Beat19 shippable preserve item``; plans unchanged. Distinct from Beat 133 (watch_lookalike→Q3) and Beats 118/119/123 (Watch CLI matrices without full cross-CLI). Proof: pytest ``-k beat134``.
Beat 135: ``watch_only_lookalike`` cross-CLI ``--json`` matrix via tmp_path (empty Queue; Watch lookalike never B-picked; open_count=0 idle): idle-decode + session-b ``--decode-only`` both idle (keep_schedule / no scheduler_delete; ``## Watch`` + ``## Heartbeat`` intact); session-b ``--dry-run``→idle (NOT dry_run); session-a ``--stub``→queued wrote_item=True open_count=1 with Watch/Heartbeat still present after write. Idle counterpart to Beat 133 + Beat 134. Proof: pytest ``-k beat135``.
Beat 136: ``watch_heartbeat_no_queue`` cross-CLI ``--json`` matrix via tmp_path (no Queue section initially; B paths idle; Session A ``--stub`` creates ``## Queue`` while keeping Watch/Heartbeat). Completes Watch idle trilogy with Beat 135 (watch_only_lookalike). Proof: pytest ``-k beat136``.
Beat 137: ``empty_queue`` cross-CLI ``--json`` matrix via tmp_path (foundational baseline after Watch idle trilogy 135/136): idle-decode + session-b ``--decode-only`` both idle (keep_schedule / no scheduler_delete; open_count=0 queue 0/10; plan unchanged); session-b ``--dry-run``→idle (NOT dry_run); session-a ``--stub``→queued wrote_item=True open_count=1 queue 1/10 (tmp write); session-a ``--no-stub``→rc=1 failed wrote_item=False plan unchanged. Proof: pytest ``-k beat137``.
Beat 138: ``watch_queue_heartbeat_empty`` cross-CLI ``--json`` matrix via tmp_path (empty Queue WITH Watch/Heartbeat markers; after plain empty_queue Beat 137): idle-decode + session-b ``--decode-only`` both idle (keep_schedule / no scheduler_delete; open_count=0 queue 0/10; plan unchanged; ``## Watch`` + ``## Heartbeat`` kept); session-b ``--dry-run``→idle (NOT dry_run); session-a ``--stub``→queued wrote_item=True open_count=1 queue 1/10 with Watch/Heartbeat still present (tmp write); session-a ``--no-stub``→rc=1 failed wrote_item=False plan unchanged markers intact. Proof: pytest ``-k beat138``.
Beat 139: ``two_queue_sections`` fail-closed cross-CLI ``--json`` matrix via tmp_path (ambiguous dual Queue sections; every path fails; plan unchanged): idle-decode + session-b ``--decode-only`` + session-b ``--dry-run`` all failed (rc=1; open_count=0 queue 0/10; keep_schedule / no scheduler_delete; NOT dry_run); session-a ``--stub`` / ``--no-stub`` / ``--stub --dry-run`` all failed wrote_item=False; plans unchanged. Proof: pytest ``-k beat139``.
Beat 140: milestone — top-level ``--help`` + public API through beat140 (A/B runners + queue/decode/serialize helpers + guards); Makefile echo reaches beat140. Proof: pytest ``-k beat140``.
Beat 141: ``shipped_only`` cross-CLI ``--json`` matrix via tmp_path (foundational post-milestone baseline; only SHIPPED items; open_count=0): idle-decode + session-b ``--decode-only`` both idle (keep_schedule / no scheduler_delete; open_count=0 queue 0/10; plan unchanged); session-b ``--dry-run``→idle (NOT dry_run); session-a ``--stub``→queued wrote_item=True open_count=1 queue 1/10 title Stub shippable change (tmp write); session-a ``--no-stub``→rc=1 failed wrote_item=False plan unchanged. Proof: pytest ``-k beat141``.
Beat 142: ``incomplete_open`` cross-CLI ``--json`` matrix via tmp_path (foundational non-pickable OPEN after shipped_only Beat 141; incomplete OPEN missing fields; open_count=0): idle-decode + session-b ``--decode-only`` both idle (keep_schedule / no scheduler_delete; open_count=0 queue 0/10; plan unchanged); session-b ``--dry-run``→idle (NOT dry_run); session-a ``--stub``→queued wrote_item=True open_count=1 queue 1/10 title Stub shippable change Heartbeat kept (tmp write); session-a ``--no-stub``→rc=1 failed wrote_item=False plan unchanged. Proof: pytest ``-k beat142``.
Beat 143: ``broken_ready_flag`` cross-CLI ``--json`` matrix via tmp_path (non-pickable OPEN with broken ready flag after incomplete_open Beat 142; open_count=0): idle-decode + session-b ``--decode-only`` both idle (keep_schedule / no scheduler_delete; open_count=0 queue 0/10; plan unchanged); session-b ``--dry-run``→idle (NOT dry_run); session-a ``--stub``→queued wrote_item=True open_count=1 queue 1/10 title Stub shippable change (tmp write); session-a ``--no-stub``→rc=1 failed wrote_item=False plan unchanged. Proof: pytest ``-k beat143``.
Beat 144: ``open_complete_not_ready`` cross-CLI ``--json`` matrix via tmp_path (completes non-pickable foundational quartet after shipped_only/incomplete_open/broken_ready_flag Beats 141–143; complete six-field OPEN but Ready≠yes; open_count=0): idle-decode + session-b ``--decode-only`` both idle (keep_schedule / no scheduler_delete; open_count=0 queue 0/10; plan unchanged); session-b ``--dry-run``→idle (NOT dry_run); session-a ``--stub``→queued wrote_item=True open_count=1 queue 1/10 title Stub shippable change (tmp write); session-a ``--no-stub``→rc=1 failed wrote_item=False plan unchanged. Proof: pytest ``-k beat144``.
Beat 145: ``queue_with_watch_heartbeat`` cross-CLI ``--json`` matrix via tmp_path (Watch *before* Queue + Heartbeat; open_count=0 idle baseline; stub queues while keeping Watch/Heartbeat markers): idle-decode + session-b ``--decode-only`` both idle (keep_schedule / no scheduler_delete; open_count=0 queue 0/10; plan unchanged; ``## Watch`` + ``## Heartbeat`` present; Watch before Queue); session-b ``--dry-run``→idle (NOT dry_run); session-a ``--stub``→queued wrote_item=True open_count=1 queue 1/10 title Stub shippable change with Watch/Heartbeat still present (tmp write); session-a ``--no-stub``→rc=1 failed wrote_item=False plan unchanged markers intact. Proof: pytest ``-k beat145``.
Beat 146: ``contract_spec`` cross-CLI ``--json`` matrix via tmp_path (foundational contract-plan baseline; open_count=0 idle; stub queues; no-stub fails): idle-decode + session-b ``--decode-only`` both idle (keep_schedule / no scheduler_delete; open_count=0 queue 0/10; plan unchanged); session-b ``--dry-run``→idle (NOT dry_run); session-a ``--stub``→queued wrote_item=True open_count=1 queue 1/10 title Stub shippable change (tmp write); session-a ``--no-stub``→rc=1 failed wrote_item=False plan unchanged. Proof: pytest ``-k beat146``.
Beat 147: ``shipped_only`` session-a ``--stub|--no-stub --dry-run --json`` matrix via tmp_path (dry-run extension of Beat 141; stub reports wrote_item=True but plan unchanged; no-stub fails): idle-decode + session-b ``--dry-run`` both idle (keep_schedule / no scheduler_delete; open_count=0 queue 0/10; plan unchanged; B NOT dry_run); session-a ``--stub --dry-run``→queued wrote_item=True open_count=1 queue 1/10 title Stub shippable change but plan UNCHANGED; session-a ``--no-stub --dry-run``→rc=1 failed wrote_item=False plan unchanged. Contrast Beat 141 (same fixture without Session A dry-run): dry-run may report wrote_item=True while plan bytes stay unchanged. Proof: pytest ``-k beat147``.
Beat 148: ``incomplete_open`` session-a ``--stub|--no-stub --dry-run --json`` matrix via tmp_path (dry-run extension of Beat 142; mirrors Beat 147 shipped_only dry-run shape; stub reports wrote_item=True but plan unchanged; no-stub fails): idle-decode + session-b ``--dry-run`` both idle (keep_schedule / no scheduler_delete; open_count=0 queue 0/10; plan unchanged; B NOT dry_run); session-a ``--stub --dry-run``→queued wrote_item=True open_count=1 queue 1/10 title Stub shippable change but plan UNCHANGED; session-a ``--no-stub --dry-run``→rc=1 failed wrote_item=False plan unchanged. Contrast Beat 142 (same fixture without Session A dry-run): dry-run may report wrote_item=True while plan bytes stay unchanged. Proof: pytest ``-k beat148``.
Beat 149: ``broken_ready_flag`` session-a ``--stub|--no-stub --dry-run --json`` matrix via tmp_path (dry-run extension of Beat 143; mirrors Beats 147–148 dry-run shape; stub reports wrote_item=True but plan unchanged; no-stub fails): idle-decode + session-b ``--dry-run`` both idle (keep_schedule / no scheduler_delete; open_count=0 queue 0/10; plan unchanged; B NOT dry_run); session-a ``--stub --dry-run``→queued wrote_item=True open_count=1 queue 1/10 title Stub shippable change but plan UNCHANGED; session-a ``--no-stub --dry-run``→rc=1 failed wrote_item=False plan unchanged. Contrast Beat 143 (same fixture without Session A dry-run): dry-run may report wrote_item=True while plan bytes stay unchanged. Proof: pytest ``-k beat149``.
Beat 150: milestone — top-level ``--help`` + public API through beat150 (A/B runners + queue/decode/serialize helpers + guards); Makefile echo reaches beat150. Proof: pytest ``-k beat150``.
Beat 151: ``open_complete_not_ready`` session-a ``--stub|--no-stub --dry-run --json`` matrix via tmp_path (completes non-pickable dry-run quartet after shipped_only/incomplete_open/broken_ready_flag Beats 147–149; stub reports wrote_item=True but plan unchanged; no-stub fails): idle-decode + session-b ``--dry-run`` both idle (keep_schedule / no scheduler_delete; open_count=0 queue 0/10; plan unchanged; B NOT dry_run); session-a ``--stub --dry-run``→queued wrote_item=True open_count=1 queue 1/10 title Stub shippable change but plan UNCHANGED; session-a ``--no-stub --dry-run``→rc=1 failed wrote_item=False plan unchanged. Contrast Beat 144 (same fixture without Session A dry-run): dry-run may report wrote_item=True while plan bytes stay unchanged. Proof: pytest ``-k beat151``.
Beat 152: ``contract_spec`` session-a ``--stub|--no-stub --dry-run --json`` matrix via tmp_path (dry-run extension of Beat 146; mirrors Beats 147–151 dry-run shape; stub reports wrote_item=True but plan unchanged; no-stub fails): idle-decode + session-b ``--dry-run`` both idle (keep_schedule / no scheduler_delete; open_count=0 queue 0/10; plan unchanged; B NOT dry_run); session-a ``--stub --dry-run``→queued wrote_item=True open_count=1 queue 1/10 title Stub shippable change but plan UNCHANGED; session-a ``--no-stub --dry-run``→rc=1 failed wrote_item=False plan unchanged. Contrast Beat 146 (same fixture without Session A dry-run): dry-run may report wrote_item=True while plan bytes stay unchanged. Proof: pytest ``-k beat152``.
Beat 153: ``queue_with_watch_heartbeat`` session-a ``--stub|--no-stub --dry-run --json`` matrix via tmp_path (dry-run extension of Beat 145; Watch *before* Queue + Heartbeat; stub reports wrote_item=True but plan unchanged with markers intact; no-stub fails): idle-decode + session-b ``--dry-run`` both idle (keep_schedule / no scheduler_delete; open_count=0 queue 0/10; plan unchanged; ``## Watch`` + ``## Heartbeat`` present; Watch before Queue; B NOT dry_run); session-a ``--stub --dry-run``→queued wrote_item=True open_count=1 queue 1/10 title Stub shippable change but plan UNCHANGED Watch/Heartbeat still present; session-a ``--no-stub --dry-run``→rc=1 failed wrote_item=False plan unchanged markers intact. Contrast Beat 145 (same fixture without Session A dry-run): dry-run may report wrote_item=True while plan bytes stay unchanged. Proof: pytest ``-k beat153``.
Beat 154: ``empty_queue`` session-a ``--stub|--no-stub --dry-run --json`` matrix via tmp_path (dry-run extension of Beat 137; mirrors Beats 147–153 dry-run shape; stub reports wrote_item=True but plan unchanged; no-stub fails): idle-decode + session-b ``--dry-run`` both idle (keep_schedule / no scheduler_delete; open_count=0 queue 0/10; plan unchanged; B NOT dry_run); session-a ``--stub --dry-run``→queued wrote_item=True open_count=1 queue 1/10 title Stub shippable change but plan UNCHANGED; session-a ``--no-stub --dry-run``→rc=1 failed wrote_item=False plan unchanged. Contrast Beat 137 (same fixture without Session A dry-run): dry-run may report wrote_item=True while plan bytes stay unchanged. Proof: pytest ``-k beat154``.
Beat 155: ``watch_queue_heartbeat_empty`` session-a ``--stub|--no-stub --dry-run --json`` matrix via tmp_path (dry-run extension of Beat 138; mirrors Beat 154 empty_queue dry-run + marker keep from 153; stub reports wrote_item=True but plan unchanged with markers intact; no-stub fails): idle-decode + session-b ``--dry-run`` both idle (keep_schedule / no scheduler_delete; open_count=0 queue 0/10; plan unchanged; ``## Watch`` + ``## Heartbeat`` present; B NOT dry_run); session-a ``--stub --dry-run``→queued wrote_item=True open_count=1 queue 1/10 title Stub shippable change but plan UNCHANGED Watch/Heartbeat still present; session-a ``--no-stub --dry-run``→rc=1 failed wrote_item=False plan unchanged markers intact. Contrast Beat 138 (same fixture without Session A dry-run): dry-run may report wrote_item=True while plan bytes stay unchanged. Proof: pytest ``-k beat155``.
Beat 156: ``watch_heartbeat_no_queue`` session-a ``--stub|--no-stub --dry-run --json`` matrix via tmp_path (dry-run extension of Beat 136; mirrors Beat 155 marker-keep dry-run but fixture has no ``## Queue`` initially; stub reports wrote_item=True but plan UNCHANGED still no Queue section; no-stub fails): idle-decode + session-b ``--dry-run`` both idle (keep_schedule / no scheduler_delete; open_count=0 queue 0/10; plan unchanged; ``## Watch`` + ``## Heartbeat`` present; no ``## Queue``; B NOT dry_run); session-a ``--stub --dry-run``→queued wrote_item=True open_count=1 queue 1/10 title Stub shippable change but plan UNCHANGED still no ``## Queue`` Watch/Heartbeat kept; session-a ``--no-stub --dry-run``→rc=1 failed wrote_item=False plan unchanged. Contrast Beat 136 (same fixture without Session A dry-run): Beat 136 stub actually creates ``## Queue``; dry-run reports wrote_item=True while plan bytes stay unchanged (no Queue created). Proof: pytest ``-k beat156``.
Beat 157: ``watch_only_lookalike`` session-a ``--stub|--no-stub --dry-run --json`` matrix via tmp_path (dry-run extension of Beat 135; Watch lookalike never B-picked; stub reports wrote_item=True but plan UNCHANGED keep markers; no-stub fails): idle-decode + session-b ``--dry-run`` both idle (keep_schedule / no scheduler_delete; open_count=0 queue 0/10; plan unchanged; ``## Watch`` + ``## Heartbeat`` present; B NOT dry_run); session-a ``--stub --dry-run``→queued wrote_item=True open_count=1 queue 1/10 title Stub shippable change but plan UNCHANGED Watch/Heartbeat still present; session-a ``--no-stub --dry-run``→rc=1 failed wrote_item=False plan unchanged. Contrast Beat 135 (same fixture without Session A dry-run): Beat 135 stub actually writes; dry-run reports wrote_item=True while plan bytes stay unchanged. Proof: pytest ``-k beat157``.
Beat 158: ``two_queue_sections`` session-a ``--stub|--no-stub --dry-run --json`` fail-closed matrix via tmp_path (dry-run extension of Beat 139; ambiguous dual Queue sections; every path still fails; wrote_item=False; plan unchanged): idle-decode + session-b ``--dry-run`` both failed (rc=1; open_count=0 queue 0/10; keep_schedule / no scheduler_delete; B NOT dry_run; plan unchanged); session-a ``--stub --dry-run``→rc=1 failed wrote_item=False plan unchanged; session-a ``--no-stub --dry-run``→rc=1 failed wrote_item=False plan unchanged. Contrast Beat 139 (same fixture broader cross-CLI including decode-only / non-dry-run stub); Beat 158 focuses the dry-run fail-closed matrix. Proof: pytest ``-k beat158``.
Beat 159: ``watch_lookalike`` dry-run / light ``--json`` matrix via tmp_path (pre-milestone pickable contrast after fail-closed Beat 158; B dry-run picks real Queue OPEN Q3 not Watch lookalike; Session A ``--stub|--no-stub --dry-run`` both light): idle-decode →picked open_count=1 keep_schedule plan unchanged; session-b ``--dry-run``→dry_run open_count=1 keep_schedule plan unchanged; session-a ``--stub|--no-stub --dry-run`` both light wrote_item=False open_count=1 ``b_pick_title=Real ready Queue item`` plan unchanged. Contrast Beat 158 (fail-closed) + Beat 133 (same fixture broader cross-CLI; Beat 159 focuses dry-run / light matrix). Proof: pytest ``-k beat159``.
Beat 160: milestone — top-level ``--help`` + public API through beat160 (A/B runners + queue/decode/serialize helpers + guards); Makefile echo reaches beat160. Proof: pytest ``-k beat160``.
Beat 161: ``one_open_ready`` dry-run / light ``--json`` matrix via tmp_path (post-milestone pickable baseline; mirrors Beat 159 ``watch_lookalike`` dry-run/light shape; B dry-run picks Q1; Session A ``--stub|--no-stub --dry-run`` both light): idle-decode →picked open_count=1 keep_schedule plan unchanged; session-b ``--dry-run``→dry_run open_count=1 keep_schedule plan unchanged; session-a ``--stub|--no-stub --dry-run`` both light wrote_item=False open_count=1 ``b_pick_title=Add fixture unit test for queue parser`` plan unchanged. Contrast Beat 159 (watch_lookalike→Q3) + Beat 160 (milestone). Proof: pytest ``-k beat161``.
Beat 162: ``two_open_ready`` dry-run / light ``--json`` matrix via tmp_path (mirrors Beat 161 ``one_open_ready`` dry-run/light shape with open_count=2 / queue 2/10; B dry-run picks Q1; Session A ``--stub|--no-stub --dry-run`` both light): idle-decode →picked open_count=2 keep_schedule plan unchanged; session-b ``--dry-run``→dry_run open_count=2 keep_schedule plan unchanged; session-a ``--stub|--no-stub --dry-run`` both light wrote_item=False open_count=2 ``b_pick_title=First ready complete item`` plan unchanged. Contrast Beat 161 (one_open_ready open_count=1). Proof: pytest ``-k beat162``.
Beat 163: ``mixed_priority`` dry-run / light ``--json`` matrix via tmp_path (mirrors Beat 161/162 dry-run/light shape; incomplete Q1 skipped; B picks first complete six-field ready OPEN Q2; open_count=1 / queue 1/10; Session A ``--stub|--no-stub --dry-run`` both light): idle-decode →picked open_count=1 keep_schedule plan unchanged; session-b ``--dry-run``→dry_run open_count=1 keep_schedule plan unchanged; session-a ``--stub|--no-stub --dry-run`` both light wrote_item=False open_count=1 ``b_pick_title=Second ready complete item`` plan unchanged. Contrast Beat 162 (two_open_ready→Q1) + Beat 129 (same fixture broader decode-only/non-dry-run A). Proof: pytest ``-k beat163``.
Beat 164: ``watch_queue_heartbeat`` dry-run / light ``--json`` matrix via tmp_path (mirrors Beat 161–163 dry-run/light shape with Watch/Heartbeat markers kept; B dry-run picks Q1; Session A ``--stub|--no-stub --dry-run`` both light): idle-decode →picked open_count=1 keep_schedule markers intact plan unchanged; session-b ``--dry-run``→dry_run open_count=1 keep_schedule markers intact plan unchanged; session-a ``--stub|--no-stub --dry-run`` both light wrote_item=False open_count=1 ``b_pick_title=Beat19 shippable preserve item`` plan unchanged. Contrast Beat 163 (mixed_priority→Q2 no markers) + Beat 134 (same fixture broader cross-CLI; Beat 164 focuses dry-run / light matrix). Proof: pytest ``-k beat164``.
Beat 165: multi-fixture pickable dry-run / light ``--json`` smoke via tmp_path (closes the pickable dry-run/light series across ``watch_lookalike`` / ``one_open_ready`` / ``two_open_ready`` / ``mixed_priority`` / ``watch_queue_heartbeat``): for each fixture idle-decode →picked correct open_count/queue/item_id keep_schedule plan unchanged; session-b ``--dry-run``→dry_run same pick ids/counts keep_schedule plan unchanged; session-a ``--stub|--no-stub --dry-run`` both light wrote_item=False correct open_count/queue/b_pick_title plan unchanged; ``watch_queue_heartbeat`` also keeps ``## Watch`` + ``## Heartbeat`` + BEAT19 markers. Contrast Beat 164 (single watch_queue_heartbeat) + Beats 159/161–163 (single-fixture series); Beat 165 consolidates. Proof: pytest ``-k beat165``.
Beat 166: multi-fixture idle dry-run ``--json`` smoke via tmp_path (idle counterpart to Beat 165 pickable smoke; consolidates Beats 154–157 across ``empty_queue`` / ``watch_queue_heartbeat_empty`` / ``watch_heartbeat_no_queue`` / ``watch_only_lookalike``): for each fixture idle-decode + session-b ``--dry-run``→both idle open_count=0 queue 0/10 keep_schedule no scheduler_delete plan unchanged (B NOT dry_run); session-a ``--stub --dry-run``→queued wrote_item=True open_count=1 queue 1/10 title Stub shippable change but plan UNCHANGED; session-a ``--no-stub --dry-run``→rc=1 failed wrote_item=False plan unchanged; Watch fixtures keep ``## Watch`` + ``## Heartbeat``; ``watch_heartbeat_no_queue`` still has no ``## Queue``. Contrast Beat 165 (pickable multi-fixture light) + Beats 154–157 (single-fixture idle dry-run); Beat 166 consolidates idle series. Proof: pytest ``-k beat166``.
Beat 167: multi-fixture non-pickable dry-run ``--json`` smoke via tmp_path (consolidates Beats 147–153 across ``shipped_only`` / ``incomplete_open`` / ``broken_ready_flag`` / ``open_complete_not_ready`` / ``contract_spec`` / ``queue_with_watch_heartbeat``; third consolidation after 165 pickable + 166 idle): for each fixture idle-decode + session-b ``--dry-run``→both idle open_count=0 queue 0/10 keep_schedule no scheduler_delete plan unchanged (B NOT dry_run); session-a ``--stub --dry-run``→queued wrote_item=True open_count=1 queue 1/10 title Stub shippable change but plan UNCHANGED; session-a ``--no-stub --dry-run``→rc=1 failed wrote_item=False plan unchanged; ``queue_with_watch_heartbeat`` keeps ``## Watch`` + ``## Heartbeat`` with Watch before Queue. Contrast Beat 166 (idle Watch/empty smoke) + Beats 147–153 (single-fixture non-pickable dry-run); Beat 167 consolidates non-pickable series. Proof: pytest ``-k beat167``.
Beat 168: multi-fixture verdict-spectrum dry-run / decode ``--json`` smoke via tmp_path (crosses pickable / idle / fail-closed after Beats 165–167 class consolidations + Beat 158 fail-closed alone): ``one_open_ready`` idle-decode + session-b ``--decode-only``→picked Q1 keep_schedule plan unchanged; session-b ``--dry-run``→dry_run Q1 keep_schedule plan unchanged; session-a ``--stub|--no-stub --dry-run`` both light wrote_item=False open_count=1 ``b_pick_title=Add fixture unit test for queue parser`` plan unchanged; ``empty_queue`` idle-decode + session-b ``--dry-run``→both idle open_count=0 queue 0/10 keep_schedule plan unchanged (B NOT dry_run); session-a ``--stub --dry-run``→queued wrote_item=True open_count=1 queue 1/10 title Stub shippable change but plan UNCHANGED; session-a ``--no-stub --dry-run``→rc=1 failed wrote_item=False plan unchanged; ``two_queue_sections`` idle-decode + session-b ``--dry-run``→both failed (rc=1; open_count=0 queue 0/10; keep_schedule / no scheduler_delete; B NOT dry_run; plan unchanged); session-a ``--stub|--no-stub --dry-run`` both failed wrote_item=False plan unchanged. Contrast Beats 165–167 (class consolidations) + Beat 158 (fail-closed alone); Beat 168 crosses pickable/idle/fail-closed in one spectrum. Proof: pytest ``-k beat168``.
Beat 169: multi-fixture idle-decode ≡ session-b ``--decode-only`` ``--json`` smoke via tmp_path (closes the idle-decode ≡ session-b ``--decode-only`` multi-fixture gap after Beat 165 pickable dry-run/light + Beat 168 spectrum with one decode-only): pickables (``watch_lookalike`` / ``one_open_ready`` / ``two_open_ready`` / ``mixed_priority`` / ``watch_queue_heartbeat``) idle-decode + session-b ``--decode-only`` both →picked same open_count/queue/item_id keep_schedule no scheduler_delete plan unchanged; ``watch_queue_heartbeat`` keeps ``## Watch`` + ``## Heartbeat`` + BEAT19 markers; idle/fail (``empty_queue`` / ``watch_only_lookalike`` idle; ``two_queue_sections`` failed) same verdict/rc/open_count/queue keep_schedule (decode-only NOT dry_run). Contrast Beat 165 (pickable dry-run/light) + Beat 168 (spectrum with one decode-only); Beat 169 is pre-milestone decode-only equivalence smoke. Next beat 170 = milestone. Proof: pytest ``-k beat169``.
Beat 170: milestone — top-level ``--help`` + public API through beat170 (A/B runners + queue/decode/serialize helpers + guards); Makefile echo reaches beat170. Proof: pytest ``-k beat170``.
Beat 171: multi-fixture Session A ``--stub`` write vs ``--stub --dry-run`` no-write ``--json`` contrast via tmp_path (post-milestone; proves dry-run reports wrote_item without mutating plan bytes; fixtures ``empty_queue`` / ``watch_queue_heartbeat_empty`` / ``watch_heartbeat_no_queue`` / ``watch_only_lookalike``): for each fixture session-a ``--stub --json``→queued wrote_item=True open_count=1 queue 1/10 title Stub shippable change plan CHANGED ``## Queue`` present (Watch fixtures keep ``## Watch`` + ``## Heartbeat``; ``watch_heartbeat_no_queue`` creates ``## Queue``); session-a ``--stub --dry-run --json``→queued wrote_item=True open_count=1 queue 1/10 title Stub shippable change but plan UNCHANGED (``watch_heartbeat_no_queue`` still no ``## Queue``; Watch fixtures keep markers). Contrast Beat 170 (milestone) + Beats 154–157/166 (dry-run-only idle matrices) + Beat 136 (stub actually creates Queue on watch_heartbeat_no_queue); Beat 171 pairs live stub write vs dry-run no-write. Proof: pytest ``-k beat171``.
Beat 172: multi-fixture non-pickable Session A ``--stub`` write vs ``--stub --dry-run`` no-write ``--json`` contrast via tmp_path (parallel to Beat 171 idle contrast; fixtures ``shipped_only`` / ``incomplete_open`` / ``broken_ready_flag`` / ``open_complete_not_ready`` / ``contract_spec`` / ``queue_with_watch_heartbeat``): for each fixture session-a ``--stub --json``→queued wrote_item=True open_count=1 queue 1/10 title Stub shippable change plan CHANGED; session-a ``--stub --dry-run --json``→queued wrote_item=True open_count=1 queue 1/10 title Stub shippable change but plan UNCHANGED; ``queue_with_watch_heartbeat`` keeps ``## Watch`` + ``## Heartbeat`` with Watch before Queue after write and dry-run. Contrast Beat 171 (idle stub write vs dry-run) + Beat 167 (non-pickable dry-run-only smoke); Beat 172 pairs live stub write vs dry-run for non-pickables. Proof: pytest ``-k beat172``.
Beat 173: multi-fixture pickable Session A light no-write ``--json`` contrast via tmp_path (completes the 171/172/173 trilogy: idle write / non-pickable write / pickable never-write; fixtures ``watch_lookalike`` / ``one_open_ready`` / ``two_open_ready`` / ``mixed_priority`` / ``watch_queue_heartbeat``): for each fixture session-a ``--stub|--no-stub --json`` (NO dry-run) both light wrote_item=False correct open_count/queue/b_pick_title plan UNCHANGED; session-a ``--stub|--no-stub --dry-run --json`` both light wrote_item=False same shape plan UNCHANGED; ``watch_queue_heartbeat`` keeps ``## Watch`` + ``## Heartbeat`` + BEAT19 markers. Contrast Beat 171 (idle stub mutates) + Beat 172 (non-pickable stub mutates) + Beat 165 (pickable dry-run light only); Beat 173 proves pickable OPEN>=1 never appends for stub|no-stub × dry-run|live. Proof: pytest ``-k beat173``.
Beat 174: multi-fixture idle/non-pickable/fail-closed Session A ``--no-stub`` live vs ``--no-stub --dry-run`` fail ``--json`` contrast via tmp_path (counterpart to Beats 171–173 stub write / light trilogy; fixtures ``empty_queue`` / ``watch_only_lookalike`` / ``shipped_only`` / ``incomplete_open`` / ``broken_ready_flag`` / ``open_complete_not_ready`` / ``contract_spec`` / ``queue_with_watch_heartbeat`` / ``two_queue_sections``): for each fixture session-a ``--no-stub --json`` (NO dry-run)→rc=1 failed wrote_item=False plan UNCHANGED; session-a ``--no-stub --dry-run --json``→rc=1 failed wrote_item=False plan UNCHANGED; Watch fixtures keep ``## Watch`` + ``## Heartbeat``; ``queue_with_watch_heartbeat`` keeps Watch before Queue. Contrast Beats 171–173 (stub write / light) + prior dry-run fail matrices; Beat 174 proves ``--no-stub`` fails live and dry-run without mutating plans. Proof: pytest ``-k beat174``.
Beat 175: multi-fixture Session A ``--candidate-json`` write vs ``--candidate-json --dry-run`` no-write ``--json`` contrast via tmp_path (same idle fixtures as Beat 171; candidate ``complete_candidate.json`` copied to tmp_path): for each fixture session-a ``--candidate-json`` ``--json``→queued wrote_item=True open_count=1 queue 1/10 title Complete candidate fixture plan CHANGED ``## Queue`` present (Watch fixtures keep ``## Watch`` + ``## Heartbeat``; ``watch_heartbeat_no_queue`` creates ``## Queue``); session-a ``--candidate-json`` ``--dry-run --json``→queued wrote_item=True open_count=1 queue 1/10 title Complete candidate fixture but plan UNCHANGED (``watch_heartbeat_no_queue`` still no ``## Queue``; Watch fixtures keep markers). Contrast Beat 171 (stub write vs dry-run) + Beat 174 (``--no-stub`` fail); Beat 175 is candidate-json write vs dry-run on idle fixtures. Proof: pytest ``-k beat175``.
Beat 176: multi-fixture idle Session A ``--candidate-json`` **incomplete** fail live vs ``--candidate-json --dry-run`` fail ``--json`` contrast via tmp_path (same idle fixtures as Beat 171/175; candidate ``incomplete_candidate.json`` copied to tmp_path): for each fixture session-a ``--candidate-json`` ``--json`` (NO dry-run)→rc=1 ok=False verdict=failed wrote_item=False open_count=0 queue 0/10 title Incomplete candidate fixture plan UNCHANGED; session-a ``--candidate-json`` ``--dry-run --json``→same fail shape plan UNCHANGED; Watch fixtures keep ``## Watch`` + ``## Heartbeat``; ``watch_heartbeat_no_queue`` still has no ``## Queue``. Contrast Beat 175 (complete candidate write vs dry-run success) + Beat 174 (``--no-stub`` fail); Beat 176 is incomplete-candidate fail live+dry-run on idle fixtures. Proof: pytest ``-k beat176``.
Beat 177: multi-fixture idle Session A ``--candidate-json`` **list** write vs ``--candidate-json --dry-run`` no-write ``--json`` contrast via tmp_path (same idle fixtures as Beat 171/175/176; candidate ``complete_candidate_list.json`` copied to tmp_path): for each fixture session-a ``--candidate-json`` ``--json`` (NO dry-run)→rc=0 ok=True verdict=queued wrote_item=True open_count=1 queue 1/10 title Complete list candidate plan CHANGED ``## Queue`` present (Watch fixtures keep ``## Watch`` + ``## Heartbeat``; ``watch_heartbeat_no_queue`` creates ``## Queue``); session-a ``--candidate-json`` ``--dry-run --json``→queued wrote_item=True open_count=1 queue 1/10 title Complete list candidate but plan UNCHANGED (``watch_heartbeat_no_queue`` still no ``## Queue``; Watch fixtures keep markers). Contrast Beat 175 (complete **dict** write vs dry-run) + Beat 176 (incomplete fail); Beat 177 is complete **list** candidate-json write vs dry-run on idle fixtures. Proof: pytest ``-k beat177``.
Beat 178: multi-fixture idle Session A ``--candidate-json`` **skip-nondict list** write vs ``--candidate-json --dry-run`` no-write ``--json`` contrast via tmp_path (same idle fixtures as Beat 171/175–177; candidate ``complete_candidate_list_skip_nondict.json`` copied to tmp_path): for each fixture session-a ``--candidate-json`` ``--json`` (NO dry-run)→rc=0 ok=True verdict=queued wrote_item=True open_count=1 queue 1/10 title Skip-nondict first dict plan CHANGED ``## Queue`` present (Watch fixtures keep ``## Watch`` + ``## Heartbeat``; ``watch_heartbeat_no_queue`` creates ``## Queue``); session-a ``--candidate-json`` ``--dry-run --json``→queued wrote_item=True open_count=1 queue 1/10 title Skip-nondict first dict but plan UNCHANGED (``watch_heartbeat_no_queue`` still no ``## Queue``; Watch fixtures keep markers). Contrast Beat 177 (complete list write vs dry-run) + Beat 30 (skip-nondict unit); Beat 178 is skip-nondict list candidate-json write vs dry-run on idle fixtures. Proof: pytest ``-k beat178``.
Beat 179: multi-fixture **non-pickable** Session A ``--candidate-json`` complete **dict** write vs ``--candidate-json --dry-run`` no-write ``--json`` contrast via tmp_path (parallel to Beat 172 stub nonpick write + Beat 175 idle candidate write; candidate ``complete_candidate.json`` copied to tmp_path): for each fixture session-a ``--candidate-json`` ``--json`` (NO dry-run)→rc=0 ok=True verdict=queued wrote_item=True open_count=1 queue 1/10 title Complete candidate fixture plan CHANGED SESSION_A_RESULT_JSON_KEYS; session-a ``--candidate-json`` ``--dry-run --json``→queued wrote_item=True open_count=1 queue 1/10 title Complete candidate fixture but plan UNCHANGED; ``queue_with_watch_heartbeat`` keeps ``## Watch`` + ``## Heartbeat`` with Watch before Queue after write and dry-run. Contrast Beat 175 (idle complete dict write vs dry-run) + Beat 172 (nonpick stub write vs dry-run); Beat 179 is complete candidate-json write vs dry-run on non-pickables. Proof: pytest ``-k beat179``.
Beat 180: milestone — top-level ``--help`` + public API through beat180 (A/B runners + queue/decode/serialize helpers + guards); Makefile echo reaches beat180. Proof: pytest ``-k beat180``.
Beat 181: multi-fixture **non-pickable** Session A ``--candidate-json`` **incomplete** fail live vs ``--candidate-json --dry-run`` fail ``--json`` contrast via tmp_path (post-milestone; parallel Beat 176 idle incomplete fail + Beat 179 nonpick complete write; candidate ``incomplete_candidate.json`` copied to tmp_path): for each fixture session-a ``--candidate-json`` ``--json`` (NO dry-run)→rc=1 ok=False verdict=failed wrote_item=False open_count=0 queue 0/10 title Incomplete candidate fixture plan UNCHANGED SESSION_A_RESULT_JSON_KEYS; session-a ``--candidate-json`` ``--dry-run --json``→same fail shape plan UNCHANGED; ``queue_with_watch_heartbeat`` keeps ``## Watch`` + ``## Heartbeat`` with Watch before Queue. Contrast Beat 176 (idle incomplete fail) + Beat 179 (nonpick complete write); Beat 181 is incomplete-candidate fail live+dry-run on non-pickables. Proof: pytest ``-k beat181``.
Beat 182: multi-fixture **non-pickable** Session A ``--candidate-json`` **list** write vs ``--candidate-json --dry-run`` no-write ``--json`` contrast via tmp_path (parallel Beat 177 idle list + Beat 179 nonpick dict; candidate ``complete_candidate_list.json`` copied to tmp_path): for each fixture session-a ``--candidate-json`` ``--json`` (NO dry-run)→rc=0 ok=True verdict=queued wrote_item=True open_count=1 queue 1/10 title Complete list candidate plan CHANGED SESSION_A_RESULT_JSON_KEYS; session-a ``--candidate-json`` ``--dry-run --json``→queued wrote_item=True open_count=1 queue 1/10 title Complete list candidate but plan UNCHANGED; ``queue_with_watch_heartbeat`` keeps ``## Watch`` + ``## Heartbeat`` with Watch before Queue after write and dry-run. Contrast Beat 177 (idle list write vs dry-run) + Beat 179 (nonpick dict write) + Beat 181 (nonpick incomplete fail); Beat 182 is complete **list** candidate-json write vs dry-run on non-pickables. Proof: pytest ``-k beat182``.
Beat 183: multi-fixture **non-pickable** Session A ``--candidate-json`` **skip-nondict list** write vs ``--candidate-json --dry-run`` no-write ``--json`` contrast via tmp_path (parallel Beat 178 idle skip-nondict + Beat 182 nonpick list; candidate ``complete_candidate_list_skip_nondict.json`` copied to tmp_path): for each fixture session-a ``--candidate-json`` ``--json`` (NO dry-run)→rc=0 ok=True verdict=queued wrote_item=True open_count=1 queue 1/10 title Skip-nondict first dict plan CHANGED SESSION_A_RESULT_JSON_KEYS; session-a ``--candidate-json`` ``--dry-run --json``→queued wrote_item=True open_count=1 queue 1/10 title Skip-nondict first dict but plan UNCHANGED; ``queue_with_watch_heartbeat`` keeps ``## Watch`` + ``## Heartbeat`` with Watch before Queue after write and dry-run. Contrast Beat 178 (idle skip-nondict) + Beat 182 (nonpick complete list); Beat 183 is skip-nondict list candidate-json write vs dry-run on non-pickables. Proof: pytest ``-k beat183``.
Beat 184: multi-fixture **pickable** Session A ``--candidate-json`` light no-write ``--json`` contrast via tmp_path (completes candidate trilogy like Beat 173 stub/no-stub: idle write / nonpick write / pickable never-write; candidate ``complete_candidate.json`` copied to tmp_path; OPEN>=1 recount-only ignores candidate-json): for each fixture session-a ``--candidate-json`` ``--json`` (NO dry-run)→rc=0 ok=True verdict=light wrote_item=False correct open_count/queue/b_pick_title plan UNCHANGED SESSION_A_RESULT_JSON_KEYS; session-a ``--candidate-json`` ``--dry-run --json``→same light shape plan UNCHANGED; ``watch_queue_heartbeat`` keeps ``## Watch`` + ``## Heartbeat`` + BEAT19 markers. Contrast Beat 173 (stub|no-stub pickable light) + Beat 179/182/183 (nonpick candidate write); Beat 184 proves pickable OPEN>=1 never appends for --candidate-json × live|dry-run. Proof: pytest ``-k beat184``.
Beat 185: multi-fixture **idle** Session A ``--candidate-json`` **empty list ``[]``** fail live vs ``--candidate-json --dry-run`` fail ``--json`` contrast via tmp_path (post–Beat 184 pickable light; empty list written to tmp_path each test; no stub fallback; same idle fixtures as Beat 171/175/176): for each fixture session-a ``--candidate-json`` ``--json`` (NO dry-run)→rc=1 ok=False verdict=failed wrote_item=False open_count=0 queue 0/10 title is None plan UNCHANGED SESSION_A_RESULT_JSON_KEYS; session-a ``--candidate-json`` ``--dry-run --json``→same fail shape plan UNCHANGED; Watch fixtures keep ``## Watch`` + ``## Heartbeat``; ``watch_heartbeat_no_queue`` still has no ``## Queue``. Contrast Beat 176 (incomplete candidate fail idle) + Beat 28/37 (empty list unit) + Beat 184 (pickable light); Beat 185 is empty-list candidate-json fail live+dry-run on idle fixtures. Proof: pytest ``-k beat185``.
Beat 186: multi-fixture **non-pickable** Session A ``--candidate-json`` **empty list ``[]``** fail live vs ``--candidate-json --dry-run`` fail ``--json`` contrast via tmp_path (parallel Beat 185 idle empty-list fail; empty list written to tmp_path each test; no stub fallback; fixtures ``shipped_only`` / ``incomplete_open`` / ``broken_ready_flag`` / ``open_complete_not_ready`` / ``contract_spec`` / ``queue_with_watch_heartbeat``): for each fixture session-a ``--candidate-json`` ``--json`` (NO dry-run)→rc=1 ok=False verdict=failed wrote_item=False open_count=0 queue 0/10 title is None plan UNCHANGED SESSION_A_RESULT_JSON_KEYS; session-a ``--candidate-json`` ``--dry-run --json``→same fail shape plan UNCHANGED; ``queue_with_watch_heartbeat`` keeps ``## Watch`` + ``## Heartbeat`` with Watch before Queue. Contrast Beat 185 (idle empty-list fail) + Beat 181 (nonpick incomplete fail); Beat 186 is empty-list candidate-json fail live+dry-run on non-pickables. Proof: pytest ``-k beat186``.
Beat 187: multi-fixture **idle** Session A ``--candidate-json`` **no-dict list** (e.g. ``[1, "x", true]``) fail live vs ``--candidate-json --dry-run`` fail ``--json`` contrast via tmp_path (parallel Beat 185 empty-list fail; candidate JSON written to tmp_path each test; no stub fallback; Beat 29 unit → multi-fixture; same idle fixtures as Beat 171/175/185): for each fixture session-a ``--candidate-json`` ``--json`` (NO dry-run)→rc=1 ok=False verdict=failed wrote_item=False open_count=0 queue 0/10 title is None plan UNCHANGED SESSION_A_RESULT_JSON_KEYS; session-a ``--candidate-json`` ``--dry-run --json``→same fail shape plan UNCHANGED; Watch fixtures keep ``## Watch`` + ``## Heartbeat``; ``watch_heartbeat_no_queue`` still has no ``## Queue``. Contrast Beat 185 (empty list ``[]`` fail idle) + Beat 29 (no-dict list unit); Beat 187 is no-dict-list candidate-json fail live+dry-run on idle fixtures. Proof: pytest ``-k beat187``.
Beat 188: multi-fixture **non-pickable** Session A ``--candidate-json`` **no-dict list** (e.g. ``[1, "x", true]``) fail live vs ``--candidate-json --dry-run`` fail ``--json`` contrast via tmp_path (parallel Beat 187 idle no-dict fail; candidate JSON written to tmp_path each test; no stub fallback; fixtures ``shipped_only`` / ``incomplete_open`` / ``broken_ready_flag`` / ``open_complete_not_ready`` / ``contract_spec`` / ``queue_with_watch_heartbeat``): for each fixture session-a ``--candidate-json`` ``--json`` (NO dry-run)→rc=1 ok=False verdict=failed wrote_item=False open_count=0 queue 0/10 title is None plan UNCHANGED SESSION_A_RESULT_JSON_KEYS; session-a ``--candidate-json`` ``--dry-run --json``→same fail shape plan UNCHANGED; ``queue_with_watch_heartbeat`` keeps ``## Watch`` + ``## Heartbeat`` with Watch before Queue. Contrast Beat 187 (idle no-dict fail) + Beat 186 (nonpick empty-list fail); Beat 188 is no-dict-list candidate-json fail live+dry-run on non-pickables. Proof: pytest ``-k beat188``.
Beat 189: multi-fixture **pickable** Session A ``--candidate-json`` **no-dict list** ``[1, "x", true]`` light no-write ``--json`` contrast via tmp_path (OPEN>=1 recount-only ignores candidate-json; parallel Beat 184 complete-candidate pickable light + Beat 187/188 no-dict fails on idle/nonpick): for each fixture session-a ``--candidate-json`` ``--json`` (NO dry-run)→rc=0 ok=True verdict=light wrote_item=False correct open_count/queue/b_pick_title plan UNCHANGED SESSION_A_RESULT_JSON_KEYS; session-a ``--candidate-json`` ``--dry-run --json``→same light shape plan UNCHANGED; ``watch_queue_heartbeat`` keeps ``## Watch`` + ``## Heartbeat`` + BEAT19 markers. Contrast Beat 184 (pickable complete candidate light) + Beat 187/188 (no-dict fail idle/nonpick); Beat 189 proves pickable OPEN>=1 ignores no-dict-list candidate too. Proof: pytest ``-k beat189``.
Beat 190: milestone — top-level ``--help`` + public API through beat190 (A/B runners + queue/decode/serialize helpers + guards); Makefile echo reaches beat190. Proof: pytest ``-k beat190``.
Beat 191: multi-fixture **pickable** Session A ``--candidate-json`` **empty list ``[]``** light no-write ``--json`` contrast via tmp_path (post-milestone; OPEN>=1 recount-only ignores candidate-json; completes empty-list trilogy after Beat 185 idle fail + Beat 186 nonpick fail; parallel Beat 189 no-dict pickable light): for each fixture session-a ``--candidate-json`` ``--json`` (NO dry-run)→rc=0 ok=True verdict=light wrote_item=False correct open_count/queue/b_pick_title plan UNCHANGED SESSION_A_RESULT_JSON_KEYS; session-a ``--candidate-json`` ``--dry-run --json``→same light shape plan UNCHANGED; ``watch_queue_heartbeat`` keeps ``## Watch`` + ``## Heartbeat`` + BEAT19 markers. Contrast Beat 185/186 (empty list fails idle/nonpick) + Beat 189 (no-dict pickable light); Beat 191 proves pickable OPEN>=1 ignores empty-list candidate too. Proof: pytest ``-k beat191``.
Beat 192: multi-fixture **pickable** Session A ``--candidate-json`` **incomplete** light no-write ``--json`` contrast via tmp_path (OPEN>=1 recount-only ignores candidate-json; completes incomplete trilogy after Beat 176 idle fail + Beat 181 nonpick fail; parallel Beat 184/189/191 pickable light for other candidate shapes; candidate ``incomplete_candidate.json`` copied to tmp_path): for each fixture session-a ``--candidate-json`` ``--json`` (NO dry-run)→rc=0 ok=True verdict=light wrote_item=False correct open_count/queue/b_pick_title plan UNCHANGED SESSION_A_RESULT_JSON_KEYS; session-a ``--candidate-json`` ``--dry-run --json``→same light shape plan UNCHANGED; ``watch_queue_heartbeat`` keeps ``## Watch`` + ``## Heartbeat`` + BEAT19 markers. Contrast Beat 176/181 (incomplete fails idle/nonpick) + Beat 191 (empty-list pickable light); Beat 192 proves pickable OPEN>=1 ignores incomplete candidate too. Proof: pytest ``-k beat192``.
Beat 193: multi-fixture **pickable** Session A ``--candidate-json`` **complete list** light no-write ``--json`` contrast via tmp_path (OPEN>=1 recount-only ignores candidate-json; completes complete-list trilogy after Beat 177 idle write + Beat 182 nonpick write; parallel Beat 184 dict / Beat 191 empty-list / Beat 192 incomplete pickable light; candidate ``complete_candidate_list.json`` copied to tmp_path): for each fixture session-a ``--candidate-json`` ``--json`` (NO dry-run)→rc=0 ok=True verdict=light wrote_item=False correct open_count/queue/b_pick_title plan UNCHANGED SESSION_A_RESULT_JSON_KEYS; session-a ``--candidate-json`` ``--dry-run --json``→same light shape plan UNCHANGED; ``watch_queue_heartbeat`` keeps ``## Watch`` + ``## Heartbeat`` + BEAT19 markers. Contrast Beat 177/182 (list write idle/nonpick) + Beat 184 (dict pickable light); Beat 193 proves pickable OPEN>=1 ignores complete list candidate too. Proof: pytest ``-k beat193``.
Beat 194: multi-fixture **pickable** Session A ``--candidate-json`` **skip-nondict list** light no-write ``--json`` contrast via tmp_path (OPEN>=1 recount-only ignores candidate-json; completes skip-nondict trilogy after Beat 178 idle write + Beat 183 nonpick write; parallel Beat 184 dict / Beat 191 empty-list / Beat 192 incomplete / Beat 193 complete-list / Beat 189 no-dict pickable light; candidate ``complete_candidate_list_skip_nondict.json`` copied to tmp_path): for each fixture session-a ``--candidate-json`` ``--json`` (NO dry-run)→rc=0 ok=True verdict=light wrote_item=False correct open_count/queue/b_pick_title plan UNCHANGED SESSION_A_RESULT_JSON_KEYS; session-a ``--candidate-json`` ``--dry-run --json``→same light shape plan UNCHANGED; ``watch_queue_heartbeat`` keeps ``## Watch`` + ``## Heartbeat`` + BEAT19 markers. Contrast Beat 178/183 (skip-nondict write idle/nonpick) + Beat 193 (complete-list pickable light); Beat 194 proves pickable OPEN>=1 ignores skip-nondict list candidate too. Proof: pytest ``-k beat194``.
Beat 195: multi-fixture **idle** Session A ``--candidate-json`` **empty object ``{}``** fail live vs ``--candidate-json --dry-run`` fail ``--json`` contrast via tmp_path (parallel Beat 185 empty-list fail / Beat 176 incomplete fail / Beat 187 no-dict fail; empty object written to tmp_path each test; no stub fallback; Beat 39 unit → multi-fixture; same idle fixtures as Beat 171/175/185/187): for each fixture session-a ``--candidate-json`` ``--json`` (NO dry-run)→rc=1 ok=False verdict=failed wrote_item=False open_count=0 queue 0/10 title is None plan UNCHANGED SESSION_A_RESULT_JSON_KEYS; session-a ``--candidate-json`` ``--dry-run --json``→same fail shape plan UNCHANGED; Watch fixtures keep ``## Watch`` + ``## Heartbeat``; ``watch_heartbeat_no_queue`` still has no ``## Queue``. Contrast Beat 185 (``[]`` fail idle) + Beat 176 (incomplete fail idle) + Beat 39 (``{}`` unit); Beat 195 is empty-object ``{}`` candidate-json fail live+dry-run on idle fixtures. Proof: pytest ``-k beat195``.
Beat 196: multi-fixture **non-pickable** Session A ``--candidate-json`` **empty object ``{}``** fail live vs ``--candidate-json --dry-run`` fail ``--json`` contrast via tmp_path (parallel Beat 195 idle empty-object fail + Beat 186 nonpick empty-list fail; empty object written to tmp_path each test; no stub fallback; Beat 39 unit → Beat 195 idle → Beat 196 nonpick; fixtures ``shipped_only`` / ``incomplete_open`` / ``broken_ready_flag`` / ``open_complete_not_ready`` / ``contract_spec`` / ``queue_with_watch_heartbeat``): for each fixture session-a ``--candidate-json`` ``--json`` (NO dry-run)→rc=1 ok=False verdict=failed wrote_item=False open_count=0 queue 0/10 title is None plan UNCHANGED SESSION_A_RESULT_JSON_KEYS; session-a ``--candidate-json`` ``--dry-run --json``→same fail shape plan UNCHANGED; ``queue_with_watch_heartbeat`` keeps ``## Watch`` + ``## Heartbeat`` with Watch before Queue. Contrast Beat 195 (``{}`` fail idle) + Beat 186 (``[]`` fail nonpick) + Beat 39 (``{}`` unit); Beat 196 is empty-object ``{}`` candidate-json fail live+dry-run on non-pickables. Proof: pytest ``-k beat196``.
Beat 197: multi-fixture **pickable** Session A ``--candidate-json`` **empty object ``{}``** light no-write ``--json`` contrast via tmp_path (OPEN>=1 recount-only ignores candidate-json; completes empty-object trilogy after Beat 195 idle fail + Beat 196 nonpick fail; parallel Beat 184 dict / Beat 189 no-dict / Beat 191 empty-list / Beat 192 incomplete / Beat 193 complete-list / Beat 194 skip-nondict pickable light; empty object written to tmp_path each test): for each fixture session-a ``--candidate-json`` ``--json`` (NO dry-run)→rc=0 ok=True verdict=light wrote_item=False correct open_count/queue/b_pick_title plan UNCHANGED SESSION_A_RESULT_JSON_KEYS; session-a ``--candidate-json`` ``--dry-run --json``→same light shape plan UNCHANGED; ``watch_queue_heartbeat`` keeps ``## Watch`` + ``## Heartbeat`` + BEAT19 markers. Contrast Beat 195/196 (``{}`` fail idle/nonpick) + Beat 191 (``[]`` pickable light); Beat 197 proves pickable OPEN>=1 ignores empty-object candidate too. Proof: pytest ``-k beat197``.
Beat 198: multi-fixture **idle** Session A ``--candidate-json`` **missing file** fail live vs ``--candidate-json --dry-run`` fail (SystemExit, not verdict=failed) via tmp_path (promotes Beat 16 unit → multi-fixture idle contrast; parallel Beat 195 idle ``{}`` fail; same idle fixtures as Beat 171/175/185/187/195; path e.g. ``does_not_exist_candidate.json`` NEVER created): for each fixture session-a ``--candidate-json`` missing ``--json`` (NO dry-run)→SystemExit nonzero ``--candidate-json`` + ``not found`` plan UNCHANGED no success JSON; session-a ``--candidate-json`` missing ``--dry-run --json``→same SystemExit plan UNCHANGED (dry-run does not skip missing-file load); Watch fixtures keep ``## Watch`` + ``## Heartbeat``; ``watch_heartbeat_no_queue`` still has no ``## Queue``. Contrast Beat 195 (``{}`` fail idle JSON rc=1) + Beat 16 (missing-file unit); Beat 198 is missing-path candidate-json CLI load-fail live+dry-run on idle fixtures. Proof: pytest ``-k beat198``.
Beat 199: multi-fixture **non-pickable** Session A ``--candidate-json`` **missing file** fail live vs ``--candidate-json --dry-run`` fail (SystemExit, not verdict=failed) via tmp_path (parallel Beat 198 idle missing-file SystemExit + Beat 196 nonpick ``{}`` fail; continues missing-file trilogy after Beat 16 unit → Beat 198 idle → Beat 199 nonpick; fixtures ``shipped_only`` / ``incomplete_open`` / ``broken_ready_flag`` / ``open_complete_not_ready`` / ``contract_spec`` / ``queue_with_watch_heartbeat``; path e.g. ``does_not_exist_candidate.json`` NEVER created): for each fixture session-a ``--candidate-json`` missing ``--json`` (NO dry-run)→SystemExit nonzero ``--candidate-json`` + ``not found`` plan UNCHANGED no success JSON; session-a ``--candidate-json`` missing ``--dry-run --json``→same SystemExit plan UNCHANGED (dry-run does not skip missing-file load); ``queue_with_watch_heartbeat`` keeps ``## Watch`` + ``## Heartbeat`` with Watch before Queue. Contrast Beat 198 (missing-file SystemExit idle) + Beat 196 (``{}`` fail nonpick) + Beat 16 (missing-file unit); Beat 199 is missing-path candidate-json CLI load-fail live+dry-run on non-pickables. Proof: pytest ``-k beat199``.
Beat 201: multi-fixture **pickable** Session A ``--candidate-json`` **missing file** still light / no-write ``--json`` contrast via tmp_path (OPEN>=1 recount-only; ``_load_candidate`` deferred inside brainstorm callback so missing path does **NOT** SystemExit on pickables — contrast Beat 198 idle SystemExit + Beat 199 nonpick SystemExit; completes missing-file trilogy after Beat 198/199; same pickable fixtures as Beat 197; path e.g. ``does_not_exist_candidate.json`` NEVER created): for each fixture session-a ``--candidate-json`` missing ``--json`` (NO dry-run)→rc=0 ok=True verdict=light wrote_item=False correct open_count/queue/b_pick_title plan UNCHANGED SESSION_A_RESULT_JSON_KEYS Must NOT raise SystemExit; session-a ``--candidate-json`` missing ``--dry-run --json``→same light shape plan UNCHANGED no SystemExit; ``watch_queue_heartbeat`` keeps ``## Watch`` + ``## Heartbeat`` + BEAT19 markers. Contrast Beat 198/199 (missing-file SystemExit idle/nonpick) + Beat 197 (``{}`` pickable light); Beat 201 proves pickable OPEN>=1 never loads missing candidate-json (deferred load). Proof: pytest ``-k beat201``.
Beat 202: multi-fixture **idle** Session A ``--candidate-json`` **invalid JSON** fail live vs ``--candidate-json --dry-run`` fail (SystemExit, not verdict=failed) via tmp_path (promotes Beat 16 unit → multi-fixture idle contrast; parallel Beat 198 idle missing-file SystemExit; same idle fixtures as Beat 171/175/185/187/195/198; writes ``{not valid json`` to tmp_path ``invalid_candidate.json`` each test; NEVER mutate fixtures): for each fixture session-a ``--candidate-json`` invalid ``--json`` (NO dry-run)→SystemExit nonzero ``--candidate-json`` + ``invalid JSON`` plan UNCHANGED no success JSON; session-a ``--candidate-json`` invalid ``--dry-run --json``→same SystemExit plan UNCHANGED (dry-run does not skip invalid-JSON load); Watch fixtures keep ``## Watch`` + ``## Heartbeat``; ``watch_heartbeat_no_queue`` still has no ``## Queue``. Contrast Beat 198 (missing-file SystemExit idle) + Beat 16 (invalid-JSON unit); Beat 202 is invalid-JSON candidate-json CLI load-fail live+dry-run on idle fixtures. Proof: pytest ``-k beat202``.
Beat 203: multi-fixture **non-pickable** Session A ``--candidate-json`` **invalid JSON** fail live vs ``--candidate-json --dry-run`` fail (SystemExit, not verdict=failed) via tmp_path (parallel Beat 202 idle invalid-JSON SystemExit + Beat 199 nonpick missing-file SystemExit; continues invalid-JSON trilogy after Beat 16 unit → Beat 202 idle → Beat 203 nonpick; fixtures ``shipped_only`` / ``incomplete_open`` / ``broken_ready_flag`` / ``open_complete_not_ready`` / ``contract_spec`` / ``queue_with_watch_heartbeat``; writes ``{not valid json`` to tmp_path ``invalid_candidate.json`` each test; NEVER mutate fixtures): for each fixture session-a ``--candidate-json`` invalid ``--json`` (NO dry-run)→SystemExit nonzero ``--candidate-json`` + ``invalid JSON`` plan UNCHANGED no success JSON; session-a ``--candidate-json`` invalid ``--dry-run --json``→same SystemExit plan UNCHANGED (dry-run does not skip invalid-JSON load); ``queue_with_watch_heartbeat`` keeps ``## Watch`` + ``## Heartbeat`` with Watch before Queue. Contrast Beat 202 (invalid-JSON SystemExit idle) + Beat 199 (missing-file SystemExit nonpick) + Beat 16 (invalid-JSON unit); Beat 203 is invalid-JSON candidate-json CLI load-fail live+dry-run on non-pickables. Proof: pytest ``-k beat203``.
Beat 204: multi-fixture **pickable** Session A ``--candidate-json`` **invalid JSON** still light / no-write ``--json`` contrast via tmp_path (OPEN>=1 recount-only; ``_load_candidate`` deferred inside brainstorm callback so invalid JSON does **NOT** SystemExit on pickables — contrast Beat 202 idle SystemExit + Beat 203 nonpick SystemExit; completes invalid-JSON trilogy after Beat 202/203; same pickable fixtures as Beat 201; writes ``{not valid json`` to tmp_path ``invalid_candidate.json`` each test; NEVER mutate fixtures): for each fixture session-a ``--candidate-json`` invalid ``--json`` (NO dry-run)→rc=0 ok=True verdict=light wrote_item=False correct open_count/queue/b_pick_title plan UNCHANGED SESSION_A_RESULT_JSON_KEYS Must NOT raise SystemExit; session-a ``--candidate-json`` invalid ``--dry-run --json``→same light shape plan UNCHANGED no SystemExit; ``watch_queue_heartbeat`` keeps ``## Watch`` + ``## Heartbeat`` + BEAT19 markers. Contrast Beat 202/203 (invalid-JSON SystemExit idle/nonpick) + Beat 201 (missing-file pickable light); Beat 204 proves pickable OPEN>=1 never loads invalid candidate-json (deferred load). Proof: pytest ``-k beat204``.
Beat 205: multi-fixture **idle** Session A ``--candidate-json`` **wrong-type** (top-level number ``42``) fail live vs ``--candidate-json --dry-run`` fail (SystemExit, not verdict=failed) via tmp_path (promotes Beat 16 unit wrong-type → multi-fixture idle contrast; starts next Beat-16 I/O series after invalid-JSON trilogy Beat 202/203/204; same idle fixtures as Beat 171/175/185/187/195/198/202; writes ``42\n`` to tmp_path ``number.json`` each test; one representative type only — keep exactly 2 tests; NEVER mutate fixtures): for each fixture session-a ``--candidate-json`` number ``--json`` (NO dry-run)→SystemExit nonzero ``--candidate-json`` + ``object or list`` + ``int`` plan UNCHANGED no success JSON; session-a ``--candidate-json`` number ``--dry-run --json``→same SystemExit plan UNCHANGED (dry-run does not skip wrong-type load); Watch fixtures keep ``## Watch`` + ``## Heartbeat``; ``watch_heartbeat_no_queue`` still has no ``## Queue``. Contrast Beat 202 (invalid-JSON SystemExit idle) + Beat 16 (wrong-type unit); Beat 205 is wrong-type number candidate-json CLI load-fail live+dry-run on idle fixtures. Proof: pytest ``-k beat205``.
Beat 206: multi-fixture **non-pickable** Session A ``--candidate-json`` **wrong-type** (top-level number ``42``) fail live vs ``--candidate-json --dry-run`` fail (SystemExit, not verdict=failed) via tmp_path (parallel Beat 205 idle wrong-type SystemExit + Beat 203 nonpick invalid-JSON SystemExit; continues wrong-type trilogy after Beat 16 unit → Beat 205 idle → Beat 206 nonpick; fixtures ``shipped_only`` / ``incomplete_open`` / ``broken_ready_flag`` / ``open_complete_not_ready`` / ``contract_spec`` / ``queue_with_watch_heartbeat``; writes ``42\n`` to tmp_path ``number.json`` each test; NEVER mutate fixtures): for each fixture session-a ``--candidate-json`` number ``--json`` (NO dry-run)→SystemExit nonzero ``--candidate-json`` + ``object or list`` + ``int`` plan UNCHANGED no success JSON; session-a ``--candidate-json`` number ``--dry-run --json``→same SystemExit plan UNCHANGED (dry-run does not skip wrong-type load); ``queue_with_watch_heartbeat`` keeps ``## Watch`` + ``## Heartbeat`` with Watch before Queue. Contrast Beat 205 (wrong-type SystemExit idle) + Beat 203 (invalid-JSON SystemExit nonpick) + Beat 16 (wrong-type unit); Beat 206 is wrong-type number candidate-json CLI load-fail live+dry-run on non-pickables. Proof: pytest ``-k beat206``.
Beat 200: milestone — top-level ``--help`` + public API through beat200 (A/B runners + queue/decode/serialize helpers + guards); Makefile echo reaches beat200. Proof: pytest ``-k beat200``.

Session A: when OPEN is 0, uses ``--stub`` (deterministic six-field fill) or
``--candidate-json``; recount-only when OPEN >= 1. Appends at most one OPEN.
Session B / idle-decode: decode-only pick or idle fire (queue 0/10); never
``scheduler_delete``. Session B ``--dry-run`` exercises the default dry-run
implement callback (records file_touch / acceptance; no repo write).

SHIPPED is callback-only: there is no CLI ``--implement stub-ship`` (or similar)
flag. Prefer pytest with ``make_fixture_ship_implement`` / ``fixture_ship_implement``
on ``tmp_path`` plans. Live prod implement stays unwired.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from src.research_implement.session_a import run_session_a_path, stub_brainstorm
from src.research_implement.session_b import dry_run_implement, run_session_b_path


def _load_candidate(path: Path | None) -> dict | None:
    """Load a Session A brainstorm candidate from JSON (dict or list).

    Beat 15: ``session-a --candidate-json`` accepts either a single candidate
    object or a JSON list; a list uses the first dict element. Incomplete
    candidates still fail-closed in Session A (Beat 14). Returns None when
    path is omitted or the list has no dict element.

    Beat 16: missing file, invalid JSON, or wrong top-level type (not object/list)
    raise ``SystemExit`` with a clear ``--candidate-json ...`` message (non-zero
    CLI failure). Callers must not mutate the plan on these paths.
    """
    if path is None:
        return None
    # Beat 32: existing non-file (directory) → clear not-a-file exit.
    if path.exists() and not path.is_file():
        raise SystemExit(f"--candidate-json is not a file: {path}")
    if not path.is_file():
        raise SystemExit(f"--candidate-json file not found: {path}")
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as err:
        raise SystemExit(f"--candidate-json invalid JSON: {err}") from err
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, list):
        for item in raw:
            if isinstance(item, dict):
                return item
        return None
    raise SystemExit(
        f"--candidate-json must be a JSON object or list of objects, got {type(raw).__name__}"
    )


def _resolve_plan(args: argparse.Namespace) -> Path:
    plan = getattr(args, "plan", None)
    log = getattr(args, "log", None)
    if plan is not None and log is not None:
        raise SystemExit("pass only one of --plan / --log")
    # Beat 33: do not use ``plan or log`` — empty string is falsy but still "passed".
    if plan is not None:
        path = plan
    elif log is not None:
        path = log
    else:
        raise SystemExit("--plan or --log is required")
    # Empty / whitespace-only path fails closed (do not coerce to Path(".")).
    text = str(path).strip()
    if not text:
        raise SystemExit("--plan/--log path is empty")
    resolved = Path(text)
    # Beat 23: missing plan/log path fail-closed (do not create).
    # Beat 31: existing non-file (directory/symlink-to-dir) → clear not-a-file exit.
    if resolved.exists() and not resolved.is_file():
        raise SystemExit(f"--plan/--log is not a file: {resolved}")
    if not resolved.is_file():
        raise SystemExit(f"--plan/--log file not found: {resolved}")
    return resolved

def _add_plan_log(p: argparse.ArgumentParser) -> None:
    g = p.add_mutually_exclusive_group(required=True)
    # Beat 33: keep as str so empty / whitespace is not coerced to Path(".").
    g.add_argument(
        "--plan",
        type=str,
        help="Living plan.md path (fixture or working copy)",
    )
    g.add_argument(
        "--log",
        type=str,
        help="Alias for --plan (e.g. logs/research-implement.md host contract)",
    )


def _run_session_b(
    plan: Path,
    *,
    as_json: bool,
    dry_run: bool = False,
) -> int:
    """Session B CLI: decode-only by default; ``dry_run`` uses dry_run_implement.

    Empty Queue still idle-fires (queue 0/10). Never ``scheduler_delete``.
    Dry-run records intended file_touch / acceptance and never writes the repo.

    Beat 17: default path never invokes implement (spy-proven); only ``dry_run``
    passes ``dry_run_implement``. fixture_ship is never CLI-wired.
    """
    if dry_run:
        result = run_session_b_path(
            plan,
            implement=dry_run_implement,
            decode_only=False,
            write=False,
        )
    else:
        # Beat 17: decode-only — never pass/call implement; never delete schedule.
        result = run_session_b_path(plan, decode_only=True, write=False)
    assert result.keep_schedule and not result.scheduler_delete_called
    if as_json:
        # Single shared SessionResult.to_dict shape (idle / decode_only / dry_run / shipped).
        print(json.dumps(result.to_dict(), indent=2, sort_keys=True))
    else:
        print(result.message)
        if result.decode_report and result.verdict in {"picked", "dry_run"}:
            if result.item is not None and f"decode pick {result.item.item_id}" not in result.message:
                print(result.decode_report)
    return 0 if result.ok else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m src.research_implement",
        description=(
            "Session A/B research-implement loop (Queue producer/consumer). "
            "Side-dev only — no Tasker yaml on prod ports, no live LLM."
        ),
        epilog=(
            "Subcommands: session-a (producer), session-b (decode / optional --dry-run implement), "
            "idle-decode (alias of session-b decode; idle fire on empty Queue)."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    a = sub.add_parser(
        "session-a",
        help="Producer: stub/candidate brainstorm when OPEN is 0; recount when OPEN>=1",
    )
    _add_plan_log(a)
    a.add_argument(
        "--candidate-json",
        type=str,
        default=None,
        help=(
            "Candidate JSON (object or list of objects) loaded when OPEN is 0 "
            "(overrides --stub); incomplete fail-closes; ignored on OPEN>=1 recount-only"
        ),
    )
    a.add_argument(
        "--stub",
        action="store_true",
        help="Use deterministic stub_brainstorm when OPEN is 0 (no LLM)",
    )
    a.add_argument(
        "--no-stub",
        action="store_true",
        help="Do not use stub; without --candidate-json empty OPEN fails fire",
    )
    a.add_argument(
        "--dry-run",
        action="store_true",
        help="Do not write the plan file",
    )
    a.add_argument(
        "--json",
        action="store_true",
        dest="as_json",
        help="Emit SessionAResult.to_dict JSON (append/queued vs recount-only/light)",
    )

    b = sub.add_parser(
        "session-b",
        help="Consumer: decode first ready OPEN Queue item (decode-only; idle on empty)",
    )
    _add_plan_log(b)
    b.add_argument(
        "--decode-only",
        action="store_true",
        default=True,
        help="Decode/pick only (default); never implements code / no prod wire-up",
    )
    b.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Run pluggable dry-run implement (record file_touch / acceptance; "
            "no repo write). Still idle-fires on empty Queue; never scheduler_delete."
        ),
    )
    b.add_argument(
        "--json",
        action="store_true",
        dest="as_json",
        help=(
            "Emit shared SessionResult.to_dict JSON "
            "(idle | picked decode_only | dry_run; never scheduler_delete)"
        ),
    )

    idle = sub.add_parser(
        "idle-decode",
        help=(
            "Decode-only Session B alias: idle fire when empty/not-ready; "
            "picked decode_only when OPEN ready; never scheduler_delete"
        ),
        description=(
            "Decode-only Session B alias. Idle fire when Queue empty/not-ready; "
            "picked decode_only when OPEN ready. Never scheduler_delete."
        ),
    )
    _add_plan_log(idle)
    idle.add_argument(
        "--json",
        action="store_true",
        dest="as_json",
        help=(
            "Emit shared SessionResult.to_dict JSON "
            "(idle | picked decode_only; same keys as session-b --json)"
        ),
    )

    args = parser.parse_args(argv)

    if args.cmd == "session-a":
        plan = _resolve_plan(args)
        # Beat 25: --stub and --no-stub are mutually exclusive.
        if getattr(args, "stub", False) and getattr(args, "no_stub", False):
            raise SystemExit("pass only one of --stub / --no-stub")
        # Beat 15: --candidate-json supplies brainstorm when OPEN==0. Load is
        # deferred inside the callback so OPEN>=1 recount-only never reads or
        # appends the candidate. Empty/non-dict JSON → None → failed fire (no stub).
        if args.candidate_json is not None:
            # Beat 37: empty / whitespace --candidate-json fails closed (no Path(".")).
            cand_text = str(args.candidate_json).strip()
            if not cand_text:
                raise SystemExit("--candidate-json path is empty")
            cand_path = Path(cand_text)

            def _brainstorm(_items, _path=cand_path):
                return _load_candidate(_path)

            brainstorm = _brainstorm
        elif args.no_stub:
            brainstorm = None
        else:
            # Default side-dev (--stub or omitted): deterministic stub, no LLM.
            brainstorm = stub_brainstorm

        result = run_session_a_path(
            plan,
            brainstorm=brainstorm,
            write=not args.dry_run,
        )
        if getattr(args, "as_json", False):
            # Single shared SessionAResult.to_dict shape (queued / light / failed).
            print(json.dumps(result.to_dict(), indent=2, sort_keys=True))
        else:
            print(result.message)
        return 0 if result.ok else 1

    if args.cmd in {"session-b", "idle-decode"}:
        plan = _resolve_plan(args)
        as_json = bool(getattr(args, "as_json", False))
        dry_run = bool(getattr(args, "dry_run", False))
        # idle-decode stays decode-only; session-b may opt into --dry-run implement.
        return _run_session_b(plan, as_json=as_json, dry_run=dry_run)

    parser.error(f"unknown command {args.cmd}")
    return 2


if __name__ == "__main__":
    sys.exit(main())
