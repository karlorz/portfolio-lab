
## [2026-05-28] retro | loop cycle: fix-bl-pypfopt-dependency
- Friction:       Deep research recommended BL shrinkage as top action — already implemented at line 310
- Miss:           Should have read the code before accepting research recommendations at face value
- Improve:        Always verify research findings against actual code before creating work items
- Generalize?:    yes (research-to-code verification pattern)
- ClaudeMd?:      yes (pypfopt now declared, sklearn.covariance safe prefix)
- WorkflowShift?: no
## [2026-07-22] retro | loop cycle: batch-dk-bandit-zero-baseline-pin
- Friction:       Live disclosed MSM as active contrib despite REGIME_WEIGHTS soft-delete
- Miss:           Bandit blend only was assumed; adaptive+noise re-inflated ε mass further
- Improve:        Pin soft-delete after every reinflation-capable stage; update integration tests that assumed all weights >0
- Generalize?:    yes (static-zero ≠ skip collect, but also ≠ vote mass)
- ClaudeMd?:      no
- WorkflowShift?: no
- Deep-research: sleeping experts no auto-reenable

## [2026-07-22] retro | loop cycle: batch-dl-bandit-soft-delete-reward-skip
- Friction:       Attribution multi-arm could still train MSM posterior after vote pin
- Miss:           Vote pin alone insufficient for bandit hygiene
- Improve:        Filter soft-delete at reward ingress; keep opt-in for shadow learning
- Generalize?:    yes
- ClaudeMd?:      no
- WorkflowShift?: no
- Deep-research: sleeping experts skip reward non-voting


## [2026-07-22] retro | loop cycle: batch-dm-dn-per-signal-cap
- Friction:       Live CAR ~85% after health renorm; 50% cap only in OnlineIC helper
- Miss:           Cap must run after every renorm stage, not only IC path (default off)
- Improve:        Final clip+waterfill after health + after turnover; disclose cap
- Generalize?:    yes (dropout renorm without max-weight → concentration risk)
- ClaudeMd?:      no
- WorkflowShift?: no
- Deep-research: sleeping experts no-update; multi-horizon IC; ESS caps

## [2026-07-22] retro | loop cycle: batch-do-recap-and-mult-hygiene
- Friction:       Live CAR still 85% after DN until refresh; mult misread
- Miss:           Analysis-floor renorm after final cap; mult not pinned
- Improve:        Cap last; mult=0 for zero baseline on all adaptive paths
- Generalize?:    yes
- ClaudeMd?:      no
- WorkflowShift?: no
- Deep-research: display multiplier 0 when weight fixed 0

## [2026-07-22] retro | loop cycle: batch-dp-inactive-mass-recap
- Friction:       Cap correct in pipeline; assign dropped inactive without renorm
- Miss:           Dashboard rollup was second concentration path
- Improve:        Zero inactive → renorm → recap; rollup safety cap
- Generalize?:    yes
- ClaudeMd?:      no
- WorkflowShift?: no
- Deep-research: awake-only renorm

## [2026-07-22] retro | loop cycle: batch-dq-concentration-sli
- Friction:       Full dashboard only hourly :15; partials fake freshness
- Miss:           No SLI on health for ensemble max weight
- Improve:        Project concentration onto health; re-project on partial
- Generalize?:    yes
- ClaudeMd?:      no
- WorkflowShift?: no
- Deep-research: observability SLIs for stale partials

## [2026-07-22] retro | loop cycle: batch-dr-rebalance-recent-orders
- Friction:       Live schema recent_orders ≠ parser orders key
- Miss:           Embedded May timestamps looked like last execution
- Improve:        Prefer payload date; accept recent_orders
- Generalize?:    yes
- ClaudeMd?:      no
- WorkflowShift?: no
- Deep-research: business date vs event timestamp

## [2026-07-22] retro | loop cycle: batch-ds-event-time-schedule
- Friction:       Daily order-history is orders.jsonl tail snapshot
- Miss:           Using write_day as last execution invents activity
- Improve:        Max order event timestamp; rewrite lag disclosure
- Generalize?:    yes
- ClaudeMd?:      no
- WorkflowShift?: no
- Deep-research: event time vs write time

## [2026-07-22] retro | loop cycle: batch-dt-pending-artifact-reconcile
- Friction:       Job pending while artifact fresh looks broken
- Miss:           Recovery map only covered sticky error, not pending
- Improve:        Map fetch-trends→google_trends.json; soft-ok pending
- Generalize?:    yes
- ClaudeMd?:      no
- WorkflowShift?: no
- Deep-research: dual-signal monitoring

## [2026-07-22] retro | loop cycle: batch-du-unhealthy-min-ic
- Friction:       Unhealthy+weak IC still contributed vote mass
- Miss:           CY only gated IC<0 for unhealthy soft floor
- Improve:        Min IC 0.08 for unhealthy; disclose soft-floor survivors
- Generalize?:    yes
- ClaudeMd?:      no
- WorkflowShift?: no
- Deep-research: IC quality gates 0.02–0.05+

## [2026-07-22] retro | loop cycle: batch-du-publish-soft-floor-rebuild
- Friction:       Full dashboard rebuild dropped soft_floor on statuses
- Miss:           Only first build path passed soft_floor map
- Improve:        Wire soft_floor into staleness rebuild; filter zero mass
- Generalize?:    yes
- ClaudeMd?:      no
- WorkflowShift?: no

## [2026-07-22] retro | loop cycle: batch-dv-ml-feature-staleness-sli
- Friction:       Stale ML features hidden in nested panel
- Miss:           Health compact had no ML freshness keys
- Improve:        Project ml_features_stale + age onto health; partial re-project
- Generalize?:    yes
- ClaudeMd?:      no
- WorkflowShift?: no
- Deep-research: feature freshness SLOs

## [2026-07-22] retro | loop cycle: batch-dw-rebalance-budget-sli
- Friction:       smart_rebalance ytd 214bps / over_budget + last_rebalance May vs order event July hidden in nested panel
- Miss:           Compact health had concentration + ML freshness SLIs but no cost-budget / dual-clock keys
- Improve:        Project rebalance_is_over_budget + controller lag onto health; partial re-project
- Generalize?:    yes (nested budget panels need compact SLIs; dual clocks for controller vs event time)
- ClaudeMd?:      no
- WorkflowShift?: no
- Deep-research: event-time vs processing-time dual-clock lag; last_execution_event_time gauges

## [2026-07-22] retro | loop cycle: batch-dx-reconcile-controller-clock
- Friction:       DW disclosed 51d lag; root cause was last_rebalance never advanced on fills
- Miss:           record_execution only path; dashboard evaluate does not record
- Improve:        reconcile_last_rebalance_from_event from rebalance_health on gate load; no invented costs
- Generalize?:    yes (event-sourced last_* clocks; controller state vs order log)
- ClaudeMd?:      no
- WorkflowShift?: no
- Deep-research: event sourcing last_rebalance from RebalanceCompleted / fill event time; venue SSOT

## [2026-07-22] retro | loop cycle: batch-dy-cost-ledger-sanitize
- Friction:       ytd 214 bps from exact-dup cost rows + zero noise; budget false over
- Miss:           add_cost was append-only without composite-key idempotency; YTD not year-scoped
- Improve:        sanitize on load (dedupe/zero); add_cost idempotent; ytd_total year filter
- Generalize?:    yes (ledger composite-key dedupe; YTD as year view)
- ClaudeMd?:      no
- WorkflowShift?: no
- Deep-research: TCA append-only ledger + composite dedupe; fiscal YTD views not mutable wipe

## [2026-07-22] retro | loop cycle: batch-dz-cost-outlier-cap
- Friction:       After DY, ytd still 163 bps; 100 bps SPY row alone blew annual 50 bps budget
- Miss:           safety.max_single_trade_cost_bps=15 existed but never applied to ledger
- Improve:        Quarantine rows > cap from YTD sum; keep audit trail; wire cap from config
- Generalize?:    yes (TCA single-trade cost caps; outliers ≠ budget burn)
- ClaudeMd?:      no
- WorkflowShift?: no
- Deep-research: max single-trade cost cap 15–50 bps; isolate outliers from budget metrics

## [2026-07-22] retro | loop cycle: batch-ea-rebuild-ledger-from-fills
- Friction:       After DZ still 63 bps from synthetic May test rows; only 3 real event days
- Miss:           Cost ledger never projected from order-fill event log (SSOT)
- Improve:        Rebuild YTD from unique fills × ETF bps / portfolio notional; archive prior
- Generalize?:    yes (event-sourced TCA projector; snapshot-rewrite dedupe)
- ClaudeMd?:      no
- WorkflowShift?: no
- Deep-research: rebuild TCA ledger from fills in notional bps; order events SSOT

## [2026-07-22] retro | loop cycle: batch-eb-paper-return-ssot
- Friction:       portfolio_paper.history ret 0.008 vs daily_pnl SSOT 0.0; multi-surface drift
- Miss:           Capture wrote daily_pnl but did not align history / performance snapshot; no compact SLI
- Improve:        paper_return_ssot module + capture side-effects + wiki_sync SSOT path; health projection
- Generalize?:    yes (write SSOT + fan-out alignment; five-surface compare)
- ClaudeMd?:      no
- WorkflowShift?: no
- Deep-research: NAV/session return SSOT; previous-day base; MTM single feed

## [2026-07-22] retro | loop cycle: batch-ec-voting-mass-quality-sli
- Friction:       1/9 healthy badge while 100% vote mass is soft_floor (MSM healthy but zero_baseline)
- Miss:           Source-count SLI ≠ voting-mass quality; operators misread badge as vote health
- Improve:        Project soft_floor_mass / healthy_mass / quality_status onto compact health
- Generalize?:    yes (count metrics vs mass-weighted portfolio SLIs; soft floor as graduated state)
- ClaudeMd?:      no
- WorkflowShift?: no
- Deep-research: soft-floor weight share SLI; IC gates; ensemble health ≠ source headcount

## [2026-07-22] retro | loop cycle: batch-ed-reentry-eligibility-sli
- Friction:       Nested reentry eligible (MSM/INTL/VIXTS) vs blocked (ALT/CARA) invisible on compact health
- Miss:           EC disclosed soft-floor mass but not multi-horizon wake eligibility
- Improve:        Project reentry_eligible/blocked counts + sources; policy=no_force_wake; no auto-wake
- Generalize?:    yes (hysteresis reentry disclose ≠ force; sleeping experts)
- ClaudeMd?:      no
- WorkflowShift?: no
- Deep-research: multi-horizon IC reentry + hysteresis; no force-wake

## [2026-07-22] retro | loop cycle: batch-ee-pending-artifact-health-sli
- Friction:       Raw cron_status fetch-trends pending while google_trends.json fresh; compact only had counts
- Miss:           DT soft-ok lived on normalize path but not compact health dual-signal keys
- Improve:        Project artifact_reconciled vs true pending_never_run onto health; no false warn
- Generalize?:    yes (dual-signal scheduler + artifact freshness SLI)
- ClaudeMd?:      no
- WorkflowShift?: no
- Deep-research: dual-signal cron pending + artifact freshness; SRE false-positive reduction

## [2026-07-22] retro | loop cycle: batch-ef-record-cost-on-execute
- Friction:       EA rebuild required because order_router never called record_execution on fills
- Miss:           Cost ledger only offline; controller lagged until DX/EA
- Improve:        On non-dry-run execute_orders, estimate ETF bps × notional and record_execution
- Generalize?:    yes (fill event = cost SSOT; dry_run never writes ledger)
- ClaudeMd?:      no
- WorkflowShift?: no
- Deep-research: TCA cost capture on fill/execution event; OMS event-driven

## [2026-07-22] retro | loop cycle: batch-fx-soft-mirror-lag-restamp
- Friction:       Sticky lag=11 critical on health_ops through dashboard soft-mirror while live 0/33
- Miss:           Soft-mirror copied nested SLI stamps; attach_shared projected onto empty dict (status elevate dead)
- Improve:        End-pipeline restamp health docs from live probe; project onto real report; consumer max(live,stamp)
- Generalize?:    yes (mirror copy ≠ metric restamp; nested SLI needs recompute after equalize)
- ClaudeMd?:      no
- WorkflowShift?: no
- Deep-research: gauge/mirror restamp staleness; soft-elevate only; private SSOT in restamp set

## [2026-07-23] retro | loop cycle: batch-hn-alerts-multi-dest
- Friction:       Live alerts priv/www equal while repo public/data lagged (0/1/1); health dual path.write_text
- Miss:           Explicit repo_path under pytest would clobber checkout; auto soft-mirror gate needed
- Improve:        write_json_multi_dest + health/generator call with repo_filename; pytest auto-skip
- Generalize?:    yes (non-authority multi-dest = serialize-once + 0o644; auto repo under pytest off)
- ClaudeMd?:      no
- WorkflowShift?: no
- Deep-research: multi-dest alerts hold; fchmod chokepoint already in HM
## [2026-07-23] retro | loop cycle: batch-ie-dashboard-health-multi-dest
- Friction:       generate_health_json single-path save_results_json; public/repo only re-EQ after soft-mirror job
- Miss:           IC multi-dest covered merge path only, not full dashboard generate
- Improve:        write_json_multi_dest public+repo, private_path=None; fallback save_results_json; 3 TDD cases
- Generalize?:    yes (every public JSON producer needs multi-dest, not only patch/merge paths)
- ClaudeMd?:      no
- WorkflowShift?: no
- Deep-research: dual schema health public vs private SSOT

## [2026-07-23] retro | loop cycle: batch-if-lag-honesty-meta-restamp
- Friction:       After multi-dest heal lagging_count=0 but mirror_lag_stamp_lagging_count=1 / source_of_truth=stamp
- Miss:           apply_lag_summary restamp rewrote gauge only, not HO honesty meta
- Improve:        Restamp sets mirror_lag_* from live probe (stamp==live post-restamp); hermetic HO live mock
- Generalize?:    yes (restamp must rewrite all consumer SLI fields, not only nested gauge)
- ClaudeMd?:      no
- WorkflowShift?: no
- Deep-research: max(live,stamp) coherent only when stamp fields restamped with probe

## [2026-07-23] retro | loop cycle: batch-ig-dashboard-graduation-cb
- Friction:       Public health.json had ops_health_* but no graduation CB; signals.health + private ops did
- Miss:           apply_ops_monitor_to_dashboard_health projected lag/kill/SH but not CB SSOT
- Improve:        Project compact + nested graduation_circuit_breaker from .circuit_breaker.json onto dashboard health
- Generalize?:    yes (every dual-surface SLI must fan-out to public dashboard, not only signals.health)
- ClaudeMd?:      no
- WorkflowShift?: no
- Deep-research: dual-SSOT projection; compact+nested keys for SPA/ops consumers
