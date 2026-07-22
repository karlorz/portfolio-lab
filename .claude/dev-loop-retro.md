
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
