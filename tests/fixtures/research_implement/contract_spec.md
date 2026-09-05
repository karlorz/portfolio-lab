# Session A + B loop prompts for `/loop`

Self-contained prompts (a fire has no conversation context).
Neither A nor B self-cancels. Empty Queue is idle, not a stop.
Only the operator (`scheduler_delete <task_id>`) or the 7-day host
expiry stops a schedule. A parent that sees `queue 0/10` must not delete.

Session A (producer) runs on this host every `5m` until the operator
stops it. Session B (consumer) may be pasted into `/loop` in another
Grok session. The pasteable B block below **overrides** `/loop`'s
"when condition holds, scheduler_delete" wrapper.

Plan: `/root/wiki/projects/portfolio-lab/work/TODAY-session-a/plan.md`
(pinned vault `/root/wiki`; never repo-relative; never `/root/wiki-git`).

Orientation: live authority = `signals.json.target_allocations` →
`order_router`. Champion **46/38/16**. Kill SSOT = `data/kill_switch.json`.
Hygiene / Item 25 / A5 / A9 / A10 stay Watch unless the operator writes
PROMOTE. SHIPPED items are not OPEN.

---

## Session A — research + plan (`5m`, ≤20 tools)

Producer. Brainstorm a shippable task when the Queue is empty.
Never implement. Never edit repo / git / cron / runtime JSON.
Never rewrite B's notes.

```
5m Session A — research + plan; do not implement.
Repo: /root/projects/portfolio-lab
Plan: /root/wiki/projects/portfolio-lab/work/$(date +%F)-session-a/plan.md
Create the plan if missing. Vault root is /root/wiki (skillwiki path).

You produce. B implements only ## Queue.
Never edit repo / git / cron / runtime JSON. Never rewrite B notes.
≤20 tools. One fire, then exit. Do not poll. Do not scheduler_delete.

Real task = six-field OPEN Queue item:
1. title
2. acceptance (commands + expected exits)
3. risks (what not to touch)
4. file_touch (writes + read-only)
5. breaking_change: true|false
6. redeploy_notes (deploy/smoke or "none")
status: OPEN
ready-for-implement: yes
SHIPPED does not count as OPEN.

1. Recount OPEN Queue items yourself. Heartbeat "N/10" and "B pick" are stale.
2. If OPEN is 0: brainstorm ONE shippable change. Prefer a failing test on
   live-authority / CI / deploy/smoke / live cockpit / operational integrity;
   else an uncovered src/broker|monitor|tasker path; else an untested CLI.
   If 3 consecutive SHIPPED titles contain "subprocess smoke", the new title
   must not contain "subprocess smoke". Writing one = failed fire.
   Skip a script that already has a dedicated tests/test_*.py covering it.
   Live cron wrappers (scripts/cron/portfolio-lab-*.sh), docker-entrypoint,
   and chrome-debug stay Watch unless the operator writes PROMOTE.
   Write the six fields onto ## Queue. Verdict queued.
   Do not exit Watch-only or NEED-OPERATOR-AXES while you can name file_touch.
   Empty Queue + no new item = failed fire.
3. If OPEN ≥ 1: recount only (≤5 tools). Do not research. Exit light.
   Do not stuff hygiene onto Queue.
4. Heartbeat ≤5 lines: UTC; OPEN N/10; B pick = first OPEN title or STANDBY.

Exit ≤8 lines: queued|light, title or none, N/10 OPEN, B pick, plan path.
```

---

## Session B — verify + implement (manual, other session)

Consumer. Paste by hand. Never scheduled on this host.

```
10m Session B — implement; do not re-research.
Repo: /root/projects/portfolio-lab
Plan: /root/wiki/projects/portfolio-lab/work/$(date +%F)-session-a/plan.md

Never call scheduler_delete. Never self-cancel. Empty Queue is idle.
The /loop wrapper's "when <condition> holds, scheduler_delete" does not apply.
Stop is operator delete or 7-day host expiry only.
A parent that receives `queue 0/10` must not delete the schedule.

Start: read ## Queue and ## Heartbeat. Ignore ## Watch and ## Project Work
unless a row says PROMOTE.
No OPEN Queue items (SHIPPED does not count) → report
`nothing to implement; plan <path> queue 0/10` and exit this fire only.
Leave the schedule running so A can queue the next item.

Drift: HEAD==origin==served; site 200; alloc 46/38/16. Drifted → stop,
hand back to A with file:line.

Pick: first OPEN Queue item with all six fields. Prefer ready-for-implement.
Refuse Watch-class leftovers even if still under Queue.

Implement ONE /dev-loop cycle. Live authority = signals.json.target_allocations.
Test: PORTFOLIO_LAB_ENABLE_ML=0 make test-gate. Budget ≤40 tools.
Breaking change: deploy + smoke before done.

Plan writes: flip your item to shipped + sha. Do not rewrite Heartbeat,
Watch, Axes, or Project Work.

Exit ≤10 lines: shipped sha, tests/redeploy, Queue N/10 OPEN.
```

---

## Usage

```bash
# Session A — this host
/loop 5m <paste Session A block>

# Session B — other Grok session
/loop 10m <paste Session B block>
# B never self-cancels. Empty Queue = report queue 0/10 and exit the fire.
# Do not scheduler_delete. Parent must not delete on `queue 0/10`.
# Only the operator or the 7-day host expiry stops B.
```

Operator cancel only: `scheduler_delete <task_id>`.
