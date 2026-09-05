"""Session A — research/plan producer (never implements).

When OPEN count is 0, consult the pluggable search/plan (brainstorm) callback
and append at most one ready six-field OPEN Queue item. The default side-dev
hook is ``stub_brainstorm`` / ``default_search_plan`` (deterministic; no LLM).
When OPEN >= 1, recount only (light exit) and do not call the callback.
Empty Queue + no new item = failed fire.

Fail-closed (Beat 14): an incomplete brainstorm/search_plan candidate
(missing any of the required six fields, or ready-for-implement not yes)
yields verdict ``failed``, ``wrote_item=False``, plan text unchanged, and
never a partial OPEN append.

JSON contract: ``SessionAResult.to_dict()`` (aliases ``to_json_dict`` /
``session_a_result_dict``) is the shared shape for CLI ``session-a --json``
and fixture tests (append/queued vs recount-only/light vs failed).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from src.research_implement.queue import (
    REQUIRED_FIELDS,
    first_b_pick,
    QueueItem,
    append_queue_item,
    count_open,
    format_queue_item,
    is_ready_yes,
    next_queue_id,
    parse_queue_items,
    queue_item_from_fields,
    render_queue_count,
)

BrainstormFn = Callable[[list[QueueItem]], QueueItem | dict | None]
# Alias: pluggable search/plan callback (same contract as brainstorm).
SearchPlanFn = BrainstormFn

STUB_TITLE = "Stub shippable change"


def stub_brainstorm(_items: list[QueueItem] | None = None) -> dict:
    """Deterministic six-field OPEN candidate for side-dev / fixtures.

    No external LLM, Hermes, or ``src.research.agent``. Callers may plug a
    real search/plan callback in place of this stub. Session A still appends
    at most one OPEN item and recounts when OPEN >= 1. Includes
    ``ready_for_implement: yes`` so the complete stub passes fail-closed
    validation (Beat 14).
    """
    return {
        "heading": STUB_TITLE,
        "title": STUB_TITLE,
        "acceptance": "pytest EXIT=0",
        "risks": "do not touch kill_switch or order_router",
        "file_touch": "write tests/test_stub_shippable.py",
        "breaking_change": False,
        "redeploy_notes": "none",
        "ready_for_implement": "yes",
    }


# Public alias for the pluggable search/plan hook default.
default_search_plan = stub_brainstorm
stub_search_plan = stub_brainstorm


# Stable CLI ``session-a --json`` / test contract keys (queued | light | failed).
SESSION_A_RESULT_JSON_KEYS: tuple[str, ...] = (
    "ok",
    "verdict",
    "open_count",
    "queue",
    "b_pick_title",
    "title",
    "wrote_item",
)

# Alias kept for callers that prefer the Session-A-named constant.
SESSION_A_RESULT_KEYS: tuple[str, ...] = SESSION_A_RESULT_JSON_KEYS


@dataclass(frozen=True)
class SessionAResult:
    ok: bool
    verdict: str  # queued | light | failed
    open_count: int
    b_pick_title: str | None
    title: str | None
    plan_text: str
    message: str
    wrote_item: bool = False

    @property
    def queue_label(self) -> str:
        return render_queue_count(self.open_count)

    def to_dict(self) -> dict[str, Any]:
        """Shared SessionAResult dict for CLI ``session-a --json`` and fixture tests.

        One shape for append (verdict queued) / recount-only (verdict light) / failed.
        Keys are exactly ``SESSION_A_RESULT_JSON_KEYS`` / ``SESSION_A_RESULT_KEYS``.
        """
        return session_a_result_dict(self)

    def to_json_dict(self) -> dict[str, Any]:
        """Alias of ``to_dict`` (stable JSON contract name)."""
        return self.to_dict()


def session_a_result_dict(result: SessionAResult) -> dict[str, Any]:
    """Build the shared Session A dict (CLI ``session-a --json`` + tests).

    One shape for all verdicts:
    - queued (append): wrote_item=True, title set, open_count after append
    - light (recount-only): wrote_item=False, title=None, open_count unchanged
    - failed: wrote_item=False, ok=False
    """
    return {
        "ok": result.ok,
        "verdict": result.verdict,
        "open_count": result.open_count,
        "queue": result.queue_label,
        "b_pick_title": result.b_pick_title,
        "title": result.title,
        "wrote_item": result.wrote_item,
    }


def _field_value_present(value: object) -> bool:
    """True when a six-field value is present (bool False counts as present)."""
    if value is None:
        return False
    if isinstance(value, bool):
        return True
    return bool(str(value).strip())


def incomplete_candidate_reasons(raw: QueueItem | dict) -> list[str]:
    """Return missing/invalid keys for a brainstorm/search_plan candidate.

    Fail-closed (Beat 14): every required six field must be present and
    non-empty (``breaking_change`` may be bool), and ``ready_for_implement``
    must be an explicit yes. Missing keys are not filled with defaults.
    """
    reasons: list[str] = []
    if isinstance(raw, QueueItem):
        fields = raw.field_map()
        for name in REQUIRED_FIELDS:
            if not _field_value_present(fields.get(name)):
                reasons.append(name)
        if not is_ready_yes(raw.ready_for_implement):
            reasons.append("ready_for_implement")
        return reasons
    if not isinstance(raw, dict):
        return ["candidate_type"]
    for name in REQUIRED_FIELDS:
        if name not in raw or not _field_value_present(raw.get(name)):
            reasons.append(name)
    if "ready_for_implement" not in raw or not is_ready_yes(
        str(raw.get("ready_for_implement") or "")
    ):
        reasons.append("ready_for_implement")
    return reasons


def _coerce_candidate(raw: QueueItem | dict, *, item_id: str) -> QueueItem:
    """Coerce a *complete* candidate (caller must fail-closed first)."""
    if isinstance(raw, QueueItem):
        if raw.item_id and raw.item_id != item_id:
            # Keep caller id if already set; otherwise stamp next id.
            item_id = raw.item_id
        return queue_item_from_fields(
            item_id=item_id,
            heading=raw.heading or raw.title or item_id,
            title=raw.title,
            acceptance=raw.acceptance,
            risks=raw.risks,
            file_touch=raw.file_touch,
            breaking_change=raw.breaking_change,
            redeploy_notes=raw.redeploy_notes,
            status=raw.status or "OPEN",
            ready_for_implement="yes",
        )
    if not isinstance(raw, dict):
        raise TypeError("brainstorm must return QueueItem, dict, or None")
    heading = str(raw.get("heading") or raw.get("title") or item_id)
    return queue_item_from_fields(
        item_id=str(raw.get("item_id") or item_id),
        heading=heading,
        title=str(raw.get("title") or ""),
        acceptance=str(raw.get("acceptance") or ""),
        risks=str(raw.get("risks") or ""),
        file_touch=str(raw.get("file_touch") or ""),
        breaking_change=raw.get("breaking_change"),
        redeploy_notes=str(raw.get("redeploy_notes") or ""),
        status=str(raw.get("status") or "OPEN"),
        ready_for_implement="yes",
    )


def run_session_a(
    plan_markdown: str,
    *,
    brainstorm: BrainstormFn | None = None,
    search_plan: SearchPlanFn | None = None,
    plan_path: str | Path | None = None,
    write_path: bool = False,
) -> SessionAResult:
    """Run one Session A fire against living-plan markdown.

    ``brainstorm`` / ``search_plan`` is the pluggable search/plan callback,
    consulted only when OPEN == 0. Prefer ``search_plan`` for new call sites;
    if both are passed, ``search_plan`` wins. It must not implement code; it
    returns at most one six-field candidate (or None → failed fire). Pass
    ``stub_brainstorm`` / ``default_search_plan`` for a deterministic fill.
    """
    if search_plan is not None:
        brainstorm = search_plan
    items = parse_queue_items(plan_markdown)
    open_n = count_open(items)
    first = next((i for i in items if i.status and i.status.upper().startswith("OPEN")), None)
    # Prefer B-pickable title for heartbeat; else first OPEN heading.

    pick = first_b_pick(items)
    b_title = (pick.title if pick else None) or (first.title if first else None)

    if open_n >= 1:
        msg = (
            f"light; title none; {render_queue_count(open_n)}; "
            f"B pick = {pick.title if pick else 'STANDBY'}; recount only"
        )
        return SessionAResult(
            ok=True,
            verdict="light",
            open_count=open_n,
            b_pick_title=pick.title if pick else None,
            title=None,
            plan_text=plan_markdown,
            message=msg,
            wrote_item=False,
        )

    if brainstorm is None:
        msg = (
            f"failed; Empty Queue + no new item = failed fire; "
            f"{render_queue_count(0)}; B pick = STANDBY"
        )
        return SessionAResult(
            ok=False,
            verdict="failed",
            open_count=0,
            b_pick_title=None,
            title=None,
            plan_text=plan_markdown,
            message=msg,
            wrote_item=False,
        )

    candidate = brainstorm(items)
    if candidate is None:
        msg = (
            f"failed; Empty Queue + no new item = failed fire; "
            f"{render_queue_count(0)}; B pick = STANDBY"
        )
        return SessionAResult(
            ok=False,
            verdict="failed",
            open_count=0,
            b_pick_title=None,
            title=None,
            plan_text=plan_markdown,
            message=msg,
            wrote_item=False,
        )

    # Beat 14 fail-closed: incomplete candidate → failed; plan unchanged;
    # never partial OPEN append (validate before coerce/format/write).
    if not isinstance(candidate, (QueueItem, dict)):
        msg = (
            f"failed; incomplete candidate missing ['candidate_type']; "
            f"{render_queue_count(0)}; B pick = STANDBY"
        )
        return SessionAResult(
            ok=False,
            verdict="failed",
            open_count=0,
            b_pick_title=None,
            title=None,
            plan_text=plan_markdown,
            message=msg,
            wrote_item=False,
        )

    reasons = incomplete_candidate_reasons(candidate)
    if reasons:
        title_hint = None
        if isinstance(candidate, QueueItem):
            title_hint = candidate.title or None
        elif isinstance(candidate, dict):
            raw_title = candidate.get("title") or candidate.get("heading")
            title_hint = str(raw_title).strip() or None if raw_title is not None else None
        msg = (
            f"failed; incomplete candidate missing {reasons}; "
            f"{render_queue_count(0)}; B pick = STANDBY"
        )
        return SessionAResult(
            ok=False,
            verdict="failed",
            open_count=0,
            b_pick_title=None,
            title=title_hint,
            plan_text=plan_markdown,
            message=msg,
            wrote_item=False,
        )

    qid = next_queue_id(items)
    item = _coerce_candidate(candidate, item_id=qid)
    # Producer contract after fail-closed gate: OPEN + ready-for-implement: yes
    item_md = format_queue_item(
        item_id=item.item_id,
        heading=item.heading or item.title or item.item_id,
        title=item.title,
        acceptance=item.acceptance,
        risks=item.risks,
        file_touch=item.file_touch,
        breaking_change=item.breaking_change,
        redeploy_notes=item.redeploy_notes,
        status="OPEN",
        ready_for_implement="yes",
    )

    new_text = append_queue_item(plan_markdown, item_md)
    new_open = count_open(parse_queue_items(new_text))
    if write_path and plan_path is not None:
        Path(plan_path).write_text(new_text, encoding="utf-8")

    msg = (
        f"queued; title {item.title}; {render_queue_count(new_open)}; "
        f"B pick = {item.title}; plan {plan_path or '-'}"
    )
    return SessionAResult(
        ok=True,
        verdict="queued",
        open_count=new_open,
        b_pick_title=item.title,
        title=item.title,
        plan_text=new_text,
        message=msg,
        wrote_item=True,
    )


def run_session_a_path(
    plan_path: str | Path,
    *,
    brainstorm: BrainstormFn | None = None,
    search_plan: SearchPlanFn | None = None,
    write: bool = True,
) -> SessionAResult:
    path = Path(plan_path)
    text = path.read_text(encoding="utf-8") if path.exists() else "# Session A plan\n\n## Queue\n\n"
    return run_session_a(
        text,
        brainstorm=brainstorm,
        search_plan=search_plan,
        plan_path=path,
        write_path=write,
    )
