"""Session B — decode/implement consumer (Queue only).

Picks the first complete six-field OPEN item with ready-for-implement: yes.
Ignores ## Watch and ## Project Work. Empty Queue is an idle fire
(``nothing to implement; … queue 0/10``) — keep the schedule; never
``scheduler_delete``.

Decode path reports the picked Q id and six fields. Pluggable ``implement``
callback defaults to ``dry_run_implement`` / ``default_implement`` (records
intended ``file_touch`` / ``acceptance`` without writing the repo). CLI
defaults to decode-only; pass ``--dry-run`` to exercise the dry-run implement
path. Live implement to prod stays unwired.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from src.research_implement.queue import (
    QueueItem,
    count_open,
    first_b_pick,
    mark_item_shipped,
    parse_queue_items,
    render_queue_count,
)

ImplementFn = Callable[[QueueItem], dict[str, Any] | None]


def dry_run_implement(item: QueueItem) -> dict[str, Any]:
    """Default implement: record intended file_touch / acceptance; write nothing.

    Side-dev / fixture hook — does not mutate the repo, does not mark SHIPPED,
    does not delete Queue rows, and never touches kill_switch / order_router /
    live authority paths. Non-mutating: a later Session B fire may pick the
    same OPEN item again (not idle).
    """
    return {
        "dry_run": True,
        "item_id": item.item_id,
        "title": item.title,
        "file_touch": item.file_touch,
        "acceptance": item.acceptance,
        "wrote_files": False,
        "sha": None,
    }


# Public alias for the pluggable implement hook default.
default_implement = dry_run_implement


class SchedulerDeleteForbidden(RuntimeError):
    """Raised if any path attempts scheduler_delete (must never happen)."""


def scheduler_delete(*_args: Any, **_kwargs: Any) -> None:
    """Hard guard: Session B must never delete the schedule on empty Queue."""
    raise SchedulerDeleteForbidden(
        "Never call scheduler_delete from Session B; empty Queue is idle "
        "(queue 0/10). Only the operator may delete the schedule."
    )


def decode_fields(item: QueueItem) -> dict[str, str]:
    """Return Q id + six required fields for CLI/decode reporting."""
    fields = item.field_map()
    return {
        "item_id": item.item_id,
        "heading": item.heading,
        "title": fields["title"],
        "acceptance": fields["acceptance"],
        "risks": fields["risks"],
        "file_touch": fields["file_touch"],
        "breaking_change": fields["breaking_change"],
        "redeploy_notes": fields["redeploy_notes"],
        "status": item.status,
        "ready_for_implement": item.ready_for_implement,
    }


def format_decode_report(item: QueueItem) -> str:
    """Human-readable decode dump: pick Q id then six fields (no implement)."""
    d = decode_fields(item)
    lines = [
        f"decode pick {d['item_id']}: {d['title'] or d['heading']}",
        f"  1. title: {d['title']}",
        f"  2. acceptance: {d['acceptance']}",
        f"  3. risks: {d['risks']}",
        f"  4. file_touch: {d['file_touch']}",
        f"  5. breaking_change: {d['breaking_change']}",
        f"  6. redeploy_notes: {d['redeploy_notes']}",
        f"  status: {d['status']}",
        f"  ready-for-implement: {d['ready_for_implement']}",
    ]
    return "\n".join(lines)


@dataclass(frozen=True)
class SessionBResult:
    ok: bool
    verdict: str  # idle | picked | dry_run | shipped | refused
    open_count: int
    item: QueueItem | None
    message: str
    plan_text: str
    keep_schedule: bool = True
    scheduler_delete_called: bool = False
    implement_result: dict[str, Any] | None = None

    @property
    def queue_label(self) -> str:
        return render_queue_count(self.open_count)

    @property
    def decode_report(self) -> str | None:
        """Structured field dump when a Q item was picked; None on idle."""
        if self.item is None:
            return None
        return format_decode_report(self.item)


def run_session_b(
    plan_markdown: str,
    *,
    implement: ImplementFn | None = None,
    plan_path: str | Path | None = None,
    decode_only: bool = False,
    write_path: bool = False,
    ship_sha: str | None = None,
) -> SessionBResult:
    """Run one Session B fire.

    Decode always uses ``## Queue`` only (Watch / Project Work ignored by parser).
    When no B-pickable OPEN item exists, return idle success with ``queue 0/10``
    and ``keep_schedule=True`` without calling ``scheduler_delete``.

    When ``decode_only`` is False, the pluggable ``implement`` callback runs
    (defaults to ``dry_run_implement`` when omitted). Dry-run records intended
    ``file_touch`` / ``acceptance`` and never writes the repo or marks SHIPPED.
    """
    # Defense in depth: bind local name so tests can assert we never call it.
    _delete = scheduler_delete  # noqa: F841

    items = parse_queue_items(plan_markdown)
    open_n = count_open(items)
    pick = first_b_pick(items)
    path_label = str(plan_path) if plan_path is not None else "<memory>"

    if pick is None:
        # Idle fire — success, keep schedule. NEVER scheduler_delete.
        # Empty / incomplete OPEN / SHIPPED-only all land here (queue 0/10).
        msg = (
            f"nothing to implement; plan {path_label} "
            f"{render_queue_count(open_n)}; keep_schedule"
        )
        return SessionBResult(
            ok=True,
            verdict="idle",
            open_count=open_n,
            item=None,
            message=msg,
            plan_text=plan_markdown,
            keep_schedule=True,
            scheduler_delete_called=False,
            implement_result=None,
        )

    if decode_only:
        # Decode path: report Q id + fields. CLI defaults here. Pass
        # decode_only=False (optionally with implement=) to exercise the
        # pluggable implement hook; implement defaults to dry_run_implement.
        report = format_decode_report(pick)
        msg = (
            f"picked; {pick.item_id} {pick.title}; "
            f"{render_queue_count(open_n)}; decode-only; plan {path_label}\n"
            f"{report}"
        )
        return SessionBResult(
            ok=True,
            verdict="picked",
            open_count=open_n,
            item=pick,
            message=msg,
            plan_text=plan_markdown,
            keep_schedule=True,
            scheduler_delete_called=False,
            implement_result=None,
        )

    # Pluggable implement — default dry_run_implement records intent only.
    fn = implement if implement is not None else default_implement
    result = fn(pick) or {}
    new_text = plan_markdown
    verdict = "picked"
    sha = ship_sha or result.get("sha")
    is_dry = bool(result.get("dry_run")) and not sha

    if sha and not is_dry:
        new_text = mark_item_shipped(
            plan_markdown,
            pick.item_id,
            str(sha),
            note=str(result.get("note") or ""),
        )
        if write_path and plan_path is not None:
            Path(plan_path).write_text(new_text, encoding="utf-8")
        verdict = "shipped"
        open_n = count_open(parse_queue_items(new_text))
        msg = (
            f"{verdict}; {pick.item_id} sha={sha or '-'}; "
            f"{render_queue_count(open_n)}; plan {path_label}"
        )
    elif is_dry:
        # Dry-run: record intended file_touch / acceptance; never write repo/plan.
        verdict = "dry_run"
        msg = (
            f"dry-run; {pick.item_id}; file_touch={pick.file_touch}; "
            f"acceptance={pick.acceptance}; {render_queue_count(open_n)}; "
            f"plan {path_label}; no repo write"
        )
    else:
        msg = (
            f"{verdict}; {pick.item_id} sha={sha or '-'}; "
            f"{render_queue_count(open_n)}; plan {path_label}"
        )

    return SessionBResult(
        ok=True,
        verdict=verdict,
        open_count=open_n,
        item=pick,
        message=msg,
        plan_text=new_text,
        keep_schedule=True,
        scheduler_delete_called=False,
        implement_result=dict(result),
    )


def run_session_b_path(
    plan_path: str | Path,
    *,
    implement: ImplementFn | None = None,
    decode_only: bool = True,
    write: bool = False,
    ship_sha: str | None = None,
) -> SessionBResult:
    path = Path(plan_path)
    text = path.read_text(encoding="utf-8")
    return run_session_b(
        text,
        implement=implement,
        plan_path=path,
        decode_only=decode_only,
        write_path=write,
        ship_sha=ship_sha,
    )
