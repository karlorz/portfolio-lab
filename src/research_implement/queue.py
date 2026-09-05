"""Six-field OPEN Queue model + markdown parse/write against ``## Queue``.

A B-pickable item is OPEN with all six fields and ``ready-for-implement: yes``.
SHIPPED does not count as OPEN. Watch / Project Work are outside this parser.

Beat 21: more than one ``## Queue`` heading is fail-closed via
``AmbiguousQueueError`` — parse / write / append refuse (no silent merge).
Session A/B return failed results without mutating the plan.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable

QUEUE_CAPACITY = 10
REQUIRED_FIELDS = (
    "title",
    "acceptance",
    "risks",
    "file_touch",
    "breaking_change",
    "redeploy_notes",
)

_SECTION_RE = re.compile(r"(?m)^##[ \t]+(.+?)\s*$")
_ITEM_HEADING_RE = re.compile(r"(?m)^###[ \t]+(Q\d+)\.\s*(.+?)\s*$")
_FIELD_RE = re.compile(
    r"(?m)^\s*([1-6])\.\s*\*\*(title|acceptance|risks|file_touch|breaking_change|redeploy_notes)\*\*\s*:\s*(.*)\s*$"
)
_STATUS_RE = re.compile(r"(?mi)^[ \t]*status[ \t]*:[ \t]*(.*?)[ \t]*$")
_READY_RE = re.compile(r"(?mi)^[ \t]*ready-for-implement[ \t]*:[ \t]*(.*?)[ \t]*$")


class AmbiguousQueueError(ValueError):
    """More than one ``## Queue`` heading — refuse to parse/write (Beat 21).

    Fail-closed: callers must not silently merge, pick-first, or mutate the plan.
    Session A/B catch this and return a failed result with plan text unchanged.
    """

    def __init__(self, count: int, message: str | None = None) -> None:
        self.count = int(count)
        if message is None:
            message = (
                f"ambiguous ## Queue: found {self.count} headings; "
                "refuse to parse/write (fail-closed)"
            )
        super().__init__(message)


def count_queue_headings(markdown: str) -> int:
    """Count ``## Queue`` section headings (case-insensitive; exact Queue token)."""
    n = 0
    for match in _SECTION_RE.finditer(markdown or ""):
        if match.group(1).strip().lower() == "queue":
            n += 1
    return n


def require_unique_queue_section(markdown: str) -> None:
    """Raise ``AmbiguousQueueError`` when markdown has more than one ``## Queue``."""
    n = count_queue_headings(markdown)
    if n > 1:
        raise AmbiguousQueueError(n)


@dataclass(frozen=True)
class QueueItem:
    """One ``## Queue`` row (OPEN or SHIPPED)."""

    item_id: str
    heading: str
    title: str = ""
    acceptance: str = ""
    risks: str = ""
    file_touch: str = ""
    breaking_change: str = ""
    redeploy_notes: str = ""
    status: str = ""
    ready_for_implement: str = ""
    raw: str = ""

    def field_map(self) -> dict[str, str]:
        return {
            "title": self.title,
            "acceptance": self.acceptance,
            "risks": self.risks,
            "file_touch": self.file_touch,
            "breaking_change": self.breaking_change,
            "redeploy_notes": self.redeploy_notes,
        }


def extract_section(markdown: str, heading: str) -> str:
    """Return body text under ``## {heading}`` until the next ``## `` heading.

    Beat 21: when ``heading`` is Queue and more than one ``## Queue`` exists,
    raise ``AmbiguousQueueError`` (fail-closed; never silently pick-first).
    """
    matches = list(_SECTION_RE.finditer(markdown))
    target = heading.strip().lower()
    if target == "queue":
        require_unique_queue_section(markdown)
    for idx, match in enumerate(matches):
        if match.group(1).strip().lower() == target:
            start = match.end()
            end = matches[idx + 1].start() if idx + 1 < len(matches) else len(markdown)
            return markdown[start:end]
    return ""


def _parse_item_block(item_id: str, heading: str, body: str) -> QueueItem:
    fields: dict[str, str] = {name: "" for name in REQUIRED_FIELDS}
    for match in _FIELD_RE.finditer(body):
        name = match.group(2)
        fields[name] = match.group(3).strip()
    status_m = _STATUS_RE.search(body)
    ready_m = _READY_RE.search(body)
    return QueueItem(
        item_id=item_id,
        heading=heading.strip(),
        title=fields["title"],
        acceptance=fields["acceptance"],
        risks=fields["risks"],
        file_touch=fields["file_touch"],
        breaking_change=fields["breaking_change"],
        redeploy_notes=fields["redeploy_notes"],
        status=(status_m.group(1).strip() if status_m else ""),
        ready_for_implement=(ready_m.group(1).strip() if ready_m else ""),
        raw=body.strip("\n"),
    )


def parse_queue_items(markdown: str) -> list[QueueItem]:
    """Parse Queue items from a living plan markdown document.

    Beat 21: raises ``AmbiguousQueueError`` when more than one ``## Queue``
    heading is present (fail-closed; never silently merge sections).
    """
    require_unique_queue_section(markdown)
    section = extract_section(markdown, "Queue")
    if not section.strip():
        # Allow callers to pass a Queue-only fragment.
        section = markdown if "### Q" in markdown else ""
    headings = list(_ITEM_HEADING_RE.finditer(section))
    items: list[QueueItem] = []
    for idx, match in enumerate(headings):
        start = match.end()
        end = headings[idx + 1].start() if idx + 1 < len(headings) else len(section)
        items.append(_parse_item_block(match.group(1), match.group(2), section[start:end]))
    return items


def is_open_status(status: str) -> bool:
    token = (status or "").strip()
    if not token:
        return False
    head = token.split(None, 1)[0].upper()
    return head == "OPEN"


def is_complete_six_field(item: QueueItem) -> bool:
    return all(bool(str(item.field_map()[name]).strip()) for name in REQUIRED_FIELDS)


def is_ready_yes(value: str) -> bool:
    return (value or "").strip().lower() in {"yes", "y", "true", "1"}


def is_b_pickable(item: QueueItem) -> bool:
    """OPEN + six fields + ready-for-implement: yes (SHIPPED never counts)."""
    return (
        is_open_status(item.status)
        and is_complete_six_field(item)
        and is_ready_yes(item.ready_for_implement)
    )


def count_open(items: Iterable[QueueItem]) -> int:
    return sum(1 for item in items if is_b_pickable(item))


def first_b_pick(items: Iterable[QueueItem]) -> QueueItem | None:
    for item in items:
        if is_b_pickable(item):
            return item
    return None


def render_queue_count(open_count: int, capacity: int = QUEUE_CAPACITY) -> str:
    return f"queue {open_count}/{capacity}"


def next_queue_id(items: Iterable[QueueItem]) -> str:
    max_n = 0
    for item in items:
        m = re.match(r"Q(\d+)$", item.item_id, flags=re.I)
        if m:
            max_n = max(max_n, int(m.group(1)))
    return f"Q{max_n + 1}"


def format_queue_item(
    *,
    item_id: str,
    heading: str,
    title: str,
    acceptance: str,
    risks: str,
    file_touch: str,
    breaking_change: str | bool,
    redeploy_notes: str,
    status: str = "OPEN",
    ready_for_implement: str = "yes",
) -> str:
    """Render one Queue row in the living-plan six-field format."""
    bc = breaking_change
    if isinstance(bc, bool):
        bc = "true" if bc else "false"
    lines = [
        f"### {item_id}. {heading}",
        f"1. **title**: {title}",
        f"2. **acceptance**: {acceptance}",
        f"3. **risks**: {risks}",
        f"4. **file_touch**: {file_touch}",
        f"5. **breaking_change**: {bc}",
        f"6. **redeploy_notes**: {redeploy_notes}",
        f"status: {status}",
        f"ready-for-implement: {ready_for_implement}",
        "",
    ]
    return "\n".join(lines)


def serialize_queue_item(item: QueueItem) -> str:
    """Serialize one parsed ``QueueItem`` back to six-field markdown."""
    return format_queue_item(
        item_id=item.item_id,
        heading=item.heading or item.title or item.item_id,
        title=item.title,
        acceptance=item.acceptance,
        risks=item.risks,
        file_touch=item.file_touch,
        breaking_change=item.breaking_change,
        redeploy_notes=item.redeploy_notes,
        status=item.status if item.status != "" else "OPEN",
        ready_for_implement=(
            item.ready_for_implement if item.ready_for_implement != "" else "yes"
        ),
    )


def serialize_queue_items(items: Iterable[QueueItem]) -> str:
    """Serialize Queue items as a section body (no ``## Queue`` heading)."""
    parts: list[str] = []
    for item in items:
        block = serialize_queue_item(item)
        parts.append(block if block.endswith("\n") else block + "\n")
    return "".join(parts)


def _create_queue_section(markdown: str, body: str) -> str:
    """Create a missing ``## Queue`` section without destroying other content.

    Beat 20: when the living plan has no ``## Queue`` heading, Session A append
    / ``write_queue_section`` add one. YAML front matter, Watch, Project Work,
    Heartbeat, and other ``##`` sections stay intact. Empty / whitespace-only
    markdown becomes a Queue-only document (no leading blank line).
    """
    block = body if body.endswith("\n") or not body else body + "\n"
    if not str(markdown).strip():
        return f"## Queue\n\n{block}"
    suffix = "" if markdown.endswith("\n") else "\n"
    return f"{markdown}{suffix}\n## Queue\n\n{block}"


def write_queue_section(
    markdown: str, items: Iterable[QueueItem] | None = None
) -> str:
    """Replace ``## Queue`` body with serialized items (parse → serialize → write).

    When ``items`` is None, re-serialize currently parsed Queue items (identity
    round-trip). Other sections (Watch / Project Work / Heartbeat) and YAML
    front matter are preserved. Passing an empty iterable clears Queue item
    rows while keeping the section.

    Beat 20: if ``## Queue`` is absent, create it (same preserve rules) rather
    than failing or rewriting the whole document.

    Beat 21: more than one ``## Queue`` → ``AmbiguousQueueError`` (no mutate).
    """
    require_unique_queue_section(markdown)
    item_list = list(parse_queue_items(markdown) if items is None else items)
    body = serialize_queue_items(item_list)
    matches = list(_SECTION_RE.finditer(markdown))
    queue_idx = None
    for idx, match in enumerate(matches):
        if match.group(1).strip().lower() == "queue":
            queue_idx = idx
            break
    if queue_idx is None:
        return _create_queue_section(markdown, body)

    start = matches[queue_idx].end()
    end = matches[queue_idx + 1].start() if queue_idx + 1 < len(matches) else len(markdown)
    before = markdown[:start].rstrip() + "\n\n"
    after = markdown[end:]
    insert = body if body.endswith("\n") or not body else body + "\n"
    if not insert.strip():
        insert = "\n"
    return before + insert + ("" if after.startswith("\n") or not after else "\n") + after


def append_queue_item(markdown: str, item_markdown: str) -> str:
    """Insert ``item_markdown`` at the end of the ``## Queue`` section.

    Only the Queue body grows; Watch / Project Work / Heartbeat (and any other
    ``##`` sections) plus YAML front matter are preserved in place.

    Beat 20: if ``## Queue`` is missing, create the section (preserving Watch /
    Heartbeat / front matter) and append the item.

    Beat 21: more than one ``## Queue`` → ``AmbiguousQueueError`` (no mutate).
    """
    require_unique_queue_section(markdown)
    matches = list(_SECTION_RE.finditer(markdown))
    queue_idx = None
    for idx, match in enumerate(matches):
        if match.group(1).strip().lower() == "queue":
            queue_idx = idx
            break
    if queue_idx is None:
        block = item_markdown if item_markdown.endswith("\n") else item_markdown + "\n"
        return _create_queue_section(markdown, block)

    end = matches[queue_idx + 1].start() if queue_idx + 1 < len(matches) else len(markdown)
    insert = item_markdown if item_markdown.endswith("\n") else item_markdown + "\n"
    # Keep a blank line before the next section when present.
    before = markdown[:end].rstrip() + "\n\n"
    after = markdown[end:]
    return before + insert + ("" if after.startswith("\n") or not after else "\n") + after


def mark_item_shipped(markdown: str, item_id: str, sha: str, note: str = "") -> str:
    """Flip ``status:`` on the named Queue item to SHIPPED (Session B plan write).

    Rewrites only the ``## Queue`` section via serialize → ``write_queue_section``.
    Other markdown sections (Watch / Project Work / Heartbeat) are preserved.

    Beat 21: more than one ``## Queue`` → ``AmbiguousQueueError`` (no mutate).
    """
    items = parse_queue_items(markdown)
    target_idx = next(
        (i for i, it in enumerate(items) if it.item_id.upper() == item_id.upper()),
        None,
    )
    if target_idx is None:
        raise KeyError(f"Queue item not found: {item_id}")
    target = items[target_idx]
    if not _STATUS_RE.search(target.raw) and not str(target.status).strip():
        raise ValueError(f"{item_id} has no status: line")
    shipped_status = f"SHIPPED `{sha}`"
    if note:
        shipped_status = f"{shipped_status} — {note}"
    updated = QueueItem(
        item_id=target.item_id,
        heading=target.heading,
        title=target.title,
        acceptance=target.acceptance,
        risks=target.risks,
        file_touch=target.file_touch,
        breaking_change=target.breaking_change,
        redeploy_notes=target.redeploy_notes,
        status=shipped_status,
        ready_for_implement="no",
        raw="",
    )
    new_items = list(items)
    new_items[target_idx] = updated
    return write_queue_section(markdown, new_items)


def queue_item_from_fields(
    *,
    item_id: str,
    heading: str,
    title: str,
    acceptance: str,
    risks: str,
    file_touch: str,
    breaking_change: str | bool,
    redeploy_notes: str,
    status: str = "OPEN",
    ready_for_implement: str = "yes",
) -> QueueItem:
    bc = breaking_change
    if isinstance(bc, bool):
        bc = "true" if bc else "false"
    raw = format_queue_item(
        item_id=item_id,
        heading=heading,
        title=title,
        acceptance=acceptance,
        risks=risks,
        file_touch=file_touch,
        breaking_change=bc,
        redeploy_notes=redeploy_notes,
        status=status,
        ready_for_implement=ready_for_implement,
    )
    return QueueItem(
        item_id=item_id,
        heading=heading,
        title=title,
        acceptance=acceptance,
        risks=risks,
        file_touch=file_touch,
        breaking_change=str(bc),
        redeploy_notes=redeploy_notes,
        status=status,
        ready_for_implement=ready_for_implement,
        raw=raw,
    )


