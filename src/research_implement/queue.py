"""Six-field OPEN Queue model + markdown parse/write against ``## Queue``.

A B-pickable item is OPEN with all six fields and ``ready-for-implement: yes``.
SHIPPED does not count as OPEN. Watch / Project Work are outside this parser.
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
    """Return body text under ``## {heading}`` until the next ``## `` heading."""
    matches = list(_SECTION_RE.finditer(markdown))
    target = heading.strip().lower()
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
    """Parse Queue items from a living plan markdown document."""
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


def append_queue_item(markdown: str, item_markdown: str) -> str:
    """Insert ``item_markdown`` at the end of the ``## Queue`` section."""
    matches = list(_SECTION_RE.finditer(markdown))
    queue_idx = None
    for idx, match in enumerate(matches):
        if match.group(1).strip().lower() == "queue":
            queue_idx = idx
            break
    if queue_idx is None:
        # Create a Queue section at the end.
        block = item_markdown if item_markdown.endswith("\n") else item_markdown + "\n"
        suffix = "" if markdown.endswith("\n") or not markdown else "\n"
        return f"{markdown}{suffix}\n## Queue\n\n{block}"

    end = matches[queue_idx + 1].start() if queue_idx + 1 < len(matches) else len(markdown)
    insert = item_markdown if item_markdown.endswith("\n") else item_markdown + "\n"
    # Keep a blank line before the next section when present.
    before = markdown[:end].rstrip() + "\n\n"
    after = markdown[end:]
    return before + insert + ("" if after.startswith("\n") or not after else "\n") + after


def mark_item_shipped(markdown: str, item_id: str, sha: str, note: str = "") -> str:
    """Flip ``status:`` on the named Queue item to SHIPPED (Session B plan write)."""
    items = parse_queue_items(markdown)
    target = next((i for i in items if i.item_id.upper() == item_id.upper()), None)
    if target is None:
        raise KeyError(f"Queue item not found: {item_id}")
    old_status_line = None
    for line in target.raw.splitlines():
        if _STATUS_RE.match(line):
            old_status_line = line
            break
    shipped = f"status: SHIPPED `{sha}`"
    if note:
        shipped = f"{shipped} — {note}"
    if old_status_line is None:
        raise ValueError(f"{item_id} has no status: line")
    # Replace only within the item's raw block occurrence after its heading.
    heading_pat = re.compile(
        rf"(?m)^(###[ \t]+{re.escape(item_id)}\.\s*.*?$)(.*?)(?=^###[ \t]+Q\d+\.|^##[ \t]|\Z)",
        flags=re.S,
    )

    def _sub(match: re.Match[str]) -> str:
        body = match.group(2)
        new_body, n = _STATUS_RE.subn(shipped, body, count=1)
        if n != 1:
            raise ValueError(f"failed to rewrite status for {item_id}")
        # shipped items are no longer ready
        new_body, _ = _READY_RE.subn("ready-for-implement: no", new_body, count=1)
        return match.group(1) + new_body

    new_md, n = heading_pat.subn(_sub, markdown, count=1)
    if n != 1:
        raise ValueError(f"failed to locate Queue heading for {item_id}")
    return new_md


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


