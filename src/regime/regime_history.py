"""Canonical daily history boundary for the regime-transition forecaster."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.paths import DATA_DIR
from src.regime.regime_transition_forecaster import REGIMES

DEFAULT_REGIME_HISTORY_PATH = DATA_DIR / "regime_log.json"
_KNOWN_REGIMES = frozenset(REGIMES)


@dataclass(frozen=True)
class DailyRegimeHistory:
    """Oldest-to-newest daily labels plus provenance and quality metadata."""

    labels: list[str]
    records: list[dict[str, str | float | None]]
    metadata: dict[str, Any]


def _parse_timestamp(value: Any) -> datetime | None:
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        parsed = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _evidence_quality(history_len: int, transition_count: int) -> str:
    if history_len < 2:
        return "insufficient"
    if transition_count < 2:
        return "prior_dominated"
    if history_len < 30:
        return "limited"
    return "observed"


def _parse_vix(value: Any) -> float | None:
    if value is None:
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(parsed):
        return None
    return parsed


def load_daily_regime_history(
    path: Path | str = DEFAULT_REGIME_HISTORY_PATH,
) -> DailyRegimeHistory:
    """Load JSONL observations and collapse to the final row of each UTC day.

    Invalid JSON, non-object rows, missing/invalid timestamps, and unknown
    regime labels are skipped and disclosed in metadata. The function is
    read-only and deliberately does not consult the legacy SQLite table.
    """
    history_path = Path(path)
    raw_row_count = 0
    malformed_row_count = 0
    unknown_regime_count = 0
    observations: list[tuple[datetime, str, float | None]] = []

    try:
        lines = history_path.open("r", encoding="utf-8")
    except OSError:
        lines = None

    if lines is not None:
        with lines:
            for line in lines:
                if not line.strip():
                    continue
                raw_row_count += 1
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    malformed_row_count += 1
                    continue
                if not isinstance(row, dict):
                    malformed_row_count += 1
                    continue
                detected_at = _parse_timestamp(row.get("detected_at"))
                if detected_at is None:
                    malformed_row_count += 1
                    continue
                regime = str(row.get("regime") or "").strip().upper()
                if regime not in _KNOWN_REGIMES:
                    unknown_regime_count += 1
                    continue
                observations.append(
                    (
                        detected_at,
                        regime,
                        _parse_vix(row.get("vix_level", row.get("vix"))),
                    )
                )

    observations.sort(key=lambda item: item[0])
    daily: dict[str, tuple[datetime, str, float | None]] = {}
    for detected_at, regime, vix in observations:
        daily[detected_at.date().isoformat()] = (detected_at, regime, vix)
    collapsed = list(daily.values())
    labels = [regime for _, regime, _ in collapsed]
    records = [
        {"d": detected_at.date().isoformat(), "r": regime.lower(), "v": vix}
        for detected_at, regime, vix in collapsed
    ]
    transition_count = sum(
        previous != current for previous, current in zip(labels, labels[1:])
    )

    metadata: dict[str, Any] = {
        "history_source": str(history_path),
        "observation_unit": "utc_day_last_observation",
        "raw_row_count": raw_row_count,
        "valid_row_count": len(observations),
        "history_len": len(labels),
        "malformed_row_count": malformed_row_count,
        "unknown_regime_count": unknown_regime_count,
        "observed_transition_count": transition_count,
        "first_observation_at": collapsed[0][0].isoformat() if collapsed else None,
        "last_observation_at": collapsed[-1][0].isoformat() if collapsed else None,
        "evidence_quality": _evidence_quality(len(labels), transition_count),
    }
    return DailyRegimeHistory(labels=labels, records=records, metadata=metadata)
