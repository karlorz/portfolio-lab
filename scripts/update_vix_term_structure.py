#!/usr/bin/env python3
"""
Update vix_term_structure.json from market.db (^VIX spot + ^VIX3M front).

Batch BV: prefer true CBOE spot ``^VIX`` when present; only fall back to
VIX3M-as-spot proxy when ^VIX is missing — and stamp provenance so operators
never confuse a proxy term structure with true spot.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

META_KEY = "_meta"


def _short_git_sha() -> Optional[str]:
    try:
        from src.dashboard.generator import _generator_git_sha_short

        return _generator_git_sha_short()
    except Exception:  # noqa: BLE001 — provenance best-effort
        return None


def _build_merged_frame(
    conn: sqlite3.Connection,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Return (merged_df, provenance) with honest spot/front sources."""
    vix3m_df = pd.read_sql_query(
        "SELECT date, close as vix3m FROM prices WHERE symbol = '^VIX3M' ORDER BY date",
        conn,
    )
    vix_df = pd.read_sql_query(
        "SELECT date, close as vix FROM prices WHERE symbol = '^VIX' ORDER BY date",
        conn,
    )

    provenance: Dict[str, Any] = {
        "vix_spot_rows": int(len(vix_df)),
        "vix3m_rows": int(len(vix3m_df)),
        "live_authoritative": False,
    }

    if vix_df.empty and not vix3m_df.empty:
        logger.warning(
            "^VIX missing in market.db — building term structure from ^VIX3M only "
            "(%d rows); contango_spot_1m will be 0 when spot==front",
            len(vix3m_df),
        )
        merged = vix3m_df.copy()
        merged["vix"] = merged["vix3m"]
        provenance.update(
            {
                "spot_source": "^VIX3M",
                "front_source": "^VIX3M",
                "spot_is_proxy": True,
                "proxy_reason": "missing_^VIX_in_market.db",
            }
        )
    elif vix3m_df.empty:
        raise RuntimeError("No ^VIX3M rows in market.db")
    else:
        # Prefer true spot; outer-ish join keeps all VIX3M sessions with spot
        # forward-filled from last ^VIX (spot sessions can lag by one day).
        merged = pd.merge(vix3m_df, vix_df, on="date", how="left")
        if merged["vix"].isna().all():
            raise RuntimeError("^VIX present but no overlapping dates with ^VIX3M")
        # Forward/back fill spot only across gaps (holidays), not invent levels
        merged["vix"] = merged["vix"].ffill().bfill()
        missing_spot = int(merged["vix"].isna().sum())
        provenance.update(
            {
                "spot_source": "^VIX",
                "front_source": "^VIX3M",
                "spot_is_proxy": False,
                "spot_ffill_note": (
                    "spot aligned to VIX3M calendar via ffill/bfill for session gaps"
                ),
                "rows_after_align": int(len(merged)),
                "spot_missing_after_fill": missing_spot,
            }
        )
        if missing_spot:
            logger.warning(
                "%d rows still missing spot after ffill/bfill; dropping those dates",
                missing_spot,
            )
            merged = merged.dropna(subset=["vix"])

    return merged, provenance


def update_vix_term_structure(
    *,
    data_dir: Optional[Path] = None,
    public_dir: Optional[Path] = None,
    market_db: Optional[Path] = None,
) -> bool:
    """Update vix_term_structure.json with VIX/VIX3M data + dual-write."""
    from src.paths import DATA_DIR, PUBLIC_DATA_DIR

    root = Path(data_dir) if data_dir is not None else Path(DATA_DIR)
    pub = Path(public_dir) if public_dir is not None else Path(PUBLIC_DATA_DIR)
    db_path = Path(market_db) if market_db is not None else (root / "market.db")
    vix_ts_path = root / "vix_term_structure.json"

    # Full rebuild from market.db (do not keep orphan proxy dates that lack
    # a current merge row — avoids sticky "unknown" regime leftovers).
    prior_n = 0
    if vix_ts_path.exists():
        try:
            with open(vix_ts_path, encoding="utf-8") as f:
                loaded = json.load(f)
            if isinstance(loaded, dict):
                prior_n = sum(
                    1
                    for k, v in loaded.items()
                    if k != META_KEY and isinstance(v, dict)
                )
            logger.info(
                "Rebuilding vix_term_structure.json (prior date entries=%d)",
                prior_n,
            )
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("Could not read existing VIX file: %s", exc)
    else:
        logger.info("Creating new vix_term_structure.json")
    current_data: Dict[str, Any] = {}

    if not db_path.exists():
        logger.error("market.db not found: %s", db_path)
        return False

    conn = sqlite3.connect(db_path)
    try:
        merged, provenance = _build_merged_frame(conn)
        logger.info(
            "Merged VIX/VIX3M data: %d rows (%s → %s); spot_source=%s proxy=%s",
            len(merged),
            merged["date"].min(),
            merged["date"].max(),
            provenance.get("spot_source"),
            provenance.get("spot_is_proxy"),
        )

        merged["vix_vix3m_ratio"] = merged["vix"] / merged["vix3m"]

        from src.data.vix_futures import VIXTermStructure

        updated_count = 0
        for _, row in merged.iterrows():
            date = str(row["date"])[:10]
            vix = float(row["vix"])
            vix3m = float(row["vix3m"])
            ratio = float(row["vix_vix3m_ratio"])

            if ratio < 0.8:
                regime = "extreme_backwardation"
            elif ratio < 0.95:
                regime = "backwardation"
            elif ratio < 1.0:
                regime = "flat"
            elif ratio < 1.15:
                regime = "contango"
            else:
                regime = "extreme_contango"

            raw = {
                "date": date,
                "vix_spot": vix,
                "front_month": vix3m,
                "third_month": vix3m,  # no VIX6M — second/third proxied to front
                "days_to_expiry_front": 0,
            }
            try:
                ts = VIXTermStructure.from_dict(raw)
                entry = ts.to_dict()
            except (TypeError, ValueError, KeyError):
                entry = {
                    "date": date,
                    "vix_spot": vix,
                    "front_month": vix3m,
                    "second_month": vix3m,
                    "third_month": vix3m,
                    "contango_1m_2m": 0.0,
                    "contango_spot_1m": (vix3m / vix - 1.0) * 100.0 if vix else 0.0,
                    "is_contango": bool(vix3m >= vix) if vix else True,
                    "days_to_expiry_front": 0,
                }
            entry["vix_vix3m_ratio"] = ratio
            entry["regime"] = regime
            entry["source"] = "market.db"
            entry["spot_source"] = provenance.get("spot_source")
            entry["front_source"] = provenance.get("front_source")
            entry["spot_is_proxy"] = bool(provenance.get("spot_is_proxy"))
            entry["as_of"] = date
            current_data[date] = entry
            updated_count += 1

        logger.info("Updated %d entries", updated_count)

        sha = _short_git_sha()
        meta = {
            "schema": "vix_term_structure/v1",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "generator_git_sha": sha,
            "generator_git_sha_status": "full" if sha else "unavailable",
            "n_dates": len(current_data),
            "date_min": min(current_data) if current_data else None,
            "date_max": max(current_data) if current_data else None,
            "live_authoritative": False,
            **provenance,
        }
        payload: Dict[str, Any] = {META_KEY: meta, **current_data}

        vix_ts_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = vix_ts_path.with_suffix(vix_ts_path.suffix + ".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
            f.write("\n")
        tmp.replace(vix_ts_path)
        logger.info(
            "Saved %s with %d date entries + _meta (spot=%s proxy=%s)",
            vix_ts_path,
            len(current_data),
            meta.get("spot_source"),
            meta.get("spot_is_proxy"),
        )

        # Dual-write public operator tree (best-effort)
        try:
            pub.mkdir(parents=True, exist_ok=True)
            pub_path = pub / "vix_term_structure.json"
            pub_tmp = pub_path.with_suffix(pub_path.suffix + ".tmp")
            with open(pub_tmp, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2)
                f.write("\n")
            pub_tmp.replace(pub_path)
            logger.info("Public dual-write %s", pub_path)
        except OSError as exc:
            logger.warning("Public dual-write failed: %s", exc)

        # Regime distribution (date entries only)
        regimes: Dict[str, int] = {}
        for entry in current_data.values():
            regime = entry.get("regime", "unknown")
            regimes[regime] = regimes.get(regime, 0) + 1
        logger.info("Regime distribution:")
        n = max(len(current_data), 1)
        for regime, count in sorted(regimes.items()):
            logger.info("  %s: %d (%.1f%%)", regime, count, count / n * 100.0)

        return True
    except Exception as e:  # noqa: BLE001 — CLI must not crash host cron
        logger.error("Error updating vix_term_structure.json: %s", e)
        return False
    finally:
        conn.close()


if __name__ == "__main__":
    success = update_vix_term_structure()
    if success:
        logger.info("Successfully updated vix_term_structure.json")
    else:
        logger.error("Failed to update vix_term_structure.json")
        raise SystemExit(1)
