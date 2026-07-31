#!/usr/bin/env python3
"""
v5.70: Performance Attribution System

Tracks each signal source's contribution to portfolio P&L, enabling
identification of which modules add alpha vs degrade performance.

Key metrics per signal source:
- Contribution to return (bps)
- Hit rate (directionally correct %)
- Sharpe contribution
- Correlation with other signals
- Win/loss ratio
- Average return when active vs inactive

Usage:
    python -m src.monitor.performance_attribution report
    python -m src.monitor.performance_attribution report --days 90
    python -m src.monitor.performance_attribution dashboard
"""

import json
import logging
import sqlite3
from src.paths import sqlite_connect, PUBLIC_DATA_DIR
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional
import numpy as np


from src.paths import DATA_DIR
from src.utils import safe_get
from src.backtest.metrics import save_results_json


__all__ = ['SIGNAL_SOURCE_META', 'SourceAttribution', 'AttributionReport', 'PerformanceAttribution', 'print_report', 'patch_save_vote']

logger = logging.getLogger(__name__)

PAPER_TRADING_DB = DATA_DIR / "paper_trading.db"

# All known signal sources with display names
SIGNAL_SOURCE_META = {
    "multi_speed_momentum": {"name": "Multi-Speed Momentum", "category": "trend", "weight_tier": "primary"},
    "cross_asset_rv": {"name": "Cross-Asset RV", "category": "meanrev", "weight_tier": "tactical"},
    "international_momentum": {"name": "International Momentum", "category": "trend", "weight_tier": "primary"},
    "alternative_data": {"name": "Alternative Data", "category": "fundamental", "weight_tier": "primary"},
    "cross_asset_regime_arb": {"name": "Cross-Asset Regime Arb", "category": "regime", "weight_tier": "tactical"},
    "unified_overlay": {"name": "Unified Overlay", "category": "orchestration", "weight_tier": "tactical"},
}


@dataclass
class SourceAttribution:
    """Attribution metrics for a single signal source."""
    source: str
    display_name: str
    category: str

    # Signal activity
    total_readings: int
    active_days: int

    # Directional accuracy
    hit_rate: Optional[float]  # None when active_days==0 (no_data); else directional hit rate
    win_rate: Optional[float]  # None when active_days==0; else positive-return rate contribution
    avg_return_bps: float    # Average daily return contribution (bps)
    total_return_bps: float  # Cumulative return contribution (bps)

    # Risk-adjusted
    sharpe_contribution: float
    max_consecutive_losses: int

    # Correlation with other signals
    avg_correlation: float   # Average pairwise correlation with all other sources

    # Weight contribution
    avg_weight: float        # Average weight assigned in ensemble
    current_weight_regime: str = "normal"

    def to_dict(self) -> dict:
        return asdict(self)

    @property
    def efficiency_ratio(self) -> float:
        """Return per-unit-of-risk efficiency."""
        if self.avg_return_bps == 0:
            return 0.0
        return 0.0 if self.hit_rate is None else self.hit_rate * abs(self.avg_return_bps) / 100


@dataclass
class AttributionReport:
    """Complete attribution report."""
    timestamp: str
    start_date: str
    end_date: str
    analysis_days: int

    # Per-source attribution
    sources: Dict[str, SourceAttribution]

    # Aggregate
    best_source: Optional[str]
    worst_source: Optional[str]
    avg_hit_rate: Optional[float]  # None when no source has active_days > 0
    avg_correlation: float

    # Signal diversity
    avg_active_sources_per_day: float
    total_sources_tracked: int

    # Special flags
    degradation_signals: List[str]  # Sources degrading performance
    top_performers: List[str]       # Sources adding most value
    status: str = "ok"  # ok | no_data

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, default=str)


class PerformanceAttribution:
    """Compute and report per-source performance attribution."""

    def __init__(self, data_dir: Optional[Path] = None):
        self.data_dir = Path(data_dir) if data_dir else DATA_DIR
        self.ensemble_db = self.data_dir / "ensemble_signals.db"
        self.attribution_dir = self.data_dir / "attribution"
        self.attribution_dir.mkdir(parents=True, exist_ok=True)

    def _get_signal_history(self, days: int = 90) -> List[Dict]:
        """Extract signal reading history from ensemble database.

        Live ``source_readings`` has thousands of rows per day. A tight
        ``LIMIT days * n_sources * 2`` only returned the latest calendar day
        (~1k rows), so attribution always looked like no_data when joined to
        multi-day returns. Prefer one latest reading per (source, date).
        """
        if not self.ensemble_db.exists():
            logger.warning("Ensemble DB not found: %s", self.ensemble_db)
            return []

        history = []
        cutoff = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
        try:
            with sqlite_connect(self.ensemble_db) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()

                # One row per source per calendar day (latest timestamp wins).
                # Window functions require SQLite 3.25+ (available on prod).
                try:
                    cursor.execute(
                        """
                        SELECT timestamp, source, value, confidence, weight, regime_fit, explanation
                        FROM (
                            SELECT
                                timestamp, source, value, confidence, weight, regime_fit, explanation,
                                ROW_NUMBER() OVER (
                                    PARTITION BY source, substr(timestamp, 1, 10)
                                    ORDER BY timestamp DESC
                                ) AS rn
                            FROM source_readings
                            WHERE substr(timestamp, 1, 10) >= ?
                        ) ranked
                        WHERE rn = 1
                        ORDER BY timestamp DESC
                        """,
                        (cutoff,),
                    )
                except sqlite3.OperationalError:
                    # Fallback: larger raw limit if window functions unavailable
                    cursor.execute(
                        """
                        SELECT timestamp, source, value, confidence, weight, regime_fit, explanation
                        FROM source_readings
                        WHERE substr(timestamp, 1, 10) >= ?
                        ORDER BY timestamp DESC
                        LIMIT ?
                        """,
                        (cutoff, max(days * 50, 5000)),
                    )

                for row in cursor.fetchall():
                    history.append({
                        "timestamp": row["timestamp"],
                        "source": row["source"],
                        "value": row["value"],
                        "confidence": row["confidence"],
                        "weight": row["weight"],
                        "regime_fit": row["regime_fit"],
                        "explanation": row["explanation"],
                    })

                # Also get ensemble votes to cross-reference
                cursor.execute(
                    """
                    SELECT timestamp, regime, consensus, agreement_ratio, equity_bias,
                           duration_bias, gold_bias, action, confidence, reasoning
                    FROM ensemble_votes
                    WHERE substr(timestamp, 1, 10) >= ?
                    ORDER BY timestamp DESC
                    LIMIT ?
                    """,
                    (cutoff, days * 4),
                )

                for row in cursor.fetchall():
                    history.append({
                        "type": "ensemble_vote",
                        "timestamp": row["timestamp"],
                        "regime": row["regime"],
                        "consensus": row["consensus"],
                        "agreement_ratio": row["agreement_ratio"],
                        "equity_bias": row["equity_bias"],
                        "duration_bias": row["duration_bias"],
                        "gold_bias": row["gold_bias"],
                        "action": row["action"],
                        "confidence": row["confidence"],
                        "reasoning": row["reasoning"],
                    })

        except (KeyError, ValueError, TypeError, AttributeError, RuntimeError, sqlite3.Error) as e:
            logger.error("Error reading signal history: %s", e)

        return history

    def _ingest_return_row(
        self,
        daily_returns: Dict[str, Dict],
        date_str: Optional[str],
        daily_return: Any,
        cumulative_return: Any = None,
        *,
        overwrite: bool = True,
    ) -> None:
        """Insert one day into the returns map when date/return are parseable."""
        if not date_str or not isinstance(date_str, str):
            return
        day = date_str[:10]
        if len(day) < 10:
            return
        try:
            ret = float(daily_return)
        except (TypeError, ValueError):
            return
        if day in daily_returns and not overwrite:
            return
        entry = daily_returns.get(day, {})
        entry["daily_return"] = ret
        if cumulative_return is not None:
            try:
                entry["cumulative_return"] = float(cumulative_return)
            except (TypeError, ValueError):
                pass
        daily_returns[day] = entry

    def _load_returns_from_jsonl(self, path: Path, *, days: int) -> Dict[str, Dict]:
        """Parse performance.jsonl / daily_pnl.jsonl into date → return map."""
        out: Dict[str, Dict] = {}
        if not path.exists():
            return out
        try:
            # Tail-ish read for large performance.jsonl
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            logger.warning("Could not read %s: %s", path, exc)
            return out
        lines = [ln for ln in text.splitlines() if ln.strip()]
        # Prefer last occurrences (later lines overwrite earlier same date)
        for line in lines:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(row, dict) or "daily_return" not in row:
                continue
            date_str = row.get("date") or row.get("timestamp")
            self._ingest_return_row(
                out,
                str(date_str) if date_str is not None else None,
                row.get("daily_return"),
                row.get("cumulative_return") or row.get("total_pnl_pct"),
                overwrite=True,
            )
        if days > 0 and out:
            # Keep the newest ``days`` calendar dates only
            keep = sorted(out.keys(), reverse=True)[:days]
            out = {k: out[k] for k in keep}
        return out

    def _get_paper_trading_returns(self, days: int = 90) -> Dict[str, Dict]:
        """Get daily returns from paper trading simulation.

        SSOT order:
        1. ``paper_trading.db`` daily_snapshots (legacy)
        2. ``daily_pnl.jsonl`` / ``daily_pnl_latest.json`` (capture_daily_pnl)
        3. ``performance.jsonl`` (eval / paper journal)
        4. ``logs/performance_summary_*.json`` (legacy summaries)
        """
        daily_returns: Dict[str, Dict] = {}
        paper_db = self.data_dir / "paper_trading.db"

        if paper_db.exists():
            try:
                with sqlite_connect(paper_db) as conn:
                    conn.row_factory = sqlite3.Row
                    cursor = conn.cursor()
                    cursor.execute("""
                        SELECT date, daily_return, cumulative_return
                        FROM daily_snapshots
                        ORDER BY date DESC
                        LIMIT ?
                    """, (days,))
                    for row in cursor.fetchall():
                        self._ingest_return_row(
                            daily_returns,
                            row["date"],
                            row["daily_return"],
                            row["cumulative_return"],
                        )
            except (KeyError, ValueError, TypeError, AttributeError, RuntimeError, sqlite3.Error) as e:
                logger.warning("Could not read paper trading DB: %s", e)

        # Live paper SSOT: daily_pnl + performance journals (paper_trading.db often absent)
        if not daily_returns:
            for name in ("daily_pnl.jsonl", "performance.jsonl"):
                loaded = self._load_returns_from_jsonl(
                    self.data_dir / name, days=max(days, 120)
                )
                for day, payload in loaded.items():
                    # Prefer earlier sources (daily_pnl over performance when both set)
                    self._ingest_return_row(
                        daily_returns,
                        day,
                        payload.get("daily_return"),
                        payload.get("cumulative_return"),
                        overwrite=False,
                    )
            # Cap to newest ``days`` after merge
            if days > 0 and daily_returns:
                keep = sorted(daily_returns.keys(), reverse=True)[:days]
                daily_returns = {k: daily_returns[k] for k in keep}

        # Fallback: check json reports
        if not daily_returns:
            logs_root = self.data_dir / "logs"
            perf_file = max(logs_root.glob("performance_summary_*.json"), default=None) if logs_root.exists() else None
            if perf_file is None:
                # Legacy global DATA_DIR path used by older tests
                perf_file = max(
                    (DATA_DIR / "logs").glob("performance_summary_*.json"),
                    default=None,
                )
            if perf_file and perf_file.exists():
                try:
                    with open(perf_file) as f:
                        data = json.load(f)
                    if "daily_returns" in data:
                        for dr in data["daily_returns"]:
                            self._ingest_return_row(
                                daily_returns,
                                dr.get("date"),
                                dr.get("return"),
                                dr.get("cumulative"),
                            )
                except (OSError, json.JSONDecodeError, KeyError, ValueError, TypeError) as e:
                    logger.warning("Could not read performance file: %s", e)

        return daily_returns

    def _compute_hit_rate(
        self,
        signal_value: float,
        subsequent_return: float,
    ) -> bool:
        """Determine if signal correctly predicted direction."""
        if abs(signal_value) < 0.05:  # Neutral signal
            return abs(subsequent_return) < 0.001  # Correct if market flat
        return (signal_value > 0 and subsequent_return > 0) or \
               (signal_value < 0 and subsequent_return < 0)

    def _compute_source_attribution(
        self,
        source_signals: List[Dict],
        daily_returns: Dict[str, Dict],
    ) -> SourceAttribution:
        """Compute attribution metrics for a single signal source."""
        if not source_signals:
            return SourceAttribution(
                source="unknown",
                display_name="Unknown",
                category="unknown",
                total_readings=0,
                active_days=0,
                hit_rate=None,
                win_rate=None,
                avg_return_bps=0.0,
                total_return_bps=0.0,
                sharpe_contribution=0.0,
                max_consecutive_losses=0,
                avg_correlation=0.0,
                avg_weight=0.0,
            )

        source = source_signals[0]["source"]
        meta = SIGNAL_SOURCE_META.get(source, {"name": source, "category": "other"})

        hits = 0
        wins = 0
        total = 0
        daily_contributions = []
        weights = []
        consecutive_losses = 0
        max_consecutive_losses = 0
        prev_negative = False

        for sig in source_signals:
            sig_date = sig["timestamp"][:10]  # Extract YYYY-MM-DD
            value = sig.get("value", 0)

            if isinstance(value, str):
                try:
                    value = float(value)
                except (ValueError, TypeError) as e:
                    logger.debug("Failed to convert signal value to float: %s", e)
                    continue

            weight = sig.get("weight", 0)
            if isinstance(weight, str):
                try:
                    weight = float(weight)
                except (ValueError, TypeError) as e:
                    logger.debug("Failed to convert signal weight to float: %s", e)
                    weight = 0
            weights.append(weight)

            if sig_date not in daily_returns:
                continue

            ret = daily_returns[sig_date].get("daily_return", 0)
            if ret is None:
                continue

            total += 1

            # Hit rate: signal direction matches return direction
            if self._compute_hit_rate(value, ret):
                hits += 1

            # Win: positive return contribution when signal active
            if ret > 0:
                wins += 1
                if prev_negative:
                    consecutive_losses = 0
                    prev_negative = False
            else:
                consecutive_losses += 1
                max_consecutive_losses = max(max_consecutive_losses, consecutive_losses)
                prev_negative = True

            # Return contribution (bps) when signal is directional
            if abs(value) > 0.05:
                contribution_bps = ret * 10000 * abs(value)
            else:
                contribution_bps = ret * 10000 * weight * 2  # Scaled by weight
            daily_contributions.append(contribution_bps)

        if total == 0:
            hit_rate = None
            win_rate = None
        else:
            hit_rate = hits / total
            win_rate = wins / total
        avg_return_bps = np.mean(daily_contributions) if daily_contributions else 0.0
        total_return_bps = np.sum(daily_contributions) if daily_contributions else 0.0

        # Sharpe contribution (annualized)
        if len(daily_contributions) > 1 and np.std(daily_contributions) > 0:
            daily_sharpe = np.mean(daily_contributions) / np.std(daily_contributions)
            sharpe_contribution = daily_sharpe * np.sqrt(252)
        else:
            sharpe_contribution = 0.0

        avg_weight = np.mean(weights) if weights else 0.0

        # Average correlation placeholder — computed across sources later
        avg_correlation = 0.0

        return SourceAttribution(
            source=source,
            display_name=meta["name"],
            category=meta.get("category", "other"),
            total_readings=len(source_signals),
            active_days=total,
            hit_rate=None if hit_rate is None else round(hit_rate, 4),
            win_rate=None if win_rate is None else round(win_rate, 4),
            avg_return_bps=round(avg_return_bps, 2),
            total_return_bps=round(total_return_bps, 2),
            sharpe_contribution=round(sharpe_contribution, 4),
            max_consecutive_losses=max_consecutive_losses,
            avg_correlation=round(avg_correlation, 4),
            avg_weight=round(avg_weight, 4),
        )

    def _compute_correlation_matrix(
        self, source_data: Dict[str, List[Dict]]
    ) -> Dict[str, float]:
        """Compute average pairwise correlation for each source."""
        # Build aligned time series of signal values
        sources = list(source_data)
        if len(sources) < 2:
            return {s: 0.0 for s in sources}

        # Get all unique dates
        all_dates = set()
        date_values = {}
        for src, signals in source_data.items():
            for sig in signals:
                d = sig["timestamp"][:10]
                all_dates.add(d)
                if d not in date_values:
                    date_values[d] = {}
                val = sig.get("value", 0)
                if isinstance(val, str):
                    try:
                        val = float(val)
                    except (ValueError, TypeError) as e:
                        logger.debug("Failed to convert signal value to float: %s", e)
                        val = 0
                date_values[d][src] = val

        sorted_dates = sorted(all_dates)
        if len(sorted_dates) < 5:
            return {s: 0.0 for s in sources}

        # Build value matrix
        matrix = np.zeros((len(sorted_dates), len(sources)))
        for i, d in enumerate(sorted_dates):
            for j, src in enumerate(sources):
                matrix[i, j] = safe_get(date_values, d, src, default=0)

        # Compute pairwise correlation for each source
        avg_corrs = []
        corr_matrix = np.corrcoef(matrix.T)
        avg_corrs = {
            src: float(np.nanmean([corr_matrix[i, j]
                                   for j in range(len(sources))
                                   if j != i and not np.isnan(corr_matrix[i, j])]))
            for i, src in enumerate(sources)
        }

        return avg_corrs

    def generate_report(self, days: int = 90) -> AttributionReport:
        """Generate complete performance attribution report."""
        logger.info("Generating attribution report over %d days...", days)

        signal_history = self._get_signal_history(days)
        daily_returns = self._get_paper_trading_returns(days)

        # Group signals by source
        source_signals: Dict[str, List[Dict]] = {}
        ensemble_votes = []

        for entry in signal_history:
            if entry.get("type") == "ensemble_vote":
                ensemble_votes.append(entry)
            else:
                src = entry.get("source", "unknown")
                if src not in source_signals:
                    source_signals[src] = []
                source_signals[src].append(entry)

        # Compute per-source attribution
        sources = {}
        source_list = []
        for src, signals in source_signals.items():
            attribution = self._compute_source_attribution(signals, daily_returns)
            sources[src] = attribution
            source_list.append(attribution)

        # Compute correlation matrix
        correlations = self._compute_correlation_matrix(source_signals)
        for src in sources:
            if src in correlations:
                sources[src].avg_correlation = round(correlations[src], 4)

        # Find best/worst
        valid_sources = [s for s in source_list if s.total_readings > 0]

        best_source = None
        worst_source = None
        if valid_sources:
            # Best by sharpe contribution
            best_source = max(valid_sources, key=lambda s: s.sharpe_contribution).source
            worst_source = min(valid_sources, key=lambda s: s.sharpe_contribution).source

        # Identify degradation signals (negative sharpe or hit rate < 0.4)
        degradation_signals = [
            s.source for s in valid_sources
            if s.sharpe_contribution < -0.1 or (s.hit_rate is not None and s.hit_rate < 0.4 and s.total_readings > 5)
        ]

        top_performers = [
            s.source for s in valid_sources
            if s.sharpe_contribution > 0.5
        ]

        sources_with_days = [s for s in source_list if getattr(s, "active_days", 0) > 0]
        if sources_with_days:
            rates = [s.hit_rate for s in sources_with_days if s.hit_rate is not None]
            avg_hit_rate = float(np.mean(rates)) if rates else None
            report_status = "ok"
        else:
            avg_hit_rate = None
            report_status = "no_data"
        avg_corr = float(np.mean([s.avg_correlation for s in valid_sources])) if valid_sources else 0.0

        # Count average active sources per day
        active_counts = []
        for entry in signal_history:
            if entry.get("type") != "ensemble_vote":
                active_counts.append(1)
        avg_active = len(set(
            e.get("source", "") for e in signal_history
            if e.get("type") != "ensemble_vote"
        ))

        now = datetime.now()
        report = AttributionReport(
            timestamp=now.isoformat(),
            start_date=(now - timedelta(days=days)).strftime("%Y-%m-%d"),
            end_date=now.strftime("%Y-%m-%d"),
            analysis_days=days,
            sources=sources,
            best_source=best_source,
            worst_source=worst_source,
            avg_hit_rate=None if avg_hit_rate is None else round(avg_hit_rate, 4),
            avg_correlation=round(avg_corr, 4),
            status=report_status,
            avg_active_sources_per_day=avg_active,
            total_sources_tracked=len(sources),
            degradation_signals=degradation_signals,
            top_performers=top_performers,
        )

        return report

    def save_report(self, report: AttributionReport) -> Path:
        """Save attribution report to private DATA_DIR and public dual-write."""
        self.attribution_dir.mkdir(parents=True, exist_ok=True)
        filename = f"attribution_{report.timestamp[:10]}.json"
        path = self.attribution_dir / filename
        payload = report.to_dict()
        # Operator lag detection: stamp code tip when available
        try:
            from src.dashboard.generator import _stamp_generator_git_sha

            payload = _stamp_generator_git_sha(payload)
        except Exception:  # noqa: BLE001 — never block attribution save on stamp
            pass

        public_dir = Path(PUBLIC_DATA_DIR) / "attribution"
        latest = public_dir / "latest.json"
        dated = public_dir / filename
        reconciled_history: list[Path] = []
        paths_identical = False
        try:
            paths_identical = path.resolve() == latest.resolve()
        except OSError:
            paths_identical = False

        try:
            from src.dashboard.generator import _attach_dual_write_provenance

            payload = _attach_dual_write_provenance(
                payload,
                private_path=path,
                public_path=latest,
                dual_write_attempted=not paths_identical,
                dual_write_ok=None if not paths_identical else True,
                paths_identical=paths_identical,
                note="public dual-write under PUBLIC_DATA_DIR/attribution/",
            )
        except Exception:  # noqa: BLE001
            pass

        save_results_json(payload, output_path=str(path))
        logger.info("Saved attribution report: %s", path)
        # Public latest for dual-tree SSOT / index consumers
        if not paths_identical:
            try:
                public_dir.mkdir(parents=True, exist_ok=True)
                try:
                    from src.dashboard.generator import _attach_dual_write_provenance

                    payload = _attach_dual_write_provenance(
                        payload,
                        private_path=path,
                        public_path=latest,
                        dual_write_attempted=True,
                        dual_write_ok=True,
                        paths_identical=False,
                        note="public dual-write under PUBLIC_DATA_DIR/attribution/",
                    )
                    save_results_json(payload, output_path=str(path))
                except Exception:  # noqa: BLE001
                    pass
                save_results_json(payload, output_path=str(latest))
                save_results_json(payload, output_path=str(dated))
                logger.info("Published attribution to public: %s", latest)
                # Batch CJ: post-sync lag/hash on private + public latest
                try:
                    from src.dashboard.generator import (
                        finalize_dual_write_provenance_after_sync,
                    )

                    payload = finalize_dual_write_provenance_after_sync(
                        payload,
                        private_path=path,
                        public_path=latest,
                        dual_write_ok=True,
                        note="post_sync attribution dual-write (Batch CJ)",
                    )
                    # dated public copy should match finalized private body
                    save_results_json(payload, output_path=str(dated))
                except Exception:  # noqa: BLE001
                    pass
                # Private DATA_DIR is the attribution SSOT. Reconcile older
                # dated shards before rebuilding the public index so a prior
                # split-brain run cannot leave business values divergent.
                reconciled_history = self.reconcile_public_history(
                    public_root=public_dir.parent
                )
                # H19/BI: keep public index catalog current without full dashboard
                try:
                    from src.dashboard.public_data_index import (
                        refresh_public_data_index_after_partial_write,
                    )
                    from src.paths import PUBLIC_DATA_DIR as _pub

                    if refresh_public_data_index_after_partial_write(
                        public_dir=Path(_pub),
                        extra_paths=[latest, dated, *reconciled_history],
                        reason="attribution_dual_write",
                    ):
                        logger.info("Refreshed public index after attribution dual-write")
                except Exception as idx_exc:  # noqa: BLE001 — never block attribution
                    logger.warning("Public index refresh after attribution failed: %s", idx_exc)
            except (OSError, TypeError, ValueError) as exc:
                logger.warning("Public attribution dual-write failed: %s", exc)
                try:
                    from src.dashboard.generator import _attach_dual_write_provenance

                    payload = _attach_dual_write_provenance(
                        payload,
                        private_path=path,
                        public_path=latest,
                        dual_write_attempted=True,
                        dual_write_ok=False,
                        paths_identical=False,
                        note=str(exc),
                    )
                    save_results_json(payload, output_path=str(path))
                except Exception:  # noqa: BLE001
                    pass
        return path

    def reconcile_public_history(
        self,
        *,
        public_root: Path | None = None,
    ) -> list[Path]:
        """Reconcile dated public attribution shards from private SSOT.

        Attribution reports are dual-written, but older deployments could
        leave same-date private/public shards produced by different runs.
        ``provenance_completeness`` describes the attempted write; it cannot
        repair a later split-brain overwrite.  For every dated shard present
        in both trees, private ``DATA_DIR`` remains authoritative and only a
        business-value mismatch triggers a plane-aware rewrite.  Private
        diagnostic paths remain private; public paths are projected by the
        shared serializer.
        """
        from src.dashboard.generator import (
            _attach_dual_write_provenance,
            finalize_dual_write_provenance_after_sync,
        )
        from src.dashboard.public_projection import public_business_values_equal

        public_attribution_dir = (
            Path(public_root) if public_root is not None else Path(PUBLIC_DATA_DIR)
        ) / "attribution"
        if not public_attribution_dir.is_dir():
            return []

        reconciled: list[Path] = []
        for private_path in sorted(self.attribution_dir.glob("attribution_*.json")):
            public_path = public_attribution_dir / private_path.name
            if not public_path.is_file():
                continue
            try:
                private_payload = json.loads(private_path.read_text(encoding="utf-8"))
                public_payload = json.loads(public_path.read_text(encoding="utf-8"))
            except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise RuntimeError(
                    f"cannot audit attribution dual-write {private_path.name}: {exc}"
                ) from exc
            if public_business_values_equal(private_payload, public_payload):
                continue

            logger.warning(
                "Reconciling attribution business drift from private SSOT: %s",
                private_path.name,
            )
            stamped = _attach_dual_write_provenance(
                private_payload,
                private_path=private_path,
                public_path=public_path,
                dual_write_attempted=True,
                dual_write_ok=True,
                paths_identical=False,
                note="historical attribution SSOT reconciliation",
            )
            finalize_dual_write_provenance_after_sync(
                stamped,
                private_path=private_path,
                public_path=public_path,
                dual_write_ok=True,
                note="historical attribution SSOT reconciliation",
            )

            reconciled_private = json.loads(private_path.read_text(encoding="utf-8"))
            reconciled_public = json.loads(public_path.read_text(encoding="utf-8"))
            if not public_business_values_equal(reconciled_private, reconciled_public):
                raise RuntimeError(
                    "attribution business equivalence failed after reconciliation: "
                    f"{private_path.name}"
                )
            reconciled.append(public_path)

        return reconciled

    def load_latest_report(self) -> Optional[AttributionReport]:
        """Load most recent attribution report."""
        files = sorted(self.attribution_dir.glob("attribution_*.json"), reverse=True)
        if not files:
            return None
        try:
            with open(files[0]) as f:
                data = json.load(f)
            sources = {}
            for src_key, src_data in data.get("sources", {}).items():
                sources[src_key] = SourceAttribution(**src_data)
            data["sources"] = sources
            # Provenance stamps (Batch AY+) are operator metadata, not dataclass fields
            for meta_key in (
                "generator_git_sha",
                "generator_git_sha_status",
                "last_full_generator_git_sha",
                "provenance_completeness",
            ):
                data.pop(meta_key, None)
            return AttributionReport(**data)
        except (OSError, json.JSONDecodeError, KeyError, ValueError, TypeError) as e:
            logger.error("Error loading report: %s", e)
            return None


def _fmt_rate(value: Optional[float], *, width: int = 8) -> str:
    """Format optional hit/win rate for console; None → n/a (no_data sources)."""
    if value is None:
        return f"{'n/a':>{width}}"
    try:
        return f"{float(value):>{width}.1%}"
    except (TypeError, ValueError):
        return f"{'n/a':>{width}}"


def print_report(report: AttributionReport):
    """Pretty-print attribution report to console."""
    logger.info("\n" + "=" * 72)
    logger.info("  PERFORMANCE ATTRIBUTION REPORT")
    logger.info("=" * 72)
    logger.info(f"  Period: {report.start_date} → {report.end_date} ({report.analysis_days} days)")
    logger.info(f"  Generated: {report.timestamp}")
    logger.info(f"  Sources tracked: {report.total_sources_tracked}")
    logger.info("")

    # Sort sources by sharpe contribution
    sorted_sources = sorted(
        report.sources.values(),
        key=lambda s: s.sharpe_contribution if s.sharpe_contribution is not None else float("-inf"),
        reverse=True,
    )

    logger.info(f"  {'Source':30} {'HitRate':>9} {'WinRate':>9} {'AvgRet':>9} {'Sharpe':>9} {'Corr':>7} {'Active':>7}")
    logger.info("  " + "-" * 80)
    for s in sorted_sources:
        avg_ret = 0.0 if s.avg_return_bps is None else float(s.avg_return_bps)
        sharpe = 0.0 if s.sharpe_contribution is None else float(s.sharpe_contribution)
        corr = 0.0 if s.avg_correlation is None else float(s.avg_correlation)
        active = 0 if s.active_days is None else int(s.active_days)
        logger.info(
            f"  {s.display_name:30}"
            f" {_fmt_rate(s.hit_rate)}"
            f" {_fmt_rate(s.win_rate)}"
            f" {avg_ret:>8.2f}"
            f" {sharpe:>8.2f}"
            f" {corr:>6.2f}"
            f" {active:>6d}"
        )
    logger.info("")

    if report.degradation_signals:
        logger.info(f"  ⚠ DEGRADATION SIGNALS (negative/weak contribution):")
        for sig in report.degradation_signals:
            src = report.sources.get(sig)
            if src:
                logger.info(f"     {src.display_name:30} sharpe={src.sharpe_contribution:+.2f} hit={'n/a' if src.hit_rate is None else f'{src.hit_rate:.1%}'}")
        logger.info("")

    if report.top_performers:
        logger.info(f"  ★ TOP PERFORMERS:")
        for sig in report.top_performers:
            src = report.sources.get(sig)
            if src:
                logger.info(f"     {src.display_name:30} sharpe={src.sharpe_contribution:+.2f} hit={'n/a' if src.hit_rate is None else f'{src.hit_rate:.1%}'}")
        logger.info("")

    if report.best_source and report.best_source in report.sources:
        best = report.sources[report.best_source]
        logger.info(f"  Best source:  {best.display_name} (Sharpe {best.sharpe_contribution:+.2f})")

    if report.worst_source and report.worst_source in report.sources:
        worst = report.sources[report.worst_source]
        logger.info(f"  Worst source: {worst.display_name} (Sharpe {worst.sharpe_contribution:+.2f})")

    average_hit_rate = (
        "n/a" if report.avg_hit_rate is None else f"{report.avg_hit_rate:.1%}"
    )
    logger.info(
        "\n  Average hit rate: %s (status=%s)",
        average_hit_rate,
        getattr(report, "status", "ok"),
    )
    logger.info(f"  Average signal correlation: {report.avg_correlation:.2f}")
    logger.info("=" * 72)
    logger.info("")


def patch_save_vote():
    """Patch EnsembleVoter._save_vote to also log source readings.

    Call this once at startup to instrument the ensemble voter.
    """
    import src.strategy.ensemble_voter as ev
    original_save = ev.EnsembleVoter._save_vote

    def patched_save(self, vote):
        # Call original
        original_save(self, vote)
        # Also save source readings
        try:
            with sqlite_connect(self.db_path) as conn:
                for reading in vote.source_votes:
                    conn.execute("""
                        INSERT INTO source_readings
                        (timestamp, source, value, confidence, weight, regime_fit, explanation)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                    """, (
                        vote.timestamp,
                        reading.source.value if hasattr(reading.source, 'value') else str(reading.source),
                        float(reading.value),
                        float(reading.confidence),
                        float(reading.weight),
                        reading.regime_fit,
                        reading.explanation[:500] if reading.explanation else "",
                    ))
        except (KeyError, ValueError, TypeError, AttributeError, RuntimeError, sqlite3.Error) as e:
            logger.warning("Failed to save source readings: %s", e)

    ev.EnsembleVoter._save_vote = patched_save
    logger.info("Patched EnsembleVoter._save_vote to log source readings")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Performance Attribution System")
    subparsers = parser.add_subparsers(dest="command")

    report_parser = subparsers.add_parser("report", help="Generate attribution report")
    report_parser.add_argument("--days", type=int, default=90, help="Analysis window (days)")
    report_parser.add_argument("--save", action="store_true", help="Save report to disk")

    dashboard_parser = subparsers.add_parser("dashboard", help="Show latest saved report")

    patch_parser = subparsers.add_parser("patch", help="Patch EnsembleVoter to log source readings")

    args = parser.parse_args()
    attributor = PerformanceAttribution()

    if args.command == "report":
        report = attributor.generate_report(days=args.days)
        # Persist before console print so a display bug never blocks artifact SSOT
        # (cron stamps error if print_report raises after a good compute).
        if args.save:
            attributor.save_report(report)
        print_report(report)
        return 0

    elif args.command == "dashboard":
        report = attributor.load_latest_report()
        if report:
            print_report(report)
        else:
            logger.warning("No saved reports found. Run 'report' first.")
        return 0

    elif args.command == "patch":
        patch_save_vote()
        logger.info("EnsembleVoter patched. Run ensemble_voter vote to populate source_readings.")
        return 0

    else:
        parser.print_help()
        return 0


if __name__ == "__main__":
    from src.utils.log_config import configure_logging
    configure_logging()
    raise SystemExit(main() or 0)
