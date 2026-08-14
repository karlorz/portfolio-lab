"""EnsembleVoter bandit mixin (Item 5 s3 ENSEMBLE-VOTER-MIXINS).
Methods extracted verbatim from src/strategy/ensemble_voter.py.
"""

import logging
from datetime import datetime
from datetime import timezone
from pathlib import Path
from src.paths import ATTRIBUTION_DIR
from src.paths import DATA_DIR
from src.signals.regime_spec import Regime
from src.signals.signal_source import SignalSource
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple
import json
logger = logging.getLogger("src.strategy.ensemble_voter")

class BanditMixin:
    def get_rebalance_config(self) -> Dict[str, Any]:
        """
        Return current regime and rebalancing parameters for the
        SmartRebalanceGate/Controller to use regime-adaptive thresholds.

        Returns:
            Dict with 'regime' key (e.g. 'normal', 'crisis', 'high_vol',
            'low_vol', 'recovery') for the rebalancing controller.
        """
        regime_map = {
            Regime.LOW_VOL: 'low_vol',
            Regime.NORMAL: 'normal',
            Regime.HIGH_VOL: 'high_vol',
            Regime.CRISIS: 'crisis',
            Regime.RECOVERY: 'recovery',
        }
        return {
            'regime': regime_map.get(self.current_regime, 'normal'),
            'regime_confidence': self.current_regime_confidence,
        }

    def update_bandit(self, signal_value: str, regime_name: str, daily_return: float):
        """Update bandit with observed return for a signal in a regime."""
        self.bandit.update(signal_value, regime_name, daily_return)
        self.bandit_observations += 1

    def _load_bandit_state(self) -> bool:
        """Load bandit history + observation count from data_path if present."""
        path = getattr(self, "bandit_state_path", None) or (
            self.data_path / "ensemble_bandit_state.json"
        )
        if not path.exists():
            return False
        try:
            with open(path, "r", encoding="utf-8") as f:
                state = json.load(f)
            if not isinstance(state, dict):
                return False
            bandit_state = state.get("bandit") or state
            if hasattr(self.bandit, "load_state"):
                self.bandit.load_state(bandit_state)
            obs = state.get("observations")
            if obs is None:
                # Derive from history length if missing
                hist = getattr(self.bandit, "_history", {}) or {}
                obs = sum(
                    len(returns)
                    for signals in hist.values()
                    if isinstance(signals, dict)
                    for returns in signals.values()
                    if isinstance(returns, list)
                )
            self.bandit_observations = int(obs or 0)
            days = state.get("reward_days", state.get("bandit_days", state.get("days")))
            if days is None:
                if self.bandit_observations > 0:
                    n_sources = max(1, len(list(SignalSource)))
                    days = max(1, self.bandit_observations // n_sources)
                else:
                    days = 0
            self.bandit_days = int(days or 0)
            logger.info(
                "Loaded ensemble bandit state from %s (observations=%s, reward_days=%s)",
                path,
                self.bandit_observations,
                self.bandit_days,
            )
            return True
        except (OSError, json.JSONDecodeError, TypeError, ValueError) as exc:
            logger.warning("Failed to load ensemble bandit state: %s", exc)
            return False

    def save_bandit_state(self) -> bool:
        """Persist bandit history + observation count atomically."""
        path = getattr(self, "bandit_state_path", None) or (
            self.data_path / "ensemble_bandit_state.json"
        )
        payload = {
            "schema_version": "ensemble-bandit-state/v1",
            "observations": int(self.bandit_observations),
            "reward_days": int(getattr(self, "bandit_days", 0) or 0),
            "bandit_days": int(getattr(self, "bandit_days", 0) or 0),
            "bandit": self.bandit.get_state() if hasattr(self.bandit, "get_state") else {},
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp_path = path.with_suffix(path.suffix + ".tmp")
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2)
                f.write("\n")
            tmp_path.replace(path)
            return True
        except (OSError, TypeError, ValueError) as exc:
            logger.warning("Failed to save ensemble bandit state: %s", exc)
            return False

    @staticmethod
    def contribution_reward_decimal(
        daily_return: float,
        *,
        value: float,
        weight: float,
    ) -> float:
        """Map one signal reading + portfolio daily return → arm reward (decimal).

        Batch BR: same credit formula as ``PerformanceAttribution._compute_source_attribution``
        (directional: ``ret * |value|``; neutral ``|value|<=0.05``: ``ret * weight * 2``),
        returned in decimal return units (not bps) for bandit updates.
        """
        ret = float(daily_return)
        val = float(value)
        w = float(weight)
        if abs(val) > 0.05:
            return ret * abs(val)
        return ret * w * 2.0

    @staticmethod
    def compute_daily_contribution_rewards(
        signals: List[Dict[str, Any]],
        daily_return: float,
        *,
        min_spread: float = 1e-12,
    ) -> Optional[Dict[str, float]]:
        """Build identifying per-source rewards for one calendar day.

        Uses the latest reading per source in ``signals``. Returns None when
        fewer than two sources or zero reward spread (non-identification).
        """
        from src.strategy.ensemble_voter import EnsembleVoter
        try:
            ret = float(daily_return)
        except (TypeError, ValueError):
            return None
        by_source: Dict[str, Dict[str, Any]] = {}
        for sig in signals:
            if not isinstance(sig, dict):
                continue
            name = sig.get("source")
            if name is None:
                continue
            src = str(name)
            # Last write wins (callers should pass chronological order)
            by_source[src] = sig
        if len(by_source) < 2:
            return None
        rewards: Dict[str, float] = {}
        for src, sig in by_source.items():
            try:
                value = float(sig.get("value", 0.0) or 0.0)
            except (TypeError, ValueError):
                continue
            try:
                weight = float(sig.get("weight", 0.0) or 0.0)
            except (TypeError, ValueError):
                weight = 0.0
            rewards[src] = EnsembleVoter.contribution_reward_decimal(
                ret, value=value, weight=weight
            )
        if len(rewards) < 2:
            return None
        vals = list(rewards.values())
        if max(vals) - min(vals) < min_spread:
            return None
        return rewards

    @staticmethod
    def load_daily_contribution_source_rewards(
        data_dir: Optional[Path] = None,
        *,
        lookback_days: int = 14,
    ) -> Optional[Tuple[Dict[str, float], Dict[str, Any]]]:
        """Load per-source rewards from *one* recent day of signal × PnL credit.

        Batch BR (B1): prefers true daily contribution over windowed
        ``avg_return_bps`` (Batch BQ). Joins ``source_readings`` (latest per
        source/day) with paper daily returns; walks newest dates first until
        an identifying multi-arm map is found.

        Returns ``(rewards, meta)`` or None. Meta includes ``as_of_date``,
        ``reward_mode``, ``live_authoritative: false``. Hermetic when
        ``data_dir`` is an explicit tmp path (no live DATA_DIR leak).
        """
        from src.strategy.ensemble_voter import EnsembleVoter
        root = Path(data_dir) if data_dir is not None else Path(DATA_DIR)
        lookback = max(int(lookback_days), 2)

        # Lazy import avoids pulling attribution/numpy heavy paths at module load
        try:
            from src.monitor.performance_attribution import PerformanceAttribution
        except ImportError:
            logger.debug("PerformanceAttribution unavailable for daily contribution rewards")
            return None

        try:
            pa = PerformanceAttribution(data_dir=root)
            history = pa._get_signal_history(days=lookback)
            daily_returns = pa._get_paper_trading_returns(days=lookback)
        except (OSError, TypeError, ValueError, AttributeError, RuntimeError) as exc:
            logger.debug("Daily contribution load failed: %s", exc)
            return None

        if not history or not daily_returns:
            return None

        # Group source readings by calendar date (latest timestamp per source/day)
        by_day: Dict[str, Dict[str, Dict[str, Any]]] = {}
        for row in history:
            if not isinstance(row, dict) or row.get("type") == "ensemble_vote":
                continue
            ts = row.get("timestamp")
            src = row.get("source")
            if not ts or not src:
                continue
            day = str(ts)[:10]
            if len(day) < 10:
                continue
            bucket = by_day.setdefault(day, {})
            # history is DESC by timestamp; first seen wins as latest
            if str(src) not in bucket:
                bucket[str(src)] = row

        # Newest return dates first
        for day in sorted(daily_returns.keys(), reverse=True):
            ret_entry = daily_returns.get(day) or {}
            try:
                ret = float(ret_entry.get("daily_return"))
            except (TypeError, ValueError):
                continue
            day_sources = by_day.get(day)
            if not day_sources or len(day_sources) < 2:
                continue
            signals = list(day_sources.values())
            rewards = EnsembleVoter.compute_daily_contribution_rewards(
                signals, daily_return=ret
            )
            if rewards is None:
                continue
            meta: Dict[str, Any] = {
                "reward_mode": "daily_contribution_source_rewards",
                "as_of_date": day,
                "arms": len(rewards),
                "reward_spread": max(rewards.values()) - min(rewards.values()),
                "live_authoritative": False,
                "portfolio_daily_return": ret,
            }
            logger.debug(
                "Loaded daily contribution rewards for %s (%d arms, spread=%.6f)",
                day,
                len(rewards),
                meta["reward_spread"],
            )
            return rewards, meta
        return None

    @staticmethod
    def load_preferred_source_rewards(
        data_dir: Optional[Path] = None,
    ) -> Tuple[Optional[Dict[str, float]], str]:
        """Prefer daily contribution rewards; fall back to windowed attribution.

        Batch BR: ``(rewards, reward_mode)``. Mode is one of
        ``daily_contribution_source_rewards``, ``attribution_source_rewards``,
        or ``none``.
        """
        from src.strategy.ensemble_voter import EnsembleVoter
        daily = EnsembleVoter.load_daily_contribution_source_rewards(data_dir)
        if daily is not None:
            rewards, meta = daily
            return rewards, str(meta.get("reward_mode") or "daily_contribution_source_rewards")
        windowed = EnsembleVoter.load_attribution_source_rewards(data_dir)
        if windowed is not None:
            return windowed, "attribution_source_rewards"
        return None, "none"

    @staticmethod
    def load_attribution_source_rewards(
        data_dir: Optional[Path] = None,
        *,
        max_age_days: Optional[float] = None,
    ) -> Optional[Dict[str, float]]:
        """Load per-source pseudo-rewards from performance attribution.

        Batch BQ: maps ``avg_return_bps / 1e4`` into decimal return units so
        multi-arm bandit updates can differentiate signals. Windowed attribution
        is a *proxy* for true daily credit assignment (linear/contextual bandit
        ideal); still identifying vs identical portfolio PnL broadcast.

        Batch BR prefers :meth:`load_daily_contribution_source_rewards` via
        :meth:`load_preferred_source_rewards` when a single-day join is available.

        Preference order (when ``data_dir`` is None → default DATA_DIR):
          1. ``{data_dir}/attribution/latest.json``
          2. Newest ``{data_dir}/attribution/attribution_*.json``
          3. Global ``ATTRIBUTION_DIR`` only when ``data_dir`` is default DATA_DIR
             (never leaks live attribution into hermetic tests that pass tmp paths)

        Returns None when missing/empty/unparseable. Never invents zeros for
        unknown sources — callers apply only keys present.
        """
        explicit_dir = data_dir is not None
        root = Path(data_dir) if explicit_dir else Path(DATA_DIR)
        attr_dir = root / "attribution"
        candidates: List[Path] = []
        latest = attr_dir / "latest.json"
        if latest.exists():
            candidates.append(latest)
        if attr_dir.exists():
            dated = sorted(attr_dir.glob("attribution_*.json"), reverse=True)
            for p in dated:
                if p not in candidates:
                    candidates.append(p)
        # Live default only: also search ATTRIBUTION_DIR when distinct
        if not explicit_dir:
            try:
                global_attr = Path(ATTRIBUTION_DIR)
                if global_attr.exists() and global_attr.resolve() != attr_dir.resolve():
                    g_latest = global_attr / "latest.json"
                    if g_latest.exists() and g_latest not in candidates:
                        candidates.insert(0, g_latest)
                    for p in sorted(global_attr.glob("attribution_*.json"), reverse=True)[:3]:
                        if p not in candidates:
                            candidates.append(p)
            except (OSError, TypeError, ValueError):
                pass

        for path in candidates:
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError, TypeError, ValueError):
                continue
            if not isinstance(payload, dict):
                continue
            sources_block = payload.get("sources")
            if not isinstance(sources_block, dict) or not sources_block:
                continue
            rewards: Dict[str, float] = {}
            for name, meta in sources_block.items():
                if not isinstance(meta, dict):
                    continue
                bps = meta.get("avg_return_bps")
                if bps is None:
                    continue
                try:
                    rewards[str(name)] = float(bps) / 10000.0
                except (TypeError, ValueError):
                    continue
            if len(rewards) >= 2:
                # Distinct values required for identification
                vals = list(rewards.values())
                spread = max(vals) - min(vals)
                if spread < 1e-12:
                    logger.info(
                        "Attribution rewards non-identifying (zero spread) from %s",
                        path,
                    )
                    continue
                logger.debug(
                    "Loaded %d attribution source rewards from %s (spread=%.6f)",
                    len(rewards),
                    path,
                    spread,
                )
                return rewards
        return None

    @staticmethod
    def _soft_delete_source_names(regime_name: str) -> set:
        """String names of REGIME_WEIGHTS soft-delete arms for a regime."""
        from src.strategy.ensemble_voter import EnsembleVoter
        zeros = EnsembleVoter._static_zero_baseline_sources(regime_name)
        names: set = set()
        for src in zeros:
            names.add(src.value if hasattr(src, "value") else str(src))
        return names

    def apply_daily_bandit_rewards(
        self,
        daily_return: float,
        regime_name: Optional[str] = None,
        sources: Optional[List[str]] = None,
        *,
        persist: bool = True,
        noise_floor: Optional[float] = None,
        source_rewards: Optional[Dict[str, float]] = None,
        reward_mode: Optional[str] = None,
        include_soft_delete_arms: bool = False,
    ) -> Dict[str, Any]:
        """Apply one day of portfolio return as reward to ensemble bandit sources.

        Production training step: maps paper/portfolio daily return into
        ``update_bandit`` for each active signal source so observations leave
        cold_start. Bandit remains advisory (not live target_allocations).

        Near-zero rewards (|r| < noise_floor, default
        ``ENSEMBLE_BANDIT_REWARD_NOISE_FLOOR`` / 1e-6) are skipped entirely —
        no arm history append, no observation increment, no reward_days step —
        so flat paper NAV / floating-point dust cannot ramp blend.

        Batch BO: multi-arm *identical* portfolio reward broadcast is skipped
        (non-identification). Batch BQ: when ``source_rewards`` maps arms to
        *differentiated* per-source returns (e.g. attribution avg_return_bps),
        multi-arm updates proceed with per-arm credit assignment.
        Batch BR: prefer daily contribution ``source_rewards`` and pass
        ``reward_mode='daily_contribution_source_rewards'`` for honesty tags.

        Batch DL: static soft-delete arms (REGIME_WEIGHTS weight 0) are excluded
        from reward updates by default — sleeping/non-voting experts must not
        train the posterior (Thompson sampling / sleeping-experts hygiene).
        Pass ``include_soft_delete_arms=True`` for explicit shadow learning.

        Returns summary with updates count and observation total.
        """
        from src.strategy.ensemble_voter import BANDIT_REWARD_NOISE_FLOOR
        try:
            reward = float(daily_return)
        except (TypeError, ValueError):
            return {
                "updates": 0,
                "observations": int(self.bandit_observations),
                "reward_days": int(getattr(self, "bandit_days", 0) or 0),
                "skipped": True,
                "reason": "invalid_daily_return",
            }

        floor = (
            float(noise_floor)
            if noise_floor is not None
            else float(BANDIT_REWARD_NOISE_FLOOR)
        )
        if floor < 0:
            floor = 0.0

        if regime_name is None:
            current = getattr(self, "current_regime", Regime.NORMAL)
            regime_name = current.name if hasattr(current, "name") else str(current)
        regime_name = str(regime_name).upper()

        # Normalize optional per-arm rewards (Batch BQ)
        per_arm: Optional[Dict[str, float]] = None
        if source_rewards:
            cleaned: Dict[str, float] = {}
            for k, v in source_rewards.items():
                try:
                    cleaned[str(k)] = float(v)
                except (TypeError, ValueError):
                    continue
            if cleaned:
                per_arm = cleaned

        if sources is None:
            if per_arm is not None:
                sources = list(per_arm.keys())
            else:
                sources = [s.value for s in SignalSource]
        sources = [str(s) for s in sources]

        # Batch DL: drop soft-delete / non-voting arms from reward training
        soft_delete_excluded: List[str] = []
        if not include_soft_delete_arms:
            soft_names = self._soft_delete_source_names(regime_name)
            if soft_names:
                kept: List[str] = []
                for s in sources:
                    if s in soft_names:
                        soft_delete_excluded.append(s)
                    else:
                        kept.append(s)
                sources = kept
                if per_arm is not None:
                    per_arm = {
                        k: v for k, v in per_arm.items() if k not in soft_names
                    }
                    if not per_arm:
                        per_arm = None
                if not sources:
                    return {
                        "updates": 0,
                        "observations": int(self.bandit_observations),
                        "reward_days": int(getattr(self, "bandit_days", 0) or 0),
                        "days": int(getattr(self, "bandit_days", 0) or 0),
                        "bandit_days": int(getattr(self, "bandit_days", 0) or 0),
                        "regime": regime_name,
                        "daily_return": reward,
                        "noise_floor": floor,
                        "skipped": True,
                        "reason": "all_arms_soft_delete_or_empty",
                        "soft_delete_excluded": soft_delete_excluded,
                        "live_authoritative": False,
                    }

        # Multi-arm identical portfolio broadcast guard (Batch BO)
        if len(sources) > 1 and per_arm is None:
            # Still respect noise floor for the scalar portfolio return path
            if abs(reward) < floor:
                logger.info(
                    "Skipping bandit reward update: |daily_return|=%.3e < noise_floor=%.3e",
                    abs(reward),
                    floor,
                )
                return {
                    "updates": 0,
                    "observations": int(self.bandit_observations),
                    "reward_days": int(getattr(self, "bandit_days", 0) or 0),
                    "days": int(getattr(self, "bandit_days", 0) or 0),
                    "bandit_days": int(getattr(self, "bandit_days", 0) or 0),
                    "daily_return": reward,
                    "noise_floor": floor,
                    "skipped": True,
                    "reason": "reward_below_noise_floor",
                }
            logger.info(
                "Skipping bandit multi-arm identical reward broadcast: "
                "daily_return=%.6f across %d arms (non-identification guard; "
                "use per-source attribution rewards when available)",
                reward,
                len(sources),
            )
            return {
                "updates": 0,
                "observations": int(self.bandit_observations),
                "reward_days": int(getattr(self, "bandit_days", 0) or 0),
                "days": int(getattr(self, "bandit_days", 0) or 0),
                "bandit_days": int(getattr(self, "bandit_days", 0) or 0),
                "regime": regime_name,
                "daily_return": reward,
                "noise_floor": floor,
                "skipped": True,
                "reason": "identical_portfolio_reward_all_arms",
                "arms_considered": len(sources),
            }

        # Build (source, reward) pairs
        pairs: List[Tuple[str, float]] = []
        if per_arm is not None:
            for src in sources:
                if src not in per_arm:
                    continue
                r = float(per_arm[src])
                if abs(r) < floor:
                    continue
                pairs.append((src, r))
            if not pairs:
                return {
                    "updates": 0,
                    "observations": int(self.bandit_observations),
                    "reward_days": int(getattr(self, "bandit_days", 0) or 0),
                    "days": int(getattr(self, "bandit_days", 0) or 0),
                    "bandit_days": int(getattr(self, "bandit_days", 0) or 0),
                    "daily_return": reward,
                    "noise_floor": floor,
                    "skipped": True,
                    "reason": "attribution_rewards_below_noise_floor",
                    "arms_considered": len(sources),
                }
            # Non-identification if multi-arm but all remaining rewards equal
            if len(pairs) > 1:
                rs = [r for _, r in pairs]
                if max(rs) - min(rs) < 1e-12:
                    return {
                        "updates": 0,
                        "observations": int(self.bandit_observations),
                        "reward_days": int(getattr(self, "bandit_days", 0) or 0),
                        "days": int(getattr(self, "bandit_days", 0) or 0),
                        "bandit_days": int(getattr(self, "bandit_days", 0) or 0),
                        "regime": regime_name,
                        "daily_return": reward,
                        "noise_floor": floor,
                        "skipped": True,
                        "reason": "identical_attribution_rewards_all_arms",
                        "arms_considered": len(pairs),
                    }
        else:
            # Single-arm explicit path (or sources already len==1)
            if abs(reward) < floor:
                logger.info(
                    "Skipping bandit reward update: |daily_return|=%.3e < noise_floor=%.3e",
                    abs(reward),
                    floor,
                )
                return {
                    "updates": 0,
                    "observations": int(self.bandit_observations),
                    "reward_days": int(getattr(self, "bandit_days", 0) or 0),
                    "days": int(getattr(self, "bandit_days", 0) or 0),
                    "bandit_days": int(getattr(self, "bandit_days", 0) or 0),
                    "daily_return": reward,
                    "noise_floor": floor,
                    "skipped": True,
                    "reason": "reward_below_noise_floor",
                }
            pairs = [(src, reward) for src in sources]

        updates = 0
        applied: Dict[str, float] = {}
        for src, r in pairs:
            self.update_bandit(str(src), regime_name, r)
            updates += 1
            applied[src] = r

        # One calendar reward day per apply, independent of arm count
        self.bandit_days = int(getattr(self, "bandit_days", 0) or 0) + 1

        if persist:
            self.save_bandit_state()

        summary: Dict[str, Any] = {
            "updates": updates,
            "observations": int(self.bandit_observations),
            "days": int(self.bandit_days),
            "bandit_days": int(self.bandit_days),
            "regime": regime_name,
            "daily_return": reward,
            "noise_floor": floor,
            "skipped": False,
            "live_authoritative": False,
        }
        if soft_delete_excluded:
            summary["soft_delete_excluded"] = soft_delete_excluded
        if per_arm is not None:
            mode = (
                str(reward_mode)
                if reward_mode
                else "attribution_source_rewards"
            )
            summary["reward_mode"] = mode
            summary["arms_updated"] = list(applied.keys())
            summary["reward_spread"] = (
                max(applied.values()) - min(applied.values()) if applied else 0.0
            )
        else:
            summary["reward_mode"] = (
                str(reward_mode) if reward_mode else "single_arm_or_scalar"
            )
            if applied:
                summary["arms_updated"] = list(applied.keys())
        return summary

    @staticmethod
    def load_latest_daily_return_from_performance(
        performance_path: Optional[Path] = None,
        *,
        max_lines: int = 200,
        prefer_daily_pnl: bool = True,
        data_dir: Optional[Path] = None,
    ) -> Optional[float]:
        """Read newest non-null daily_return for bandit training.

        Prefer ``daily_pnl_latest.json`` (capture_daily_pnl SSOT) when present
        and |return| is finite — avoids replaying flat-NAV micro-noise rows
        that historically polluted performance.jsonl. Falls back to
        performance.jsonl tail.
        """
        root = Path(data_dir) if data_dir is not None else Path(DATA_DIR)
        if prefer_daily_pnl:
            pnl_path = root / "daily_pnl_latest.json"
            if pnl_path.exists():
                try:
                    payload = json.loads(pnl_path.read_text(encoding="utf-8"))
                    if isinstance(payload, dict) and "daily_return" in payload:
                        return float(payload["daily_return"])
                except (OSError, json.JSONDecodeError, TypeError, ValueError) as exc:
                    logger.debug("daily_pnl_latest read failed: %s", exc)

        path = Path(performance_path) if performance_path is not None else root / "performance.jsonl"
        if not path.exists():
            return None
        try:
            # Efficient-ish tail for moderate files
            with open(path, "rb") as f:
                f.seek(0, 2)
                size = f.tell()
                block = min(size, 64 * 1024)
                f.seek(max(0, size - block))
                chunk = f.read().decode("utf-8", errors="replace")
            lines = [ln for ln in chunk.splitlines() if ln.strip()][-max_lines:]
            for line in reversed(lines):
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not isinstance(row, dict):
                    continue
                if "daily_return" not in row:
                    continue
                try:
                    return float(row["daily_return"])
                except (TypeError, ValueError):
                    continue
            return None
        except OSError as exc:
            logger.debug("performance.jsonl read failed: %s", exc)
            return None

    def apply_goal_risk_budget(self, base_allocation: dict) -> dict:
        """Scale allocation weights based on investment goals from goals.json.

        Reads goals.json via src.config.goals, computes risk budget multiplier,
        and shifts allocation toward safer assets proportionally.
        """
        try:
            from src.config.goals import load_goals, get_risk_budget_multiplier
            goals = load_goals()
            risk_mult = get_risk_budget_multiplier(goals)
        except (ImportError, OSError, KeyError, ValueError, TypeError, json.JSONDecodeError) as e:
            logger.warning("Failed to load goals for risk budget, using risk_mult=1.0: %s", e)
            risk_mult = 1.0

        if risk_mult >= 1.0:
            return base_allocation

        safe_assets = {"SHY", "IEF", "BIL", "TLT"}
        total = sum(base_allocation.values()) if base_allocation else 1.0
        if total == 0:
            return base_allocation

        shifted = {}
        risky_reduction = 0.0
        for asset, weight in base_allocation.items():
            if asset in safe_assets:
                shifted[asset] = weight
            else:
                reduced = weight * risk_mult
                shifted[asset] = reduced
                risky_reduction += weight - reduced

        # Redistribute reduced risk to safe assets proportionally
        safe_total = sum(shifted.get(a, 0) for a in safe_assets if a in shifted)
        if safe_total > 0 and risky_reduction > 0:
            for asset in safe_assets:
                if asset in shifted:
                    shifted[asset] += risky_reduction * (shifted[asset] / safe_total)

        # Renormalize
        new_total = sum(shifted.values())
        if new_total == 0:
            return base_allocation
        return {k: v / new_total * total for k, v in shifted.items()}
