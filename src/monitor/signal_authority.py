"""Live authority guards for signals.json multi-destination writers.

Hard rule: only ``signals.json.target_allocations`` is order-routing authority.
Partial and full producers must never leave a hollow twin (missing TA) on any
destination. Fan-out uses one serialized body → public + private + optional
repo soft-mirror (same bytes).
"""

from __future__ import annotations

import json
import logging
import math
import os
import tempfile
from dataclasses import dataclass
from numbers import Real
from pathlib import Path
from typing import Any, Mapping, MutableMapping, Optional, Sequence

from src.paths import BASE_ALLOCATION

logger = logging.getLogger(__name__)

AUTHORITY_REQUIRED_KEYS: tuple[str, ...] = ("target_allocations",)

# Champion sleeve symbols (see src.paths.BASE_ALLOCATION). Non-champion keys
# may appear under crisis/vol overrides; require at least these three weights.
_CHAMPION_SYMBOLS: tuple[str, ...] = ("SPY", "GLD", "TLT")
CHAMPION_TARGET_ALLOCATIONS: dict[str, float] = dict(BASE_ALLOCATION)
_SUM_TOLERANCE = 1e-2  # allow mild float drift after renorm


class AuthorityValidationError(ValueError):
    """Raised when a signals payload lacks live-routing authority fields."""


def validate_authority_payload(
    payload: Mapping[str, Any],
    *,
    require_champion_symbols: bool = True,
) -> None:
    """Require ``target_allocations`` suitable for order_router.

    - Must be a non-empty dict of symbol → weight
    - Weights finite, in (0, 1]
    - Sum ≈ 1 within tolerance
    - Default: SPY/GLD/TLT all present (champion or regime override sleeves)
    """
    if not isinstance(payload, Mapping):
        raise AuthorityValidationError("signals payload must be a mapping")
    ta = payload.get("target_allocations")
    if not isinstance(ta, Mapping) or not ta:
        raise AuthorityValidationError(
            "signals payload missing non-empty target_allocations (live authority)"
        )
    total = 0.0
    for sym, weight in ta.items():
        if not isinstance(sym, str) or not sym:
            raise AuthorityValidationError(f"invalid allocation symbol: {sym!r}")
        try:
            w = float(weight)
        except (TypeError, ValueError) as exc:
            raise AuthorityValidationError(
                f"target_allocations[{sym!r}] not numeric: {weight!r}"
            ) from exc
        if not (0.0 < w <= 1.0):
            raise AuthorityValidationError(
                f"target_allocations[{sym!r}]={w} outside (0, 1]"
            )
        total += w
    if abs(total - 1.0) > _SUM_TOLERANCE:
        raise AuthorityValidationError(
            f"target_allocations sum={total:.6f} not within {_SUM_TOLERANCE} of 1.0"
        )
    if require_champion_symbols:
        missing = [s for s in _CHAMPION_SYMBOLS if s not in ta]
        if missing:
            raise AuthorityValidationError(
                f"target_allocations missing required symbols: {missing}"
            )


def is_champion_target_allocations(payload: Mapping[str, Any]) -> bool:
    """Return True only for the currently approved champion live allocation."""
    if not isinstance(payload, Mapping):
        return False
    ta = payload.get("target_allocations")
    if not isinstance(ta, Mapping):
        return False
    if set(ta.keys()) != set(CHAMPION_TARGET_ALLOCATIONS.keys()):
        return False
    for sym, expected in CHAMPION_TARGET_ALLOCATIONS.items():
        try:
            observed = float(ta[sym])
        except (TypeError, ValueError, KeyError):
            return False
        if not math.isfinite(observed) or abs(observed - expected) > 1e-9:
            return False
    return True


def merge_signals_patch(
    base: MutableMapping[str, Any],
    patch: Mapping[str, Any],
    *,
    allowed_top_keys: Optional[Sequence[str]] = None,
) -> dict[str, Any]:
    """Shallow-merge patch into a copy of base; never drop authority keys.

    When ``allowed_top_keys`` is set, only those keys from ``patch`` are applied
    (plus nested merge for ``health`` if both sides have dict health).
    """
    if not isinstance(base, Mapping):
        raise AuthorityValidationError("merge base must be a mapping")
    out: dict[str, Any] = dict(base)
    if not isinstance(patch, Mapping):
        return out
    keys = (
        list(allowed_top_keys)
        if allowed_top_keys is not None
        else [k for k in patch.keys() if k != "target_allocations"]
    )
    for key in keys:
        if key not in patch:
            continue
        if key == "target_allocations":
            # Never overwrite authority from a partial patch unless explicitly
            # validated and complete — partials must not wipe TA.
            continue
        val = patch[key]
        if (
            key == "health"
            and isinstance(val, Mapping)
            and isinstance(out.get("health"), Mapping)
        ):
            merged_h = dict(out["health"])
            merged_h.update(dict(val))
            out["health"] = merged_h
        else:
            out[key] = val
    # Preserve TA from base always
    if "target_allocations" in base:
        out["target_allocations"] = base["target_allocations"]
    validate_authority_payload(out)
    return out


def default_repo_signals_path() -> Path:
    """Repo checkout soft-mirror path for signals.json (not live PUBLIC_DATA_DIR)."""
    return default_repo_public_data_path("signals.json")


def default_repo_public_data_path(filename: str) -> Path:
    """Repo checkout soft-mirror path under public/data (not live PUBLIC_DATA_DIR)."""
    name = str(filename or "").lstrip("/")
    try:
        from src.paths import PROJECT_ROOT

        return Path(PROJECT_ROOT) / "public" / "data" / name
    except Exception:  # noqa: BLE001
        return Path("public/data") / name


def is_ephemeral_write_path(path: Path | str | None) -> bool:
    """True when *path* lives under pytest / plab isolation temp trees."""
    if path is None:
        return False
    text = str(path).replace("\\", "/")
    if not text:
        return False
    if "plab-pytest" in text:
        return True
    if "/pytest-of-" in text:
        return True
    if "/tmp/pytest-" in text or text.startswith("/tmp/pytest-"):
        return True
    if "/var/folders/" in text and "pytest" in text:
        return True
    return False


def is_production_ssot_path(path: Path | str | None) -> bool:
    """True when *path* resolves under live lab private/public data trees.

    Tests isolate PUBLIC_DATA_DIR to ``/tmp/plab-pytest-public.*`` but often
    leave ``DATA_DIR`` / repo ``public/data`` pointing at production. Multi-dest
    fan-out must not rewrite those SSOT paths while ``PYTEST_CURRENT_TEST`` is
    set (Batch HP private alerts/health pollution).
    """
    if path is None:
        return False
    if is_ephemeral_write_path(path):
        return False
    try:
        resolved = Path(path).expanduser().resolve()
    except OSError:
        resolved = Path(path).expanduser()
    text = str(resolved).replace("\\", "/")
    try:
        from src.paths import PROJECT_ROOT

        project = str(Path(PROJECT_ROOT).resolve()).replace("\\", "/")
    except Exception:  # noqa: BLE001
        project = "/root/projects/portfolio-lab"
    production_prefixes = (
        f"{project}/data",
        f"{project}/public/data",
        "/var/www/portfolio-lab/data",
    )
    return any(
        text == prefix or text.startswith(prefix + "/")
        for prefix in production_prefixes
    )


def _should_skip_production_ssot_write(path: Path | str | None) -> bool:
    """Skip production SSOT writes under pytest unless explicitly allowed."""
    if path is None:
        return False
    if os.environ.get("PORTFOLIO_LAB_ALLOW_LIVE_PUBLIC", "0") == "1":
        return False
    if not os.environ.get("PYTEST_CURRENT_TEST"):
        return False
    return is_production_ssot_path(path)


def _atomic_write_text(path: Path, text: str, *, mode: int = 0o644) -> None:
    """Atomic replace with explicit mode (mkstemp defaults to 0o600).

    Public dashboard JSON under Caddy must be world-readable (0o644). Without
    fchmod/chmod, multi-dest fan-out leaves sticky 0600 → HTTPS 403 while SPA
    still 200 (Batch HJ/HK).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=str(path.parent),
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(text)
            fh.flush()
            os.fsync(fh.fileno())
        # mkstemp → 0o600; set readable mode before replace so dest inherits it.
        try:
            os.chmod(tmp_name, mode)
        except OSError as exc:
            logger.warning("chmod %s → %s failed: %s", tmp_name, oct(mode), exc)
        os.replace(tmp_name, path)
    except Exception:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise


@dataclass
class MultiDestWriteResult:
    wrote_public: bool = False
    wrote_private: bool = False
    wrote_repo: bool = False
    public_path: Optional[str] = None
    private_path: Optional[str] = None
    repo_path: Optional[str] = None
    skipped_reason: Optional[str] = None


def serialize_signals_payload(payload: Mapping[str, Any]) -> str:
    """Canonical, browser-parseable JSON body used for multi-dest equality."""
    try:
        from src.backtest.metrics import _json_serializer as _default
    except Exception:  # noqa: BLE001 — keep fan-out usable offline
        _default = str  # type: ignore[assignment]

    def strict_json_value(value: Any) -> Any:
        if isinstance(value, Mapping):
            return {key: strict_json_value(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [strict_json_value(item) for item in value]
        if isinstance(value, Real) and not isinstance(value, bool):
            return value if math.isfinite(float(value)) else None
        return value

    return json.dumps(
        strict_json_value(dict(payload)),
        indent=2,
        default=_default,
        allow_nan=False,
    ) + "\n"


def write_signals_multi_dest(
    payload: Mapping[str, Any],
    *,
    public_path: Path | str | None = None,
    private_path: Path | str | None = None,
    repo_path: Path | str | None = None,
    soft_mirror_repo: bool = True,
    validate: bool = True,
) -> MultiDestWriteResult:
    """Validate authority, serialize once, fan-out same bytes to dests.

    Raises ``AuthorityValidationError`` before any write when validation fails
    (callers that want soft-skip should catch). Existing good files are left
    unchanged on validation failure.
    """
    if validate:
        validate_authority_payload(payload)

    text = serialize_signals_payload(payload)
    result = MultiDestWriteResult()

    pub = Path(public_path) if public_path is not None else None
    priv = Path(private_path) if private_path is not None else None
    # Auto soft-mirror to checkout public/data only when caller did not pass an
    # explicit repo_path. Under pytest, skip auto soft-mirror so unit tests that
    # monkeypatch PUBLIC_DATA_DIR cannot clobber the real repo tree (Case F/G
    # still work by passing repo_path or patching default_repo_signals_path).
    auto_repo = soft_mirror_repo and repo_path is None
    if auto_repo and os.environ.get("PYTEST_CURRENT_TEST"):
        auto_repo = False
    repo = Path(repo_path) if repo_path is not None else (
        default_repo_signals_path() if auto_repo else None
    )

    if pub is not None:
        if _should_skip_production_ssot_write(pub):
            logger.warning(
                "signals multi-dest skipped production public under pytest: %s",
                pub,
            )
            result.skipped_reason = f"public:pytest-ssot-guard:{pub}"
        else:
            _atomic_write_text(pub, text)
            result.wrote_public = True
            result.public_path = str(pub)

    if priv is not None:
        try:
            if pub is None or priv.resolve() != pub.resolve():
                if _should_skip_production_ssot_write(priv):
                    logger.warning(
                        "signals multi-dest skipped production private under pytest: %s",
                        priv,
                    )
                    reason = f"private:pytest-ssot-guard:{priv}"
                    result.skipped_reason = (
                        f"{result.skipped_reason};{reason}"
                        if result.skipped_reason
                        else reason
                    )
                else:
                    _atomic_write_text(priv, text)
                    result.wrote_private = True
                    result.private_path = str(priv)
        except OSError as exc:
            logger.warning("private signals twin write failed: %s", exc)
            result.skipped_reason = f"private:{exc}"

    if soft_mirror_repo and repo is not None:
        try:
            if pub is not None and repo.resolve() == pub.resolve():
                result.wrote_repo = False
            elif priv is not None and repo.resolve() == priv.resolve():
                result.wrote_repo = False
            elif _should_skip_production_ssot_write(repo):
                logger.warning(
                    "signals multi-dest skipped production repo soft-mirror under pytest: %s",
                    repo,
                )
                reason = f"repo:pytest-ssot-guard:{repo}"
                result.skipped_reason = (
                    f"{result.skipped_reason};{reason}"
                    if result.skipped_reason
                    else reason
                )
            else:
                _atomic_write_text(repo, text)
                result.wrote_repo = True
                result.repo_path = str(repo)
        except OSError as exc:
            logger.warning("repo soft-mirror signals write failed: %s", exc)
            if result.skipped_reason:
                result.skipped_reason += f";repo:{exc}"
            else:
                result.skipped_reason = f"repo:{exc}"

    return result


def try_write_signals_multi_dest(
    payload: Mapping[str, Any],
    **kwargs: Any,
) -> MultiDestWriteResult:
    """Like ``write_signals_multi_dest`` but logs and skips on authority failure."""
    try:
        return write_signals_multi_dest(payload, **kwargs)
    except AuthorityValidationError as exc:
        logger.error(
            "Refusing signals multi-dest write (authority gate): %s",
            exc,
        )
        return MultiDestWriteResult(skipped_reason=str(exc))


def serialize_json_payload(payload: Mapping[str, Any] | Any) -> str:
    """Canonical JSON body for non-authority multi-dest fan-out (alerts, etc.)."""
    try:
        from src.backtest.metrics import _json_serializer as _default
    except Exception:  # noqa: BLE001
        _default = str  # type: ignore[assignment]
    if isinstance(payload, Mapping):
        body: Any = dict(payload)
    else:
        body = payload
    return json.dumps(body, indent=2, default=_default) + "\n"


def write_json_multi_dest(
    payload: Mapping[str, Any] | Any,
    *,
    public_path: Path | str | None = None,
    private_path: Path | str | None = None,
    repo_path: Path | str | None = None,
    soft_mirror_repo: bool = True,
    repo_filename: str | None = None,
) -> MultiDestWriteResult:
    """Serialize once and fan-out same bytes (no authority gate).

    Batch HN: alerts.json and other operator JSON need the same 0o644 multi-dest
    contract as signals so health dual-write cannot leave repo public/data stale
    or sticky 0600 under Caddy.
    """
    text = serialize_json_payload(payload)
    result = MultiDestWriteResult()

    pub = Path(public_path) if public_path is not None else None
    priv = Path(private_path) if private_path is not None else None
    auto_repo = soft_mirror_repo and repo_path is None
    if auto_repo and os.environ.get("PYTEST_CURRENT_TEST"):
        auto_repo = False
    if repo_path is not None:
        repo = Path(repo_path)
    elif auto_repo:
        fname = repo_filename
        if fname is None and pub is not None:
            fname = pub.name
        repo = default_repo_public_data_path(fname or "artifact.json")
    else:
        repo = None

    if pub is not None:
        if _should_skip_production_ssot_write(pub):
            logger.warning(
                "json multi-dest skipped production public under pytest: %s",
                pub,
            )
            result.skipped_reason = f"public:pytest-ssot-guard:{pub}"
        else:
            _atomic_write_text(pub, text)
            result.wrote_public = True
            result.public_path = str(pub)

    if priv is not None:
        try:
            if pub is None or priv.resolve() != pub.resolve():
                if _should_skip_production_ssot_write(priv):
                    logger.warning(
                        "json multi-dest skipped production private under pytest: %s",
                        priv,
                    )
                    reason = f"private:pytest-ssot-guard:{priv}"
                    result.skipped_reason = (
                        f"{result.skipped_reason};{reason}"
                        if result.skipped_reason
                        else reason
                    )
                else:
                    _atomic_write_text(priv, text)
                    result.wrote_private = True
                    result.private_path = str(priv)
        except OSError as exc:
            logger.warning("private multi-dest write failed: %s", exc)
            result.skipped_reason = f"private:{exc}"

    if soft_mirror_repo and repo is not None:
        try:
            if pub is not None and repo.resolve() == pub.resolve():
                result.wrote_repo = False
            elif priv is not None and repo.resolve() == priv.resolve():
                result.wrote_repo = False
            elif _should_skip_production_ssot_write(repo):
                logger.warning(
                    "json multi-dest skipped production repo soft-mirror under pytest: %s",
                    repo,
                )
                reason = f"repo:pytest-ssot-guard:{repo}"
                result.skipped_reason = (
                    f"{result.skipped_reason};{reason}"
                    if result.skipped_reason
                    else reason
                )
            else:
                _atomic_write_text(repo, text)
                result.wrote_repo = True
                result.repo_path = str(repo)
        except OSError as exc:
            logger.warning("repo soft-mirror multi-dest write failed: %s", exc)
            if result.skipped_reason:
                result.skipped_reason += f";repo:{exc}"
            else:
                result.skipped_reason = f"repo:{exc}"

    return result
