"""Compatibility shim: FRED readiness checks moved to ``src.data.fred_readiness``.

Keeps the ``src.monitor.fred_readiness`` import path importable for one-way
consumers (dashboard / monitor / tests) and preserves ``python -m`` usage.
"""

from src.data.fred_readiness import (  # noqa: F401
    FRED_READINESS_SCHEMA_VERSION,
    assess_fred_readiness,
    main,
    resolve_fred_operating_mode,
)

if __name__ == "__main__":
    from src.utils.log_config import configure_logging

    configure_logging()
    raise SystemExit(main())
