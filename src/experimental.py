"""
Experimental feature flags for portfolio-lab.

ML features (torch/xgboost) consume ~140MB RAM and cause OOM in full test suites.
They are DISABLED by default. Set PORTFOLIO_LAB_ENABLE_ML=1 to enable.

Usage:
    from src.experimental import ML_ENABLED, require_ml

    if not ML_ENABLED:
        return stub_result()

    # or as a decorator:
    @require_ml
    def train_model():
        import torch  # safe — only reached when ML_ENABLED
        ...
"""

import os


def _is_ml_enabled() -> bool:
    """Check if ML features should be loaded. Default: OFF."""
    return os.environ.get("PORTFOLIO_LAB_ENABLE_ML", "0") == "1"


ML_ENABLED: bool = _is_ml_enabled()


def require_ml(func):
    """Decorator: skip function if ML is disabled, return None."""
    def wrapper(*args, **kwargs):
        if not ML_ENABLED:
            return None
        return func(*args, **kwargs)
    wrapper.__name__ = func.__name__
    wrapper.__doc__ = func.__doc__
    return wrapper


def ml_import_error(module_name: str) -> ImportError:
    """Standard error message when ML imports are blocked."""
    return ImportError(
        f"{module_name} requires ML libraries (torch/xgboost) which are disabled by default. "
        f"Set PORTFOLIO_LAB_ENABLE_ML=1 to enable."
    )
