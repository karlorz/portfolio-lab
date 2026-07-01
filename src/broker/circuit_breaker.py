"""
Circuit breaker for broker operations using pybreaker.

Provides a global singleton ``broker_breaker`` that wraps broker API calls,
short-circuiting them when a configurable failure threshold is reached.

The circuit breaker is *optional* -- if pybreaker is not installed, all calls
pass through transparently (no-op mode).

Environment variables
---------------------
BROKER_CIRCUIT_FAIL_MAX : int
    Consecutive failures before opening the circuit (default: 3)
BROKER_CIRCUIT_RESET_TIMEOUT : int
    Seconds to wait before transitioning from open to half-open (default: 60)
"""

import functools
import logging
import os
from typing import Any, Callable, Dict, Optional, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")

# ---------------------------------------------------------------------------
# Optional pybreaker import -- graceful fallback when the library is absent
# ---------------------------------------------------------------------------
try:
    import pybreaker  # type: ignore[import-untyped]

    PYBREAKER_AVAILABLE = True
    CircuitBreakerError = pybreaker.CircuitBreakerError
except ImportError:
    PYBREAKER_AVAILABLE = False
    pybreaker = None  # type: ignore[assignment]

    class CircuitBreakerError(Exception):
        """Stub -- never raised when pybreaker is not installed."""


# ---------------------------------------------------------------------------
# Exception that the inner error handler raises to signal a counted failure
# ---------------------------------------------------------------------------
class BrokerError(Exception):
    """Raised when a broker API call fails (network, API error, etc.).

    This exception is caught by ``BrokerCircuitBreaker`` and counted as a
    failure against the circuit breaker threshold.  Callers of the broker
    method should catch this and return their normal fallback value (e.g.
    ``None`` for submit_order, ``[]`` for get_positions) so the public
    interface remains unchanged.
    """


def _state_name(state: Any) -> str:
    """Return a normalized pybreaker state name for logging and branching."""
    return str(state.name if hasattr(state, "name") else state).lower()


def _rollback_live_ramp_for_open_circuit(old_state: str, new_state: str) -> Optional[Dict[str, Any]]:
    """Roll back live-ramp exposure when the broker circuit opens.

    The import is intentionally lazy because ``alpaca.py`` imports this module.
    """
    try:
        from src.broker.alpaca import LiveTransitionManager

        manager = LiveTransitionManager()
        result = manager.rollback(f"broker circuit breaker opened: {old_state} -> {new_state}")
    except (ImportError, OSError, ValueError, TypeError, RuntimeError) as exc:
        logger.warning("Broker circuit breaker rollback hook failed: %s", exc)
        return None

    if result.get("success"):
        logger.warning("Live ramp rollback triggered by broker circuit breaker open: %s", result)
    else:
        logger.info(
            "Live ramp rollback skipped after broker circuit breaker open: %s",
            result.get("reason"),
        )
    return result


# ---------------------------------------------------------------------------
# State-change listener
# ---------------------------------------------------------------------------
class _StateChangeListener:
    """Logs circuit breaker state transitions for observability."""

    # These hooks are called unconditionally by pybreaker; provide no-op
    # stubs so the listener duck-types correctly.
    @staticmethod
    def before_call(*args: Any, **kwargs: Any) -> None:
        pass

    @staticmethod
    def failure(*args: Any, **kwargs: Any) -> None:
        pass

    @staticmethod
    def success(*args: Any, **kwargs: Any) -> None:
        pass

    def state_change(
        self,
        cb: Any,
        old_state: Any,
        new_state: Any,
    ) -> None:
        old_name = _state_name(old_state)
        new_name = _state_name(new_state)
        logger.info(
            "Broker circuit breaker state change: %s -> %s",
            old_name,
            new_name,
        )
        if new_name != "open" or old_name == "open":
            return
        try:
            _rollback_live_ramp_for_open_circuit(old_name, new_name)
        except Exception as exc:
            logger.warning("Broker circuit breaker rollback hook failed: %s", exc)


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------
class BrokerCircuitBreaker:
    """Wraps pybreaker's ``CircuitBreaker`` for broker API calls.

    Usage as a decorator (short-circuits to ``None`` when open)::

        @broker_breaker
        def submit_order(self, order):
            ...

    Usage with ``.call()`` (caller is responsible for the fallback)::

        def get_positions(self):
            try:
                return broker_breaker.call(self._impl)
            except (CircuitBreakerError, BrokerError):
                return []
    """

    def __init__(self) -> None:
        self.fail_max: int = int(
            os.environ.get("BROKER_CIRCUIT_FAIL_MAX", "3")
        )
        self.reset_timeout: int = int(
            os.environ.get("BROKER_CIRCUIT_RESET_TIMEOUT", "60")
        )
        # Our own counter as a fallback when pybreaker attribute is
        # inaccessible or when pybreaker is unavailable.
        self._our_failure_count: int = 0

        if PYBREAKER_AVAILABLE:
            self._breaker = pybreaker.CircuitBreaker(
                fail_max=self.fail_max,
                reset_timeout=self.reset_timeout,
            )
            self._breaker.add_listener(_StateChangeListener())
        else:
            self._breaker = None

    # -- decorator interface ----------------------------------------------

    def __call__(
        self, func: Callable[..., T]
    ) -> Callable[..., Optional[T]]:
        """Decorate *func* so it is called through the circuit breaker.

        When the circuit is **open** the decorated call returns ``None``
        without executing the body (short-circuit fallback).
        When the circuit is **closed** or **half-open** the original function
        runs; exceptions raised by the function are counted as failures by
        pybreaker.  ``BrokerError`` is converted to ``None`` for callers.
        """
        if not PYBREAKER_AVAILABLE:
            return func

        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Optional[T]:
            return self._call_decorated(func, *args, **kwargs)

        return wrapper

    # -- explicit call interface ------------------------------------------

    def call(self, func: Callable[..., T], *args: Any, **kwargs: Any) -> T:
        """Call *func* through the circuit breaker.

        Raises
        ------
        CircuitBreakerError
            When the circuit is open (caller should handle fallback).
        BrokerError
            When the wrapped function raised a broker error (already counted
            as a failure by pybreaker).
        Exception
            Unexpected exceptions are re-raised unchanged.
        """
        if not PYBREAKER_AVAILABLE or self._breaker is None:
            return func(*args, **kwargs)

        try:
            result = self._breaker.call(func, *args, **kwargs)
            self._our_failure_count = 0
            return result
        except pybreaker.CircuitBreakerError:
            self._our_failure_count = self.fail_max
            raise
        except Exception:
            self._our_failure_count += 1
            raise

    # -- internal: decorator helper --------------------------------------

    def _call_decorated(
        self, func: Callable[..., T], *args: Any, **kwargs: Any
    ) -> Optional[T]:
        """Call through breaker; convert open-circuit and BrokerError to None.

        This is the fallback bridge for the ``@broker_breaker`` decorator.
        """
        try:
            return self.call(func, *args, **kwargs)
        except CircuitBreakerError:
            logger.warning(
                "Circuit breaker is OPEN -- call to %s blocked (fallback)",
                func.__name__,
            )
            return None
        except BrokerError:
            # Already counted by pybreaker as a failure; return the safe
            # fallback value so callers see the same interface.
            return None

    # -- state introspection ----------------------------------------------

    @property
    def state(self) -> str:
        """Current breaker state: ``"closed"``, ``"open"``, or ``"half-open"``.

        Returns ``"closed"`` when pybreaker is unavailable (no-op mode).
        """
        if not PYBREAKER_AVAILABLE or self._breaker is None:
            return "closed"
        return self._breaker.current_state

    @property
    def fail_count(self) -> int:
        """Number of consecutive failures tracked by the breaker."""
        if PYBREAKER_AVAILABLE and self._breaker is not None:
            # Prefer our own counter since we increment it for every
            # exception we observe; pybreaker's internal  fail_counter
            # is authoritative when available but may not be synchronised
            # if the breaker was reset externally.
            return max(
                self._our_failure_count,
                getattr(self._breaker, "fail_counter", 0),
            )
        return self._our_failure_count

    def reset(self) -> None:
        """Reset the circuit breaker to closed state and clear count.

        Intended for test teardown so state does not leak across tests.
        """
        self._our_failure_count = 0
        if PYBREAKER_AVAILABLE and self._breaker is not None:
            try:
                self._breaker.close()
            except Exception as exc:
                # close() may fail in edge cases; reset anyway
                logger.warning(
                    "Failed to close broker circuit breaker during reset: %s: %s",
                    type(exc).__name__,
                    exc,
                )


# ---------------------------------------------------------------------------
# Fallback helpers
# ---------------------------------------------------------------------------

def fallback_positions() -> list:
    """Return empty list for get_positions when circuit is open."""
    return []


def fallback_order() -> None:
    """Return None for submit_order when circuit is open."""
    return None


# ---------------------------------------------------------------------------
# Global singleton + convenience accessor
# ---------------------------------------------------------------------------

broker_breaker = BrokerCircuitBreaker()


def get_circuit_state() -> Dict[str, Any]:
    """Return current circuit breaker state for dashboard integration.

    Returns
    -------
    dict
        ``{"state": ..., "fail_count": ..., "reset_timeout": ...}``
    """
    return {
        "state": broker_breaker.state,
        "fail_count": broker_breaker.fail_count,
        "reset_timeout": broker_breaker.reset_timeout,
    }
