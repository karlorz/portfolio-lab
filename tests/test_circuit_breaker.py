"""
Tests for the broker circuit breaker wrapper (``src.broker.circuit_breaker``).

pybreaker is mocked so the tests are fully hermetic and do not require the
real library to be installed.
"""
import time
from unittest.mock import MagicMock, patch, ANY

import pytest

from src.broker.circuit_breaker import (
    PYBREAKER_AVAILABLE,
    BrokerCircuitBreaker,
    BrokerError,
    CircuitBreakerError,
    broker_breaker,
    get_circuit_state,
    fallback_positions,
    fallback_order,
)


# =====================================================================
# Fixture: create a fresh BrokerCircuitBreaker with a mocked pybreaker
# =====================================================================

@pytest.fixture
def mock_pybreaker():
    """Patch pybreaker on the circuit_breaker module so a fresh
    BrokerCircuitBreaker uses a controlled mock instead of the real lib."""
    with patch("src.broker.circuit_breaker.pybreaker") as mpb:
        with patch("src.broker.circuit_breaker.PYBREAKER_AVAILABLE", True):
            # Simulate a pybreaker.CircuitBreaker instance
            fake_breaker = MagicMock()
            fake_breaker.current_state = "closed"
            fake_breaker.fail_counter = 0

            # state objects with .name for listener
            fake_closed = MagicMock()
            fake_closed.name = "closed"
            fake_open = MagicMock()
            fake_open.name = "open"
            fake_half = MagicMock()
            fake_half.name = "half-open"

            fake_breaker.state = fake_closed
            mpb.CircuitBreaker.return_value = fake_breaker
            mpb.CircuitBreakerError = CircuitBreakerError

            yield mpb, fake_breaker


@pytest.fixture
def cb(mock_pybreaker):
    """Return a fresh BrokerCircuitBreaker wired to a mocked pybreaker."""
    mpb, fake_breaker = mock_pybreaker
    return BrokerCircuitBreaker()


# =====================================================================
# Tests
# =====================================================================

class TestBrokerCircuitBreaker:
    """Tests for BrokerCircuitBreaker wrapping pybreaker."""

    # -- 1. Initial state ----------------------------------------------

    def test_starts_closed(self, cb: BrokerCircuitBreaker):
        """The circuit breaker starts in the 'closed' state."""
        assert cb.state == "closed"

    # -- 2. Transition to open after fail_max consecutive failures -----

    def test_transitions_to_open_after_fail_max_failures(
        self, cb: BrokerCircuitBreaker, mock_pybreaker
    ):
        """After fail_max consecutive failures via .call(), state becomes open."""
        mpb, fake_breaker = mock_pybreaker

        def _fail(*args, **kwargs):
            raise BrokerError("boom")

        # Simulate the breaker tripping on the 3rd failure.
        # pybreaker raises CircuitBreakerError on the 4th call.
        fake_breaker.call.side_effect = [
            BrokerError("fail1"),
            BrokerError("fail2"),
            BrokerError("fail3"),
            CircuitBreakerError("Open!"),
        ]
        fake_breaker.current_state = "open"
        fake_breaker.state.name = "open"

        # First three calls: each raises BrokerError through pybreaker.
        for i in range(3):
            with pytest.raises(BrokerError):
                cb.call(_fail)
            assert fake_breaker.call.call_count == i + 1

        # Fourth call: circuit is open -> CircuitBreakerError
        fake_breaker.current_state = "open"
        with pytest.raises(CircuitBreakerError):
            cb.call(lambda: "ok")

        # The breaker wrapper should report open state.
        assert cb.state == "open"
        assert cb.fail_count == cb.fail_max

    # -- 3. Returns to half-open after reset_timeout -------------------

    def test_half_open_after_reset_timeout(
        self, cb: BrokerCircuitBreaker, mock_pybreaker
    ):
        """When pybreaker transitions to half-open, state reflects it."""
        mpb, fake_breaker = mock_pybreaker

        # Simulate half-open state after timeout
        fake_breaker.current_state = "half-open"
        fake_breaker.state.name = "half-open"
        fake_breaker.call.side_effect = None
        fake_breaker.call.return_value = "recovered"

        # Verify state
        assert cb.state == "half-open"

        # A call should succeed (half-open allows trial requests)
        result = cb.call(lambda: "recovered")
        assert result == "recovered"

    # -- 4. Fallback functions -----------------------------------------

    def test_fallback_positions_returns_empty_list(self):
        """fallback_positions() returns [] when circuit is open."""
        assert fallback_positions() == []

    def test_fallback_order_returns_none(self):
        """fallback_order() returns None when circuit is open."""
        assert fallback_order() is None

    # -- 5. get_circuit_state() ----------------------------------------

    def test_get_circuit_state_returns_correct_dict(
        self, cb: BrokerCircuitBreaker
    ):
        """get_circuit_state() returns the expected shape."""
        state = get_circuit_state()
        assert isinstance(state, dict)
        assert "state" in state
        assert "fail_count" in state
        assert "reset_timeout" in state
        assert isinstance(state["fail_count"], int)
        assert isinstance(state["reset_timeout"], int)

    # -- Additional: decorator interface -------------------------------

    def test_decorator_returns_none_on_open(
        self, cb: BrokerCircuitBreaker, mock_pybreaker
    ):
        """The @broker_breaker decorator returns None when circuit is open."""
        mpb, fake_breaker = mock_pybreaker

        fake_breaker.current_state = "open"
        fake_breaker.state.name = "open"
        fake_breaker.call.side_effect = CircuitBreakerError("Open!")

        @cb
        def my_func():
            return "should-not-reach"

        result = my_func()
        assert result is None

    def test_decorator_catches_broker_error(
        self, cb: BrokerCircuitBreaker, mock_pybreaker
    ):
        """The @broker_breaker decorator converts BrokerError to None."""
        mpb, fake_breaker = mock_pybreaker

        fake_breaker.current_state = "closed"
        fake_breaker.state.name = "closed"
        fake_breaker.call.side_effect = BrokerError("fail")

        @cb
        def my_func():
            raise BrokerError("fail")

        result = my_func()
        assert result is None

    # -- Additional: config via env vars --------------------------------

    def test_config_from_env(
        self, mock_pybreaker
    ):
        """Constructor reads BROKER_CIRCUIT_FAIL_MAX and RESET_TIMEOUT from env."""
        import os
        os.environ["BROKER_CIRCUIT_FAIL_MAX"] = "7"
        os.environ["BROKER_CIRCUIT_RESET_TIMEOUT"] = "120"
        try:
            cb = BrokerCircuitBreaker()
            assert cb.fail_max == 7
            assert cb.reset_timeout == 120
        finally:
            del os.environ["BROKER_CIRCUIT_FAIL_MAX"]
            del os.environ["BROKER_CIRCUIT_RESET_TIMEOUT"]

    # -- Additional: no-op mode when pybreaker unavailable -------------

    def test_noop_mode(self):
        """When pybreaker is unavailable, calls pass through transparently."""
        with patch("src.broker.circuit_breaker.PYBREAKER_AVAILABLE", False):
            cb = BrokerCircuitBreaker()
            assert cb.state == "closed"

            # call() should just invoke the function
            result = cb.call(lambda: 42)
            assert result == 42

            # Decorator should be transparent
            @cb
            def my_func():
                return "hello"

            assert my_func() == "hello"


class TestBrokerCircuitBreakerSingleton:
    """Tests against the global ``broker_breaker`` singleton."""

    def test_singleton_is_instance(self):
        """broker_breaker is a BrokerCircuitBreaker instance."""
        assert isinstance(broker_breaker, BrokerCircuitBreaker)

    def test_singleton_get_circuit_state_is_callable(self):
        """get_circuit_state on singleton returns a dict."""
        state = get_circuit_state()
        assert isinstance(state, dict)
