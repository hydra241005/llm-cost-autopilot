"""Circuit breaker with a rolling failure window.

A provider that is failing should be skipped quickly rather than absorbing a
full timeout budget on every request. The breaker opens only when a provider is
*both* meaningfully broken (at least ``failure_threshold`` failures) *and*
broken at a meaningful rate (at least ``failure_rate`` of recent calls) inside a
rolling window. Requiring both prevents a burst of five failures inside a
thousand healthy calls from cutting off a working provider.

State machine::

    CLOSED ──(threshold and rate exceeded)──▶ OPEN
      ▲                                        │
      │                                  (cooldown elapsed)
      │                                        ▼
      └───────(trial call succeeds)───── HALF_OPEN ──(trial call fails)──▶ OPEN

In ``HALF_OPEN`` exactly one trial call is admitted. Its result decides whether
the breaker closes or re-opens for another cooldown.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

from autopilot.domain.enums import BreakerState
from autopilot.domain.interfaces import Clock
from autopilot.infrastructure.clock import SystemClock
from autopilot.infrastructure.observability.logging import get_logger

_log = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class BreakerPolicy:
    """Thresholds governing when a circuit opens and how it recovers.

    Attributes:
        failure_threshold: Minimum failures in the window before opening.
        failure_rate: Minimum fraction of failed calls before opening.
        window_s: Width of the rolling observation window in seconds.
        cooldown_s: How long the circuit stays open before a trial call.
    """

    failure_threshold: int = 5
    failure_rate: float = 0.5
    window_s: float = 30.0
    cooldown_s: float = 20.0


@dataclass(slots=True)
class _Window:
    """Timestamped outcomes inside the rolling window."""

    successes: deque[float] = field(default_factory=deque)
    failures: deque[float] = field(default_factory=deque)

    def prune(self, cutoff: float) -> None:
        """Drop observations older than ``cutoff``."""
        for series in (self.successes, self.failures):
            while series and series[0] < cutoff:
                series.popleft()

    def clear(self) -> None:
        """Forget every observation."""
        self.successes.clear()
        self.failures.clear()


class CircuitBreaker:
    """Tracks health for one key (a provider or a model) and gates calls.

    The breaker is a pure state machine over injected time; it never sleeps and
    never performs I/O, so its behaviour is fully testable with a frozen clock.
    """

    def __init__(
        self,
        key: str,
        policy: BreakerPolicy | None = None,
        *,
        clock: Clock | None = None,
    ) -> None:
        """Create a breaker.

        Args:
            key: Identifier this breaker guards, used for log context.
            policy: Thresholds; defaults to the blueprint values.
            clock: Time source, injected by tests.
        """
        self.key = key
        self._policy = policy or BreakerPolicy()
        self._clock = clock or SystemClock()
        self._window = _Window()
        self._state = BreakerState.CLOSED
        self._opened_at_monotonic: float | None = None
        self._trial_in_flight = False

    @property
    def policy(self) -> BreakerPolicy:
        """Thresholds governing this breaker."""
        return self._policy

    @property
    def state(self) -> BreakerState:
        """Current state, after applying any elapsed cooldown."""
        self._maybe_half_open()
        return self._state

    @property
    def failure_count(self) -> int:
        """Failures currently inside the rolling window."""
        self._prune()
        return len(self._window.failures)

    def allows(self) -> bool:
        """Return whether a call may proceed right now.

        An open circuit rejects calls until its cooldown elapses. A half-open
        circuit admits exactly one trial call.
        """
        state = self.state
        if state is BreakerState.CLOSED:
            return True
        if state is BreakerState.OPEN:
            return False
        if self._trial_in_flight:
            return False
        self._trial_in_flight = True
        return True

    def record_success(self) -> None:
        """Record a successful call, closing a half-open circuit."""
        now = self._clock.monotonic()
        self._prune()
        self._window.successes.append(now)
        if self._state is BreakerState.HALF_OPEN:
            self._close()

    def record_failure(self) -> None:
        """Record a failed call, opening the circuit if thresholds are breached."""
        now = self._clock.monotonic()
        self._prune()
        self._window.failures.append(now)

        if self._state is BreakerState.HALF_OPEN:
            # The trial call failed; serve another full cooldown.
            self._open()
            return
        if self._state is BreakerState.CLOSED and self._should_open():
            self._open()

    def reset(self) -> None:
        """Force the circuit closed and forget observed history."""
        self._close()

    def _should_open(self) -> bool:
        """Return whether both the count and rate thresholds are breached."""
        failures = len(self._window.failures)
        total = failures + len(self._window.successes)
        if failures < self._policy.failure_threshold or total == 0:
            return False
        return (failures / total) >= self._policy.failure_rate

    def _open(self) -> None:
        """Transition to OPEN and start the cooldown."""
        self._state = BreakerState.OPEN
        self._opened_at_monotonic = self._clock.monotonic()
        self._trial_in_flight = False
        _log.warning(
            "breaker.opened",
            key=self.key,
            failures=len(self._window.failures),
            cooldown_s=self._policy.cooldown_s,
        )

    def _close(self) -> None:
        """Transition to CLOSED and clear the window."""
        was = self._state
        self._state = BreakerState.CLOSED
        self._opened_at_monotonic = None
        self._trial_in_flight = False
        self._window.clear()
        if was is not BreakerState.CLOSED:
            _log.info("breaker.closed", key=self.key, previous_state=was.value)

    def _maybe_half_open(self) -> None:
        """Move an open circuit to HALF_OPEN once its cooldown has elapsed."""
        if self._state is not BreakerState.OPEN or self._opened_at_monotonic is None:
            return
        if self._clock.monotonic() - self._opened_at_monotonic >= self._policy.cooldown_s:
            self._state = BreakerState.HALF_OPEN
            self._trial_in_flight = False
            _log.info("breaker.half_open", key=self.key)

    def _prune(self) -> None:
        """Drop observations that have aged out of the window."""
        self._window.prune(self._clock.monotonic() - self._policy.window_s)
