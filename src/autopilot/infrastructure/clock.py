"""Time sources.

Time is injected rather than read directly so that the circuit breaker, the
metrics window, and anything else with a clock dependency can be tested
deterministically instead of with ``sleep``.
"""

from __future__ import annotations

import time
from datetime import UTC, datetime, timedelta


class SystemClock:
    """Real wall-clock and monotonic time.

    Implements the :class:`~autopilot.domain.interfaces.Clock` protocol.
    """

    def now(self) -> datetime:
        """Return the current UTC time."""
        return datetime.now(UTC)

    def monotonic(self) -> float:
        """Return a monotonic timer value in seconds.

        Monotonic rather than wall-clock, so NTP corrections cannot make a
        rolling window appear to move backwards.
        """
        return time.monotonic()


class FrozenClock:
    """A manually advanced clock for tests and simulations.

    Implements the :class:`~autopilot.domain.interfaces.Clock` protocol.
    """

    def __init__(self, start: datetime | None = None, monotonic_start: float = 0.0) -> None:
        """Create a clock pinned to ``start``.

        Args:
            start: Initial wall-clock time; defaults to the Unix epoch in UTC.
            monotonic_start: Initial monotonic reading in seconds.
        """
        self._now = start or datetime(1970, 1, 1, tzinfo=UTC)
        self._monotonic = monotonic_start

    def now(self) -> datetime:
        """Return the pinned UTC time."""
        return self._now

    def monotonic(self) -> float:
        """Return the pinned monotonic reading."""
        return self._monotonic

    def advance(self, seconds: float) -> None:
        """Move both time sources forward by ``seconds``."""
        self._monotonic += seconds
        self._now += timedelta(seconds=seconds)
